# Copyright 2024 The Langfun Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for REST models."""

from typing import Any
import unittest
from unittest import mock
import langfun.core as lf
from langfun.core.llms import rest
import pyglove as pg
import requests


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'content': [(
          f'hello with temperature={json.get("temperature")}, '
          f'top_k={json.get("top_k")}, '
          f'top_p={json.get("top_p")}, '
          f'max_tokens={json.get("max_tokens")}, '
          f'stop={json.get("stop_sequences")}.'
      )],
  }).encode()
  return response


def mock_requests_post_exception(error):
  def _mock_requests(url: str, json: dict[str, Any], **kwargs):
    del url, json, kwargs
    raise error
  return _mock_requests


def mock_requests_post_error(status_code, error_type, error_message):
  def _mock_requests(url: str, json: dict[str, Any], **kwargs):
    del url, json, kwargs
    response = requests.Response()
    response.status_code = status_code
    response._content = pg.to_json_str(
        {
            'error': {
                'type': error_type,
                'message': error_message,
            }
        }
    ).encode()
    return response

  return _mock_requests


class RestTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._lm = rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(
            model='test-model',
            prompt=x.text,
            temperature=0.0,
            top_k=0.1,
            top_p=0.2,
            stop_sequences=['\n'],
            max_tokens=4096,
        ),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]),
        headers=dict(api_key='fake_key'),
    )

  def test_call(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      self.assertEqual(self._lm.model_id, 'unknown')
      response = self._lm(
          'hello', temperature=0.0, top_k=0.1, top_p=0.2, stop=['\n'])
      self.assertEqual(
          response.text,
          (
              'hello with temperature=0.0, top_k=0.1, top_p=0.2, '
              "max_tokens=4096, stop=['\\n']."
          ),
      )
      self.assertIsInstance(response.usage, lf.UsageNotAvailable)

  def test_call_errors(self):
    for status_code, error_type, error_message in [
        (429, 'rate_limit', 'Rate limit exceeded.'),
        (529, 'service_unavailable', 'Service unavailable.'),
        (500, 'bad_request', 'Bad request.'),
    ]:
      with mock.patch('requests.Session.post') as mock_mm_request:
        mock_mm_request.side_effect = mock_requests_post_error(
            status_code, error_type, error_message
        )
        with self.assertRaisesRegex(
            Exception, f'.*{status_code}: .*{error_message}'
        ):
          self._lm('hello', max_attempts=1)

    for error, expected_lm_error_cls, expected_lm_error_msg in [
        (
            requests.exceptions.Timeout('Timeout.'),
            lf.TemporaryLMError,
            'Timeout.',
        ),
        (
            requests.exceptions.ReadTimeout('Read timeout.'),
            lf.TemporaryLMError,
            'Read timeout.',
        ),
        (
            requests.exceptions.ConnectTimeout('Connect timeout.'),
            lf.TemporaryLMError,
            'Connect timeout.',
        ),
        (
            TimeoutError('Timeout error.'),
            lf.TemporaryLMError,
            'Timeout error.',
        ),
        (
            requests.exceptions.ConnectionError('REJECTED_CLIENT_THROTTLED'),
            lf.TemporaryLMError,
            'REJECTED_CLIENT_THROTTLED',
        ),
        (
            requests.exceptions.ConnectionError(
                "('Connection aborted.', TimeoutError('The write operation"
                " timed out'))"
            ),
            lf.TemporaryLMError,
            'Connection aborted',
        ),
        (
            requests.exceptions.ConnectionError('Connection error.'),
            lf.LMError,
            'Connection error.',
        ),
        (
            ConnectionError('Connection error.'),
            lf.LMError,
            'Connection error.',
        ),
    ]:
      with mock.patch('requests.Session.post') as mock_post:
        mock_post.side_effect = mock_requests_post_exception(error)
        with self.assertRaisesRegex(
            expected_lm_error_cls, expected_lm_error_msg
        ):
          self._lm._sample_single(lf.UserMessage('hello'))


class ContentFilteringErrorTest(unittest.TestCase):
  """TDD tests for content filtering classification at the REST layer."""

  def setUp(self):
    super().setUp()
    self._lm = rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(
            model='test-model',
            prompt=x.text,
            max_tokens=4096,
        ),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        headers=dict(api_key='fake_key'),
    )

  def test_content_filtered_error_is_lm_error(self):
    """ContentFilteredError should be a subclass of LMError."""
    self.assertTrue(issubclass(lf.ContentFilteredError, lf.LMError))

  def test_content_filtered_error_not_retryable(self):
    """ContentFilteredError should NOT be retryable (raw retry won't help)."""
    self.assertFalse(issubclass(lf.ContentFilteredError, lf.RetryableLMError))

  def test_content_filtered_error_not_context_limit(self):
    """ContentFilteredError != ContextLimitError (distinct failure modes)."""
    self.assertFalse(issubclass(lf.ContentFilteredError, lf.ContextLimitError))
    self.assertFalse(issubclass(lf.ContextLimitError, lf.ContentFilteredError))

  def test_content_filtered_error_instantiation(self):
    """ContentFilteredError should be instantiable with a message."""
    error = lf.ContentFilteredError('400: Output blocked by content filter')
    self.assertIn('blocked', str(error))

  def test_error_classification(self):
    """Verify _error() maps (status_code, message) to the correct error class."""
    cases = [
        # (status_code, message, expected_class, description)
        # Content-filtered 400s.
        (
            400,
            'Output blocked by content filtering policy',
            lf.ContentFilteredError,
            'Claude content filtering',
        ),
        (
            400,
            'Request blocked due to SAFETY reasons',
            lf.ContentFilteredError,
            'Gemini safety block',
        ),
        (
            400,
            'content_filter triggered for this request',
            lf.ContentFilteredError,
            'OpenAI content_filter',
        ),
        (
            400,
            'blocked by safety filter',
            lf.ContentFilteredError,
            'Generic safety filter',
        ),
        # Non-filtered 400.
        (400, 'Invalid request format', lf.LMError, '400 non-filter'),
        # Other status codes must NOT be reclassified.
        (429, 'Rate limit exceeded', lf.RateLimitError, '429 rate limit'),
        (500, 'Internal server error', lf.TemporaryLMError, '500 server'),
        (503, 'Service unavailable', lf.TemporaryLMError, '503 unavailable'),
        # Non-400 with filter-like text must NOT be ContentFilteredError.
        (
            403,
            'Output blocked by content filtering policy',
            lf.LMError,
            '403 with filter text',
        ),
    ]
    for status_code, message, expected_cls, desc in cases:
      with self.subTest(desc=desc, status_code=status_code):
        error = self._lm._error(status_code, message)
        self.assertIsInstance(error, expected_cls)
        if (
            expected_cls is lf.LMError
            and expected_cls is not lf.ContentFilteredError
        ):
          self.assertNotIsInstance(error, lf.ContentFilteredError)

  def test_sample_raises_content_filtered_error_on_400(self):
    """_sample_single should raise ContentFilteredError for filtered 400."""
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_requests_post_error(
          400,
          'invalid_request_error',
          'Output blocked by content filtering policy',
      )
      with self.assertRaises(lf.ContentFilteredError):
        self._lm._sample_single(lf.UserMessage('test'))

  def test_sample_raises_plain_lm_error_on_non_filter_400(self):
    """_sample_single should raise LMError for non-filter 400."""
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_requests_post_error(
          400,
          'invalid_request_error',
          'Malformed JSON in request body',
      )
      with self.assertRaises(lf.LMError) as ctx:
        self._lm._sample_single(lf.UserMessage('test'))
      self.assertNotIsInstance(ctx.exception, lf.ContentFilteredError)


if __name__ == '__main__':
  unittest.main()
