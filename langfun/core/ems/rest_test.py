# Copyright 2025 The Langfun Authors
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
"""Tests for REST embedding models."""

from typing import Any
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core.ems import rest
import pyglove as pg
import requests


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'embedding': [0.1, 0.2, 0.3],
      'input_text': json.get('text', ''),
  }).encode()
  return response


def mock_requests_post_error(status_code):
  def _mock_requests(url: str, json: dict[str, Any], **kwargs):
    del url, json, kwargs
    response = requests.Response()
    response.status_code = status_code
    response._content = b'error'
    return response
  return _mock_requests


def mock_requests_post_exception(error):
  def _mock_requests(url: str, json: dict[str, Any], **kwargs):
    del url, json, kwargs
    raise error
  return _mock_requests


class RESTEmbeddingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._em = rest.REST(
        api_endpoint='https://fake-embedding-api.com',
        request=lambda msg: dict(text=msg.text),
        result=lambda x: lf.EmbeddingResult(
            embedding=x['embedding'],
        ),
        headers=dict(api_key='fake_key'),
    )

  def test_embed(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      result = self._em('hello')
      self.assertEqual(result.embedding, [0.1, 0.2, 0.3])

  def test_embed_with_message(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      result = self._em(lf.UserMessage('hello'))
      self.assertEqual(result.embedding, [0.1, 0.2, 0.3])

  def test_error_status_codes(self):
    for status_code, expected_cls in [
        (429, lf.RateLimitError),
        (500, lf.TemporaryLMError),
        (502, lf.TemporaryLMError),
        (503, lf.TemporaryLMError),
        (529, lf.TemporaryLMError),
        (499, lf.TemporaryLMError),
        (400, lf.LMError),
    ]:
      with mock.patch('requests.Session.post') as mock_post:
        mock_post.side_effect = mock_requests_post_error(status_code)
        with self.assertRaises(expected_cls):
          self._em._embed(lf.UserMessage('hello'))

  def test_timeout_errors(self):
    for error in [
        requests.exceptions.Timeout('Timeout.'),
        requests.exceptions.ReadTimeout('Read timeout.'),
        requests.exceptions.ConnectTimeout('Connect timeout.'),
        TimeoutError('Timeout error.'),
    ]:
      with mock.patch('requests.Session.post') as mock_post:
        mock_post.side_effect = mock_requests_post_exception(error)
        with self.assertRaises(lf.TemporaryLMError):
          self._em._embed(lf.UserMessage('hello'))

  def test_ssl_error(self):
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_requests_post_exception(
          requests.exceptions.SSLError('SSL error.')
      )
      with self.assertRaises(lf.TemporaryLMError):
        self._em._embed(lf.UserMessage('hello'))

  def test_connection_errors(self):
    for error_msg, expected_cls in [
        ('REJECTED_CLIENT_THROTTLED', lf.TemporaryLMError),
        ('UNREACHABLE_NO_RESPONSE', lf.TemporaryLMError),
        ('UNREACHABLE_ERROR', lf.TemporaryLMError),
        ('Connection reset by peer', lf.TemporaryLMError),
        ('Remote end closed connection', lf.TemporaryLMError),
        ('Connection aborted', lf.TemporaryLMError),
        ('Unknown connection error', lf.LMError),
    ]:
      with mock.patch('requests.Session.post') as mock_post:
        mock_post.side_effect = mock_requests_post_exception(
            requests.exceptions.ConnectionError(error_msg)
        )
        with self.assertRaises(expected_cls):
          self._em._embed(lf.UserMessage('hello'))

  def test_embed_retries_on_temporary_error(self):
    """RED: __call__ should retry on TemporaryLMError using max_attempts."""
    em = rest.REST(
        api_endpoint='https://fake-embedding-api.com',
        request=lambda msg: dict(text=msg.text),
        result=lambda x: lf.EmbeddingResult(embedding=x['embedding']),
        headers=dict(api_key='fake_key'),
        max_attempts=3,
        retry_interval=0,
        exponential_backoff=False,
    )
    call_count = 0
    original_embed = em._embed

    def flaky_embed(message):
      nonlocal call_count
      call_count += 1
      if call_count < 3:
        raise lf.TemporaryLMError('Server error')
      return original_embed(message)

    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      with mock.patch.object(em, '_embed', side_effect=flaky_embed):
        result = em('hello')
        self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
        self.assertEqual(call_count, 3)

  def test_embed_retry_exhaustion_raises(self):
    """RED: After max_attempts, error should propagate."""
    from langfun.core import concurrent as lf_concurrent  # pylint: disable=g-import-not-at-top

    em = rest.REST(
        api_endpoint='https://fake-embedding-api.com',
        request=lambda msg: dict(text=msg.text),
        result=lambda x: lf.EmbeddingResult(embedding=x['embedding']),
        headers=dict(api_key='fake_key'),
        max_attempts=2,
        retry_interval=0,
        exponential_backoff=False,
    )

    def always_fail(message):
      raise lf.TemporaryLMError('Persistent server error')

    with mock.patch.object(em, '_embed', side_effect=always_fail):
      with self.assertRaises(lf_concurrent.RetryError):
        em('hello')

  # ===== ADVERSARIAL / RED TEAM TESTS (CL/918044373) =====

  def test_non_retryable_lm_error_not_retried(self):
    """RED TEAM: Non-RetryableLMError must NOT trigger retry logic.

    LMInputError inherits from LMError, NOT RetryableLMError. The retry
    wrapper in __call__ should let it propagate immediately without any
    retry attempts. If retried, a permanent input error would waste N
    API calls and add unnecessary latency.
    """
    em = rest.REST(
        api_endpoint='https://fake-embedding-api.com',
        request=lambda msg: dict(text=msg.text),
        result=lambda x: lf.EmbeddingResult(embedding=x['embedding']),
        headers=dict(api_key='fake_key'),
        max_attempts=3,
        retry_interval=0,
        exponential_backoff=False,
    )
    call_count = 0

    def non_retryable_fail(_):
      nonlocal call_count
      call_count += 1
      raise lf.LMInputError('Invalid input format')

    with mock.patch.object(em, '_embed', side_effect=non_retryable_fail):
      with self.assertRaises(lf.LMInputError):
        em('hello')

    self.assertEqual(
        call_count, 1,
        f'Non-retryable LMInputError was retried {call_count} times — '
        'should propagate immediately without retry.',
    )

  def test_generic_runtime_error_not_retried(self):
    """RED TEAM: Generic RuntimeError must NOT be retried by __call__.

    The retry wrapper only catches RetryableLMError. Any other exception
    type (RuntimeError, ValueError, etc.) should propagate immediately.
    """
    em = rest.REST(
        api_endpoint='https://fake-embedding-api.com',
        request=lambda msg: dict(text=msg.text),
        result=lambda x: lf.EmbeddingResult(embedding=x['embedding']),
        headers=dict(api_key='fake_key'),
        max_attempts=3,
        retry_interval=0,
        exponential_backoff=False,
    )
    call_count = 0

    def generic_fail(_):
      nonlocal call_count
      call_count += 1
      raise RuntimeError('Unexpected internal error')

    with mock.patch.object(em, '_embed', side_effect=generic_fail):
      with self.assertRaises(RuntimeError):
        em('hello')

    self.assertEqual(
        call_count, 1,
        f'Generic RuntimeError was retried {call_count} times — '
        'should propagate immediately without retry.',
    )


if __name__ == '__main__':
  unittest.main()
