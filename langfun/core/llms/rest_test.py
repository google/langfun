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

import time
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


def mock_streaming_post(
    chunks, delay_per_chunk=0, status_code=200,
):
  """Returns a mock session.post that returns a streaming Response."""
  def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
    del url, json, timeout, stream, kwargs
    response = requests.Response()
    response.status_code = status_code
    response.headers['Content-Type'] = 'application/json'
    def _delayed_iter(chunk_size=1, decode_unicode=False):
      del chunk_size, decode_unicode
      for chunk in chunks:
        if delay_per_chunk > 0:
          time.sleep(delay_per_chunk)
        yield chunk
    response.iter_content = _delayed_iter
    response._content = False
    response.close = lambda: None
    return response
  return _mock_post


class TotalTimeoutTest(unittest.TestCase):
  """Tests for total-request deadline enforcement via stream=True."""

  def _make_lm(self, timeout=120.0):
    return rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(prompt=x.text),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        timeout=timeout,
    )

  def test_total_deadline_fires_on_slow_chunked_response(self):
    """A slow chunked response should hit the total deadline."""
    lm = self._make_lm(timeout=0.3)
    chunks = [b'chunk'] * 10  # 10 x 0.05s = 0.5s > 0.3s deadline
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          chunks, delay_per_chunk=0.05
      )
      with self.assertRaises(lf.TemporaryLMError) as ctx:
        lm._sample_single(lf.UserMessage('hello'))
      self.assertIn('deadline', str(ctx.exception).lower())

  def test_fast_response_unaffected_by_deadline(self):
    """A fast response should complete normally despite a deadline."""
    lm = self._make_lm(timeout=5.0)
    valid_json = pg.to_json_str({'content': ['hello']}).encode()
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post([valid_json])
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, 'hello')

  def test_timeout_none_disables_deadline(self):
    """timeout=None should disable deadline enforcement entirely."""
    lm = self._make_lm(timeout=None)
    valid_json = pg.to_json_str({'content': ['hello']}).encode()
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post([valid_json])
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, 'hello')

  def test_per_operation_timeout_is_tuple(self):
    """_per_operation_timeout should return (connect, read) tuple."""
    for timeout_val, expected in [
        (120.0, (60.0, 120.0)),
        (30.0, (30.0, 30.0)),
        (300.0, (60.0, 300.0)),
    ]:
      with self.subTest(timeout=timeout_val):
        lm = self._make_lm(timeout=timeout_val)
        self.assertEqual(lm._per_operation_timeout, expected)
    # timeout=None case.
    lm = self._make_lm(timeout=None)
    self.assertIsNone(lm._per_operation_timeout)

  def test_stream_true_passed_to_post(self):
    """session.post() should always receive stream=True."""
    lm = self._make_lm(timeout=10.0)
    valid_json = pg.to_json_str({'content': ['hello']}).encode()
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post([valid_json])
      lm._sample_single(lf.UserMessage('test'))
      mock_post.assert_called_once()
      _, kwargs = mock_post.call_args
      self.assertTrue(kwargs.get('stream', False))

  def test_per_read_timeout_still_triggers(self):
    """requests.ReadTimeout should still become TemporaryLMError."""
    lm = self._make_lm(timeout=10.0)
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_requests_post_exception(
          requests.exceptions.ReadTimeout('Read timeout.')
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_error_response_works_with_streaming(self):
    """Streaming should not break error-status-code handling (e.g. 429)."""
    lm = self._make_lm(timeout=10.0)
    error_body = pg.to_json_str({
        'error': {'type': 'rate_limit', 'message': 'Rate limit exceeded.'}
    }).encode()
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [error_body], status_code=429
      )
      with self.assertRaises(lf.RateLimitError):
        lm._sample_single(lf.UserMessage('hello'))


class AdversarialStreamingTest(unittest.TestCase):
  """Red-team tests: adversarial scenarios targeting stream+deadline logic.

  These tests expose edge cases and bugs introduced or amplified by the
  switch to stream=True with wall-clock deadline enforcement.
  """

  def _make_lm(self, timeout=120.0):
    return rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(prompt=x.text),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        timeout=timeout,
    )

  # ---- Bug 1: Streaming-induced errors misclassified as non-retryable ----

  def test_incomplete_read_during_streaming_should_be_retryable(self):
    """IncompleteRead during streaming is retryable.

    stream=True makes IncompleteRead a common failure when the server
    drops the connection mid-body. This is a transient network error
    that should be TemporaryLMError for retry.
    """
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'{"content": ["par'
        raise requests.exceptions.ChunkedEncodingError(
            'Connection broken: IncompleteRead'
            '(18 bytes read, 100 more expected)'
        )
      response.iter_content = _failing_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      # Should be TemporaryLMError (retryable transient network error).
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- Bug 2: BrokenPipeError during streaming not retryable ----

  def test_broken_pipe_during_streaming_should_be_retryable(self):
    """BrokenPipeError during streaming is retryable.

    BrokenPipeError during iter_content is a transient network failure.
    It is a subclass of builtin ConnectionError (caught by the except
    block), but the message 'Broken pipe' must match a retryable
    pattern.
    """
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        raise BrokenPipeError('[Errno 32] Broken pipe')
      response.iter_content = _failing_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      # Should be TemporaryLMError — broken pipe is transient.
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- Bug 3: Resource leak — response.close() not called on error ----

  def test_response_not_closed_on_non_timeout_streaming_error(self):
    """Response is closed on non-timeout streaming errors.

    _read_response_with_deadline must call response.close() on all
    error paths, not just the deadline-timeout path. With stream=True,
    an unconsumed streaming response holds the underlying TCP connection
    open.
    """
    lm = self._make_lm(timeout=10.0)

    close_mock = mock.MagicMock()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        # 'Connection aborted' matches a retryable pattern, so this
        # becomes TemporaryLMError — but response.close() is never called.
        raise requests.exceptions.ChunkedEncodingError(
            'Connection aborted.'
        )
      response.iter_content = _failing_iter
      response.close = close_mock
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

    # response.close() should have been called for cleanup, but it wasn't.
    close_mock.assert_called()

  # ---- Bug 4: timeout=0 — verify fix ----

  def test_timeout_zero_enforces_instant_deadline(self):
    """timeout=0 enforces an instant deadline.

    Consistent with _per_operation_timeout which treats 0 as a value
    (not None). Previously, `if self.timeout` (truthiness) treated 0
    as falsy, disabling the deadline. Fixed to
    `if self.timeout is not None`.
    """
    lm = self._make_lm(timeout=0)

    # _per_operation_timeout correctly treats 0 as a value (not None):
    self.assertEqual(lm._per_operation_timeout, (0.0, 0.0))

    # After fix: timeout=0 should enforce an instant deadline.
    valid_json = pg.to_json_str({'content': ['instant timeout']}).encode()
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [valid_json], delay_per_chunk=0.01
      )
      # Deadline = now + 0 → fires immediately on first chunk.
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- Bug 5: Empty 200 body → unhandled JSONDecodeError ----

  def test_empty_body_200_unhandled_json_error(self):
    """Empty 200 body produces LMError, not unhandled JSONDecodeError.

    Server returns HTTP 200 with empty body (0 chunks). Streaming reads
    no chunks, sets _content=b''. Then response.json() raises
    json.JSONDecodeError, which must be caught and wrapped in LMError.

    More likely with stream=True: server sends headers (200 OK) then
    drops the connection before sending any body data.
    """
    lm = self._make_lm(timeout=10.0)

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post([], status_code=200)
      # JSONDecodeError should be caught and wrapped in LMError,
      # but it propagates unhandled.
      with self.assertRaises((lf.LMError, lf.TemporaryLMError)):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- Edge case: Deadline includes header wait ----

  def test_deadline_budget_consumed_by_header_wait(self):
    """Header wait consumes deadline budget.

    Deadline is set BEFORE session.post(). If header negotiation (TCP
    connect + TLS + HTTP headers) consumes most of the timeout budget,
    the body reading has almost no time left.
    """
    lm = self._make_lm(timeout=0.4)
    valid_json = pg.to_json_str({'content': ['fast body']}).encode()

    def _slow_header_post(
        url, json=None, timeout=None, stream=False, **kwargs
    ):
      del url, json, timeout, stream, kwargs
      time.sleep(0.6)  # Headers arrive after 0.6s > 0.4s deadline
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _instant_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield valid_json  # Body arrives instantly
      response.iter_content = _instant_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _slow_header_post
      with self.assertRaises(lf.TemporaryLMError) as ctx:
        lm._sample_single(lf.UserMessage('hello'))
      self.assertIn('deadline', str(ctx.exception).lower())

  # ---- Edge case: Complete data discarded on last-chunk deadline ----

  def test_complete_data_discarded_on_last_chunk_deadline(self):
    """All data received but deadline fires on last chunk.

    The complete (valid) response data is thrown away and TimeoutError
    is raised.
    """
    lm = self._make_lm(timeout=0.25)
    valid_json = pg.to_json_str({'content': ['complete result']}).encode()
    half = len(valid_json) // 2

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [valid_json[:half], valid_json[half:]],
          delay_per_chunk=0.15,  # 2 × 0.15 = 0.30s > 0.25s deadline
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- Edge case: Deadline bypassed by pre-loaded _content ----

  def test_deadline_bypassed_by_preloaded_content(self):
    """Deadline bypassed when _content is pre-loaded.

    _read_response_with_deadline checks
    `if response._content is not False: return` before enforcing
    the deadline. If _content is pre-populated (any value other than
    the sentinel False), deadline enforcement is completely skipped.

    This relies on the internal requests library sentinel value
    _content=False. If requests ever changes this sentinel (e.g., to
    None), deadline enforcement silently breaks for all streaming
    responses.
    """
    lm = self._make_lm(timeout=0.001)
    response = requests.Response()
    response.status_code = 200
    # Pre-populate _content (simulates already-buffered body)
    response._content = pg.to_json_str({'content': ['bypass']}).encode()

    # Deadline is 100 seconds in the past — should definitely fire.
    deadline = time.monotonic() - 100
    # But the guard clause skips deadline enforcement entirely:
    lm._read_response_with_deadline(response, deadline)
    # _parse_response succeeds, defeating the timeout:
    result = lm._parse_response(response)
    self.assertEqual(result.samples[0].response.text, 'bypass')


class ComprehensiveStreamingRobustnessTest(unittest.TestCase):
  """Production-level robustness tests for stream=True + deadline.

  Guards against regressions across all identified risk areas:
  error classification, resource cleanup, deadline boundaries,
  data integrity, HTTP errors, and Trawler proxy interaction.
  """

  def _make_lm(self, timeout=120.0):
    return rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(prompt=x.text),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        timeout=timeout,
    )

  # ======== A. Error Classification: streaming-specific errors ========

  def test_connection_reset_during_body_read_is_retryable(self):
    """ConnectionResetError during iter_content is retryable.

    Common in prod when server closes TCP mid-stream.
    """
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        raise ConnectionResetError('[Errno 104] Connection reset by peer')
      response.iter_content = _failing_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_generic_connection_error_stays_non_retryable(self):
    """ConnectionError with unknown message stays non-retryable.

    Fixes should not over-broaden retryability.
    """
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        raise requests.exceptions.ConnectionError(
            'Some novel error not in retryable patterns'
        )
      response.iter_content = _failing_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.LMError) as ctx:
        lm._sample_single(lf.UserMessage('hello'))
      # Must NOT be TemporaryLMError (retryable).
      self.assertNotIsInstance(ctx.exception, lf.TemporaryLMError)

  def test_ssl_error_during_body_read_is_retryable(self):
    """SSLError during iter_content is retryable.

    Verifies existing SSLError handler works post-CL.
    """
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        raise requests.exceptions.SSLError(
            'EOF occurred in violation of protocol (_ssl.c:2427)'
        )
      response.iter_content = _failing_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_read_timeout_during_iter_content_is_retryable(self):
    """ReadTimeout during iter_content is retryable.

    Verifies existing Timeout handler works post-CL.
    """
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'first chunk'
        raise requests.exceptions.ReadTimeout(
            'Read timed out. (read timeout=10)'
        )
      response.iter_content = _failing_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ======== B. Resource Cleanup ========

  def test_response_closed_on_deadline_timeout(self):
    """Positive test: response.close() IS called when deadline fires."""
    lm = self._make_lm(timeout=0.2)
    close_mock = mock.MagicMock()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _slow_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        while True:
          time.sleep(0.1)
          yield b'data'
      response.iter_content = _slow_iter
      response.close = close_mock
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

    close_mock.assert_called()

  def test_response_not_closed_on_success(self):
    """response.close() should NOT be called on successful streaming read."""
    lm = self._make_lm(timeout=10.0)
    valid_json = pg.to_json_str({'content': ['success']}).encode()
    close_mock = mock.MagicMock()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield valid_json
      response.iter_content = _iter
      response.close = close_mock
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, 'success')

    close_mock.assert_not_called()

  def test_response_closed_on_incomplete_read(self):
    """After fix: response.close() called on IncompleteRead error."""
    lm = self._make_lm(timeout=10.0)
    close_mock = mock.MagicMock()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        raise requests.exceptions.ChunkedEncodingError(
            'Connection broken: IncompleteRead(5 bytes read)'
        )
      response.iter_content = _failing_iter
      response.close = close_mock
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

    close_mock.assert_called()

  # ======== C. Deadline Boundary Conditions ========

  def test_very_small_timeout_enforces_deadline(self):
    """timeout=0.001 fires almost immediately — boundary test."""
    lm = self._make_lm(timeout=0.001)
    valid_json = pg.to_json_str({'content': ['tiny timeout']}).encode()

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [valid_json], delay_per_chunk=0.05
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_deadline_fires_midway_through_many_chunks(self):
    """Deadline fires midway through a 20-chunk stream.

    20 chunks at 0.05s each = 1.0s total. Deadline at 0.3s fires
    around chunk 6.
    """
    lm = self._make_lm(timeout=0.3)
    chunks = [b'x' * 100] * 20

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          chunks, delay_per_chunk=0.05
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_timeout_none_disables_all_enforcement(self):
    """timeout=None disables both per-operation timeout and deadline."""
    lm = self._make_lm(timeout=None)
    self.assertIsNone(lm._per_operation_timeout)

    valid_json = pg.to_json_str({'content': ['no timeout']}).encode()
    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [valid_json], delay_per_chunk=0.01
      )
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, 'no timeout')

  # ======== D. Data Integrity ========

  def test_multi_chunk_json_reassembly_correct(self):
    """JSON body split across 3 chunks reassembles correctly.

    Split at arbitrary byte boundaries.
    """
    lm = self._make_lm(timeout=10.0)
    valid_json = pg.to_json_str({'content': ['hello world']}).encode()
    # Split at arbitrary boundaries
    c1, c2, c3 = valid_json[:10], valid_json[10:20], valid_json[20:]

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post([c1, c2, c3])
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, 'hello world')

  def test_large_fast_response_unaffected(self):
    """Many chunks arriving instantly complete within deadline.

    Regression test: streaming overhead should not slow happy path.
    """
    lm = self._make_lm(timeout=10.0)
    large_text = 'A' * 5000
    valid_json = pg.to_json_str({'content': [large_text]}).encode()
    # Split into ~50 chunks of 100 bytes
    chunks = [valid_json[i:i+100] for i in range(0, len(valid_json), 100)]

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(chunks)
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, large_text)

  def test_json_decode_error_on_malformed_body_wrapped(self):
    """Malformed JSON in 200 response → LMError (not unhandled)."""
    lm = self._make_lm(timeout=10.0)

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [b'this is not valid JSON'], status_code=200
      )
      with self.assertRaises(lf.LMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ======== E. HTTP Error Responses with Streaming Body ========

  def test_error_500_body_read_via_streaming(self):
    """HTTP 500 error body is read via streaming before raising.

    Body must be fully read before the status code check.
    """
    lm = self._make_lm(timeout=10.0)
    error_body = pg.to_json_str({
        'error': {'type': 'internal', 'message': 'Server error'}
    }).encode()

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [error_body], status_code=500
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_error_429_rate_limit_via_streaming(self):
    """HTTP 429 rate limit: body read via streaming before raising."""
    lm = self._make_lm(timeout=10.0)
    error_body = pg.to_json_str({
        'error': {'type': 'rate_limit', 'message': 'Too many requests'}
    }).encode()

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = mock_streaming_post(
          [error_body], status_code=429
      )
      with self.assertRaises(lf.RateLimitError):
        lm._sample_single(lf.UserMessage('hello'))

  # ======== F. Trawler Proxy Interaction ========

  def test_trawler_prebuffered_body_reads_correctly(self):
    """Trawler pre-buffers body; deadline check reads BytesIO instantly.

    Simulates Trawler: body pre-buffered in memory (BytesIO),
    _content=False. Trawler ignores stream=True and buffers the entire
    response via FetchWithParams.
    """
    lm = self._make_lm(timeout=10.0)
    valid_json = pg.to_json_str({'content': ['trawler response']}).encode()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False
      # Trawler: body in memory, iter_content reads instantly
      def _instant_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield valid_json
      response.iter_content = _instant_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      result = lm._sample_single(lf.UserMessage('test'))
      self.assertEqual(result.samples[0].response.text, 'trawler response')

  def test_trawler_long_fetch_exceeds_client_deadline(self):
    """Trawler blocks during FetchWithParams; client deadline fires.

    Trawler's own fetch_timeout_ms should fire first in prod, but if
    the client deadline is shorter, it fires on the first body chunk.
    """
    lm = self._make_lm(timeout=0.3)
    valid_json = pg.to_json_str({'content': ['slow trawler']}).encode()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      # Simulate Trawler blocking during FetchWithParams
      time.sleep(0.5)  # 0.5s > 0.3s client deadline
      response = requests.Response()
      response.status_code = 200
      response._content = False
      def _instant_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield valid_json  # Body instant, but deadline already expired
      response.iter_content = _instant_iter
      response.close = lambda: None
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError) as ctx:
        lm._sample_single(lf.UserMessage('hello'))
      self.assertIn('deadline', str(ctx.exception).lower())

  def test_trawler_timeout_raises_temporary_error(self):
    """Trawler timeout raises TemporaryLMError.

    REJECTED_DEADLINE_EXCEEDED during session.post() raises
    requests.Timeout which is caught as TemporaryLMError.
    """
    lm = self._make_lm(timeout=10.0)

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = requests.exceptions.Timeout(
          'REJECTED_DEADLINE_EXCEEDED'
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_trawler_throttle_raises_temporary_error(self):
    """Trawler throttle raises TemporaryLMError.

    REJECTED_CLIENT_THROTTLED during session.post() raises
    requests.ConnectionError which is caught as TemporaryLMError.
    """
    lm = self._make_lm(timeout=10.0)

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = requests.exceptions.ConnectionError(
          'REJECTED_CLIENT_THROTTLED'
      )
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))


class RedTeamStreamingTest(unittest.TestCase):
  """Red-team tests for stream+deadline robustness.

  These tests target edge cases found by adversarial testing that the
  basic tests don't cover: exception handling in response.close(),
  ValueError/KeyError scope, resource leaks, and timeout validation.
  """

  def _make_lm(self, timeout=120.0):
    return rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(prompt=x.text),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        timeout=timeout,
    )

  # ---- close() error handling on deadline timeout ----

  def test_close_raises_oserror_on_deadline_still_returns_temporary_error(self):
    """response.close() raising OSError should not escape as unhandled."""
    lm = self._make_lm(timeout=0.2)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False

      def _slow_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        while True:
          time.sleep(0.1)
          yield b'data'

      response.iter_content = _slow_iter
      response.close = lambda: (_ for _ in ()).throw(
          OSError('[Errno 9] Bad file descriptor')
      )
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_close_raises_attributeerror_on_deadline_still_returns_temporary(
      self,
  ):
    """response.close() raising AttributeError should not escape."""
    lm = self._make_lm(timeout=0.2)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False

      def _slow_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        while True:
          time.sleep(0.1)
          yield b'data'

      response.iter_content = _slow_iter

      def _broken_close():
        raise AttributeError("'NoneType' object has no attribute 'close'")

      response.close = _broken_close
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- close() error handling on non-timeout error path ----

  def test_close_raises_on_error_path_does_not_mask_retryable_error(self):
    """close() raising on error path should not mask the original error."""
    lm = self._make_lm(timeout=10.0)

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False

      def _failing_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'partial'
        raise requests.exceptions.ChunkedEncodingError('Connection aborted.')

      response.iter_content = _failing_iter

      def _broken_close():
        raise RuntimeError('Error closing response')

      response.close = _broken_close
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      # The original ChunkedEncodingError should propagate as
      # TemporaryLMError, not be masked by RuntimeError from close().
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- ValueError/KeyError scope ----

  def test_valueerror_from_request_function_propagates(self):
    """ValueError from user's request() should propagate, not be masked."""
    lm = rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: (_ for _ in ()).throw(
            ValueError('Image modality not supported by this model')
        ),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        timeout=10.0,
    )

    with mock.patch('requests.Session.post'):
      with self.assertRaises(ValueError) as ctx:
        lm._sample_single(lf.UserMessage('hello'))
      self.assertNotIsInstance(ctx.exception, lf.LMError)

  def test_keyerror_from_request_function_propagates(self):
    """KeyError from user's request() should propagate, not be masked."""

    def _bad_request(prompt, sampling_options):
      del sampling_options
      _ = prompt.metadata['required_field']
      return {}

    lm = rest.REST(
        api_endpoint='https://fake-api.com',
        request=_bad_request,
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]
        ),
        timeout=10.0,
    )

    with mock.patch('requests.Session.post'):
      with self.assertRaises(KeyError):
        lm._sample_single(lf.UserMessage('hello'))

  def test_valueerror_from_result_function_wrapped_in_lm_error(self):
    """ValueError from result() parsing IS correctly wrapped in LMError."""

    def _broken_result(json_response):
      raise ValueError(
          "Expected 'content' key in response but got: "
          + str(list(json_response.keys()))
      )

    lm = rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(prompt=x.text),
        result=_broken_result,
        timeout=10.0,
    )

    valid_json = pg.to_json_str({'unexpected_key': 'value'}).encode()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = valid_json
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.LMError):
        lm._sample_single(lf.UserMessage('hello'))

  # ---- Resource leak: TimeoutError from iter_content ----

  def test_timeout_error_from_iter_content_closes_response(self):
    """TimeoutError from iter_content must close response (no leak)."""
    lm = self._make_lm(timeout=10.0)
    close_mock = mock.MagicMock()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False

      def _timeout_iter(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        yield b'first chunk ok'
        # socket.timeout IS TimeoutError in Python 3.3+
        raise TimeoutError('Read timed out. (read timeout=10)')

      response.iter_content = _timeout_iter
      response.close = close_mock
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

    close_mock.assert_called()

  # ---- First-chunk error cleanup ----

  def test_connection_error_on_first_chunk_closes_response(self):
    """Error on first chunk: response.close() should still be called."""
    lm = self._make_lm(timeout=10.0)
    close_mock = mock.MagicMock()

    def _mock_post(url, json=None, timeout=None, stream=False, **kwargs):
      del url, json, timeout, stream, kwargs
      response = requests.Response()
      response.status_code = 200
      response._content = False

      def _fail_immediately(chunk_size=1, decode_unicode=False):
        del chunk_size, decode_unicode
        raise ConnectionResetError('[Errno 104] Connection reset by peer')

      response.iter_content = _fail_immediately
      response.close = close_mock
      return response

    with mock.patch('requests.Session.post') as mock_post:
      mock_post.side_effect = _mock_post
      with self.assertRaises(lf.TemporaryLMError):
        lm._sample_single(lf.UserMessage('hello'))

    close_mock.assert_called()

  # ---- Timeout edge cases ----

  def test_negative_timeout_clamped_to_zero(self):
    """Negative timeout should be clamped to 0, not produce invalid tuple."""
    lm = self._make_lm(timeout=-1)
    connect, read = lm._per_operation_timeout
    self.assertGreaterEqual(connect, 0, 'Connect timeout should be >= 0')
    self.assertGreaterEqual(read, 0, 'Read timeout should be >= 0')

  @unittest.skipUnless(
      hasattr(rest, 'internet_access'),
      'Requires Google-internal internet_access module',
  )
  def test_timeout_zero_trawler_consistent(self):
    """timeout=0: Trawler should get 0ms, not 30 minutes."""
    lm = self._make_lm(timeout=0)
    with mock.patch.object(
        rest.internet_access, 'maybe_enable_trawler'
    ) as mock_trawler:
      lm.session()
      mock_trawler.assert_called_once()
      _, kwargs = mock_trawler.call_args
      self.assertEqual(kwargs['fetch_timeout_ms'], 0)
      self.assertEqual(kwargs['request_deadline_ms'], 0)


if __name__ == '__main__':
  unittest.main()
