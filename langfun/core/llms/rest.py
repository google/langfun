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
"""Base class for language models through REST APIs."""

import functools
import time
from typing import Annotated, Any, Callable

import langfun.core as lf
# Placeholder for Google-internal internet access import.
import requests


class REST(lf.LanguageModel):
  """Base class for language models accessed via REST APIs.

  The `REST` class provides a foundation for implementing language models
  that are accessed through RESTful endpoints. It handles the details of
  making HTTP requests, managing sessions, and handling common errors like
  timeouts and connection issues.

  Subclasses need to implement the `request` and `result` methods to
  convert Langfun messages to API-specific request formats and to parse
  API responses back into `LMSamplingResult` objects. They also need to
  provide the `api_endpoint` and can override `headers` for authentication.
  """

  api_endpoint: Annotated[
      str,
      'The endpoint of the REST API.'
  ]

  request: Annotated[
      Callable[[lf.Message, lf.LMSamplingOptions], dict[str, Any]],
      'A function to convert a Langfun message to a JSON request.'
  ]

  result: Annotated[
      Callable[[dict[str, Any]], lf.LMSamplingResult],
      'A function to convert a JSON response to an LMSamplingResult.'
  ]

  model: Annotated[
      str | None,
      'Model ID.'
  ] = None

  headers: Annotated[
      dict[str, Any] | None,
      'The headers for the REST API.'
  ] = None

  inactivity_timeout: Annotated[
      float | None,
      (
          'Short per-chunk INACTIVITY bound in seconds. This is the maximum '
          'time allowed to elapse between two successive received chunks (it '
          'resets every time a chunk arrives). A genuinely dead connection '
          'fast-fails after this window, while a live-but-slow generation '
          'that keeps emitting bytes is NOT killed. When None, falls back to '
          '`timeout`, preserving the historical single-timeout behavior. Used '
          'both as the requests read timeout (socket idle) and as an in-loop '
          'check in _read_response_with_deadline.'
      ),
  ] = None

  max_total_timeout: Annotated[
      float | None,
      (
          'Long TOTAL wall-clock budget in seconds for a single response. '
          'This bounds the entire response duration regardless of how '
          'steadily bytes arrive, letting a healthy multi-hour generation '
          'complete. When None, falls back to `timeout`. Set this large '
          '(e.g. 14400 for 4h) together with a small `inactivity_timeout` '
          '(e.g. 120) to support slow-but-live long generations.'
      ),
  ] = None

  @functools.cached_property
  def _api_initialized(self) -> bool:
    """Returns whether the API is initialized."""
    self._initialize()
    return True

  def _initialize(self) -> None:
    """Initializes the API. Subclasses can override."""

  def session(self) -> requests.Session:
    assert self._api_initialized
    s = self._session()
    # Placeholder for Google-internal session adapter.
    s.headers.update(self.headers or {})
    return s

  def _session(self) -> requests.Session:
    """Creates a new session."""
    return requests.Session()

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('_api_initialized', None)

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    assert self._api_initialized
    return self._parallel_execute_with_currency_control(
        self._sample_single, prompts
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    try:
      with self.session() as session:
        total_timeout = self._effective_total_timeout
        deadline = (
            (time.monotonic() + total_timeout)
            if total_timeout is not None
            else None
        )
        response = session.post(
            self.api_endpoint,
            json=self.request(prompt, self.sampling_options),
            timeout=self._per_operation_timeout,
            stream=True,
        )
        self._read_response_with_deadline(response, deadline)
        return self._parse_response(response)
    except (
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        TimeoutError,
    ) as e:
      raise lf.TemporaryLMError(str(e)) from e
    except requests.exceptions.SSLError as e:
      # SSLEOFError during handshake is typically transient (load balancer
      # drops, network instability, etc.) and should be retried.
      raise lf.TemporaryLMError(str(e)) from e
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        ConnectionError,
        ConnectionResetError,
    ) as e:
      error_message = str(e)
      if 'REJECTED_CLIENT_THROTTLED' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'UNREACHABLE_NO_RESPONSE' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'UNREACHABLE_ERROR' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'Connection reset by peer' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'Remote end closed connection' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'Connection aborted' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'IncompleteRead' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      if 'Broken pipe' in error_message:
        raise lf.TemporaryLMError(error_message) from e
      raise lf.LMError(error_message) from e

  @property
  def _effective_inactivity_timeout(self) -> float | None:
    """Short per-chunk inactivity bound (resets on each received chunk).

    Falls back to self.timeout when not explicitly configured, preserving the
    historical single-timeout behavior for callers that only set `timeout`.
    """
    if self.inactivity_timeout is not None:
      return self.inactivity_timeout
    return self.timeout

  @property
  def _effective_total_timeout(self) -> float | None:
    """Long total wall-clock budget for a single response.

    Falls back to self.timeout when not explicitly configured.
    """
    if self.max_total_timeout is not None:
      return self.max_total_timeout
    return self.timeout

  @property
  def _per_operation_timeout(self):
    """Per-operation (connect, read) timeout tuple for the requests library.

    - connect: bounded to 60s (no server needs more to accept a TCP
      connection).
    - read: the per-socket idle timeout == the inactivity bound. requests
      raises ReadTimeout if no bytes arrive within this window, which is
      exactly the dead-connection fast-fail we want. It does NOT bound the
      total response time (that is enforced by _read_response_with_deadline).
    """
    inactivity = self._effective_inactivity_timeout
    if inactivity is None:
      return None
    inactivity = max(0.0, inactivity)
    return (min(60.0, inactivity), inactivity)

  def _read_response_with_deadline(
      self, response: requests.Response, deadline: float | None
  ) -> None:
    """Reads response body, enforcing inactivity + total-request deadlines.

    When stream=True, session.post() returns after HTTP headers are received.
    This method reads the body in chunks and enforces TWO independent bounds:

    - Inactivity bound (short, `_effective_inactivity_timeout`): the maximum
      time allowed BETWEEN two successive chunks. It resets every time a chunk
      is received, so a live-but-slow generation that keeps emitting bytes is
      never killed, while a genuinely dead connection fast-fails. This is also
      enforced at the socket layer via the read timeout in
      `_per_operation_timeout` (which covers the case where iter_content blocks
      because the server sends nothing at all); the in-loop check below
      additionally covers slow trickles observed between yielded chunks.
    - Total bound (long, via `deadline`): the absolute wall-clock budget for
      the whole response. Lets a healthy multi-hour generation complete.

    If either bound is exceeded, the response is closed (which immediately
    closes the underlying socket) and TimeoutError is raised.

    After successful read, sets response._content so that response.json()
    and response.content work normally for _parse_response().

    Args:
      response: A streaming requests.Response (from stream=True).
      deadline: Monotonic clock TOTAL deadline (from time.monotonic()), or
        None to disable total-deadline enforcement.
    """
    # If body was already buffered (non-streaming or content pre-loaded),
    # there is nothing to read.
    if response._content is not False:  # pylint: disable=protected-access,g-bool-id-comparison
      return
    inactivity = self._effective_inactivity_timeout
    chunks = []
    last_chunk_time = time.monotonic()
    try:
      for chunk in response.iter_content(chunk_size=65536):
        now = time.monotonic()
        # Inactivity bound: time since the previous chunk (or since the start
        # of the read for the first chunk). Resets on every received chunk.
        if inactivity is not None and now - last_chunk_time > inactivity:
          raise TimeoutError(
              f'No response data received for {inactivity}s '
              '(inactivity timeout).'
          )
        chunks.append(chunk)
        last_chunk_time = now
        # Total bound: absolute wall-clock budget for the whole response.
        if deadline is not None and now > deadline:
          raise TimeoutError(
              'Response exceeded total deadline of '
              f'{self._effective_total_timeout}s.'
          )
    except BaseException:
      # Close on any error to prevent TCP connection leaks. Protect close()
      # with try/except to prevent close errors from masking the original.
      try:
        response.close()
      except Exception:  # pylint: disable=broad-except
        pass
      raise
    # Set internal cache so response.json() / response.content work normally.
    response._content = b''.join(chunks)  # pylint: disable=protected-access  # pyrefly: ignore[bad-assignment]

  # Content filtering patterns observed from various LLM providers.
  # These are best-effort substring heuristics derived from real API error
  # messages, as providers do not formally document exact error strings.
  _CONTENT_FILTER_PATTERNS = (
      'content filtering',  # Anthropic/Claude: "Output blocked by content
      #   filtering policy"
      'content_filter',  # OpenAI: "content_filter triggered for this
      #   request"
      'output blocked',  # Anthropic/Claude: "Output blocked by ..."
      'blocked by safety',  # Google/Gemini: "blocked by safety filter"
      'blocked due to safety',  # Google/Gemini: "blocked due to SAFETY reasons"
      'safety filter',  # Generic: covers variations across providers.
  )

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 429:
      error_cls = lf.RateLimitError
    elif status_code in (
        500,  # Server side issue (might be bug).
        502,  # Bad gateway (upstream issue, might retry).
        503,  # Servers currently under load, retry after a brief wait.
        529,  # Overloaded, retry after a brief wait.
        499,  # Client Closed Request
    ):
      error_cls = lf.TemporaryLMError
    elif status_code == 400:  # Bad Request — providers use this for both
      # malformed requests AND content policy violations
      # (input prompt or output blocked by safety).
      # We disambiguate via message patterns below.
      content_lower = (
          content.lower() if isinstance(content, str) else str(content).lower()
      )
      if any(p in content_lower for p in self._CONTENT_FILTER_PATTERNS):
        error_cls = lf.ContentFilteredError
      else:
        error_cls = lf.LMError
    else:
      error_cls = lf.LMError
    return error_cls(f'{status_code}: {content}')

  def _response_to_message_dict(self, response: requests.Response) -> Any:
    """Converts an HTTP response into the message dict expected by `result`.

    Default implementation assumes a single buffered JSON body. Subclasses
    whose API returns a streamed (e.g. Server-Sent Events) body override this
    to reassemble the full message dict before it reaches `result()`.

    Args:
      response: The HTTP response returned by the API.

    Returns:
      The parsed message dict to be passed to `result`.
    """
    return response.json()

  def _parse_response(self, response: requests.Response) -> lf.LMSamplingResult:
    """Parses the LLM response."""
    if response.status_code == 200:
      try:
        return self.result(self._response_to_message_dict(response))
      except (ValueError, KeyError) as e:
        raise lf.LMError(str(e)) from e
    else:
      raise self._error(response.status_code, response.content)  # pyrefly: ignore[bad-argument-type]

  @property
  def max_concurrency(self) -> int | None:  # pyrefly: ignore[bad-override]
    """Returns the max concurrency for this model."""
    rate_limits = self.model_info.rate_limits
    if rate_limits is not None:
      return self.estimate_max_concurrency(
          max_requests_per_minute=rate_limits.max_requests_per_minute,
          max_tokens_per_minute=rate_limits.max_tokens_per_minute
      )
    return None
