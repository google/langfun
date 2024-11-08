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
from typing import Annotated, Any, Callable

import langfun.core as lf
import requests


class REST(lf.LanguageModel):
  """REST-based language model."""

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

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model or 'unknown'

  @functools.cached_property
  def _api_initialized(self) -> bool:
    """Returns whether the API is initialized."""
    self._initialize()
    return True

  def _initialize(self) -> None:
    """Initializes the API. Subclasses can override."""

  @functools.cached_property
  def _session(self) -> requests.Session:
    assert self._api_initialized
    s = requests.Session()
    s.headers.update(self.headers or {})
    return s

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('_session', None)
    self.__dict__.pop('_api_initialized', None)

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    assert self._api_initialized
    return self._parallel_execute_with_currency_control(
        self._sample_single, prompts
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    try:
      response = self._session.post(
          self.api_endpoint,
          json=self.request(prompt, self.sampling_options),
          timeout=self.timeout,
      )
      return self._parse_response(response)
    except ConnectionError as e:
      raise lf.LMError(str(e)) from e

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 429:
      error_cls = lf.RateLimitError
    elif status_code in (
        500,  # Server side issue (might be bug).
        502,  # Bad gateway (upstream issue, might retry).
        503,  # Servers currently under load, retry after a brief wait.
        529,  # Overloaded, retry after a brief wait.
    ):
      error_cls = lf.TemporaryLMError
    else:
      error_cls = lf.LMError
    return error_cls(f'{status_code}: {content}')

  def _parse_response(self, response: requests.Response) -> lf.LMSamplingResult:
    """Parses Anthropic's response."""
    if response.status_code == 200:
      return self.result(response.json())
    else:
      raise self._error(response.status_code, response.content)
