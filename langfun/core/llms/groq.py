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
"""Language models from Groq."""

import functools
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
import pyglove as pg
import requests


SUPPORTED_MODELS_AND_SETTINGS = {
    # Refer https://console.groq.com/docs/models
    'llama3-8b-8192': pg.Dict(max_tokens=8192, max_concurrency=16),
    'llama3-70b-8192': pg.Dict(max_tokens=8192, max_concurrency=16),
    'llama2-70b-4096': pg.Dict(max_tokens=4096, max_concurrency=16),
    'mixtral-8x7b-32768': pg.Dict(max_tokens=32768, max_concurrency=16),
    'gemma-7b-it': pg.Dict(max_tokens=8192, max_concurrency=16),
}


class GroqError(Exception):  # pylint: disable=g-bad-exception-name
  """Base class for Groq errors."""


class RateLimitError(GroqError):
  """Error for rate limit reached."""


class OverloadedError(GroqError):
  """Groq's server is temporarily overloaded."""


_CHAT_COMPLETE_API_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'


@lf.use_init_args(['model'])
class Groq(lf.LanguageModel):
  """Groq LLMs through REST APIs (OpenAI compatible).

  See https://platform.openai.com/docs/api-reference/chat
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  multimodal: Annotated[bool, 'Whether this model has multimodal support.'] = (
      False
  )

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'GROQ_API_KEY'."
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None
    self.__dict__.pop('_api_initialized', None)
    self.__dict__.pop('_session', None)

  @functools.cached_property
  def _api_initialized(self):
    api_key = self.api_key or os.environ.get('GROQ_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GROQ_API_KEY` with your Groq API key.'
      )
    self._api_key = api_key
    return True

  @functools.cached_property
  def _session(self) -> requests.Session:
    assert self._api_initialized
    s = requests.Session()
    s.headers.update({
        'Authorization': f'Bearer {self._api_key}',
        'Content-Type': 'application/json',
    })
    return s

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @property
  def max_concurrency(self) -> int:
    return SUPPORTED_MODELS_AND_SETTINGS[self.model].max_concurrency

  def _get_request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # `logprobs` and `top_logprobs` flags are not supported on Groq yet.
    args = dict(
        model=self.model,
        n=options.n,
        stream=False,
    )

    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.max_tokens is not None:
      args['max_tokens'] = options.max_tokens
    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.stop:
      args['stop'] = options.stop
    return args

  def _content_from_message(self, prompt: lf.Message) -> list[dict[str, Any]]:
    """Converts an message to Groq's content protocol (list of dicts)."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = []
    for chunk in prompt.chunk():
      if isinstance(chunk, str):
        item = dict(type='text', text=chunk)
      elif (
          self.multimodal
          and isinstance(chunk, lf_modalities.Image)
          and chunk.uri
      ):
        # NOTE(daiyip): Groq only support image URL.
        item = dict(type='image_url', image_url=chunk.uri)
      else:
        raise ValueError(f'Unsupported modality object: {chunk!r}.')
      content.append(item)
    return content

  def _message_from_choice(self, choice: dict[str, Any]) -> lf.Message:
    """Converts Groq's content protocol to message."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = choice['message']['content']
    if isinstance(content, str):
      return lf.AIMessage(content)
    return lf.AIMessage.from_chunks(
        [x['text'] for x in content if x['type'] == 'text']
    )

  def _parse_response(self, response: requests.Response) -> lf.LMSamplingResult:
    """Parses Groq's response."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/object
    if response.status_code == 200:
      output = response.json()
      samples = [
          lf.LMSample(self._message_from_choice(choice), score=0.0)
          for choice in output['choices']
      ]
      usage = output['usage']
      return lf.LMSamplingResult(
          samples,
          usage=lf.LMSamplingUsage(
              prompt_tokens=usage['prompt_tokens'],
              completion_tokens=usage['completion_tokens'],
              total_tokens=usage['total_tokens'],
          ),
      )
    else:
      # https://platform.openai.com/docs/guides/error-codes/api-errors
      if response.status_code == 429:
        error_cls = RateLimitError
      elif response.status_code in (500, 502, 503):
        error_cls = OverloadedError
      else:
        error_cls = GroqError
      raise error_cls(f'{response.status_code}: {response.content}')

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    assert self._api_initialized
    return self._parallel_execute_with_currency_control(
        self._sample_single,
        prompts,
        retry_on_errors=(RateLimitError, OverloadedError),
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    request = dict()
    request.update(self._get_request_args(self.sampling_options))
    request.update(
        dict(
            messages=[
                dict(role='user', content=self._content_from_message(prompt))
            ]
        )
    )
    try:
      response = self._session.post(
          _CHAT_COMPLETE_API_ENDPOINT,
          json=request,
          timeout=self.timeout,
      )
      return self._parse_response(response)
    except ConnectionError as e:
      raise OverloadedError(str(e)) from e


class GroqLlama3_8B(Groq):  # pylint: disable=invalid-name
  """Llama3-8B with 8K context window.

  See: https://huggingface.co/meta-llama/Meta-Llama-3-8B
  """

  model = 'llama3-8b-8192'


class GroqLlama3_70B(Groq):  # pylint: disable=invalid-name
  """Llama3-70B with 8K context window.

  See: https://huggingface.co/meta-llama/Meta-Llama-3-70B
  """

  model = 'llama3-70b-8192'


class GroqLlama2_70B(Groq):  # pylint: disable=invalid-name
  """Llama2-70B with 4K context window.

  See: https://huggingface.co/meta-llama/Llama-2-70b
  """

  model = 'llama2-70b-4096'


class GroqMistral_8x7B(Groq):  # pylint: disable=invalid-name
  """Mixtral 8x7B with 32K context window.

  See: https://huggingface.co/meta-llama/Llama-2-70b
  """

  model = 'mixtral-8x7b-32768'


class GroqGemma7B_IT(Groq):  # pylint: disable=invalid-name
  """Gemma 7B with 8K context window.

  See: https://huggingface.co/google/gemma-1.1-7b-it
  """

  model = 'gemma-7b-it'
