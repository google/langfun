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

import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import rest
import pyglove as pg


SUPPORTED_MODELS_AND_SETTINGS = {
    # Refer https://console.groq.com/docs/models
    # Price in US dollars at https://groq.com/pricing/ as of 2024-10-10.
    'llama-3.2-3b-preview': pg.Dict(
        max_tokens=8192,
        max_concurrency=64,
        cost_per_1k_input_tokens=0.00006,
        cost_per_1k_output_tokens=0.00006,
    ),
    'llama-3.2-1b-preview': pg.Dict(
        max_tokens=8192,
        max_concurrency=64,
        cost_per_1k_input_tokens=0.00004,
        cost_per_1k_output_tokens=0.00004,
    ),
    'llama-3.1-70b-versatile': pg.Dict(
        max_tokens=8192,
        max_concurrency=16,
        cost_per_1k_input_tokens=0.00059,
        cost_per_1k_output_tokens=0.00079,
    ),
    'llama-3.1-8b-instant': pg.Dict(
        max_tokens=8192,
        max_concurrency=32,
        cost_per_1k_input_tokens=0.00005,
        cost_per_1k_output_tokens=0.00008,
    ),
    'llama3-70b-8192': pg.Dict(
        max_tokens=8192,
        max_concurrency=16,
        cost_per_1k_input_tokens=0.00059,
        cost_per_1k_output_tokens=0.00079,
    ),
    'llama3-8b-8192': pg.Dict(
        max_tokens=8192,
        max_concurrency=32,
        cost_per_1k_input_tokens=0.00005,
        cost_per_1k_output_tokens=0.00008,
    ),
    'llama2-70b-4096': pg.Dict(
        max_tokens=4096,
        max_concurrency=16,
    ),
    'mixtral-8x7b-32768': pg.Dict(
        max_tokens=32768,
        max_concurrency=16,
        cost_per_1k_input_tokens=0.00024,
        cost_per_1k_output_tokens=0.00024,
    ),
    'gemma2-9b-it': pg.Dict(
        max_tokens=8192,
        max_concurrency=32,
        cost_per_1k_input_tokens=0.0002,
        cost_per_1k_output_tokens=0.0002,
    ),
    'gemma-7b-it': pg.Dict(
        max_tokens=8192,
        max_concurrency=32,
        cost_per_1k_input_tokens=0.00007,
        cost_per_1k_output_tokens=0.00007,
    ),
    'whisper-large-v3': pg.Dict(
        max_tokens=8192,
        max_concurrency=16,
    ),
    'whisper-large-v3-turbo': pg.Dict(
        max_tokens=8192,
        max_concurrency=16,
    )
}


@lf.use_init_args(['model'])
class Groq(rest.REST):
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

  api_endpoint: str = 'https://api.groq.com/openai/v1/chat/completions'

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None

  def _initialize(self):
    api_key = self.api_key or os.environ.get('GROQ_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GROQ_API_KEY` with your Groq API key.'
      )
    self._api_key = api_key

  @property
  def headers(self) -> dict[str, Any]:
    return {
        'Authorization': f'Bearer {self._api_key}',
        'Content-Type': 'application/json',
    }

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @property
  def max_concurrency(self) -> int:
    return SUPPORTED_MODELS_AND_SETTINGS[self.model].max_concurrency

  def estimate_cost(
      self,
      num_input_tokens: int,
      num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    cost_per_1k_input_tokens = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_input_tokens', None
    )
    cost_per_1k_output_tokens = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_output_tokens', None
    )
    if cost_per_1k_input_tokens is None or cost_per_1k_output_tokens is None:
      return None
    return (
        cost_per_1k_input_tokens * num_input_tokens
        + cost_per_1k_output_tokens * num_output_tokens
    ) / 1000

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request = dict()
    request.update(self._request_args(sampling_options))
    request.update(
        dict(
            messages=[
                dict(role='user', content=self._content_from_message(prompt))
            ]
        )
    )
    return request

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
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

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    samples = [
        lf.LMSample(self._message_from_choice(choice), score=0.0)
        for choice in json['choices']
    ]
    usage = json['usage']
    return lf.LMSamplingResult(
        samples,
        usage=lf.LMSamplingUsage(
            prompt_tokens=usage['prompt_tokens'],
            completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'],
            estimated_cost=self.estimate_cost(
                num_input_tokens=usage['prompt_tokens'],
                num_output_tokens=usage['completion_tokens'],
            ),
        ),
    )

  def _message_from_choice(self, choice: dict[str, Any]) -> lf.Message:
    """Converts Groq's content protocol to message."""
    # Refer: https://platform.openai.com/docs/api-reference/chat/create
    content = choice['message']['content']
    if isinstance(content, str):
      return lf.AIMessage(content)
    return lf.AIMessage.from_chunks(
        [x['text'] for x in content if x['type'] == 'text']
    )


class GroqLlama3_2_3B(Groq):  # pylint: disable=invalid-name
  """Llama3.2-3B with 8K context window.

  See: https://huggingface.co/meta-llama/Llama-3.2-3B
  """

  model = 'llama-3.2-3b-preview'


class GroqLlama3_2_1B(Groq):  # pylint: disable=invalid-name
  """Llama3.2-1B with 8K context window.

  See: https://huggingface.co/meta-llama/Llama-3.2-1B
  """

  model = 'llama-3.2-3b-preview'


class GroqLlama3_8B(Groq):  # pylint: disable=invalid-name
  """Llama3-8B with 8K context window.

  See: https://huggingface.co/meta-llama/Meta-Llama-3-8B
  """

  model = 'llama3-8b-8192'


class GroqLlama3_1_70B(Groq):  # pylint: disable=invalid-name
  """Llama3.1-70B with 8K context window.

  See: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md   # pylint: disable=line-too-long
  """

  model = 'llama-3.1-70b-versatile'


class GroqLlama3_1_8B(Groq):  # pylint: disable=invalid-name
  """Llama3.1-8B with 8K context window.

  See: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md   # pylint: disable=line-too-long
  """

  model = 'llama-3.1-8b-instant'


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


class GroqGemma2_9B_IT(Groq):  # pylint: disable=invalid-name
  """Gemma2 9B with 8K context window.

  See: https://huggingface.co/google/gemma-2-9b-it
  """

  model = 'gemma2-9b-it'


class GroqGemma_7B_IT(Groq):  # pylint: disable=invalid-name
  """Gemma 7B with 8K context window.

  See: https://huggingface.co/google/gemma-1.1-7b-it
  """

  model = 'gemma-7b-it'


class GroqWhisper_Large_v3(Groq):  # pylint: disable=invalid-name
  """Whisper Large V3 with 8K context window.

  See: https://huggingface.co/openai/whisper-large-v3
  """

  model = 'whisper-large-v3'


class GroqWhisper_Large_v3Turbo(Groq):  # pylint: disable=invalid-name
  """Whisper Large V3 Turbo with 8K context window.

  See: https://huggingface.co/openai/whisper-large-v3-turbo
  """

  model = 'whisper-large-v3-turbo'
