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
from langfun.core.llms import openai_compatible
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
class Groq(openai_compatible.OpenAICompatible):
  """Groq LLMs through REST APIs (OpenAI compatible).

  See https://platform.openai.com/docs/api-reference/chat
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'GROQ_API_KEY'."
      ),
  ] = None

  api_endpoint: str = 'https://api.groq.com/openai/v1/chat/completions'

  @property
  def headers(self) -> dict[str, Any]:
    api_key = self.api_key or os.environ.get('GROQ_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GROQ_API_KEY` with your Groq API key.'
      )
    headers = super().headers
    headers.update({
        'Authorization': f'Bearer {api_key}',
    })
    return headers

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

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # `logprobs` and `top_logprobs` flags are not supported on Groq yet.
    args = super()._request_args(options)
    args.pop('logprobs', None)
    args.pop('top_logprobs', None)
    return args


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
