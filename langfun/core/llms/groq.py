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
"""Language models from Groq."""

import datetime
import functools
import os
from typing import Annotated, Any, Final

import langfun.core as lf
from langfun.core.llms import openai_compatible
import pyglove as pg


class GroqModelInfo(lf.ModelInfo):
  """Groq model info."""

  LINKS = dict(
      models='https://console.groq.com/docs/models',
      pricing='https://groq.com/pricing/',
      rate_limits='https://console.groq.com/docs/rate-limits',
      error_codes='https://console.groq.com/docs/errors',
  )

  provider: Final[str] = 'Groq'  # pylint: disable=invalid-name


SUPPORTED_MODELS = [
    #
    # Llama models.
    #
    GroqModelInfo(
        model_id='llama-3.3-70b-versatile',
        in_service=True,
        model_type='instruction-tuned',
        description='Llama 3.3 70B model on Groq (Production)',
        url='https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct',
        release_date=datetime.datetime(2024, 12, 6),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.59,
            cost_per_1m_output_tokens=0.79,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=1_000,
            max_tokens_per_minute=120_000,
        ),
    ),
    GroqModelInfo(
        model_id='llama-3.3-70b-specdec',
        in_service=True,
        model_type='instruction-tuned',
        description='Llama 3.3 70B model on Groq (Production)',
        url='https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct',
        release_date=datetime.datetime(2024, 12, 6),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=8_192,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.59,
            cost_per_1m_output_tokens=0.99,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=30_000,
        ),
    ),
    GroqModelInfo(
        model_id='llama-3.2-1b-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='Llama 3.2 1B model on Groq (Preview)',
        url='https://huggingface.co/meta-llama/Llama-3.2-1B',
        release_date=datetime.datetime(2024, 12, 6),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.04,
            cost_per_1m_output_tokens=0.04,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=30_000,
        ),
    ),
    GroqModelInfo(
        model_id='llama-3.2-3b-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='Llama 3.2 3B model on Groq (Preview)',
        url='https://huggingface.co/meta-llama/Llama-3.2-3B',
        release_date=datetime.datetime(2024, 12, 6),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.06,
            cost_per_1m_output_tokens=0.06,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=30_000,
        ),
    ),
    GroqModelInfo(
        model_id='llama-3.2-11b-vision-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='Llama 3.2 11B vision model on Groq (Preview)',
        url='https://huggingface.co/meta-llama/Llama-3.2-11B-Vision',
        release_date=datetime.datetime(2024, 12, 6),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.18,
            cost_per_1m_output_tokens=0.18,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=30_000,
        ),
    ),
    GroqModelInfo(
        model_id='llama-3.2-90b-vision-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='Llama 3.2 90B vision model on Groq (Preview)',
        url='https://huggingface.co/meta-llama/Llama-3.2-90B-Vision',
        release_date=datetime.datetime(2024, 12, 6),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.9,
            cost_per_1m_output_tokens=0.9,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=30_000,
        ),
    ),
    #
    # DeepSeek models
    #
    GroqModelInfo(
        model_id='deepseek-r1-distill-llama-70b',
        in_service=True,
        model_type='thinking',
        description='DeepSeek R1 distilled from Llama 70B (Preview)',
        url='https://console.groq.com/docs/model/deepseek-r1-distill-llama-70b',
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        # TODO(daiyip): Pricing needs to be computed based on the number of
        # input/output tokens.
        pricing=None,
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=1_000,
            max_tokens_per_minute=120_000,
        ),
    ),
    GroqModelInfo(
        model_id='deepseek-r1-distill-llama-70b-specdec',
        in_service=True,
        model_type='thinking',
        description='DeepSeek R1 distilled from Llama 70B (Preview)',
        url='https://console.groq.com/docs/model/deepseek-r1-distill-llama-70b',
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        # TODO(daiyip): Pricing needs to be computed based on the number of
        # input/output tokens.
        pricing=None,
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=60_000,
        ),
    ),
    #
    # Gemma models.
    #
    GroqModelInfo(
        model_id='gemma2-9b-it',
        in_service=True,
        model_type='instruction-tuned',
        description='Google Gemma 2 9B model on Groq.',
        url='https://huggingface.co/google/gemma-2-9b-it',
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=8_192,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.2,
            cost_per_1m_output_tokens=0.2,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=200,
            max_tokens_per_minute=30_000,
        ),
    ),
    #
    # Mixtral models.
    #
    GroqModelInfo(
        model_id='mixtral-8x7b-32768',
        in_service=True,
        model_type='instruction-tuned',
        description='Mixtral 8x7B model on Groq (Production)',
        url='https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1',
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=32_768,
            max_output_tokens=None,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.24,
            cost_per_1m_output_tokens=0.24,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Developer tier.
            max_requests_per_minute=100,
            max_tokens_per_minute=25_000,
        ),
    ),
]

_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


@lf.use_init_args(['model'])
class Groq(openai_compatible.OpenAICompatible):
  """Groq LLMs through REST APIs (OpenAI compatible).

  See https://platform.openai.com/docs/api-reference/chat
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]
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

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    return _SUPPORTED_MODELS_BY_ID[self.model]

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

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # `logprobs` and `top_logprobs` flags are not supported on Groq yet.
    args = super()._request_args(options)
    args.pop('logprobs', None)
    args.pop('top_logprobs', None)
    return args


class GroqLlama33_70B_Versatile(Groq):  # pylint: disable=invalid-name
  """Llama3.2-3B with 128K context window."""
  model = 'llama-3.3-70b-versatile'


class GroqLlama33_70B_SpecDec(Groq):  # pylint: disable=invalid-name
  """Llama3.3-70B with 8K context window."""
  model = 'llama-3.3-70b-specdec'


class GroqLlama32_1B(Groq):  # pylint: disable=invalid-name
  """Llama3.2-1B."""
  model = 'llama-3.2-1b-preview'


class GroqLlama32_3B(Groq):  # pylint: disable=invalid-name
  """Llama3.2-3B."""
  model = 'llama-3.2-3b-preview'


class GroqLlama32_11B_Vision(Groq):  # pylint: disable=invalid-name
  """Llama3.2-11B vision."""
  model = 'llama-3.2-11b-vision-preview'


class GroqLlama32_90B_Vision(Groq):  # pylint: disable=invalid-name
  """Llama3.2-90B vision."""
  model = 'llama-3.2-90b-vision-preview'


class GroqDeepSeekR1_DistillLlama_70B(Groq):  # pylint: disable=invalid-name
  """DeepSeek R1 distilled from Llama 70B."""
  model = 'deepseek-r1-distill-llama-70b'


class GroqDeepSeekR1_DistillLlama_70B_SpecDec(Groq):  # pylint: disable=invalid-name
  """DeepSeek R1 distilled from Llama 70B (SpecDec)."""
  model = 'deepseek-r1-distill-llama-70b-specdec'


class GroqMistral_8x7B(Groq):  # pylint: disable=invalid-name
  """Mixtral 8x7B."""
  model = 'mixtral-8x7b-32768'


class GroqGemma2_9B_IT(Groq):  # pylint: disable=invalid-name
  """Gemma2 9B."""
  model = 'gemma2-9b-it'


#
# Register Groq models so they can be retrieved with LanguageModel.get().
#


def _groq_model(model: str, *args, **kwargs):
  model = model.removeprefix('groq://')
  return Groq(model, *args, **kwargs)


def _register_groq_models():
  """Registers Groq models."""
  for m in SUPPORTED_MODELS:
    lf.LanguageModel.register('groq://' + m.model_id, _groq_model)

_register_groq_models()
