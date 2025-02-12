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
"""Language models from DeepSeek."""

import datetime
import functools
import os
from typing import Annotated, Any, Final

import langfun.core as lf
from langfun.core.llms import openai_compatible
import pyglove as pg


class DeepSeekModelInfo(lf.ModelInfo):
  """DeepSeek model info."""

  LINKS = dict(
      models='https://api-docs.deepseek.com/quick_start/pricing',
      pricing='https://api-docs.deepseek.com/quick_start/pricing',
      rate_limits='https://api-docs.deepseek.com/quick_start/rate_limit',
      error_codes='https://api-docs.deepseek.com/quick_start/error_codes',
  )

  provider: Final[str] = 'DeepSeek'  # pylint: disable=invalid-name

  api_model_name: Annotated[
      str,
      'The model name used in the DeepSeek API.'
  ]


SUPPORTED_MODELS = [
    DeepSeekModelInfo(
        model_id='deepseek-r1',
        in_service=True,
        model_type='thinking',
        api_model_name='deepseek-reasoner',
        description='DeepSeek Reasoner model (01/20/2025).',
        url='https://api-docs.deepseek.com/news/news250120',
        release_date=datetime.datetime(2025, 1, 20),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=64_000,
            max_output_tokens=8_000,
            max_cot_tokens=32_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.14,
            cost_per_1m_input_tokens=0.55,
            cost_per_1m_output_tokens=2.19,
        ),
        # No rate limits is enforced by DeepSeek for now.
        rate_limits=None
    ),
    DeepSeekModelInfo(
        model_id='deepseek-v3',
        in_service=True,
        model_type='instruction-tuned',
        api_model_name='deepseek-chat',
        description='DeepSeek V3 model (12/26/2024).',
        url='https://api-docs.deepseek.com/news/news1226',
        release_date=datetime.datetime(2024, 12, 26),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=64_000,
            max_output_tokens=8_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.07,
            cost_per_1m_input_tokens=0.27,
            cost_per_1m_output_tokens=1.1,
        ),
        # No rate limits is enforced by DeepSeek for now.
        rate_limits=None
    ),
]

_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


# DeepSeek API uses an API format compatible with OpenAI.
# Reference: https://api-docs.deepseek.com/
@lf.use_init_args(['model'])
class DeepSeek(openai_compatible.OpenAICompatible):
  """DeepSeek model."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]
      ),
      'The name of the model to use.',
  ]

  api_endpoint: str = 'https://api.deepseek.com/chat/completions'

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'DEEPSEEK_API_KEY'."
      ),
  ] = None

  @property
  def headers(self) -> dict[str, Any]:
    api_key = self.api_key or os.environ.get('DEEPSEEK_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `DEEPSEEK_API_KEY` with your DeepSeek API key.'
      )
    headers = super().headers
    headers.update({
        'Authorization': f'Bearer {api_key}',
    })
    return headers

  @functools.cached_property
  def model_info(self) -> DeepSeekModelInfo:
    return _SUPPORTED_MODELS_BY_ID[self.model]

  def _request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # NOTE(daiyip): Replace model name with the API model name instead of the
    # model ID.
    args = super()._request_args(options)
    args['model'] = self.model_info.api_model_name
    return args

  @classmethod
  def dir(cls):
    return [m.model_id for m in SUPPORTED_MODELS if m.in_service]


class DeepSeekR1(DeepSeek):
  """DeepSeek Reasoner model.

  Currently it is powered by DeepSeek-R1 model, 64k input context, 8k max
  output, 32k max CoT output.
  """

  model = 'deepseek-r1'


class DeepSeekV3(DeepSeek):
  """DeepSeek Chat model.

  Currently, it is powered by DeepSeek-V3 model, 64K input contenxt window and
  8k max output tokens.
  """

  model = 'deepseek-v3'


def _register_deepseek_models():
  """Registers DeepSeek models."""
  for m in SUPPORTED_MODELS:
    lf.LanguageModel.register(m.model_id, DeepSeek)

_register_deepseek_models()
