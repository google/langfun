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
"""Language models from MiniMax."""

import datetime
import functools
import os
from typing import Annotated, Any, Final

import langfun.core as lf
from langfun.core.llms import openai_compatible
import pyglove as pg


class MiniMaxModelInfo(lf.ModelInfo):
  """MiniMax model info."""

  LINKS = dict(
      models='https://platform.minimaxi.com/document/Models',
      pricing='https://platform.minimaxi.com/document/Price',
  )

  provider: Final[str] = 'MiniMax'  # pylint: disable=invalid-name


SUPPORTED_MODELS = [
    MiniMaxModelInfo(
        model_id='MiniMax-M2.7',
        in_service=True,
        model_type='instruction-tuned',
        description='MiniMax M2.7 model with 1M context window.',
        url='https://platform.minimaxi.com/document/Models',
        release_date=datetime.datetime(2025, 3, 1),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=65_536,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=1.1,
            cost_per_1m_output_tokens=4.4,
        ),
        rate_limits=None,
    ),
    MiniMaxModelInfo(
        model_id='MiniMax-M2.7-highspeed',
        in_service=True,
        model_type='instruction-tuned',
        description='MiniMax M2.7 Highspeed model with 1M context window.',
        url='https://platform.minimaxi.com/document/Models',
        release_date=datetime.datetime(2025, 3, 1),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=65_536,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.55,
            cost_per_1m_output_tokens=2.2,
        ),
        rate_limits=None,
    ),
]

_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


# MiniMax API uses an API format compatible with OpenAI.
# Reference: https://platform.minimaxi.com/document/ChatCompletion%20v2
@lf.use_init_args(['model'])
class MiniMax(openai_compatible.OpenAIChatCompletionAPI):
  """MiniMax models.

  **Quick Start:**

  ```python
  import langfun as lf

  # Call MiniMax-M2.7 using API key from environment variable
  # 'MINIMAX_API_KEY'.
  lm = lf.llms.MiniMaxM27()
  r = lm('Who are you?')
  print(r)
  ```

  **Setting up API key:**

  The MiniMax API key can be specified in following ways:

  1. At model instantiation:

     ```python
     lm = lf.llms.MiniMaxM27(api_key='MY_API_KEY')
     ```
  2. via environment variable `MINIMAX_API_KEY`.

  **References:**

  *   https://platform.minimaxi.com/document/ChatCompletion%20v2
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]
      ),
      'The name of the model to use.',
  ]

  api_endpoint: str = 'https://api.minimax.io/v1/chat/completions'

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'MINIMAX_API_KEY'."
      ),
  ] = None

  @property
  def headers(self) -> dict[str, Any]:
    api_key = self.api_key or os.environ.get('MINIMAX_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `MINIMAX_API_KEY` with your MiniMax API key.'
      )
    headers = super().headers
    headers.update({
        'Authorization': f'Bearer {api_key}',
    })
    return headers

  @functools.cached_property
  def model_info(self) -> MiniMaxModelInfo:
    return _SUPPORTED_MODELS_BY_ID[self.model]

  def _request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    args = super()._request_args(options)
    # MiniMax requires temperature in (0.0, 1.0], clamp 0.0 to 0.01.
    if 'temperature' in args and args['temperature'] == 0.0:
      args['temperature'] = 0.01
    return args

  @classmethod
  def dir(cls):
    return [m.model_id for m in SUPPORTED_MODELS if m.in_service]


class MiniMaxM27(MiniMax):  # pylint: disable=invalid-name
  """MiniMax M2.7 model with 1M context window."""

  model = 'MiniMax-M2.7'


class MiniMaxM27Highspeed(MiniMax):  # pylint: disable=invalid-name
  """MiniMax M2.7 Highspeed model with 1M context window."""

  model = 'MiniMax-M2.7-highspeed'


def _register_minimax_models():
  """Registers MiniMax models."""
  for m in SUPPORTED_MODELS:
    lf.LanguageModel.register(m.model_id, MiniMax)

_register_minimax_models()
