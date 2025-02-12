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
"""Language models from Google GenAI."""

import functools
import os
from typing import Annotated, Literal

import langfun.core as lf
from langfun.core.llms import gemini
import pyglove as pg


@lf.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class GenAI(gemini.Gemini):
  """Language models provided by Google GenAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE,
          [
              m.model_id for m in gemini.SUPPORTED_MODELS
              if m.provider == 'Google GenAI' or (
                  isinstance(m.provider, pg.hyper.OneOf)
                  and 'Google GenAI' in m.provider.candidates
              )
          ]
      ),
      'The name of the model to use.',
  ]

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'GOOGLE_API_KEY'. "
          'Get an API key at https://ai.google.dev/tutorials/setup'
      ),
  ] = None

  api_version: Annotated[
      Literal['v1beta', 'v1alpha'],
      'The API version to use.'
  ] = 'v1beta'

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    return super().model_info.clone(
        override=dict(provider='Google GenAI')
    )

  @property
  def api_endpoint(self) -> str:
    api_key = self.api_key or os.environ.get('GOOGLE_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GOOGLE_API_KEY` with your Google Cloud API key. '
          'Check out '
          'https://cloud.google.com/api-keys/docs/create-manage-api-keys '
          'for more details.'
      )
    return (
        f'https://generativelanguage.googleapis.com/{self.api_version}'
        f'/models/{self.model}:generateContent?'
        f'key={api_key}'
    )


# pylint: disable=invalid-name

#
# Experimental models.
#


class Gemini2ProExp_20250205(GenAI):
  """Gemini 2.0 Pro experimental model launched on 02/05/2025."""
  model = 'gemini-2.0-pro-exp-02-05'


class Gemini2FlashThinkingExp_20250121(GenAI):
  """Gemini 2.0 Flash Thinking model launched on 01/21/2025."""
  api_version = 'v1beta'
  model = 'gemini-2.0-flash-thinking-exp-01-21'
  timeout = None


class GeminiExp_20241206(GenAI):
  """Gemini Experimental model launched on 12/06/2024."""
  model = 'gemini-exp-1206'


#
# Production models.
#


class Gemini2Flash(GenAI):
  """Gemini 2.0 Flash model (latest stable)."""
  model = 'gemini-2.0-flash'


class Gemini2Flash_001(GenAI):
  """Gemini 2.0 Flash model launched on 02/05/2025."""
  model = 'gemini-2.0-flash-001'


class Gemini2FlashLitePreview_20250205(GenAI):
  """Gemini 2.0 Flash lite preview model launched on 02/05/2025."""
  model = 'gemini-2.0-flash-lite-preview-02-05'


class Gemini15Pro(GenAI):
  """Gemini 1.5 Pro latest stable model."""
  model = 'gemini-1.5-pro'


class Gemini15Pro_002(GenAI):
  """Gemini 1.5 Pro stable version 002."""
  model = 'gemini-1.5-pro-002'


class Gemini15Pro_001(GenAI):
  """Gemini 1.5 Pro stable version 001."""
  model = 'gemini-1.5-pro-001'


class Gemini15Flash(GenAI):
  """Gemini 1.5 Flash latest model."""
  model = 'gemini-1.5-flash'


class Gemini15Flash_002(GenAI):
  """Gemini 1.5 Flash model stable version 002."""
  model = 'gemini-1.5-flash-002'


class Gemini15Flash_001(GenAI):
  """Gemini 1.5 Flash model stable version 001."""
  model = 'gemini-1.5-flash-001'


class Gemini15Flash8B(GenAI):
  """Gemini 1.5 Flash 8B modle (latest stable)."""
  model = 'gemini-1.5-flash-8b'


class Gemini15Flash8B_001(GenAI):
  """Gemini 1.5 Flash 8B model (version 001)."""
  model = 'gemini-1.5-flash-8b-001'


# For backward compatibility.
GeminiPro1_5 = Gemini15Pro
GeminiFlash1_5 = Gemini15Flash

# pylint: enable=invalid-name


def _genai_model(model: str, *args, **kwargs) -> GenAI:
  model = model.removeprefix('google_genai://')
  return GenAI(model=model, *args, **kwargs)


def _register_genai_models():
  """Register GenAI models."""
  for m in gemini.SUPPORTED_MODELS:
    lf.LanguageModel.register('google_genai://' + m.model_id, _genai_model)

_register_genai_models()
