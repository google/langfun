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

import os
from typing import Annotated, Literal

import langfun.core as lf
from langfun.core.llms import gemini
import pyglove as pg


@lf.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class GenAI(gemini.Gemini):
  """Language models provided by Google GenAI."""

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

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'GenAI({self.model})'

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


class GeminiFlash2_0ThinkingExp_20241219(GenAI):  # pylint: disable=invalid-name
  """Gemini Flash 2.0 Thinking model launched on 12/19/2024."""

  api_version = 'v1alpha'
  model = 'gemini-2.0-flash-thinking-exp-1219'
  timeout = None


class GeminiFlash2_0Exp(GenAI):  # pylint: disable=invalid-name
  """Gemini Flash 2.0 model launched on 12/11/2024."""

  model = 'gemini-2.0-flash-exp'


class GeminiExp_20241206(GenAI):  # pylint: disable=invalid-name
  """Gemini Experimental model launched on 12/06/2024."""

  model = 'gemini-exp-1206'


class GeminiExp_20241114(GenAI):  # pylint: disable=invalid-name
  """Gemini Experimental model launched on 11/14/2024."""

  model = 'gemini-exp-1114'


class GeminiPro1_5(GenAI):  # pylint: disable=invalid-name
  """Gemini Pro latest model."""

  model = 'gemini-1.5-pro-latest'


class GeminiPro1_5_002(GenAI):  # pylint: disable=invalid-name
  """Gemini Pro latest model."""

  model = 'gemini-1.5-pro-002'


class GeminiPro1_5_001(GenAI):  # pylint: disable=invalid-name
  """Gemini Pro latest model."""

  model = 'gemini-1.5-pro-001'


class GeminiFlash1_5(GenAI):  # pylint: disable=invalid-name
  """Gemini Flash latest model."""

  model = 'gemini-1.5-flash-latest'


class GeminiFlash1_5_002(GenAI):  # pylint: disable=invalid-name
  """Gemini Flash 1.5 model stable version 002."""

  model = 'gemini-1.5-flash-002'


class GeminiFlash1_5_001(GenAI):  # pylint: disable=invalid-name
  """Gemini Flash 1.5 model stable version 001."""

  model = 'gemini-1.5-flash-001'


class GeminiPro1(GenAI):  # pylint: disable=invalid-name
  """Gemini 1.0 Pro model."""

  model = 'gemini-1.0-pro'
