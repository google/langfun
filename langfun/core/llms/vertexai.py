# Copyright 2023 The Langfun Authors
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
"""Vertex AI generative models."""

import functools
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core.llms import gemini
import pyglove as pg

try:
  # pylint: disable=g-import-not-at-top
  from google import auth as google_auth
  from google.auth import credentials as credentials_lib
  from google.auth.transport import requests as auth_requests
  # pylint: enable=g-import-not-at-top

  Credentials = credentials_lib.Credentials
except ImportError:
  google_auth = None
  credentials_lib = None
  auth_requests = None
  Credentials = Any


@lf.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAI(gemini.Gemini):
  """Language model served on VertexAI with REST API."""

  project: Annotated[
      str | None,
      (
          'Vertex AI project ID. Or set from environment variable '
          'VERTEXAI_PROJECT.'
      ),
  ] = None

  location: Annotated[
      str | None,
      (
          'Vertex AI service location. Or set from environment variable '
          'VERTEXAI_LOCATION.'
      ),
  ] = None

  credentials: Annotated[
      Credentials | None,
      (
          'Credentials to use. If None, the default credentials to the '
          'environment will be used.'
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    if google_auth is None:
      raise ValueError(
          'Please install "langfun[llm-google-vertex]" to use Vertex AI models.'
      )
    self._project = None
    self._credentials = None

  def _initialize(self):
    project = self.project or os.environ.get('VERTEXAI_PROJECT', None)
    if not project:
      raise ValueError(
          'Please specify `project` during `__init__` or set environment '
          'variable `VERTEXAI_PROJECT` with your Vertex AI project ID.'
      )

    location = self.location or os.environ.get('VERTEXAI_LOCATION', None)
    if not location:
      raise ValueError(
          'Please specify `location` during `__init__` or set environment '
          'variable `VERTEXAI_LOCATION` with your Vertex AI service location.'
      )

    self._project = project
    self._location = location

    credentials = self.credentials
    if credentials is None:
      # Use default credentials.
      credentials = google_auth.default(
          scopes=['https://www.googleapis.com/auth/cloud-platform']
      )
    self._credentials = credentials

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'VertexAI({self.model})'

  @functools.cached_property
  def _session(self):
    assert self._api_initialized
    assert self._credentials is not None
    assert auth_requests is not None
    s = auth_requests.AuthorizedSession(self._credentials)
    s.headers.update(self.headers or {})
    return s

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:generateContent'
    )


class VertexAIGeminiFlash2_0ThinkingExp_20241219(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini Flash 2.0 Thinking model launched on 12/19/2024."""

  api_version = 'v1alpha'
  model = 'gemini-2.0-flash-thinking-exp-1219'
  timeout = None


class VertexAIGeminiFlash2_0Exp(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 2.0 Flash model."""

  model = 'gemini-2.0-flash-exp'


class VertexAIGeminiExp_20241206(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini Experimental model launched on 12/06/2024."""

  model = 'gemini-exp-1206'


class VertexAIGeminiExp_20241114(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini Experimental model launched on 11/14/2024."""

  model = 'gemini-exp-1114'


class VertexAIGeminiPro1_5(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-latest'


class VertexAIGeminiPro1_5_002(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-002'


class VertexAIGeminiPro1_5_001(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-001'


class VertexAIGeminiFlash1_5(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash'


class VertexAIGeminiFlash1_5_002(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-002'


class VertexAIGeminiFlash1_5_001(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-001'


class VertexAIGeminiPro1(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.0 Pro model."""

  model = 'gemini-1.0-pro'
