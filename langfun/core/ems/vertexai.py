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
"""Vertex AI embedding models."""

import os
import random
from typing import Annotated, Any

import langfun.core as lf
from langfun.core.ems import rest
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


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAI(rest.REST):
  """Gemini embedding models served on Vertex AI.

  **Quick Start:**

  ```python
  import langfun as lf

  em = lf.ems.VertexAI(
      model='gemini-embedding-002',
      project='my-project', location='us-central1'
  )
  result = em('hello world')
  print(result.embedding)
  ```
  """

  model: Annotated[
      str | None,
      'Model name.',
  ] = None

  project: Annotated[
      str | list[str] | None,
      (
          'Vertex AI project ID(s). Can be a single project ID or a list of '
          'project IDs for load balancing (random selection per request). '
          'Or set from environment variable VERTEXAI_PROJECT.'
      ),
  ] = None

  location: Annotated[
      str | None,
      (
          'Vertex AI service location. Or set from environment variable '
          'VERTEXAI_LOCATION.'
      ),
  ] = 'us-central1'

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
    self._projects = None
    self._credentials = None

  def _initialize(self):
    project = self.project or os.environ.get('VERTEXAI_PROJECT', None)
    if not project:
      raise ValueError(
          'Please specify `project` during `__init__` or set environment '
          'variable `VERTEXAI_PROJECT` with your Vertex AI project ID.'
      )

    if isinstance(project, str) and ',' in project:
      project = [p.strip() for p in project.split(',')]

    if isinstance(project, str):
      self._projects = [project]
    else:
      self._projects = list(project)

    location = self.location or os.environ.get('VERTEXAI_LOCATION', None)
    if not location:
      raise ValueError(
          'Please specify `location` during `__init__` or set environment '
          'variable `VERTEXAI_LOCATION` with your Vertex AI service location.'
      )

    self._location = location

    credentials = self.credentials
    if credentials is None:
      # Use default credentials.
      credentials, _ = google_auth.default(
          scopes=['https://www.googleapis.com/auth/cloud-platform']
      )
    self._credentials = credentials

  @property
  def _project(self) -> str:
    """Returns a project ID. Randomly selects from list if multiple."""
    if len(self._projects) == 1:
      return self._projects[0]
    return random.choice(self._projects)

  def session(self):
    assert self._api_initialized
    assert self._credentials is not None
    assert auth_requests is not None
    s = auth_requests.AuthorizedSession(self._credentials)
    s.headers.update(self.headers or {})
    return s

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    project = self._project
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:embedContent'
    )

  def request(self, message: lf.Message) -> dict[str, Any]:
    parts = [{'text': message.text}]
    if message.referred_modalities:
      for _, modality in message.referred_modalities.items():
        if hasattr(modality, 'to_bytes'):
          import base64  # pylint: disable=g-import-not-at-top

          parts.append({
              'inline_data': {
                  'mime_type': 'image/png',
                  'data': base64.b64encode(modality.to_bytes()).decode('utf-8'),
              }
          })
          break

    request_body = {'content': {'parts': parts}}
    options = self.embedding_options
    if options.task_type is not None:
      request_body['taskType'] = options.task_type
    if options.output_dimensionality is not None:
      request_body['outputDimensionality'] = options.output_dimensionality
    return request_body

  def result(self, json_response: dict[str, Any]) -> lf.EmbeddingResult:
    embedding = json_response.get('embedding', {})
    values = embedding.get('values', [])
    return lf.EmbeddingResult(embedding=values)


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAIPredictAPI(VertexAI):
  """Vertex AI embedding models that use the Predict API.

  This is for models like `gemini-embedding-001`, `text-embedding-005`, and
  `text-multilingual-embedding-002` which use the `:predict` endpoint with
  `instances[].content` request format.
  """

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    project = self._project
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:predict'
    )

  def request(self, message: lf.Message) -> dict[str, Any]:
    if message.referred_modalities:
      raise ValueError(
          f'{self.model!r} does not support multimodal input. '
          f'Got modalities: {list(message.referred_modalities.keys())}. '
          f'Use a multimodal embedding model such as '
          f'VertexAIGeminiEmbedding2 instead.'
      )
    request_body = {
        'instances': [{'content': message.text}],
    }
    parameters = {}
    options = self.embedding_options
    if options.task_type is not None:
      parameters['task_type'] = options.task_type
    if options.output_dimensionality is not None:
      parameters['outputDimensionality'] = options.output_dimensionality
    if parameters:
      request_body['parameters'] = parameters
    return request_body

  def result(self, json_response: dict[str, Any]) -> lf.EmbeddingResult:
    prediction = json_response['predictions'][0]
    embeddings = prediction.get('embeddings', {})
    values = embeddings.get('values', [])
    stats = embeddings.get('statistics', {})
    return lf.EmbeddingResult(
        embedding=values,
        usage=lf.EmbeddingUsage(
            prompt_tokens=stats.get('token_count', 0),
        ),
    )


class VertexAIGeminiEmbedding2(VertexAI):  # pylint: disable=invalid-name
  """Gemini Embedding v2 model on Vertex AI.

  Uses the EmbedContent API. Supports multimodal inputs (text, image, audio,
  video, PDF) with 3072-dimensional output.
  """

  model = 'gemini-embedding-2-preview'
  location = 'us-central1'


class VertexAIGeminiEmbedding1(VertexAIPredictAPI):  # pylint: disable=invalid-name
  """Gemini Embedding v1 model on Vertex AI.

  Uses the Predict API. Supports text-only input with 3072-dimensional output.
  """

  model = 'gemini-embedding-001'
  location = 'us-central1'


class VertexAITextEmbedding005(VertexAIPredictAPI):  # pylint: disable=invalid-name
  """Text Embedding 005 model on Vertex AI.

  Uses the Predict API. English-optimized, 768-dimensional output by default.
  """

  model = 'text-embedding-005'
  location = 'us-central1'


class VertexAITextMultilingualEmbedding002(VertexAIPredictAPI):  # pylint: disable=invalid-name
  """Text Multilingual Embedding 002 model on Vertex AI.

  Uses the Predict API. Supports 100+ languages, 768-dimensional output
  by default.
  """

  model = 'text-multilingual-embedding-002'
  location = 'us-central1'


def _register_vertexai_models():
  for model_id, cls in [
      ('gemini-embedding-2-preview', VertexAIGeminiEmbedding2),
      ('gemini-embedding-001', VertexAIGeminiEmbedding1),
      ('text-embedding-005', VertexAITextEmbedding005),
      ('text-multilingual-embedding-002', VertexAITextMultilingualEmbedding002),
  ]:
    lf.EmbeddingModel.register(model_id, cls)


_register_vertexai_models()
