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
"""OpenAI embedding models."""

import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core.ems import rest
import pyglove as pg


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class OpenAI(rest.REST):
  """OpenAI embedding models.

  **Quick Start:**

  ```python
  import langfun as lf

  em = lf.ems.OpenAI(model='text-embedding-3-small')
  result = em('hello world')
  print(result.embedding)
  ```

  **Setting up API key:**

  The OpenAI API key can be specified in following ways:

  1. At model instantiation:

     ```python
     em = lf.ems.OpenAI(
         model='text-embedding-3-small', api_key='MY_API_KEY'
     )
     ```
  2. via environment variable `OPENAI_API_KEY`.
  """

  model: Annotated[
      str | None,
      'Model name.',
  ] = None

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'OPENAI_API_KEY'."
      ),
  ] = None

  organization: Annotated[
      str | None,
      (
          'Organization. If None, the key will be read from environment '
          "variable 'OPENAI_ORGANIZATION'."
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None
    self._organization = None

  def _initialize(self) -> None:
    api_key = self.api_key or os.environ.get('OPENAI_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `OPENAI_API_KEY` with your OpenAI API key.'
      )
    self._api_key = api_key
    self._organization = self.organization or os.environ.get(
        'OPENAI_ORGANIZATION', None
    )

  @property
  def api_endpoint(self) -> str:
    return 'https://api.openai.com/v1/embeddings'

  @property
  def headers(self) -> dict[str, Any]:
    assert self._api_initialized
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self._api_key}',
    }
    if self._organization:
      headers['OpenAI-Organization'] = self._organization
    return headers

  def request(self, message: lf.Message) -> dict[str, Any]:
    if message.referred_modalities:
      raise ValueError(
          f'{self.model!r} does not support multimodal input. '
          f'Got modalities: {list(message.referred_modalities.keys())}.'
      )
    request_body = {
        'model': self.model,
        'input': message.text,
    }
    options = self.embedding_options
    if options.output_dimensionality is not None:
      request_body['dimensions'] = options.output_dimensionality
    return request_body

  def result(self, json_response: dict[str, Any]) -> lf.EmbeddingResult:
    embedding = json_response['data'][0]['embedding']
    usage_data = json_response.get('usage', {})
    return lf.EmbeddingResult(
        embedding=embedding,
        usage=lf.EmbeddingUsage(
            prompt_tokens=usage_data.get('prompt_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0),
        ),
    )


class TextEmbedding3Small(OpenAI):  # pylint: disable=invalid-name
  """OpenAI text-embedding-3-small model.

  Dimensions: 1536 (default), supports custom dimensions.
  Max input tokens: 8191.
  """

  model = 'text-embedding-3-small'


class TextEmbedding3Large(OpenAI):  # pylint: disable=invalid-name
  """OpenAI text-embedding-3-large model.

  Dimensions: 3072 (default), supports custom dimensions.
  Max input tokens: 8191.
  """

  model = 'text-embedding-3-large'


class TextEmbeddingAda002(OpenAI):  # pylint: disable=invalid-name
  """OpenAI text-embedding-ada-002 model.

  Dimensions: 1536 (fixed, does not support custom dimensions).
  Max input tokens: 8191.
  """

  model = 'text-embedding-ada-002'


def _register_openai_models():
  for model_id, cls in [
      ('text-embedding-3-small', TextEmbedding3Small),
      ('text-embedding-3-large', TextEmbedding3Large),
      ('text-embedding-ada-002', TextEmbeddingAda002),
  ]:
    lf.EmbeddingModel.register(model_id, cls)


_register_openai_models()
