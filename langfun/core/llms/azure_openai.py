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
"""OpenAI models hosted on Azure."""
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core.llms import openai
import pyglove as pg


@lf.use_init_args(['model', 'deployment_name'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class AzureOpenAI(openai.OpenAI):
  """Azure OpenAI models.

  **Quick Start:**

  ```python
  import langfun as lf

  # Call GPT-4o on Azure using API key from environment variable
  # 'AZURE_OPENAI_API_KEY'.
  lm = lf.llms.AzureOpenAI(
      model='gpt-4o',
      deployment_name='my-gpt4o-deployment',
      api_version='2024-08-01-preview',
      azure_endpoint='https://my-resource.openai.azure.com/',
  )
  r = lm('Who are you?')
  print(r)
  ```

  **Setting up API key:**

  The Azure OpenAI API key can be specified in following ways:

  1. At model instantiation:

     ```python
     lm = lf.llms.AzureOpenAI(..., api_key='MY_API_KEY')
     ```
  2. via environment variable `AZURE_OPENAI_API_KEY`.
  """

  deployment_name: Annotated[
      str,
      'The name of the Azure OpenAI deployment.'
  ] = 'gpt-4'

  api_version: Annotated[
      str,
      'The API version for Azure OpenAI.'
  ] = '2023-05-15'

  azure_endpoint: Annotated[
      str,
      (
          'The base URL for Azure OpenAI '
          '(e.g. "https://<your-resource>.openai.azure.com/")'
      )
  ] = 'https://api.openai.azure.com/'

  def _initialize(self):
    # Authentication
    self._api_key = self.api_key or os.environ.get('AZURE_OPENAI_API_KEY')
    if not self._api_key:
      raise ValueError(
          'Azure OpenAI requires an API key. Please provide '
          'via `api_key` argument or AZURE_OPENAI_API_KEY '
          'environment variable.'
      )

    # Endpoint construction
    self._api_endpoint = (
        f"{self.azure_endpoint.rstrip('/')}/openai/deployments/"
        f"{self.deployment_name}/chat/completions"
        f"?api-version={self.api_version}"
    )

  @property
  def api_endpoint(self) -> str:
    return self._api_endpoint

  @property
  def headers(self) -> dict[str, Any]:
    headers = super().headers
    headers.update({
        'api-key': self._api_key,
    })
    return headers
