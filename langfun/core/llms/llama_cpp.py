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
"""Language models from llama.cpp."""

from typing import Annotated
from langfun.core.llms import openai_compatible
import pyglove as pg


@pg.use_init_args(['url', 'model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class LlamaCppRemote(openai_compatible.OpenAIChatCompletionAPI):
  """LLaMA C++ models served via a remote server.

  This class provides an interface to interact with language models
  hosted on a LLaMA C++ server, which is compatible with the OpenAI
  Chat Completions API format.

  **Quick Start:**

  Assuming a LLaMA C++ server is running at `http://localhost:8080`,
  you can interact with it as follows:

  ```python
  import langfun as lf

  # If model name is not specified, it will use server's default.
  lm = lf.llms.LlamaCppRemote(url='http://localhost:8080')
  r = lm('Who are you?')
  print(r)
  ```

  **References:**

  *   https://github.com/ggerganov/llama.cpp/tree/master/examples/server
  """
  url: Annotated[
      str,
      'The URL of the LLaMA C++ server.',
  ]

  model: Annotated[
      str,
      'The name of the model to use.',
  ] = ''

  @property
  def api_endpoint(self) -> str:
    return self.url + '/completion'

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'LLaMAC++({self.model or ""})'

