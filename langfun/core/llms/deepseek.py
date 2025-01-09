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

import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core.llms import openai_compatible
import pyglove as pg

SUPPORTED_MODELS_AND_SETTINGS = {
    # pylint: disable=g-line-too-long
    # TODO(yifenglu): The RPM and TPM are arbitrary numbers. Update them once DeepSeek provides concrete guidelines.
    # DeepSeek doesn't control the rate limit at the moment: https://api-docs.deepseek.com/quick_start/rate_limit
    # The cost is based on: https://api-docs.deepseek.com/quick_start/pricing
    'deepseek-chat': pg.Dict(
        in_service=True,
        rpm=100,
        tpm=1000000,
        cost_per_1k_input_tokens=0.00014,
        cost_per_1k_output_tokens=0.00028,
    ),
}


# DeepSeek API uses an API format compatible with OpenAI.
# Reference: https://api-docs.deepseek.com/
@lf.use_init_args(['model'])
class DeepSeek(openai_compatible.OpenAICompatible):
  """DeepSeek model."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
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

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'DeepSeek({self.model})'

  @property
  def max_concurrency(self) -> int:
    rpm = SUPPORTED_MODELS_AND_SETTINGS[self.model].get('rpm', 0)
    tpm = SUPPORTED_MODELS_AND_SETTINGS[self.model].get('tpm', 0)
    return self.rate_to_max_concurrency(
        requests_per_min=rpm, tokens_per_min=tpm
    )

  def estimate_cost(
      self, num_input_tokens: int, num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    cost_per_1k_input_tokens = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_input_tokens', None
    )
    cost_per_1k_output_tokens = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_output_tokens', None
    )
    if cost_per_1k_output_tokens is None or cost_per_1k_input_tokens is None:
      return None
    return (
        cost_per_1k_input_tokens * num_input_tokens
        + cost_per_1k_output_tokens * num_output_tokens
    ) / 1000

  @classmethod
  def dir(cls):
    return [k for k, v in SUPPORTED_MODELS_AND_SETTINGS.items() if v.in_service]


class DeepSeekChat(DeepSeek):
  """DeepSeek Chat model.

  Currently, it is powered by DeepSeek-V3 model, 64K input contenxt window and
  8k max output tokens.
  """

  model = 'deepseek-chat'
