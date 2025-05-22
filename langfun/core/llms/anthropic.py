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
"""Language models from Anthropic."""

import datetime
import functools
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.data.conversion import anthropic as anthropic_conversion  # pylint: disable=unused-import
from langfun.core.llms import rest
import pyglove as pg


class AnthropicModelInfo(lf.ModelInfo):
  """Anthropic model info."""

  # Constants for supported MIME types.
  INPUT_IMAGE_TYPES = [
      'image/png',
      'image/jpeg',
      'image/gif',
      'image/webp',
  ]
  INPUT_DOC_TYPES = [
      'application/pdf',
  ]

  LINKS = dict(
      models='https://docs.anthropic.com/claude/docs/models-overview',
      pricing='https://www.anthropic.com/pricing#anthropic-api',
      rate_limits='https://docs.anthropic.com/en/api/rate-limits',
      error_codes='https://docs.anthropic.com/en/api/errors',
  )

  class RateLimits(lf.ModelInfo.RateLimits):
    """Rate limits for Anthropic models."""

    max_input_tokens_per_minute: int
    max_output_tokens_per_minute: int

    @property
    def max_tokens_per_minute(self) -> int:
      return (self.max_input_tokens_per_minute
              + self.max_output_tokens_per_minute)


SUPPORTED_MODELS = [
    AnthropicModelInfo(
        model_id='claude-4-opus-20250514',
        provider='Anthropic',
        in_service=True,
        description='Claude 4 Opus model (5/14/2025).',
        release_date=datetime.datetime(2025, 5, 14),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.5,
            cost_per_1m_input_tokens=15,
            cost_per_1m_output_tokens=75,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=100_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-4-sonnet-20250514',
        provider='Anthropic',
        in_service=True,
        description='Claude 4 Sonnet model (5/14/2025).',
        release_date=datetime.datetime(2025, 5, 14),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=100_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    # 3.5 Sonnet models.
    AnthropicModelInfo(
        model_id='claude-3-5-sonnet-latest',
        alias_for='claude-3-5-sonnet-20241022',
        provider='Anthropic',
        in_service=True,
        description='Claude 3.5 Sonnet model (latest).',
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-7-sonnet-20250219',
        provider='Anthropic',
        in_service=True,
        description='Claude 3.7 Sonnet model (2/19/2025).',
        release_date=datetime.datetime(2025, 2, 19),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=100_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-5-sonnet-20241022',
        provider='Anthropic',
        in_service=True,
        description='Claude 3.5 Sonnet model (10/22/2024).',
        release_date=datetime.datetime(2024, 10, 22),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4@20250514',
        alias_for='claude-opus-4-20250514',
        provider='VertexAI',
        in_service=True,
        description='Claude 4 Opus model served on VertexAI (5/14/2025).',
        release_date=datetime.datetime(2025, 5, 14),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.5,
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=75.0,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-sonnet-4@20250514',
        alias_for='claude-sonnet-4@20250514',
        provider='VertexAI',
        in_service=True,
        description='Claude 4 Sonnet model served on VertexAI (5/14/2025).',
        release_date=datetime.datetime(2025, 5, 14),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=15.0,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-5-sonnet-v2@20241022',
        alias_for='claude-3-5-sonnet-20241022',
        provider='VertexAI',
        in_service=True,
        description='Claude 3.5 Sonnet model served on VertexAI (10/22/2024).',
        release_date=datetime.datetime(2024, 10, 22),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-7-sonnet@20250219',
        alias_for='claude-3-7-sonnet-20250219',
        provider='VertexAI',
        in_service=True,
        description='Claude 3.7 Sonnet model served on VertexAI (02/19/2025).',
        release_date=datetime.datetime(2025, 2, 19),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    # 3.5 Haiku models.
    AnthropicModelInfo(
        model_id='claude-3-5-haiku-latest',
        alias_for='claude-3-5-haiku-20241022',
        provider='Anthropic',
        in_service=True,
        description='Claude 3.5 Haiku v2 model (10/22/2024).',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.08,
            cost_per_1m_input_tokens=0.8,
            cost_per_1m_output_tokens=4,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-5-haiku-20241022',
        provider='Anthropic',
        in_service=True,
        description='Claude 3.5 Haiku v2 model (10/22/2024).',
        release_date=datetime.datetime(2024, 10, 22),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.08,
            cost_per_1m_input_tokens=0.8,
            cost_per_1m_output_tokens=4,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-5-haiku@20241022',
        alias_for='claude-3-5-haiku-20241022',
        provider='VertexAI',
        in_service=True,
        description='Claude 3.5 Haiku model served on VertexAI (10/22/2024).',
        release_date=datetime.datetime(2024, 10, 22),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.08,
            cost_per_1m_input_tokens=0.8,
            cost_per_1m_output_tokens=4,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    # 3.0 Opus models.
    AnthropicModelInfo(
        model_id='claude-3-opus-latest',
        alias_for='claude-3-opus-20240229',
        provider='Anthropic',
        in_service=True,
        description='Claude 3 Opus model (latest).',
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.5,
            cost_per_1m_input_tokens=15,
            cost_per_1m_output_tokens=75,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-opus-20240229',
        provider='Anthropic',
        in_service=True,
        description='Claude 3 Opus model (02/29/2024).',
        release_date=datetime.datetime(2024, 2, 29),
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.5,
            cost_per_1m_input_tokens=15,
            cost_per_1m_output_tokens=75,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-opus@20240229',
        alias_for='claude-3-opus-20240229',
        provider='VertexAI',
        in_service=True,
        description='Claude 3 Opus model served on VertexAI (02/29/2024).',
        release_date=datetime.datetime(2024, 2, 29),
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.5,
            cost_per_1m_input_tokens=15,
            cost_per_1m_output_tokens=75,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    # 3.0 Sonnet models.
    AnthropicModelInfo(
        model_id='claude-3-sonnet-20240229',
        provider='Anthropic',
        in_service=True,
        description='Claude 3 Sonnet model (02/29/2024).',
        release_date=datetime.datetime(2024, 2, 29),
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-sonnet@20240229',
        alias_for='claude-3-sonnet-20240229',
        provider='VertexAI',
        in_service=True,
        description='Claude 3 Sonnet model served on VertexAI (02/29/2024).',
        release_date=datetime.datetime(2024, 2, 29),
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    # 3.0 Haiku models.
    AnthropicModelInfo(
        model_id='claude-3-haiku-20240307',
        provider='Anthropic',
        in_service=True,
        description='Claude 3 Haiku model (03/07/2024).',
        release_date=datetime.datetime(2024, 3, 7),
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=0.25,
            cost_per_1m_output_tokens=1.25,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-3-haiku@20240307',
        alias_for='claude-3-haiku-20240307',
        provider='Anthropic',
        in_service=True,
        description='Claude 3 Haiku model served on VertexAI (03/07/2024).',
        release_date=datetime.datetime(2024, 3, 7),
        input_modalities=AnthropicModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=0.25,
            cost_per_1m_output_tokens=1.25,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=400_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
]


_SUPPORTED_MODELS_BY_MODEL_ID = {m.model_id: m for m in SUPPORTED_MODELS}


@lf.use_init_args(['model'])
class Anthropic(rest.REST):
  """Anthropic LLMs (Claude) through REST APIs.

  See https://docs.anthropic.com/claude/reference/messages_post
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]
      ),
      'The name of the model to use.',
  ]

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'ANTHROPIC_API_KEY'."
      ),
  ] = None

  api_endpoint: str = 'https://api.anthropic.com/v1/messages'

  api_version: Annotated[
      str,
      'Anthropic API version.'
  ] = '2023-06-01'

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None

  def _initialize(self):
    api_key = self.api_key or os.environ.get('ANTHROPIC_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `ANTHROPIC_API_KEY` with your Anthropic API key.'
      )
    self._api_key = api_key

  @property
  def headers(self) -> dict[str, Any]:
    return {
        'x-api-key': self._api_key,
        'anthropic-version': self.api_version,
        'content-type': 'application/json',
        'anthropic-beta': 'output-128k-2025-02-19',
    }

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    mi = _SUPPORTED_MODELS_BY_MODEL_ID[self.model]
    if mi.provider != 'Anthropic':
      assert mi.alias_for is not None
      mi = _SUPPORTED_MODELS_BY_MODEL_ID[mi.alias_for]
      assert mi.provider == 'Anthropic', mi
    return mi

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request = dict()
    request.update(self._request_args(sampling_options))

    def modality_check(chunk: Any) -> Any:
      if isinstance(chunk, lf_modalities.Mime):
        if not self.supports_input(chunk.mime_type):
          raise ValueError(f'Unsupported modality: {chunk!r}.')
      return chunk

    if system_message := prompt.get('system_message'):
      assert isinstance(system_message, lf.SystemMessage), type(system_message)
      request['system'] = system_message.text

    messages = [
        prompt.as_format('anthropic', chunk_preprocessor=modality_check)
    ]
    request.update(messages=messages)
    return request

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # Authropic requires `max_tokens` to be specified.
    max_tokens = (
        options.max_tokens or self.model_info.context_length.max_output_tokens
    )
    args = dict(
        model=self.model,
        max_tokens=max_tokens,
        stream=False,
    )
    if options.stop:
      args['stop_sequences'] = options.stop
    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.top_k is not None:
      args['top_k'] = options.top_k
    if options.top_p is not None:
      args['top_p'] = options.top_p
    if options.max_thinking_tokens is not None:
      args['thinking'] = {
          'type': 'enabled',
          # Minimum budget is 1,024 tokens.
          'budget_tokens': options.max_thinking_tokens,
      }
      # max_tokens, which is thinking tokens + response tokens, must be greater
      # than the thinking tokens.
      if args['max_tokens'] < options.max_thinking_tokens:
        args['max_tokens'] += options.max_thinking_tokens

      # Thinking isn’t compatible with temperature, top_p, or top_k.
      # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking
      args.pop('temperature', None)
      args.pop('top_k', None)
      args.pop('top_p', None)
    return args

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    message = lf.Message.from_value(json, format='anthropic')
    input_tokens = json['usage']['input_tokens']
    output_tokens = json['usage']['output_tokens']
    return lf.LMSamplingResult(
        [lf.LMSample(message)],
        usage=lf.LMSamplingUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 413 and b'Prompt is too long' in content:
      return lf.ContextLimitError(f'{status_code}: {content}')
    return super()._error(status_code, content)


class Claude4(Anthropic):
  """Base class for Claude 4 models."""


# pylint: disable=invalid-name
class Claude4Opus_20250514(Claude4):
  """Claude 4 Opus model 20250514."""

  model = 'claude-4-opus-20250514'


# pylint: disable=invalid-name
class Claude4Sonnet_20250514(Claude4):
  """Claude 4 Sonnet model 20250514."""

  model = 'claude-4-sonnet-20250514'


class Claude37(Anthropic):
  """Base class for Claude 3.7 models."""


# pylint: disable=invalid-name
class Claude37Sonnet_20250219(Claude37):
  """Claude 3.7 Sonnet model (latest)."""

  model = 'claude-3-7-sonnet-20250219'


class Claude35(Anthropic):
  """Base class for Claude 3.5 models."""


class Claude35Sonnet(Claude35):
  """Claude 3.5 Sonnet model (latest)."""
  model = 'claude-3-5-sonnet-latest'


class Claude35Sonnet_20241022(Claude35):  # pylint: disable=invalid-name
  """Claude 3.5 Sonnet model (10/22/2024)."""
  model = 'claude-3-5-sonnet-20241022'


class Claude35Haiku(Claude35):
  """Claude 3.5 Haiku model (latest)."""
  model = 'claude-3-5-haiku-latest'


class Claude35Haiku_20241022(Claude35):  # pylint: disable=invalid-name
  """Claude 3.5 Haiku model (10/22/2024)."""
  model = 'claude-3-5-haiku-20241022'


class Claude3(Anthropic):
  """Base class for Claude 3 models."""


class Claude3Opus(Claude3):
  """Claude 3 Opus model (latest)."""

  model = 'claude-3-opus-latest'


class Claude3Opus_20240229(Claude3):  # pylint: disable=invalid-name
  """Claude 3 Opus model (02/29/2024)."""

  model = 'claude-3-opus-20240229'


class Claude3Sonnet(Claude3):
  """Claude 3 Sonnet model."""

  model = 'claude-3-sonnet-20240229'


class Claude3Sonnet_20240229(Claude3):  # pylint: disable=invalid-name
  """Claude 3 Sonnet model (02/29/2024)."""

  model = 'claude-3-sonnet-20240229'


class Claude3Haiku(Claude3):
  """Claude 3 Haiku model."""

  model = 'claude-3-haiku-20240307'


class Claude3Haiku_20240307(Claude3):  # pylint: disable=invalid-name
  """Claude 3 Haiku model (03/07/2024)."""

  model = 'claude-3-haiku-20240307'


def _register_anthropic_models():
  """Registers Anthropic models."""
  for m in SUPPORTED_MODELS:
    if m.provider == 'Anthropic':
      lf.LanguageModel.register(m.model_id, Anthropic)

_register_anthropic_models()
