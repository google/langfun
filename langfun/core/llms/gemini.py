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
"""Gemini REST API (Shared by Google GenAI and Vertex AI)."""

import datetime
import functools
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.data.conversion import gemini as gemini_conversion  # pylint: disable=unused-import
from langfun.core.llms import rest
import pyglove as pg


class GeminiModelInfo(lf.ModelInfo):
  """Gemini model info."""

  # Constants for supported MIME types.
  INPUT_IMAGE_TYPES = [
      'image/png',
      'image/jpeg',
      'image/webp',
      'image/heic',
      'image/heif',
  ]

  INPUT_AUDIO_TYPES = [
      'audio/aac',
      'audio/flac',
      'audio/mp3',
      'audio/m4a',
      'audio/mpeg',
      'audio/mpga',
      'audio/mp4',
      'audio/opus',
      'audio/pcm',
      'audio/wav',
      'audio/webm',
  ]

  INPUT_VIDEO_TYPES = [
      'video/mov',
      'video/mpeg',
      'video/mpegps',
      'video/mpg',
      'video/mp4',
      'video/webm',
      'video/wmv',
      'video/x-flv',
      'video/3gpp',
      'video/quicktime',
  ]

  INPUT_DOC_TYPES = [
      'application/pdf',
      'text/plain',
      'text/csv',
      'text/html',
      'text/xml',
      'text/x-script.python',
      'application/json',
  ]

  ALL_SUPPORTED_INPUT_TYPES = (
      INPUT_IMAGE_TYPES
      + INPUT_AUDIO_TYPES
      + INPUT_VIDEO_TYPES
      + INPUT_DOC_TYPES
  )

  LINKS = dict(
      models='https://ai.google.dev/gemini-api/docs/models/gemini',
      pricing='https://ai.google.dev/gemini-api/docs/pricing',
      rate_limits='https://ai.google.dev/gemini-api/docs/models/gemini',
      error_codes='https://ai.google.dev/gemini-api/docs/troubleshooting?lang=python#error-codes',
  )

  class Pricing(lf.ModelInfo.Pricing):
    """Pricing for Gemini models."""

    cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k: Annotated[
        float | None,
        (
            'The cost per 1M cached input tokens for prompts longer than 128K. '
            'If None, the 128k constraint is not applicable.'
        )
    ] = None

    cost_per_1m_input_tokens_with_prompt_longer_than_128k: Annotated[
        float | None,
        (
            'The cost per 1M input tokens for prompts longer than 128K. '
            'If None, the 128k constraint is not applicable.'
        )
    ] = None

    cost_per_1m_output_tokens_with_prompt_longer_than_128k: Annotated[
        float | None,
        (
            'The cost per 1M output tokens for prompts longer than 128K.'
            'If None, the 128k constraint is not applicable.'
        )
    ] = None

    def estimate_cost(self, usage: lf.LMSamplingUsage) -> float | None:
      """Estimates the cost of using the model. Subclass could override.

      Args:
        usage: The usage information of the model.

      Returns:
        The estimated cost in US dollars. If None, cost estimating is not
        supported on the model.
      """
      if (usage.prompt_tokens is None
          or usage.prompt_tokens < 128_000
          or not self.cost_per_1m_input_tokens_with_prompt_longer_than_128k):
        return super().estimate_cost(usage)

      return (
          self.cost_per_1m_input_tokens_with_prompt_longer_than_128k
          * usage.prompt_tokens
          + self.cost_per_1m_output_tokens_with_prompt_longer_than_128k
          * (usage.total_tokens - usage.prompt_tokens)
      ) / 1000_000

  experimental: Annotated[
      bool,
      (
          'If True, the model is experimental and may retire without notice.'
      )
  ] = False


# !!! PLEASE KEEP MODELS SORTED BY MODEL FAMILY AND RELEASE DATE !!!


SUPPORTED_MODELS = [
    #
    # Production models.
    #
    # Gemini 2.5 Flash Preview 0520
    GeminiModelInfo(
        model_id='gemini-2.5-flash-preview-05-20',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Flash.',
        release_date=datetime.datetime(2025, 5, 20),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.15,
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.6,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 2.5 Pro Preview
    GeminiModelInfo(
        model_id='gemini-2.5-pro-preview-05-06',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Pro.',
        release_date=datetime.datetime(2025, 5, 6),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.25,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        )
    ),
    # Gemini 2.5 Flash Preview
    GeminiModelInfo(
        model_id='gemini-2.5-flash-preview-04-17',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Flash.',
        release_date=datetime.datetime(2025, 4, 17),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.15,
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.6,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 2.5 Pro Preview
    GeminiModelInfo(
        model_id='gemini-2.5-pro-preview-03-25',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Pro.',
        release_date=datetime.datetime(2025, 3, 25),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.25,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-2.5-pro-exp-03-25',
        in_service=True,
        provider=pg.oneof(['VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Pro.',
        release_date=datetime.datetime(2025, 3, 25),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.25,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=20,
            max_tokens_per_minute=1_000_000,
        ),
    ),
    # Gemini 2.0 Flash.
    GeminiModelInfo(
        model_id='gemini-2.0-flash',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.0 Flash model.',
        release_date=datetime.datetime(2025, 2, 5),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.025,
            cost_per_1m_input_tokens=0.1,
            cost_per_1m_output_tokens=0.4,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-2.0-flash-001',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.0 Flash model (version 001).',
        release_date=datetime.datetime(2025, 2, 5),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.025,
            cost_per_1m_input_tokens=0.1,
            cost_per_1m_output_tokens=0.4,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 2.0 Flash Lite.
    GeminiModelInfo(
        model_id='gemini-2.0-flash-lite-preview-02-05',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.0 Lite preview model.',
        release_date=datetime.datetime(2025, 2, 5),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.01875,
            cost_per_1m_input_tokens=0.075,
            cost_per_1m_output_tokens=0.3,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 1.5 Flash.
    GeminiModelInfo(
        model_id='gemini-1.5-flash',
        alias_for='gemini-1.5-flash-002',
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        in_service=True,
        model_type='instruction-tuned',
        description='Gemini 1.5 Flash model (latest stable).',
        release_date=datetime.datetime(2024, 9, 30),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.01875,
            cost_per_1m_input_tokens=0.075,
            cost_per_1m_output_tokens=0.3,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.0375,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=0.15,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=0.6,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-1.5-flash-001',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 1.5 Flash model (version 001).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.01875,
            cost_per_1m_input_tokens=0.075,
            cost_per_1m_output_tokens=0.3,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.0375,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=0.15,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=0.6,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-1.5-flash-002',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 1.5 Flash model (version 002).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.01875,
            cost_per_1m_input_tokens=0.075,
            cost_per_1m_output_tokens=0.3,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.0375,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=0.15,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=0.6,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 1.5 Flash-8B.
    GeminiModelInfo(
        model_id='gemini-1.5-flash-8b',
        in_service=True,
        provider='Google GenAI',
        model_type='instruction-tuned',
        description='Gemini 1.5 Flash 8B model (latest stable).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.01,
            cost_per_1m_input_tokens=0.0375,
            cost_per_1m_output_tokens=0.15,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.02,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=0.075,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=0.3,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-1.5-flash-8b-001',
        in_service=True,
        provider='Google GenAI',
        model_type='instruction-tuned',
        description='Gemini 1.5 Flash 8B model (version 001).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.01,
            cost_per_1m_input_tokens=0.0375,
            cost_per_1m_output_tokens=0.15,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.02,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=0.075,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=0.3,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 1.5 Pro.
    GeminiModelInfo(
        model_id='gemini-1.5-pro',
        alias_for='gemini-1.5-pro-002',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 1.5 Pro model (latest stable).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=2_097_152,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=5,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.625,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.5,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=10,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=1000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-1.5-pro-001',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 1.5 Pro model (version 001).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=2_097_152,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=5,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.625,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.5,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=10,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=1000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-1.5-pro-002',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 1.5 Pro model (version 002).',
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=2_097_152,
            max_output_tokens=8_192,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=5,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.625,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.5,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=10,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=1000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    #
    # Experimental models.
    #
    GeminiModelInfo(
        model_id='gemini-2.0-pro-exp-02-05',
        in_service=True,
        experimental=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.0 Pro experimental model (02/05/2025).',
        release_date=datetime.datetime(2025, 2, 5),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-2.0-flash-thinking-exp-01-21',
        in_service=True,
        experimental=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='thinking',
        description=(
            'Gemini 2.0 Flash thinking experimental model (01/21/2025).'
        ),
        release_date=datetime.datetime(2025, 1, 21),
        input_modalities=GeminiModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-exp-1206',
        in_service=True,
        experimental=True,
        provider='Google GenAI',
        model_type='instruction-tuned',
        description='Gemini year 1 experimental model (12/06/2024)',
        release_date=datetime.datetime(2025, 1, 21),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
    ),
    GeminiModelInfo(
        model_id='learnlm-1.5-pro-experimental',
        in_service=True,
        experimental=True,
        provider='Google GenAI',
        model_type='instruction-tuned',
        description='Gemini experimental model on learning science principles.',
        url='https://ai.google.dev/gemini-api/docs/learnlm',
        release_date=datetime.datetime(2024, 11, 19),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=8_192,
        ),
    ),
]


_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


@pg.use_init_args(['model'])
class Gemini(rest.REST):
  """Language models provided by Google GenAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]
      ),
      'The name of the model to use.',
  ]

  @functools.cached_property
  def model_info(self) -> GeminiModelInfo:
    return _SUPPORTED_MODELS_BY_ID[self.model]

  @classmethod
  def dir(cls):
    return [m.model_id for m in SUPPORTED_MODELS if m.in_service]

  @property
  def headers(self):
    return {
        'Content-Type': 'application/json; charset=utf-8',
    }

  def request(
      self, prompt: lf.Message, sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    request = dict(
        generationConfig=self._generation_config(prompt, sampling_options)
    )
    def modality_conversion(chunk: str | lf.Modality) -> Any:
      if isinstance(chunk, lf_modalities.Mime):
        try:
          return chunk.make_compatible(
              self.model_info.input_modalities + ['text/plain']
          )
        except lf.ModalityError as e:
          raise lf.ModalityError(f'Unsupported modality: {chunk!r}') from e
      return chunk

    if system_message := prompt.get('system_message'):
      assert isinstance(system_message, lf.SystemMessage), type(system_message)
      request['systemInstruction'] = system_message.as_format(
          'gemini', chunk_preprocessor=modality_conversion
      )

    contents = []
    contents.append(
        prompt.as_format('gemini', chunk_preprocessor=modality_conversion)
    )
    request['contents'] = contents
    return request

  def _generation_config(
      self, prompt: lf.Message, options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns a dict as generation config for prompt and LMSamplingOptions."""
    config = dict(
        temperature=options.temperature,
        maxOutputTokens=options.max_tokens,
        candidateCount=options.n,
        topK=options.top_k,
        topP=options.top_p,
        stopSequences=options.stop,
        seed=options.random_seed,
        responseLogprobs=options.logprobs,
        logprobs=options.top_logprobs,
    )

    if json_schema := prompt.metadata.get('json_schema'):
      if not isinstance(json_schema, dict):
        raise ValueError(
            f'`json_schema` must be a dict, got {json_schema!r}.'
        )
      json_schema = pg.to_json(json_schema)
      config['responseSchema'] = json_schema
      config['responseMimeType'] = 'application/json'
      prompt.metadata.formatted_text = (
          prompt.text
          + '\n\n [RESPONSE FORMAT (not part of prompt)]\n'
          + pg.to_json_str(json_schema, json_indent=2)
      )
    if options.max_thinking_tokens is not None:
      config['thinkingConfig'] = {
          'thinkingBudget': options.max_thinking_tokens
      }

    return config

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    candidates = json.get('candidates')
    if candidates is None:
      raise lf.TemporaryLMError(
          'No candidates found in response. This is a Gemini API issue that '
          'happens occasionally, and retrying should fix it. '
      )
    messages = [
        lf.Message.from_value(candidate['content'], format='gemini')
        for candidate in candidates
    ]
    usage = json['usageMetadata']
    input_tokens = usage['promptTokenCount']
    # NOTE(daiyip): We saw cases that `candidatesTokenCount` is not present.
    # Therefore, we use 0 as the default value.
    output_tokens = usage.get('candidatesTokenCount', 0)
    thinking_tokens = usage.get('thoughtsTokenCount', 0)
    total_tokens = usage.get('totalTokenCount', 0)

    return lf.LMSamplingResult(
        [lf.LMSample(message) for message in messages],
        usage=lf.LMSamplingUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            completion_tokens_details={
                'thinking_tokens': thinking_tokens,
            },
        ),
    )

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if (
        status_code == 400
        and b'exceeds the maximum number of tokens' in content
    ):
      return lf.ContextLimitError(f'{status_code}: {content}')
    return super()._error(status_code, content)
