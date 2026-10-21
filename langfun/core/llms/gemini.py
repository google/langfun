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
import re
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.data.conversion import gemini as gemini_conversion  # pylint: disable=unused-import
from langfun.core.llms import rest
import pyglove as pg
import requests


class GeminiEmptyGenerationError(lf.LMError):
  """Gemini returned a candidate with no answer content, deterministically.

  Three response signatures produce an answer-less candidate as an EXPECTED
  function of the request rather than as a transient server fault:

  * ``finishReason == 'MAX_TOKENS'`` with no non-thought part -- thinking
    consumed the shared decode budget before a single answer token was
    emitted.
  * ``finishReason == 'STOP'`` with ``candidatesTokenCount == 0`` -- the model
    terminated normally after emitting nothing at all.
  * no ``content`` on the candidate at all (absent or null) -- nothing was
    generated, so there is not even an empty message to parse.

  All three are reproduced by re-sending the same request, so this subclasses
  the NON-retryable ``lf.LMError`` -- deliberately NOT
  ``lf.EmptyGenerationError`` / ``lf.RetryableLMError``. Without this, the
  first two reach ``LanguageModel.sample()`` as empty text, which raises the
  retryable ``lf.EmptyGenerationError`` and re-sends the full prompt up to
  ``max_attempts`` times -- burning quota and latency only to fail with an
  opaque error -- while the third escapes as a bare ``KeyError('content')``.
  Mirrors ``beyond_api.BeyondThinkingBudgetExhaustedError``.
  """


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
      # Use 128K+ pricing if applicable
      if (
          usage.prompt_tokens is not None
          and usage.prompt_tokens >= 128_000
          and self.cost_per_1m_input_tokens_with_prompt_longer_than_128k
      ):
        # Calculate non-cached prompt tokens
        non_cached_prompt_tokens = (
            usage.prompt_tokens - usage.cached_prompt_tokens
        )
        cost = (
            self.cost_per_1m_input_tokens_with_prompt_longer_than_128k
            * non_cached_prompt_tokens
        )

        # Add cost for cached prompt tokens (if pricing is available)
        if (
            self.cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k
            is not None
        ):
          cost += (
              self.cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k
              * usage.cached_prompt_tokens
          )
        else:
          # Fall back to regular 128K+ input pricing for cached tokens
          cost += (
              self.cost_per_1m_input_tokens_with_prompt_longer_than_128k
              * usage.cached_prompt_tokens
          )

        # Add cost for output tokens
        cost += (
            self.cost_per_1m_output_tokens_with_prompt_longer_than_128k  # pyrefly: ignore[unsupported-operation]
            * usage.completion_tokens
        )

        return cost / 1000_000

      # Use regular pricing (delegates to parent class which handles cached
      # tokens)
      return super().estimate_cost(usage)

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
    # Gemini 4 Argon
    GeminiModelInfo(
        model_id='gemini-4-argon',
        in_service=True,
        provider='VertexAI',
        model_type='instruction-tuned',
        description=(
            'Gemini 4 Argon: Multimodal model for complex reasoning and'
            ' agentic workflows.'
        ),
        release_date=datetime.datetime(2026, 10, 13),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        # pricing, rate_limits and knowledge_cutoff are left unset, so they
        # default to None, which callers already read as "unknown". Launch
        # pricing is introductory and scheduled to rise, so pinning a rate here
        # would go stale with no signal to the reader.
    ),
    # Gemini 3.1 Pro Preview
    GeminiModelInfo(
        model_id='gemini-3.1-pro-preview',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 3.1 Pro Preview.',
        release_date=datetime.datetime(2026, 2, 19),
        knowledge_cutoff=datetime.date(2025, 1, 31),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.2,
            cost_per_1m_input_tokens=2.0,
            cost_per_1m_output_tokens=12.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.4,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=4.0,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=18.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 3.1 Flash Lite Preview
    GeminiModelInfo(
        model_id='gemini-3.1-flash-lite-preview',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 3.1 Flash Lite Preview.',
        # https://ai.google.dev/gemini-api/docs/changelog#03-03-2026
        release_date=datetime.datetime(2026, 3, 3),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            # https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-lite-preview
            cost_per_1m_cached_input_tokens=0.025,
            cost_per_1m_input_tokens=0.25,
            cost_per_1m_output_tokens=1.5,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    GeminiModelInfo(
        model_id='gemini-3.1-flash-lite',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 3.1 Flash Lite.',
        release_date=datetime.datetime(2026, 5, 7),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.025,
            cost_per_1m_input_tokens=0.25,
            cost_per_1m_output_tokens=1.5,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 3 Pro Preview
    GeminiModelInfo(
        model_id='gemini-3-pro-preview',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 3 Pro Preview.',
        release_date=datetime.datetime(2025, 11, 18),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.2,
            cost_per_1m_input_tokens=2.0,
            cost_per_1m_output_tokens=12.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.4,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=4.0,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=18.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 3 Pro Image Preview
    GeminiModelInfo(
        model_id='gemini-3-pro-image-preview',
        in_service=True,
        experimental=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description=(
            'Gemini 3 Pro Image Preview for high-fidelity image generation,'
            ' editing, and visual reasoning.'
        ),
        release_date=datetime.datetime(2025, 12, 9),
        input_modalities=GeminiModelInfo.INPUT_IMAGE_TYPES
        + GeminiModelInfo.INPUT_DOC_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=65_536,
            max_output_tokens=32_768,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=200,
            max_tokens_per_minute=1_000_000,
        ),
    ),
    # Gemini 3 Flash Preview
    GeminiModelInfo(
        model_id='gemini-3-flash-preview',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description=(
            'Gemini 3 Flash Preview: High-efficiency, low-latency multimodal'
            ' model optimized for agentic workflows.'
        ),
        release_date=datetime.datetime(2025, 12, 17),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.05,
            cost_per_1m_input_tokens=0.50,
            cost_per_1m_output_tokens=3.00,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2_000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 3.5 Flash
    GeminiModelInfo(
        model_id='gemini-3.5-flash',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description=(
            'Gemini 3.5 Flash: High-efficiency, low-latency multimodal'
            ' model optimized for agentic workflows.'
        ),
        release_date=datetime.datetime(2026, 5, 19),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.15,
            cost_per_1m_input_tokens=1.50,
            cost_per_1m_output_tokens=9.00,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2_000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 3.7 Flash
    GeminiModelInfo(
        model_id='gemini-3.7-flash',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description=(
            'Gemini 3.7 Flash: High-efficiency, low-latency multimodal'
            ' model optimized for agentic workflows.'
        ),
        release_date=datetime.datetime(2026, 8, 13),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.075,
            cost_per_1m_input_tokens=0.75,
            cost_per_1m_output_tokens=3.75,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2_000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 3.8 Flash
    GeminiModelInfo(
        model_id='gemini-3.8-flash',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description=(
            'Gemini 3.8 Flash: High-efficiency, low-latency multimodal'
            ' model optimized for agentic workflows.'
        ),
        release_date=datetime.datetime(2026, 9, 2),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.075,
            cost_per_1m_input_tokens=0.75,
            cost_per_1m_output_tokens=3.75,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2_000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 2.5 Flash
    GeminiModelInfo(
        model_id='gemini-2.5-flash',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Flash GA.',
        release_date=datetime.datetime(2025, 6, 17),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.030,
            cost_per_1m_input_tokens=0.30,
            cost_per_1m_output_tokens=2.50,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 2.5 Pro
    GeminiModelInfo(
        model_id='gemini-2.5-pro',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Pro GA.',
        release_date=datetime.datetime(2025, 6, 17),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.250,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.50,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=15.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
    # Gemini 2.5 Pro 0605
    GeminiModelInfo(
        model_id='gemini-2.5-pro-preview-06-05',
        in_service=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Pro.',
        release_date=datetime.datetime(2025, 6, 5),
        input_modalities=GeminiModelInfo.ALL_SUPPORTED_INPUT_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_048_576,
            max_output_tokens=65_536,
        ),
        pricing=GeminiModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.250,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.50,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=15.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
    ),
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
            cost_per_1m_cached_input_tokens=0.125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.250,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.50,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=15.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=4_000_000,
        ),
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
            cost_per_1m_cached_input_tokens=0.125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.250,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.50,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=15.0,
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
            cost_per_1m_cached_input_tokens=0.125,
            cost_per_1m_input_tokens=1.25,
            cost_per_1m_output_tokens=10.0,
            cost_per_1m_cached_input_tokens_with_prompt_longer_than_128k=0.250,
            cost_per_1m_input_tokens_with_prompt_longer_than_128k=2.50,
            cost_per_1m_output_tokens_with_prompt_longer_than_128k=15.0,
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
        model_id='gemini-2.5-flash-image-preview',
        in_service=True,
        experimental=True,
        provider=pg.oneof(['Google GenAI', 'VertexAI']),
        model_type='instruction-tuned',
        description='Gemini 2.5 Flash Image Preview model.',
        release_date=datetime.datetime(2025, 8, 17),
        input_modalities=GeminiModelInfo.INPUT_IMAGE_TYPES
        + GeminiModelInfo.INPUT_DOC_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=32_768,
            max_output_tokens=32_768,
        ),
    ),
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

# A 1M-token budget needs about 32 hops of 32K tokens; the cap only stops a
# server that keeps returning CONTINUATION without ever finishing.
_MAX_CONTINUATION_HOPS = 64

# How many times one generation may be granted a fresh output budget after
# spending an entire one without emitting a single answer token.
#
# This is deliberately NOT folded into `_MAX_CONTINUATION_HOPS`, because the
# two bound different things. A `CONTINUATION` hop stays inside the budget the
# caller already authorized -- it is the same generation arriving in 32K-token
# instalments. A budget extension authorizes spending the caller never asked
# for, because `maxOutputTokens` accumulates across continuation requests and
# the service refuses a resume that does not raise it. Two extensions bound a
# starved request at 4x its nominal cost in the worst case, which covers the
# observed failure -- thinking overran the budget once -- without letting a
# pathological prompt spend without limit.
_MAX_OUTPUT_BUDGET_EXTENSIONS = 2

# Part keys that carry content the caller would see. `text` is handled
# separately because a starved candidate comes back as `{'text': ''}`: a text
# part that holds no answer.
_ANSWER_PART_KEYS = (
    'functionCall',
    'inlineData',
    'fileData',
    'executableCode',
    'codeExecutionResult',
)


def _gemini_major_version(model_id: str) -> int:
  """Returns N for a `gemini-N...` model ID, or 0 for unversioned IDs."""
  match = re.match(r'gemini-(\d+)', model_id)
  return int(match.group(1)) if match else 0


def _has_answer_part(candidate: dict[str, Any]) -> bool:
  """Returns True if `candidate` carries content the caller would receive.

  Thought parts are excluded: they are the model's internal reasoning, which
  `Message.from_value` routes to `message.thought` rather than to the answer,
  so a candidate holding nothing else has produced nothing for the caller.

  Args:
    candidate: A `candidates[i]` entry from a `generateContent` response.

  Returns:
    Whether any non-thought part holds text or a non-text payload.
  """
  for part in (candidate.get('content') or {}).get('parts') or []:
    if part.get('thought'):
      continue
    if part.get('text') or any(key in part for key in _ANSWER_PART_KEYS):
      return True
  return False


def _continues_text(part: dict[str, Any], next_part: dict[str, Any]) -> bool:
  """Returns True if `next_part` continues the text or thought of `part`."""
  return (
      isinstance(part.get('text'), str)
      and isinstance(next_part.get('text'), str)
      and bool(part.get('thought')) == bool(next_part.get('thought'))
  )


def _merge_continuation_hops(payloads: list[dict[str, Any]]) -> dict[str, Any]:
  """Merges the responses of a continued generation into a single response.

  Parts are concatenated in order, re-joining a text part that a hop boundary
  split, since `Message.from_chunks` would otherwise insert a separator there.
  Token counts are summed, as every hop is billed. All other fields, e.g.
  `finishReason`, come from the last hop.

  Args:
    payloads: Single-candidate `generateContent` responses, in hop order.

  Returns:
    A `generateContent` response combining all hops.
  """
  parts = []
  usage = {}
  for payload in payloads:
    content = payload['candidates'][0].get('content') or {}
    hop_parts = list(content.get('parts') or [])
    if parts and hop_parts and _continues_text(parts[-1], hop_parts[0]):
      first = hop_parts.pop(0)
      parts[-1] = {
          **parts[-1],
          **first,
          'text': parts[-1]['text'] + first['text'],
      }
    parts.extend(hop_parts)
    for key, value in (payload.get('usageMetadata') or {}).items():
      if isinstance(value, int):
        value += usage.get(key, 0)
      usage[key] = value
  last = payloads[-1]
  candidate = dict(last['candidates'][0])
  candidate['content'] = dict(candidate.get('content') or {}, parts=parts)
  return dict(last, candidates=[candidate], usageMetadata=usage)


@pg.use_init_args(['model'])
class Gemini(rest.REST):
  """Base class for Gemini models served on Google GenAI and Vertex AI.

  This class implements the Gemini API protocol, shared by
  `lf.llms.GoogleGenAI` and `lf.llms.VertexAI`, providing common request
  formatting and response parsing for Gemini models.

  It is not intended to be used directly. Please use `lf.llms.GoogleGenAI` or
  `lf.llms.VertexAI` instead.
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]
      ),
      'The name of the model to use.',
  ]

  response_modalities: pg.typing.Annotated[
      list[str] | None,
      'Response modalities. It is needed for models whose response modalities '
      + 'are more than plain text.',
  ] = None

  @functools.cached_property
  def model_info(self) -> GeminiModelInfo:
    return _SUPPORTED_MODELS_BY_ID[self.model]

  @classmethod
  def dir(cls):  # pyrefly: ignore[bad-override]
    return [m.model_id for m in SUPPORTED_MODELS if m.in_service]

  @property
  def headers(self):  # pyrefly: ignore[bad-override]
    return {
        'Content-Type': 'application/json; charset=utf-8',
    }

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    """Samples a prompt, resuming a generation cut at the per-request cap."""
    with self._as_lm_errors():
      request = self.request(prompt, self.sampling_options)
      # Resolved once so that all requests of a generation go to the same
      # project; `api_endpoint` may pick a different project on each call.
      endpoint = self.api_endpoint
      response = self._post(request, endpoint)
      hops: list[dict[str, Any]] = []
      # Retried here rather than by `_sample()`, which would restart from hop 0.
      post_continuation = lf.with_retry(
          self._post_continuation,
          retry_on_errors=lf.RetryableLMError,
          max_attempts=self.max_attempts,
          retry_interval=self.retry_interval,
          exponential_backoff=self.exponential_backoff,
          max_retry_interval=self.max_retry_interval,
      )
      budget_extensions = 0
      # Carries a raised output ceiling forward. Rebuilding every resume from
      # `request` would re-send the ceiling the service just rejected, so a
      # second extension would never actually extend anything.
      base = request
      while (hop := self._continuation_hop(response)) is not None:
        candidate = hop['candidates'][0]
        resumed = dict(
            base, continuationToken=candidate['continuationToken']
        )
        # A `MAX_TOKENS` hop reaches here only when it produced no answer at
        # all. Its budget is gone, and the service accumulates that budget
        # across continuations, so resuming under the same ceiling is
        # refused. Raising it authorizes spending the caller did not ask for,
        # which is why extensions are bounded separately and tightly.
        starved = candidate.get('finishReason') == 'MAX_TOKENS'
        if starved:
          if budget_extensions == _MAX_OUTPUT_BUDGET_EXTENSIONS:
            break
          extended = self._extend_output_budget(resumed)
          if extended is None:
            break
          budget_extensions += 1
          resumed = extended
          base = extended
        hops.append(hop)
        if len(hops) == _MAX_CONTINUATION_HOPS:
          raise lf.LMError(
              f'Model {self.model_id} did not finish within '
              f'{_MAX_CONTINUATION_HOPS} requests.'
          )
        # Long decode: each hop resends the original request plus the token.
        try:
          response = post_continuation(resumed, endpoint)
        except lf.LMError:
          # Resuming a starved hop is a rescue attempt for a generation that
          # would otherwise be reported as empty, so its failure must not
          # replace that diagnosis -- which names the budget and the knobs
          # that control it -- with an error about the rescue. Drop the hop
          # and let the normal path report on what was really generated.
          #
          # A failure resuming an ordinary CONTINUATION hop is not
          # speculative: that generation was proceeding normally, and hiding
          # the error would turn a real failure into a silent truncation.
          if not starved:
            raise
          hops.pop()
          break
      if not hops:
        return self._parse_response(response)
      try:
        final = self._response_to_message_dict(response)
        if final.get('candidates'):
          final = _merge_continuation_hops(hops + [final])
        return self.result(final)
      except (ValueError, KeyError) as e:
        raise lf.LMError(str(e)) from e

  def _post_continuation(
      self, request: dict[str, Any], endpoint: str
  ) -> requests.Response:
    """POSTs a continuation request, raising if it was not successful."""
    response = self._post(request, endpoint)
    if response.status_code != 200:
      raise self._error(response.status_code, response.content)  # pyrefly: ignore[bad-argument-type]
    return response

  def _extend_output_budget(
      self, request: dict[str, Any]
  ) -> dict[str, Any] | None:
    """Returns `request` with room for the answer its budget never reached.

    `maxOutputTokens` is a running total over a continued generation, not a
    per-response limit, so a generation that spent all of it has to be
    resumed under a higher ceiling or the service refuses with
    ``max_output_tokens (N) has already been exhausted``. Doubling gives the
    answer a budget the size of the one thinking just consumed, which is the
    smallest increase with a principled justification: the observed failure
    is thinking overrunning the budget, and the answer still has to fit.

    Args:
      request: The resume request, carrying the exhausted ceiling.

    Returns:
      The request with a raised `generationConfig.maxOutputTokens`, or None
      when no higher ceiling can be named -- in which case the caller should
      stop rather than send a request the service will reject.
    """
    config = dict(request.get('generationConfig') or {})
    current = config.get('maxOutputTokens')
    if current is None:
      # No ceiling was requested, so the service applied its own default and
      # the only higher value this library can state is the model's maximum.
      context_length = self.model_info.context_length
      raised = context_length.max_output_tokens if context_length else None
      if raised is None:
        return None
    else:
      raised = current * 2
    config['maxOutputTokens'] = raised
    return dict(request, generationConfig=config)

  def _continuation_hop(
      self, response: requests.Response
  ) -> dict[str, Any] | None:
    """Returns the payload of `response` if it stopped to be continued.

    Two different stops are resumable, and the service signals both by
    attaching a `continuationToken`:

    * ``CONTINUATION`` -- the generation hit the per-response wire limit with
      budget left. Resuming costs nothing extra.
    * ``MAX_TOKENS`` with no answer part -- the output budget was spent
      entirely on thinking, so there is nothing to return. Resuming is the
      only way to get an answer out of the tokens already paid for; the
      alternative is `GeminiEmptyGenerationError`, i.e. billing the caller
      for the thinking and delivering nothing. The caller of this method is
      responsible for raising `maxOutputTokens` on the resumed request, since
      the service accumulates that budget across continuations.

    A ``MAX_TOKENS`` candidate that DID produce answer content is not
    resumed. There the budget did exactly what the caller asked of it, and
    silently spending more would override an explicit instruction.

    Args:
      response: The HTTP response of the hop just completed.

    Returns:
      The parsed payload when the generation can be continued, else None.
    """
    if response.status_code != 200:
      return None
    try:
      payload = self._response_to_message_dict(response)
    except ValueError:
      return None
    candidates = payload.get('candidates') or []
    if len(candidates) != 1 or not candidates[0].get('continuationToken'):
      return None
    finish_reason = candidates[0].get('finishReason')
    if finish_reason == 'CONTINUATION':
      return payload
    if finish_reason == 'MAX_TOKENS' and not _has_answer_part(candidates[0]):
      return payload
    return None

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
              self.model_info.input_modalities + ['text/plain']  # pyrefly: ignore[unsupported-operation]
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
    request['contents'] = contents  # pyrefly: ignore[bad-assignment]
    request['toolConfig'] = {
        'functionCallingConfig': {
            'mode': 'NONE',
        }
    }
    if sampling_options.extras:
      request.update(sampling_options.extras)
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
      config['responseMimeType'] = 'application/json'  # pyrefly: ignore[bad-assignment]
      prompt.metadata.formatted_text = (  # pyrefly: ignore[missing-attribute]
          prompt.text
          + '\n\n [RESPONSE FORMAT (not part of prompt)]\n'
          + pg.to_json_str(json_schema, json_indent=2)
      )
    thinking_config_data = {}
    if options.max_thinking_tokens is not None:
      thinking_config_data['includeThoughts'] = options.max_thinking_tokens > 0
      thinking_config_data['thinkingBudget'] = options.max_thinking_tokens
    if options.thinking_level is not None:
      thinking_config_data['thinkingLevel'] = options.thinking_level
    if thinking_config_data:
      config['thinkingConfig'] = thinking_config_data  # pyrefly: ignore[bad-assignment]

    # Supported since Gemini 3, except by lite and image generation models.
    if (
        _gemini_major_version(self.model_id) >= 3
        and 'lite' not in self.model_id
        and not (
            self.response_modalities and 'IMAGE' in self.response_modalities
        )
    ):
      config['mediaResolution'] = 'MEDIA_RESOLUTION_HIGH'  # pyrefly: ignore[bad-assignment]

    if self.response_modalities:
      config['responseModalities'] = self.response_modalities
      if 'IMAGE' in self.response_modalities:
        config.pop('responseLogprobs', None)
        config.pop('logprobs', None)
    return config

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    candidates = json.get('candidates')
    if candidates is None:
      raise lf.TemporaryLMError(
          'No candidates found in response. This is a Gemini API issue that '
          'happens occasionally, and retrying should fix it. '
      )

    usage = json['usageMetadata']
    # NOTE(daiyip): We saw cases that `candidatesTokenCount` is not present.
    # Therefore, we use 0 as the default value.
    input_tokens = usage.get('promptTokenCount', 0)
    output_tokens = usage.get('candidatesTokenCount', 0)
    thinking_tokens = usage.get('thoughtsTokenCount', 0)
    total_tokens = usage.get('totalTokenCount', 0)
    cached_tokens = usage.get('cachedContentTokenCount', 0)

    messages = []
    for candidate in candidates:
      finish_reason = candidate.get('finishReason')
      content = candidate.get('content')
      if content is None:
        raise self._no_answer_error(
            'the candidate carries no content at all',
            finish_reason,
            output_tokens,
            thinking_tokens,
        )
      message = lf.Message.from_value(content, format='gemini')
      if finish_reason:
        message.metadata['finish_reason'] = finish_reason
      self._check_answer_present(
          message, finish_reason, output_tokens, thinking_tokens
      )
      messages.append(message)

    return lf.LMSamplingResult(
        [lf.LMSample(message) for message in messages],
        usage=lf.LMSamplingUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=cached_tokens,
            completion_tokens_details={
                'thinking_tokens': thinking_tokens,
            },
        ),
    )

  def _check_answer_present(
      self,
      message: lf.Message,
      finish_reason: str | None,
      output_tokens: int,
      thinking_tokens: int,
  ) -> None:
    """Fails fast when a parsed candidate carries no answer, deterministically.

    Only the two token-budget signatures documented on
    `GeminiEmptyGenerationError` are reclassified here; its third signature
    (no `content` at all) is caught by `result()` before parsing. Every other
    answer-less case (an unrelated `finishReason`, a function-call-only
    candidate that did emit tokens, a missing `finishReason`) is left alone so
    the pre-existing retryable `lf.EmptyGenerationError` path keeps handling it
    -- those may genuinely be transient.

    Args:
      message: The parsed candidate. Its text holds the non-thought answer
        content (thought parts are routed to `message.thought`), so empty text
        means no answer was returned.
      finish_reason: The candidate's `finishReason`, if any.
      output_tokens: Response-level `candidatesTokenCount`. With `n > 1` this
        aggregates all candidates, which only makes the `STOP` check stricter
        (never falsely positive).
      thinking_tokens: Response-level `thoughtsTokenCount`, for diagnostics.

    Raises:
      GeminiEmptyGenerationError: If the candidate is deterministically
        answer-less.
    """
    if message.text:
      return

    if finish_reason == 'MAX_TOKENS':
      cause = (
          'the decode budget was exhausted before any answer token was '
          f'emitted ({thinking_tokens} thinking tokens were produced)'
      )
    elif finish_reason == 'STOP' and output_tokens == 0:
      cause = 'the model stopped without emitting any output token'
    else:
      return

    raise self._no_answer_error(
        cause, finish_reason, output_tokens, thinking_tokens
    )

  def _no_answer_error(
      self,
      cause: str,
      finish_reason: str | None,
      output_tokens: int,
      thinking_tokens: int,
  ) -> GeminiEmptyGenerationError:
    """Builds the error reported for a deterministically answer-less candidate.

    Args:
      cause: A phrase completing 'returned no answer content: ...', naming
        which signature was matched.
      finish_reason: The candidate's `finishReason`, if any.
      output_tokens: Response-level `candidatesTokenCount`.
      thinking_tokens: Response-level `thoughtsTokenCount`.

    Returns:
      The error to raise, carrying the diagnosis and the knobs to change.
    """
    return GeminiEmptyGenerationError(
        f'Model {self.model_id} returned no answer content: {cause} '
        f'(finishReason={finish_reason}, candidatesTokenCount={output_tokens}, '
        f'thoughtsTokenCount={thinking_tokens}). This is determined by the '
        'request, so retrying it as-is would reproduce the same result. '
        'Consider raising `max_tokens`, lowering the thinking budget '
        '(`max_thinking_tokens` / `thinking_level`), or shortening the prompt.'
    )

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 400 and (
        b'exceeds the maximum number of tokens' in content  # pyrefly: ignore[unsupported-operation]
        or b'Reduce the input token count and try again.' in content  # pyrefly: ignore[unsupported-operation]
        or b'Request payload size exceeds the limit' in content  # pyrefly: ignore[unsupported-operation]
        or b'Request contains text fields that are too large' in content  # pyrefly: ignore[unsupported-operation]
    ):
      return lf.ContextLimitError(f'{status_code}: {content}')
    return super()._error(status_code, content)
