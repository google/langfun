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

import collections
import contextvars
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

  The rule is NARROW by construction, because raising is itself expensive:
  the error is non-retryable, so every signature admitted here ends a live
  episode. Only a signature a re-send provably cannot change earns that, and
  each one names the mechanism that makes the re-send futile:

  * ``promptFeedback.blockReason`` set -- the prompt was blocked before
    decoding started, so no candidate was ever generated.
  * ``finishReason == 'MAX_TOKENS'`` with ``candidatesTokenCount == 0`` --
    thinking consumed the shared decode budget before a single answer token
    was emitted.
  * ``finishReason == 'MAX_TOKENS'`` with ``candidatesTokenCount > 0`` -- the
    truncated hop was billed for answer tokens it did not return, and no
    continuation recovered them. The budget did reach the answer, so this one
    is not starvation and the error says so. More answer than thought tokens
    is a misfire that ran into the cap instead, raised as
    ``MalformedFunctionCallError``.
  * ``finishReason == 'STOP'`` with ``candidatesTokenCount == 0`` AND
    ``thoughtsTokenCount == 0`` -- the model terminated normally having been
    billed for no work at all, so there is no lost work to re-request.
  * no ``content`` on the candidate at all (absent or null) -- nothing was
    generated, so there is not even an empty message to parse.
  * ``candidates`` present but EMPTY *and* a block reason given -- a response
    with no candidate at all, which otherwise reaches ``LanguageModel`` as a
    bare ``IndexError``.

  Everything else that renders no answer -- a tool-protocol finish reason
  (``MALFORMED_FUNCTION_CALL``, ...), a candidate-level content-policy flag
  (``SAFETY``, ``RECITATION``, ``BLOCKLIST``, ...), a ``STOP`` over non-zero
  token counts, an absent or unrecognised ``finishReason``, an empty
  ``candidates`` list with nothing to explain it -- is left on the RETRYABLE
  path (``lf.EmptyGenerationError`` / ``lf.TemporaryLMError``), because it is
  the path that recovers them. A closed rule was tried here and measured
  against 30 live ``gemini-4-argon`` trajectories: 28 of 28 aborts it caused
  were ``MALFORMED_FUNCTION_CALL`` on a mid-episode turn that declared no
  tools, each billed 521-4,336 answer and 983-6,020 thinking tokens --
  transient faults the next sample recovers from, converted into fatal ones.

  A candidate-level policy flag is kept on that path for the same reason, and
  the API contract is what settles it: ``SAFETY``, ``RECITATION``,
  ``BLOCKLIST``, ``PROHIBITED_CONTENT``, ``SPII``, ``LANGUAGE`` and
  ``IMAGE_SAFETY`` are defined as flags raised on *the response candidate
  content* -- the tokens that happened to be sampled -- which is precisely
  what the next sample changes. Only ``promptFeedback.blockReason`` is
  defined on the prompt ("the prompt was blocked and no candidates are
  returned. Rephrase the prompt."), and only it is admitted above. Reporting
  a sampled-content flag as deterministic would end an episode over an
  outcome a re-sample can clear, which is the expensive direction of this
  trade.

  ``candidatesTokenCount > 0`` is NOT evidence that an answer was returned.
  Measured across 11 answer-less ``gemini-4-argon`` generations,
  ``candidatesTokenCount`` ran from 51,936 to 64,417 while the parsed text was
  empty. The only sound evidence of an unrendered answer is the parts
  themselves, which is what ``_carries_unrendered_content`` inspects: a
  ``functionCall`` / ``functionResponse`` part is a real answer that
  ``GeminiMessageConverter.from_value`` drops, and it is the one exemption.
  Nor is a non-zero count evidence of a truncation hiding under a ``STOP``:
  that laundering is closed where it happens, in
  ``_merge_continuation_hops``, which keeps a chain that rendered nothing
  labelled ``MAX_TOKENS`` instead of taking the last hop's ``STOP``.

  The signatures admitted are reproduced by re-sending the same request, so
  this subclasses the NON-retryable ``lf.LMError`` -- deliberately NOT
  ``lf.EmptyGenerationError`` / ``lf.RetryableLMError``. Every recovery that
  does work is already spent by the time it is raised: a candidate carrying a
  ``continuationToken`` is resumed in-request by ``_continuation_hop``, and an
  explicit output ceiling is extended across up to
  ``_MAX_PINNED_BUDGET_HOPS`` requests for exactly the answer tokens still
  owed. Re-sending the prompt instead buys nothing and costs ``max_attempts``
  full generations -- 80 of them under common configurations, each billed for
  the whole decode budget.

  It stays a distinct class rather than ``lf.EmptyGenerationError`` so the
  diagnosis -- finish reason, block reason, decode budget, token counts, and
  the remedy that fits the signature matched -- survives into the message, and
  so no signature escapes as a bare ``KeyError('content')`` or
  ``IndexError``.
  Mirrors ``beyond_api.BeyondThinkingBudgetExhaustedError``.
  """


class MalformedFunctionCallError(lf.EmptyGenerationError):
  """Gemini misfired into the output cap: a function call that never closed.

  The request declares no tools, yet the model emits a function call. A short
  one comes back as ``finishReason == 'MALFORMED_FUNCTION_CALL'`` with empty
  content; `result()` logs it and leaves it to the retryable empty-generation
  path under ``max_attempts``. This error is for the call that never
  terminates: an answer-less ``MAX_TOKENS`` or ``CONTINUATION`` candidate that
  billed more answer tokens than thoughts. Each such attempt can cost the whole
  output budget, so it is never continuation-resumed, and a query draws at most
  ``_CAP_MISFIRE_MAX_ATTEMPTS`` of them (fewer if ``max_attempts`` is lower),
  counted across all of its retries.
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
# server that keeps returning CONTINUATION without ever finishing. It is the
# bound for a generation that pins no ceiling; a pinned generation is cut off
# by `_MAX_PINNED_BUDGET_HOPS` long before this.
_MAX_CONTINUATION_HOPS = 64

# How many requests a generation that pinned `max_tokens` may span. A resume
# asks for the tokens already spent plus the answer tokens still owed, so it
# raises the ceiling by at most the pinned cap and four requests hold the whole
# generation to 4x what the caller asked for: 1x -> 2x -> 3x -> 4x. Cutting it
# off there is what keeps the cap meaningful; handing the rest of the
# generation to the service default instead was measured answering a 512-token
# request with ~75 KB of text.
_MAX_PINNED_BUDGET_HOPS = 4

# Samples misfired into the output cap that one query may draw, across all of
# its retries, in place of `max_attempts` when that is larger (up to 100 in
# agent configs): each one can cost the whole pinned budget. Short misfires are
# cheap and stay under `max_attempts`.
_CAP_MISFIRE_MAX_ATTEMPTS = 3
# Seconds before the first re-sample, drawn at random and doubled per attempt.
_CAP_MISFIRE_RETRY_INTERVAL = (1, 5)
_CAP_MISFIRE_MAX_RETRY_INTERVAL = 30
# Cap misfires drawn so far in the current `Gemini.sample()` call, by `id` of
# the prompt; see there.
_cap_misfires_drawn: contextvars.ContextVar[collections.Counter[int]] = (
    contextvars.ContextVar('_cap_misfires_drawn')
)


def _gemini_major_version(model_id: str) -> int:
  """Returns N for a `gemini-N...` model ID, or 0 for unversioned IDs."""
  match = re.match(r'gemini-(\d+)', model_id)
  return int(match.group(1)) if match else 0


def _continues_text(part: dict[str, Any], next_part: dict[str, Any]) -> bool:
  """Returns True if `next_part` continues the text or thought of `part`."""
  return (
      isinstance(part.get('text'), str)
      and isinstance(next_part.get('text'), str)
      and bool(part.get('thought')) == bool(next_part.get('thought'))
  )


def _with_output_budget(
    request: dict[str, Any], max_output_tokens: int
) -> dict[str, Any]:
  """Returns a copy of `request` resuming under a `max_output_tokens` ceiling.

  `maxOutputTokens` is a running total over a whole generation, not a
  per-response limit, so a hop that reports `MAX_TOKENS` reports that ceiling
  spent. A resume that asserts the spent ceiling again is rejected before it
  starts:

      max_output_tokens (1024) has already been exhausted. Because
      max_output_tokens is accumulated across continuation requests, pass a
      higher max_output_tokens value to continue generation.

  Raising the ceiling is the resume the service accepts, and it is the one
  that keeps the caller's `max_tokens` meaningful: it authorizes exactly the
  tokens already spent plus the answer tokens still owed, so the generation
  cannot return more answer than was asked for.

  The raised value needs no upper clamp. Measured against the service,
  `maxOutputTokens` of 65,537, 131,072 and 1,000,000 -- all above the model's
  own 65,536-token output maximum -- are each accepted with HTTP 200, so there
  is no threshold above which the ceiling has to be dropped. Dropping it would
  be the worst option available anyway: it hands the rest of the generation to
  the service default, which is the unbounded behaviour the ceiling exists to
  prevent.

  Args:
    request: A `generateContent` request body.
    max_output_tokens: The ceiling the resume should assert.

  Returns:
    A copy of `request` with `generationConfig.maxOutputTokens` set to
    `max_output_tokens`.
  """
  return dict(
      request,
      generationConfig=dict(
          request.get('generationConfig') or {},
          maxOutputTokens=max_output_tokens,
      ),
  )


def _answer_text(candidate: dict[str, Any]) -> str:
  """Returns the non-thought answer text a response candidate carries.

  Read through the same parser `result()` uses, so that `''` here means
  exactly what an empty `Message.text` means there: thought parts routed
  away, empty text parts fused out, modalities counted as answer content.

  Args:
    candidate: A single `generateContent` response candidate.

  Returns:
    The candidate's answer text, empty when it carries no answer.
  """
  return lf.Message.from_value(
      candidate.get('content') or {'parts': []}, format='gemini'
  ).text


def _merge_continuation_hops(payloads: list[dict[str, Any]]) -> dict[str, Any]:
  """Merges the responses of a continued generation into a single response.

  Parts are concatenated in order, re-joining a text part that a hop boundary
  split, since `Message.from_chunks` would otherwise insert a separator there.
  Bare `{'text': ''}` parts -- what a hop whose whole budget went to thinking
  returns -- are kept verbatim: `Message.from_chunks` writes no separator after
  an empty chunk and strips the fused text, so they contribute nothing to the
  parsed message wherever they land. Token counts are summed, as every hop is
  billed. All other fields, e.g. `finishReason`, come from the last hop.

  `finishReason` is the one exception, and only when the chain rendered no
  answer: a `MAX_TOKENS` hop followed by a hop that adds nothing and reports
  `STOP` (0 tokens, empty text) is a budget exhaustion the last hop would
  otherwise launder into a clean stop. Measured on `ex 118`, that laundering
  cost 80 full 65K-token regenerations, because `_check_answer_present` saw
  `('STOP', summed count)` -- a signature that looks recoverable -- instead of
  the `MAX_TOKENS` it was. Taking the truncation over the stop is sound only
  while the merged candidate renders nothing: once any hop returned answer
  text, the chain did finish and the last hop's reason is the true one.

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
  if not _answer_text(candidate) and any(
      payload['candidates'][0].get('finishReason') == 'MAX_TOKENS'
      for payload in payloads
  ):
    candidate['finishReason'] = 'MAX_TOKENS'
  return dict(last, candidates=[candidate], usageMetadata=usage)


def _carries_unrendered_content(candidate: dict[str, Any]) -> bool:
  """Returns True if a candidate holds an answer the converter does not render.

  `GeminiMessageConverter.from_value` drops `functionCall` and
  `functionResponse` parts, so a tool-calling candidate parses to empty text
  while carrying a real answer. Those parts remain reachable even though
  `request()` hard-sets `toolConfig.functionCallingConfig.mode` to `'NONE'`,
  because `sampling_options.extras` can override the request it built.

  Asking the parts directly is the evidence that replaces the unsound
  `candidatesTokenCount > 0` proxy: a token count says the service was billed,
  not that anything was returned.

  Args:
    candidate: A single `generateContent` candidate.

  Returns:
    True if any part is a function call or a function response.
  """
  parts = (candidate.get('content') or {}).get('parts') or ()
  return any(
      'functionCall' in part or 'functionResponse' in part for part in parts
  )


def _is_cap_misfire(
    finish_reason: str | None, output_tokens: int, thinking_tokens: int
) -> bool:
  """Returns True if an answer-less candidate misfired into the output cap.

  Thinking starvation also returns no answer under `MAX_TOKENS`, but it spends
  the budget on thoughts. A misfire spends it on answer tokens that come back
  as no text: an unterminated function call.

  Args:
    finish_reason: The candidate's `finishReason`.
    output_tokens: `candidatesTokenCount` of the response.
    thinking_tokens: `thoughtsTokenCount` of the response.

  Returns:
    Whether the candidate misfired into the cap.
  """
  return (
      finish_reason in ('MAX_TOKENS', 'CONTINUATION')
      and output_tokens > thinking_tokens
  )


def _hop_signature(payload: dict[str, Any]) -> tuple[str | None, int, int]:
  """Returns a hop's `finishReason` and its own answer and thinking tokens."""
  usage = payload.get('usageMetadata') or {}
  return (
      payload['candidates'][0].get('finishReason'),
      usage.get('candidatesTokenCount') or 0,
      usage.get('thoughtsTokenCount') or 0,
  )


def _is_misfire_hop(payload: dict[str, Any]) -> bool:
  """Returns True if a hop misfired, so resuming it would return nothing."""
  candidate = payload['candidates'][0]
  return (
      not _answer_text(candidate)
      and not _carries_unrendered_content(candidate)
      and _is_cap_misfire(*_hop_signature(payload))
  )


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

  def sample(
      self,
      prompts: list[str | lf.Message],
      *,
      cache_seed: int = 0,
      **kwargs,
  ) -> list[lf.LMSamplingResult]:
    """Samples `prompts`, with one cap-misfire budget per prompt."""
    # A short misfire is retried by `LanguageModel.sample()`, and a transient
    # error such as a 429 by `_parallel_execute_with_currency_control`, both by
    # sampling the prompt anew. Counting cap misfires here, above both, bounds
    # them per query rather than per retry.
    token = _cap_misfires_drawn.set(collections.Counter())
    try:
      return super().sample(prompts, cache_seed=cache_seed, **kwargs)
    finally:
      _cap_misfires_drawn.reset(token)

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    assert self._api_initialized
    # Read on this thread, as the prompts may be sampled on others. Called
    # outside `sample()`, the budget is per call.
    misfires = _cap_misfires_drawn.get(collections.Counter())
    return self._parallel_execute_with_currency_control(
        functools.partial(self._sample_with_misfire_budget, misfires=misfires),
        prompts,
    )

  def _sample_with_misfire_budget(
      self, prompt: lf.Message, misfires: collections.Counter[int]
  ) -> lf.LMSamplingResult:
    """Samples a prompt, drawing a fresh sample when it misfires into the cap.

    Raises `lf.concurrent.RetryError` once the query has drawn
    `_cap_misfire_max_attempts()` such misfires, which is not retried further:
    each one can cost the whole output budget. Short misfires and transport
    errors escape this loop and are retried under `max_attempts` as before,
    and the misfires drawn before them still count.

    Args:
      prompt: The prompt to sample.
      misfires: Cap misfires drawn so far in this query, by `id` of the prompt.

    Returns:
      The sampling result.
    """

    def sample_once(prompt: lf.Message) -> lf.LMSamplingResult:
      try:
        return self._sample_single(prompt)
      except MalformedFunctionCallError:
        misfires[id(prompt)] += 1
        raise

    return lf.with_retry(
        sample_once,
        retry_on_errors=MalformedFunctionCallError,
        max_attempts=self._cap_misfire_max_attempts() - misfires[id(prompt)],
        retry_interval=_CAP_MISFIRE_RETRY_INTERVAL,
        max_retry_interval=_CAP_MISFIRE_MAX_RETRY_INTERVAL,
    )(prompt)

  def _cap_misfire_max_attempts(self) -> int:
    """Returns how many cap misfires a query may draw.

    Never more than `max_attempts`: a caller that sets it lower, e.g. to 1 to
    run its own retries, has asked for no more than that.
    """
    return min(_CAP_MISFIRE_MAX_ATTEMPTS, self.max_attempts)

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    """Samples a prompt, resuming a generation cut at the per-request cap."""
    with self._as_lm_errors():
      request = self.request(prompt, self.sampling_options)
      # Resolved once so that all requests of a generation go to the same
      # project; `api_endpoint` may pick a different project on each call.
      endpoint = self.api_endpoint
      response = self._post(request, endpoint)
      hops: list[dict[str, Any]] = []
      # The output ceiling the caller pinned, if any. `maxOutputTokens` is a
      # running total over the whole generation covering thinking AND answer,
      # so a `MAX_TOKENS` hop reports it spent and the only resume the service
      # accepts is one that raises it. What the caller is owed is answer
      # tokens, so the two are tracked apart: `maxOutputTokens` bounds the
      # total, `requested_max_tokens` bounds the answer.
      requested_max_tokens = (request.get('generationConfig') or {}).get(
          'maxOutputTokens'
      )
      cumulative_candidates_tokens = 0
      cumulative_total_tokens = 0
      budget_hops = 1
      # Set only when a hop reports no `candidatesTokenCount` at all, which
      # the service does occasionally. There is no arithmetic to do without a
      # count, so the weaker signal it stands in for is used instead.
      answered_without_count = False
      continuation_request = request
      # Retried here rather than by `_sample()`, which would restart from hop 0.
      post_continuation = lf.with_retry(
          self._post_continuation,
          retry_on_errors=lf.RetryableLMError,
          max_attempts=self.max_attempts,
          retry_interval=self.retry_interval,
          exponential_backoff=self.exponential_backoff,
          max_retry_interval=self.max_retry_interval,
      )
      while (hop := self._continuation_hop(response)) is not None:
        if _is_misfire_hop(hop):
          # Re-sampled, never resumed, which would only extend the runaway
          # call. Judged here on this hop's own counts: `result()` sees them
          # summed over the hops, where earlier thinking can mask the misfire.
          raise self._misfire_error(*_hop_signature(hop))
        hops.append(hop)
        if len(hops) == _MAX_CONTINUATION_HOPS:
          raise lf.LMError(
              f'Model {self.model_id} did not finish within '
              f'{_MAX_CONTINUATION_HOPS} requests.'
          )
        candidate = hop['candidates'][0]
        usage = hop.get('usageMetadata') or {}
        hop_answer_tokens = usage.get('candidatesTokenCount')
        if hop_answer_tokens is None:
          answered_without_count = answered_without_count or bool(
              _answer_text(candidate)
          )
          hop_answer_tokens = 0
        cumulative_candidates_tokens += hop_answer_tokens
        cumulative_total_tokens += hop_answer_tokens + (
            usage.get('thoughtsTokenCount') or 0
        )
        if (
            candidate.get('finishReason') == 'MAX_TOKENS'
            and requested_max_tokens is not None
        ):
          remaining_answer_tokens = (
              requested_max_tokens - cumulative_candidates_tokens
          )
          if (
              remaining_answer_tokens <= 0
              or answered_without_count
              or budget_hops == _MAX_PINNED_BUDGET_HOPS
          ):
            # The caller already has every answer token they asked for, or
            # has answer content that cannot be counted, or the generation
            # has spent its bound. Stop resuming and report what the
            # generation itself did -- a truncated answer, or the answer-less
            # diagnosis.
            hops.pop()
            break
          budget_hops += 1
          # Thinking spent part of the cap, so the answer tokens still owed
          # only fit above what the whole generation has burned so far. Asking
          # for exactly that keeps the resume accepted -- it raises the
          # running total -- without authorizing an answer longer than the one
          # the caller asked for.
          continuation_request = _with_output_budget(
              request, cumulative_total_tokens + remaining_answer_tokens
          )
        try:
          response = post_continuation(
              dict(
                  continuation_request,
                  continuationToken=candidate['continuationToken'],
              ),
              endpoint,
          )
        except (lf.LMError, lf.concurrent.RetryError):
          if candidate.get('finishReason') == 'CONTINUATION':
            # A mid-flight generation really did fail; hiding that would turn
            # a failure into a silently truncated answer.
            raise
          # Resuming an exhausted budget is best effort -- the token can have
          # expired, or the service can refuse the resume for a reason the
          # ceiling does not explain. Fall back to reporting what the
          # generation itself did rather than the resume's own error.
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

  def _continuation_hop(
      self, response: requests.Response
  ) -> dict[str, Any] | None:
    """Returns the payload of `response` if it stopped to be continued."""
    if response.status_code != 200:
      return None
    try:
      payload = self._response_to_message_dict(response)
    except ValueError:
      return None
    candidates = payload.get('candidates') or []
    # `CONTINUATION` is the per-response cap: the generation is mid-flight.
    # `MAX_TOKENS` is budget exhaustion -- when thinking spends the shared
    # decode budget the service still offers a `continuationToken`, and
    # following it is what turns an answer-less generation into an answer.
    # Either way the token is only honoured when the service actually sent
    # one, which it omits once the generation is genuinely over.
    if (
        len(candidates) == 1
        and candidates[0].get('finishReason') in ('CONTINUATION', 'MAX_TOKENS')
        and candidates[0].get('continuationToken')
    ):
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

    usage = json.get('usageMetadata') or {}
    # NOTE(daiyip): We saw cases that `candidatesTokenCount` is not present.
    # Therefore, we use 0 as the default value.
    input_tokens = usage.get('promptTokenCount', 0)
    output_tokens = usage.get('candidatesTokenCount', 0)
    thinking_tokens = usage.get('thoughtsTokenCount', 0)
    total_tokens = usage.get('totalTokenCount', 0)
    cached_tokens = usage.get('cachedContentTokenCount', 0)
    # A blocked prompt is reported here, not on a candidate -- there is no
    # candidate to report it on.
    block_reason = (json.get('promptFeedback') or {}).get('blockReason')

    if not candidates:
      # `candidates` present but EMPTY is a response that answered nothing.
      # Left alone, it returns `LMSamplingResult([])` and `LanguageModel`
      # fails on `result.samples[0]` with a bare `IndexError` that names
      # neither the block reason nor the model, so it never reaches that
      # point: a blocked prompt is reported as the deterministic failure it
      # is, and anything else as the transient one it is.
      self._check_answer_present(
          lf.AIMessage(''),
          {},
          None,
          output_tokens,
          thinking_tokens,
          block_reason,
      )
      raise lf.TemporaryLMError(
          f'Model {self.model_id} returned an empty candidate list with no '
          f'block reason (candidatesTokenCount={output_tokens}, '
          f'thoughtsTokenCount={thinking_tokens}). Nothing was generated and '
          'nothing explains why, which is the transient shape of this API; '
          'retrying should fix it.'
      )

    messages = []
    for candidate in candidates:
      finish_reason = candidate.get('finishReason')
      content = candidate.get('content')
      if finish_reason == 'MALFORMED_FUNCTION_CALL':
        # A short misfire. It is cheap and a fresh sample usually clears it, so
        # it stays on the retryable empty-generation path under `max_attempts`.
        pg.logging.warning(
            self._misfire_message(
                finish_reason,
                output_tokens,
                thinking_tokens,
                candidate.get('finishMessage'),
            )
            + ' Retrying it as an empty generation.'
        )
      if content is None:
        raise self._no_answer_error(
            'the candidate carries no content at all',
            (
                'Nothing was generated, so no budget knob applies: this is '
                'either a transient service fault or a response the service '
                'withheld.'
            ),
            finish_reason,
            output_tokens,
            thinking_tokens,
            block_reason,
        )
      message = lf.Message.from_value(content, format='gemini')
      if finish_reason:
        message.metadata['finish_reason'] = finish_reason
      self._check_answer_present(
          message,
          candidate,
          finish_reason,
          output_tokens,
          thinking_tokens,
          block_reason,
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
      candidate: dict[str, Any],
      finish_reason: str | None,
      output_tokens: int,
      thinking_tokens: int,
      block_reason: str | None,
  ) -> None:
    """Fails fast when a parsed candidate carries no RECOVERABLE answer.

    A misfire into the output cap (see `_is_cap_misfire`) is raised first, as
    the retryable `MalformedFunctionCallError`. A short misfire
    (`MALFORMED_FUNCTION_CALL`) is logged by `result()` and returns below.

    Raising here spends the caller's whole generation: the error is
    non-retryable, so the episode that was mid-flight ends. That is the right
    trade only for a signature a re-send cannot change. The rule is therefore
    NARROW, and every arm of it names the mechanism that makes a re-send
    futile:

    * the prompt never reached decoding (`promptFeedback.blockReason`) -- the
      API defines that field on the prompt, returns no candidate with it, and
      names rephrasing the prompt as the remedy;
    * the decode budget was exhausted (`MAX_TOKENS`, including a continuation
      chain that `_merge_continuation_hops` keeps marked as one) -- the
      in-request rescues are already spent by the time this is reached;
    * the model stopped having generated literally nothing (`STOP` with no
      answer AND no thinking tokens) -- it was billed for no work, so there is
      no work in flight to recover.

    Everything else returns, and `LanguageModel.sample()` raises the RETRYABLE
    `lf.EmptyGenerationError` for it. A closed rule was tried here and
    measured: across 30 live `gemini-4-argon` trajectories, 28 of 28
    `GeminiEmptyGenerationError` aborts were `MALFORMED_FUNCTION_CALL` on a
    mid-episode turn that declared no tools, each billed 521-4,336 answer and
    983-6,020 thinking tokens -- a transient fault the very next sample
    recovers from, turned into a fatal abort. Non-zero `candidatesTokenCount`
    or `thoughtsTokenCount` is exactly that signature: the service did work
    and lost it, which is what a retry is for. The futile-retry case the
    closed rule was built for is not reached by widening the rule, but by
    `_merge_continuation_hops` refusing to relabel an exhausted budget `STOP`.

    A candidate-level content-policy flag (`SAFETY`, `RECITATION`,
    `BLOCKLIST`, ...) returns for the same reason. The API raises those on the
    response candidate content, i.e. on the tokens that were sampled, so the
    next sample is exactly what can clear them -- unlike
    `promptFeedback.blockReason`, which the API raises on the prompt and
    returns with no candidate at all.

    There are two ways out before any of this. Rendered text is an answer. A
    `functionCall` / `functionResponse` part is an answer the converter drops,
    which `_carries_unrendered_content` checks directly rather than inferring
    from `candidatesTokenCount > 0` -- a count that is billed for thinking and
    for tokens a truncated hop never returned.

    Each signature carries its own remedy, because they do not share one: a
    starved budget wants less thinking or more room, a budget that reached the
    answer wants only more room, and a blocked prompt or a model that stopped
    on its own wants neither.

    Args:
      message: The parsed candidate. Its text holds the non-thought answer
        content (thought parts are routed to `message.thought`), so empty text
        means nothing was rendered.
      candidate: The raw candidate, for parts the converter does not render.
      finish_reason: The candidate's `finishReason`, if any.
      output_tokens: Response-level `candidatesTokenCount`. It only ever
        chooses between two diagnoses of the same failure here; it is never
        read as evidence that an answer exists.
      thinking_tokens: Response-level `thoughtsTokenCount`, for diagnostics.
      block_reason: Response-level `promptFeedback.blockReason`, if any.

    Raises:
      MalformedFunctionCallError: If the candidate misfired into the cap.
      GeminiEmptyGenerationError: If the candidate carries no answer.
    """
    if message.text:
      return

    if _carries_unrendered_content(candidate):
      # A tool call is a real answer; the converter just does not render it.
      return

    if _is_cap_misfire(finish_reason, output_tokens, thinking_tokens):
      # More answer than thinking tokens and no text: a stray function call,
      # not starvation. A fresh sample often succeeds, so it is retryable.
      raise self._misfire_error(finish_reason, output_tokens, thinking_tokens)

    if block_reason:
      cause = (
          'the prompt was blocked before generation started '
          f'({block_reason}), so no answer was ever produced'
      )
      remedy = (
          'No budget knob applies: the request never reached decoding. '
          'Change the prompt or the safety configuration.'
      )
    elif finish_reason == 'MAX_TOKENS' and output_tokens > 0:
      # The decode budget did reach the answer, so this is not starvation: the
      # truncated hop was billed for answer tokens it did not return, and no
      # `continuationToken` -- or no hop following one -- recovered them.
      cause = (
          f'{output_tokens} answer tokens were billed on a MAX_TOKENS-'
          'truncated hop that returned no non-empty text part, and no '
          f'continuation recovered them ({thinking_tokens} thinking tokens '
          'were produced)'
      )
      remedy = (
          'The budget did reach the answer, so spending less of it on '
          'thinking is not the fix: raise `max_tokens` so the answer fits, '
          'or check whether the candidate returned a non-text part that is '
          'reported here as no answer.'
      )
    elif finish_reason == 'MAX_TOKENS':
      cause = (
          'the decode budget was exhausted before any answer token was '
          f'emitted ({thinking_tokens} thinking tokens were produced)'
      )
      remedy = (
          'Consider raising `max_tokens`, lowering the thinking budget '
          '(`max_thinking_tokens` / `thinking_level`), or shortening the '
          'prompt; a request that starves its decode budget tends to starve '
          'it again.'
      )
    elif finish_reason == 'STOP' and output_tokens == 0 and (
        thinking_tokens == 0
    ):
      cause = 'the model stopped without emitting any output token'
      remedy = (
          'The decode budget was not the constraint, so `max_tokens` is not '
          'the knob: look at the prompt and the safety configuration.'
      )
    else:
      # Everything left is transient, and the retry path is the one that
      # recovers it:
      #
      # * a tool-protocol fault (`MALFORMED_FUNCTION_CALL`,
      #   `UNEXPECTED_TOOL_CALL`) -- it fires mid-episode on turns that
      #   declared no tools at all, so there is no tool declaration to
      #   correct and the next sample simply succeeds;
      # * a candidate-level content-policy flag (`SAFETY`, `RECITATION`,
      #   `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `LANGUAGE`,
      #   `IMAGE_SAFETY`) -- the API raises these on the response candidate
      #   content, so they describe the tokens that were sampled and the next
      #   sample is what clears them. The prompt-level counterpart, which a
      #   re-send genuinely cannot change, is `promptFeedback.blockReason`
      #   and it is claimed by the first arm above;
      # * `STOP` over non-zero `candidatesTokenCount` / `thoughtsTokenCount`
      #   -- the service generated and then lost the work. A truncation is no
      #   longer hiding in here: `_merge_continuation_hops` keeps an exhausted
      #   budget labelled `MAX_TOKENS`, which the arm above claims;
      # * an absent or unrecognised `finishReason` -- an unclassified
      #   signature is not evidence of determinism, and guessing that it is
      #   costs a live episode on every false positive.
      #
      # `LanguageModel.sample()` sees the empty text and raises the retryable
      # `lf.EmptyGenerationError`, bounded by the caller's `max_attempts`.
      return

    raise self._no_answer_error(
        cause,
        remedy,
        finish_reason,
        output_tokens,
        thinking_tokens,
        block_reason,
    )

  def _no_answer_error(
      self,
      cause: str,
      remedy: str,
      finish_reason: str | None,
      output_tokens: int,
      thinking_tokens: int,
      block_reason: str | None,
  ) -> GeminiEmptyGenerationError:
    """Builds the error reported for a deterministically answer-less candidate.

    Args:
      cause: A phrase completing 'returned no answer content: ...', naming
        which signature was matched.
      remedy: A sentence naming what to change for that signature. The
        signatures do not share a remedy, so it is passed in rather than
        written once here.
      finish_reason: The candidate's `finishReason`, if any.
      output_tokens: Response-level `candidatesTokenCount`.
      thinking_tokens: Response-level `thoughtsTokenCount`.
      block_reason: Response-level `promptFeedback.blockReason`, if any.

    Returns:
      The error to raise, carrying the diagnosis, the knob to change, and the
      cost of the retry it declines to make.
    """
    return GeminiEmptyGenerationError(
        f'Model {self.model_id} returned no answer content: {cause} '
        f'(finishReason={finish_reason}, blockReason={block_reason}, '
        f'candidatesTokenCount={output_tokens}, '
        f'thoughtsTokenCount={thinking_tokens}). Any `continuationToken` the '
        'service offered was already followed, raising `max_tokens`, if '
        'pinned, for the answer tokens still owed across up to '
        f'{_MAX_PINNED_BUDGET_HOPS} requests. {remedy} '
        'This error is not retryable: the signature is a function of the '
        'request, so re-sending the prompt would reproduce it at a cost of '
        f'up to max_attempts={self.max_attempts} generations, every one of '
        'them billed in full.'
    )

  def _misfire_message(
      self,
      finish_reason: str | None,
      output_tokens: int,
      thinking_tokens: int,
      finish_message: str | None = None,
  ) -> str:
    """Returns the diagnosis of a misfired candidate, for logs and errors."""
    details = (
        f'finishReason={finish_reason}, candidatesTokenCount={output_tokens}, '
        f'thoughtsTokenCount={thinking_tokens}'
    )
    if finish_message:
      # User content, up to several KB, so only its length, as in `beyond_api`.
      details += f', finishMessageChars={len(finish_message)}'
    return (
        f'Model {self.model_id} misfired: it emitted a function call, which '
        f'the request does not declare, instead of text ({details}).'
    )

  def _misfire_error(
      self, finish_reason: str | None, output_tokens: int, thinking_tokens: int
  ) -> MalformedFunctionCallError:
    """Logs and returns the error for a misfire that ran into the cap."""
    error = MalformedFunctionCallError(
        'Empty generation. '
        + self._misfire_message(finish_reason, output_tokens, thinking_tokens)
        + ' The call ran into the output cap, which is not thinking starvation.'
        ' A fresh sample often succeeds, so samples are drawn up to'
        f' {self._cap_misfire_max_attempts()} per query, each costing up to the'
        ' cap.'
    )
    pg.logging.warning(str(error))
    return error

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 400 and (
        b'exceeds the maximum number of tokens' in content  # pyrefly: ignore[unsupported-operation]
        or b'Reduce the input token count and try again.' in content  # pyrefly: ignore[unsupported-operation]
        or b'Request payload size exceeds the limit' in content  # pyrefly: ignore[unsupported-operation]
        or b'Request contains text fields that are too large' in content  # pyrefly: ignore[unsupported-operation]
    ):
      return lf.ContextLimitError(f'{status_code}: {content}')
    return super()._error(status_code, content)
