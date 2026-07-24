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
import json
import os
from typing import Annotated, Any, Literal

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
    def max_tokens_per_minute(self) -> int:  # pyrefly: ignore[bad-override]
      return (self.max_input_tokens_per_minute
              + self.max_output_tokens_per_minute)


SUPPORTED_MODELS = [
    AnthropicModelInfo(
        model_id='claude-opus-4-6',
        provider='Anthropic',
        in_service=True,
        description='Claude 4.6 Opus model (02/05/2026).',
        release_date=datetime.datetime(2026, 2, 5),
        knowledge_cutoff=datetime.date(2025, 8, 31),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5,
            cost_per_1m_output_tokens=25,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=400_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-7',
        provider='Anthropic',
        in_service=True,
        description='Claude Opus 4.7 model.',
        release_date=datetime.datetime(2026, 2, 5),
        knowledge_cutoff=datetime.date(2026, 1, 31),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=25.0,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=400_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-8',
        provider='Anthropic',
        in_service=True,
        description='Claude Opus 4.8 model.',
        release_date=datetime.datetime(2026, 8, 5),
        knowledge_cutoff=datetime.date(2026, 7, 31),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=25.0,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=400_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-5',
        provider='Anthropic',
        in_service=True,
        description='Claude Opus 5 model.',
        # release_date and knowledge_cutoff intentionally omitted: Opus 5 is a
        # dateless/pinned snapshot and neither date is doc-grounded. Both fields
        # default to None (unknown), matching the convention used by most other
        # entries in this list rather than shipping fabricated dates.
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=25.0,
        ),
        # UNVERIFIED: no public/internal doc grounds Opus 5 quota; these
        # rate_limits are copied from the Opus 4.8 entry as a best-effort
        # placeholder. Update once official Opus 5 limits are published.
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=400_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-haiku-4-5-20251001',
        provider='Anthropic',
        in_service=True,
        description='Claude 4.5 Haiku model (10/15/2025).',
        release_date=datetime.datetime(2025, 10, 15),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=64_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.1,
            cost_per_1m_input_tokens=1,
            cost_per_1m_output_tokens=5,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=4_000_000,
            max_output_tokens_per_minute=800_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-sonnet-4-5-20250929',
        provider='Anthropic',
        in_service=True,
        description='Claude 4.5 Sonnet model (9/29/2025).',
        release_date=datetime.datetime(2025, 9, 29),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=64_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            # This rate limit is a total limit that applies to combined traffic
            # across both Sonnet 4 and Sonnet 4.5.
            max_requests_per_minute=4000,
            max_input_tokens_per_minute=2_000_000,
            max_output_tokens_per_minute=400_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-5-20251101',
        provider='Anthropic',
        in_service=True,
        description='Claude 4.5 Opus model (11/01/2025).',
        release_date=datetime.datetime(2025, 11, 1),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=64_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5,
            cost_per_1m_output_tokens=25,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # Tier 4 rate limits
            max_requests_per_minute=2000,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=400_000,
        ),
    ),
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
        model_id='claude-haiku-4-5@20251001',
        alias_for='claude-haiku-4-5-20251001',
        provider='VertexAI',
        in_service=True,
        description='Claude 4.5 Haiku model served on VertexAI (10/15/2025).',
        release_date=datetime.datetime(2025, 10, 15),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=64_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            # For global endpoint
            cost_per_1m_cached_input_tokens=0.1,
            cost_per_1m_input_tokens=1,
            cost_per_1m_output_tokens=5,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # For global endpoint
            max_requests_per_minute=2500,
            max_input_tokens_per_minute=200_000,
            max_output_tokens_per_minute=0,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-sonnet-4-5@20250929',
        alias_for='claude-sonnet-4-5-20250929',
        provider='VertexAI',
        in_service=True,
        description='Claude 4.5 Sonnet model (9/29/2025).',
        release_date=datetime.datetime(2025, 9, 29),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=64_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            # For global endpoint
            cost_per_1m_cached_input_tokens=0.3,
            cost_per_1m_input_tokens=3,
            cost_per_1m_output_tokens=15,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            # For global endpoint
            max_requests_per_minute=1500,
            max_input_tokens_per_minute=200_000,
            max_output_tokens_per_minute=0,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-6@latest',
        alias_for='claude-opus-4-6',
        provider='VertexAI',
        in_service=True,
        description='Claude 4.6 Opus model served on VertexAI (02/05/2026).',
        release_date=datetime.datetime(2026, 2, 5),
        knowledge_cutoff=datetime.date(2025, 8, 31),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5,
            cost_per_1m_output_tokens=25,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-7@latest',
        alias_for='claude-opus-4-7',
        provider='VertexAI',
        in_service=True,
        description='Claude Opus 4.7 model served on VertexAI.',
        release_date=datetime.datetime(2026, 2, 5),
        knowledge_cutoff=datetime.date(2026, 1, 31),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=25.0,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-8@latest',
        alias_for='claude-opus-4-8',
        provider='VertexAI',
        in_service=True,
        description='Claude Opus 4.8 model served on VertexAI.',
        release_date=datetime.datetime(2026, 8, 5),
        knowledge_cutoff=datetime.date(2026, 7, 31),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_000_000,
            max_output_tokens=128_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=25.0,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
            max_output_tokens_per_minute=80_000,
        ),
    ),
    AnthropicModelInfo(
        model_id='claude-opus-4-5@20251101',
        alias_for='claude-opus-4-5-20251101',
        provider='VertexAI',
        in_service=True,
        description='Claude 4.5 Opus model served on VertexAI (11/01/2025).',
        release_date=datetime.datetime(2025, 11, 1),
        input_modalities=(
            AnthropicModelInfo.INPUT_IMAGE_TYPES
            + AnthropicModelInfo.INPUT_DOC_TYPES
        ),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=64_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.5,
            cost_per_1m_input_tokens=5,
            cost_per_1m_output_tokens=25,
        ),
        rate_limits=AnthropicModelInfo.RateLimits(
            max_requests_per_minute=100,
            max_input_tokens_per_minute=1_000_000,
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


def _apply_cache_breakpoints(
    request: dict[str, Any],
    *,
    cache_system: bool = True,
    cache_last_message: bool = True,
) -> dict[str, Any]:
  """Mutates `request` in place to add Anthropic ephemeral cache_control breakpoints.

  Anthropic's prompt-caching API requires per-block markers inside
  messages[].content[].cache_control and system[].cache_control. Top-level
  cache_control is a no-op for caching large static prefixes.

  Args:
    request: An Anthropic request dict produced by Anthropic.request(). May
      contain 'system' as a bare string (current Anthropic.request shape) or a
      list of content blocks.
    cache_system: If True and request has a 'system' field, ensure it is in
      list-of-blocks form and stamp cache_control on the LAST system block.
    cache_last_message: If True, stamp cache_control on the LAST content block
      of the LAST entry in request['messages'].

  Returns:
    The mutated request (same object).

  """
  if not request:
    return request

  cache_ctrl = {'type': 'ephemeral'}

  if cache_system and 'system' in request and request['system']:
    sys = request['system']
    if isinstance(sys, str):
      request['system'] = [
          {'type': 'text', 'text': sys, 'cache_control': cache_ctrl}
      ]
    elif isinstance(sys, list) and sys:
      # Add it to the last element if not already present
      if 'cache_control' not in sys[-1]:
        sys[-1]['cache_control'] = cache_ctrl

  if cache_last_message and 'messages' in request and request['messages']:
    msgs = request['messages']
    last_msg = msgs[-1]
    if 'content' in last_msg and last_msg['content']:
      content = last_msg['content']
      if isinstance(content, list) and content:
        if 'cache_control' not in content[-1]:
          content[-1]['cache_control'] = cache_ctrl

  return request


@lf.use_init_args(['model'])
class Anthropic(rest.REST):
  """Anthropic Claude models.

  **Quick Start:**

  ```python
  import langfun as lf

  # Call Claude 3.5 Sonnet using API key from environment variable
  # 'ANTHROPIC_API_KEY'.
  lm = lf.llms.Claude35Sonnet()
  r = lm('Who are you?')
  print(r)
  ```

  **Setting up API key:**

  The Anthropic API key can be specified in following ways:

  1. At model instantiation:

     ```python
     lm = lf.llms.Claude35Sonnet(api_key='MY_API_KEY')

  2. via environment variable `ANTHROPIC_API_KEY`.

  **References:**

  *   https://docs.anthropic.com/claude/reference/messages_post
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

  thinking: Annotated[
      bool | None,
      (
          'Whether to enable thinking/reasoning mode. If True, enables '
          'thinking (adaptive for Claude 4.6+, manual with budget for older '
          'models). If None and max_thinking_tokens is set, thinking is '
          'enabled for backward compatibility. If False, thinking is disabled.'
      ),
  ] = None

  effort: Annotated[
      Literal['low', 'medium', 'high', 'xhigh', 'max'] | None,
      'Thinking depth for models supporting extended thinking (low, medium,'
      + ' high, xhigh, max).',
  ] = 'high'

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
  def headers(self) -> dict[str, Any]:  # pyrefly: ignore[bad-override]
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

  @property
  def _use_adaptive_thinking(self) -> bool:
    return self.model is not None and (
        'claude-opus-4-7' in self.model_id
        or 'claude-opus-4-8' in self.model_id
        or 'claude-opus-5' in self.model_id
    )

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
        if chunk.is_text:
          return chunk.to_text()
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

    # Anthropic prompt caching is enabled by default for all requests. The
    # cache_control stamps are additive — Anthropic ignores them on
    # non-cacheable models, so there is no behavioral risk for callers using
    # legacy/non-cacheable Claude variants. Mirrors Gemini 2.5/3.x implicit
    # prompt caching: users do nothing; the system maximizes cache hits.
    _apply_cache_breakpoints(request)

    return request

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    # Authropic requires `max_tokens` to be specified.
    max_tokens = (
        options.max_tokens or self.model_info.context_length.max_output_tokens  # pyrefly: ignore[missing-attribute]
    )
    args = dict(
        model=self.model,
        max_tokens=max_tokens,
        # Stream the response. Without this, Vertex/Anthropic buffers the
        # ENTIRE response and emits a single JSON blob only at completion, so
        # no bytes flow until the generation finishes -- any generation that
        # needs longer than the read timeout to produce its first (and only)
        # byte dies with a read timeout. With stream=True the server emits
        # Server-Sent Events incrementally; _reassemble_sse() below rebuilds
        # the full message + usage, and rest.py's inactivity/total deadline
        # split lets a slow-but-live generation run for hours.
        stream=True,
    )
    if options.stop:
      args['stop_sequences'] = options.stop
    if options.temperature is not None:
      args['temperature'] = options.temperature
    if options.top_k is not None:
      args['top_k'] = options.top_k
    if options.top_p is not None:
      args['top_p'] = options.top_p
    # Determine if thinking should be enabled.
    thinking_enabled = False
    if self.thinking:
      thinking_enabled = True
    elif self.thinking is None and options.max_thinking_tokens is not None:
      # Backward compatibility: max_thinking_tokens implies thinking=True.
      thinking_enabled = True
    # self.thinking is False -> no thinking regardless.

    if thinking_enabled:
      if self._use_adaptive_thinking:
        args['thinking'] = {
            'type': 'adaptive',
        }
        if self.model is not None and (
            'claude-opus-4-7' in self.model
            or 'claude-opus-4-8' in self.model
            or 'claude-opus-5' in self.model
        ):
          args['thinking']['display'] = 'summarized'

        effort = options.reasoning_effort or self.effort
        if effort:
          args['output_config'] = {'effort': effort}
      else:
        budget = options.max_thinking_tokens
        if budget is None:
          # Default to 50% of the total capacity, ensuring at least 1024.
          budget = max(1024, args['max_tokens'] // 2)

        args['thinking'] = {
            'type': 'enabled',
            'budget_tokens': budget,
        }
        # max_tokens, which is thinking tokens + response tokens, must be
        # greater than the thinking tokens.
        if args['max_tokens'] <= budget:
          args['max_tokens'] += budget

        # Ensure max_tokens does not exceed model's absolute hard capacity.
        model_cap = self.model_info.context_length.max_output_tokens  # pyrefly: ignore[missing-attribute]
        if args['max_tokens'] > model_cap:
          args['max_tokens'] = model_cap

        # If forced to clamp max_tokens, ensure budget remains strictly less.
        if budget >= args['max_tokens']:
          # Reserve 1024 tokens for final text response, ensuring budget is
          # valid.
          budget = max(1024, args['max_tokens'] - 1024)
          args['thinking']['budget_tokens'] = budget

      # Thinking isn't compatible with temperature, top_p, or top_k.
      # https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#important-considerations-when-using-extended-thinking
      args.pop('temperature', None)
      args.pop('top_k', None)
      args.pop('top_p', None)

    # Claude Opus 4.7, 4.8 and 5 do not support temperature, top_p, or top_k.
    if self.model is not None and (
        'claude-opus-4-7' in self.model
        or 'claude-opus-4-8' in self.model
        or 'claude-opus-5' in self.model
    ):
      args.pop('temperature', None)
      args.pop('top_k', None)
      args.pop('top_p', None)

    if options.extras:
      args.update(options.extras)
    return args

  def result(self, response_json: dict[str, Any]) -> lf.LMSamplingResult:
    message = lf.Message.from_value(response_json, format='anthropic')
    input_tokens = response_json['usage']['input_tokens']
    output_tokens = response_json['usage']['output_tokens']
    return lf.LMSamplingResult(
        [lf.LMSample(message)],
        usage=lf.LMSamplingUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )

  def _response_to_message_dict(self, response: Any) -> dict[str, Any]:
    """Builds the buffered Anthropic message dict consumed by `result`.

    With body `stream=True`, Vertex/Anthropic returns a Server-Sent Events
    body instead of a single JSON object. This reassembles that event stream
    back into the exact same dict shape the non-streaming Messages API would
    have returned (`role`, `content` blocks, `stop_reason`, `usage`), so that
    `result()` (and `lf.Message.from_value(..., format='anthropic')`) stay
    correct and unchanged.

    For robustness (and to keep buffered-JSON unit tests working), a body that
    is already a single JSON object is parsed directly.

    Args:
      response: The streaming (or buffered) HTTP response from the API.

    Returns:
      The reassembled Anthropic message dict (role/content/stop_reason/usage).
    """
    raw = response.content
    text = raw.decode('utf-8') if isinstance(raw, (bytes, bytearray)) else raw
    stripped = text.lstrip()
    # A buffered (non-streamed) JSON body starts with '{'. An SSE body starts
    # with an 'event:' / 'data:' line.
    if stripped.startswith('{'):
      return json.loads(text)
    return self._reassemble_sse(text)

  def _reassemble_sse(self, text: str) -> dict[str, Any]:
    """Reassembles an Anthropic Messages SSE stream into a message dict.

    Handles the Anthropic streaming event sequence:
      message_start -> (content_block_start,
                        content_block_delta*, content_block_stop)* ->
      message_delta -> message_stop

    Rebuilds full text/thinking content AND token usage:
    - message_start: provides the message skeleton incl. input_tokens.
    - content_block_start: initializes a content block at its index.
    - content_block_delta: appends text_delta / thinking_delta /
      signature_delta / input_json_delta fragments to that block.
    - content_block_stop: finalizes any accumulated tool-use JSON.
    - message_delta: carries the final stop_reason and usage.output_tokens.
    - message_stop: terminator.

    Args:
      text: The full Server-Sent Events response body as text.

    Returns:
      The reassembled Anthropic message dict, equivalent to the non-streaming
      Messages API response.
    """
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    json_buffers: dict[int, str] = {}
    final_usage: dict[str, Any] = {}
    saw_message_stop = False

    for raw_line in text.splitlines():
      line = raw_line.strip()
      if not line.startswith('data:'):
        continue
      data_str = line[len('data:') :].strip()
      if not data_str or data_str == '[DONE]':
        continue
      try:
        event = json.loads(data_str)
      except json.JSONDecodeError as e:
        # A well-formed Anthropic SSE stream emits exactly one valid-JSON
        # payload per `data:` line; keep-alives are comment (`:`) or
        # `event: ping` lines (skipped by the `data:` check above) and
        # `[DONE]`/empty payloads are handled above. A `data:` line that
        # fails to parse here is therefore a CORRUPT/TRUNCATED event, not a
        # benign keep-alive. Silently `continue`-ing would drop that
        # fragment -- and if it was a content_block_delta, the reconstructed
        # text would be silently truncated while message_stop/stop_reason
        # still arrive, defeating the terminal sentinel below. The buffered
        # path fails loudly on a malformed body (json.loads -> ValueError ->
        # LMError) and never returns partial content; preserve that
        # no-silent-corruption guarantee here by raising a RETRYABLE error
        # (mid-stream corruption is transient, matching the empty-stream and
        # truncated-stream guards).
        raise lf.TemporaryLMError(
            'Anthropic SSE stream contained a malformed data event '
            f'({data_str[:120]!r}); cannot guarantee complete content, '
            'retrying.'
        ) from e
      etype = event.get('type')

      if etype == 'message_start':
        message = dict(event['message'])
        message['content'] = []
      elif etype == 'content_block_start':
        idx = event['index']
        block = dict(event.get('content_block', {}))
        blocks[idx] = block
        if block.get('type') == 'tool_use':
          json_buffers[idx] = ''
      elif etype == 'content_block_delta':
        idx = event['index']
        delta = event.get('delta', {})
        dtype = delta.get('type')
        block = blocks.setdefault(idx, {})
        if dtype == 'text_delta':
          block.setdefault('type', 'text')
          block['text'] = block.get('text', '') + delta.get('text', '')
        elif dtype == 'thinking_delta':
          block.setdefault('type', 'thinking')
          block['thinking'] = block.get('thinking', '') + delta.get(
              'thinking', ''
          )
        elif dtype == 'signature_delta':
          block['signature'] = block.get('signature', '') + delta.get(
              'signature', ''
          )
        elif dtype == 'input_json_delta':
          json_buffers[idx] = json_buffers.get(idx, '') + delta.get(
              'partial_json', ''
          )
      elif etype == 'content_block_stop':
        idx = event['index']
        buf = json_buffers.get(idx)
        if buf:
          try:
            blocks[idx]['input'] = json.loads(buf)
          except json.JSONDecodeError:
            blocks[idx]['input'] = {}
      elif etype == 'message_delta':
        delta = event.get('delta', {})
        if message is not None:
          for key in ('stop_reason', 'stop_sequence'):
            if key in delta:
              message[key] = delta[key]
        usage = event.get('usage')
        if usage:
          final_usage.update(usage)
      elif etype == 'error':
        # An in-stream `error` event arrives on an HTTP 200 body. Map the
        # Anthropic error type to the equivalent HTTP status and reuse the
        # buffered-path classification (self._error) so transient failures
        # (overloaded_error -> 529 -> TemporaryLMError; rate_limit_error ->
        # 429 -> RateLimitError; api_error -> 500 -> TemporaryLMError) stay
        # RETRYABLE, while genuinely permanent ones (invalid_request_error ->
        # 400) remain permanent. A bare ValueError/lf.LMError here would be
        # downgraded to a PERMANENT error by _parse_response and silently lose
        # the retry the buffered 529/429 path already gets.
        error_type_to_status = {
            'invalid_request_error': 400,
            'authentication_error': 401,
            'permission_error': 403,
            'not_found_error': 404,
            'request_too_large': 413,
            'rate_limit_error': 429,
            'api_error': 500,
            'overloaded_error': 529,
        }
        err = event.get('error') or {}
        status = error_type_to_status.get(err.get('type'), 500)
        # Anthropic._error inspects `content` as bytes, so encode it.
        raise self._error(status, json.dumps(err or event).encode('utf-8'))
      elif etype == 'message_stop':
        saw_message_stop = True
      # 'ping' and other unrecognized events need no handling.

    if message is None:
      # A 200 whose SSE body never produced a message_start (empty body, only
      # keep-alives, or a dropped/garbled stream) is a transient anomaly, not a
      # permanent client error. Raise a RETRYABLE error -- a bare ValueError
      # here would be downgraded to a permanent lf.LMError by _parse_response.
      raise lf.TemporaryLMError(
          'Anthropic SSE stream produced no message (empty body or no '
          'message_start event); retrying.'
      )
    # Assemble content blocks in index order.
    message['content'] = [blocks[i] for i in sorted(blocks)]
    # Merge usage: message_start carries input_tokens (and an initial
    # output_tokens); message_delta carries the FINAL output_tokens.
    usage = dict(message.get('usage') or {})
    usage.update(final_usage)
    message['usage'] = usage
    # TERMINAL SENTINEL: only accept a stream that actually completed. A
    # cleanly-closed-but-incomplete 200 (no message_stop, or stop_reason still
    # null because message_delta never arrived) would otherwise be returned as
    # silently truncated text with an under-counted output_tokens. Require BOTH
    # the message_stop terminator AND a non-null stop_reason; otherwise raise a
    # RETRYABLE error so the request is retried rather than silently accepted.
    if not saw_message_stop or message.get('stop_reason') is None:
      raise lf.TemporaryLMError(
          'Anthropic SSE stream ended without a terminal message_stop and a '
          'non-null stop_reason (truncated/incomplete 200 response); retrying.'
      )
    return message

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 413 and b'Prompt is too long' in content:  # pyrefly: ignore[unsupported-operation]
      return lf.ContextLimitError(f'{status_code}: {content}')
    if status_code == 400 and b'prompt is too long' in content:  # pyrefly: ignore[unsupported-operation]
      return lf.ContextLimitError(f'{status_code}: {content}')
    return super()._error(status_code, content)


class Claude46(Anthropic):
  """Base class for Claude 4.6 models."""


# pylint: disable=invalid-name
class Claude5Opus(Anthropic):
  """Claude Opus 5 model."""

  model = 'claude-opus-5'


class Claude48Opus(Anthropic):
  """Claude Opus 4.8 model."""

  model = 'claude-opus-4-8'


class Claude47Opus(Anthropic):
  """Claude Opus 4.7 model."""

  model = 'claude-opus-4-7'


class Claude46Opus(Claude46):
  """Claude 4.6 Opus model."""

  model = 'claude-opus-4-6'


class Claude45(Anthropic):
  """Base class for Claude 4.5 models."""


# pylint: disable=invalid-name
class Claude45Haiku_20251001(Claude45):
  """Claude 4.5 Haiku model 20251001."""

  model = 'claude-haiku-4-5-20251001'


# pylint: disable=invalid-name
class Claude45Sonnet_20250929(Claude45):
  """Claude 4.5 Sonnet model 20250929."""

  model = 'claude-sonnet-4-5-20250929'


# pylint: disable=invalid-name
class Claude45Opus_20251101(Claude45):
  """Claude 4.5 Opus model 20251101."""

  model = 'claude-opus-4-5-20251101'


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
