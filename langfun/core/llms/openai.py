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
"""Language models from OpenAI."""

import datetime
import functools
import os
from typing import Annotated, Any, Final

import langfun.core as lf
from langfun.core.llms import openai_compatible
import pyglove as pg


class OpenAIModelInfo(lf.ModelInfo):
  """OpenAI model info."""

  # Constants for supported MIME types.
  INPUT_IMAGE_TYPES = [
      'image/png',
      'image/jpeg',
      'image/gif',
      'image/webp',
  ]

  LINKS = dict(
      models='https://platform.openai.com/docs/models',
      pricing='https://openai.com/api/pricing/',
      rate_limits='https://platform.openai.com/docs/guides/rate-limits',
      error_codes='https://platform.openai.com/docs/guides/error-codes',
  )

  provider: Final[str] = 'OpenAI'  # pylint: disable=invalid-name


#
# !!! Please sort models by model family and model_id (time descending).
#

SUPPORTED_MODELS = [
    # GPT-4.1 models
    OpenAIModelInfo(
        model_id='gpt-4.1',
        alias_for='gpt-4.1-2025-04-14',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4.1 model (latest stable).',
        url='https://platform.openai.com/docs/models/gpt-4.1',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1_047_576,
            max_output_tokens=32_768,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.50,
            cost_per_1m_input_tokens=2.0,
            cost_per_1m_output_tokens=8.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=30_000_000,
        ),
    ),
    # o3 models.
    OpenAIModelInfo(
        model_id='o3',
        alias_for='o3-2025-04-16',
        in_service=True,
        model_type='thinking',
        description='GPT O3 model.',
        url='https://platform.openai.com/docs/models/o3',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=100_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=2.5,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=40.0,
        ),
        # Set to 1/3 of Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=3_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    # o4-mini models.
    OpenAIModelInfo(
        model_id='o4-mini',
        alias_for='o4-mini-2025-04-16',
        in_service=True,
        model_type='thinking',
        description='GPT O4-mini model.',
        url='https://platform.openai.com/docs/models/o4-mini',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=100_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.275,
            cost_per_1m_input_tokens=1.1,
            cost_per_1m_output_tokens=4.4,
        ),
        # Set to 1/3 of Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=50_000_000,
        ),
    ),
    # GPT-4.5 models
    OpenAIModelInfo(
        model_id='gpt-4.5-preview-2025-02-27',
        alias_for='gpt-4.5-preview-2025-02-27',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4.5 preview model.',
        url='https://platform.openai.com/docs/models#gpt-4-5',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=37.5,
            cost_per_1m_input_tokens=75,
            cost_per_1m_output_tokens=150.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    # o3-mini models.
    OpenAIModelInfo(
        model_id='o3-mini',
        alias_for='o3-mini-2025-01-31',
        in_service=True,
        model_type='thinking',
        description='GPT O3-mini model (latest).',
        url='https://platform.openai.com/docs/models#o3-mini',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=100_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.55,
            cost_per_1m_input_tokens=1.1,
            cost_per_1m_output_tokens=4.4,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='o3-mini-2025-01-31',
        in_service=True,
        model_type='thinking',
        description='GPT O3-mini model (01/31/2025).',
        url='https://platform.openai.com/docs/models#o3-mini',
        release_date=datetime.datetime(2025, 1, 31),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=100_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.55,
            cost_per_1m_input_tokens=1.1,
            cost_per_1m_output_tokens=4.4,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    # o1-mini models.
    OpenAIModelInfo(
        model_id='o1-mini',
        alias_for='o1-mini-2024-09-12',
        in_service=True,
        model_type='thinking',
        description='GPT O1-mini model (latest).',
        url='https://platform.openai.com/docs/models#o1',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=65_536,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.55,
            cost_per_1m_input_tokens=1.1,
            cost_per_1m_output_tokens=4.4,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='o1-mini-2024-09-12',
        in_service=True,
        model_type='thinking',
        description='GPT O1-mini model (09/12/2024).',
        url='https://platform.openai.com/docs/models#o1',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=65_536,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.55,
            cost_per_1m_input_tokens=1.1,
            cost_per_1m_output_tokens=4.4,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='o1-preview',
        alias_for='o1-preview-2024-09-12',
        in_service=True,
        model_type='thinking',
        description='GPT O1-preview model (latest).',
        url='https://platform.openai.com/docs/models#o1',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=32_768,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=7.5,
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='o1-preview-2024-09-12',
        in_service=True,
        model_type='thinking',
        description='GPT O1-preview model (09/12/2024).',
        url='https://platform.openai.com/docs/models#o1',
        release_date=datetime.datetime(2024, 9, 12),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=32_768,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=7.5,
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    # o1 models.
    OpenAIModelInfo(
        model_id='o1',
        alias_for='o1-2024-12-17',
        in_service=True,
        model_type='thinking',
        description='GPT O1 model (latest).',
        url='https://platform.openai.com/docs/models#o1',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=100_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=7.5,
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='o1-2024-12-17',
        in_service=True,
        model_type='thinking',
        description='GPT O1 model (12/17/2024).',
        url='https://platform.openai.com/docs/models#o1',
        release_date=datetime.datetime(2024, 12, 17),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=200_000,
            max_output_tokens=100_000,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=7.5,
            cost_per_1m_input_tokens=15.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    # GPT-4o-mini models
    OpenAIModelInfo(
        model_id='gpt-4o-mini',
        alias_for='gpt-4o-mini-2024-07-18',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o mini model (latest).',
        url='https://platform.openai.com/docs/models#gpt-4o-mini',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.075,
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.6,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4o-mini-2024-07-18',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o mini model (07/18/2024).',
        url='https://platform.openai.com/docs/models#gpt-4o-mini',
        release_date=datetime.datetime(2024, 7, 18),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=0.075,
            cost_per_1m_input_tokens=0.15,
            cost_per_1m_output_tokens=0.6,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    # GPT-4o models
    OpenAIModelInfo(
        model_id='gpt-4o',
        alias_for='gpt-4o-2024-08-06',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o model (latest stable).',
        url='https://platform.openai.com/docs/models#gpt-4o',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.25,
            cost_per_1m_input_tokens=2.5,
            cost_per_1m_output_tokens=10.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4o-2024-11-20',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o model (11/20/2024).',
        url='https://platform.openai.com/docs/models#gpt-4o',
        release_date=datetime.datetime(2024, 11, 20),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.25,
            cost_per_1m_input_tokens=2.5,
            cost_per_1m_output_tokens=10.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4o-2024-08-06',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o model (08/06/2024).',
        url='https://platform.openai.com/docs/models#gpt-4o',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=1.25,
            cost_per_1m_input_tokens=2.5,
            cost_per_1m_output_tokens=10.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4o-2024-05-13',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o model (05/13/2024).',
        url='https://platform.opedsnai.com/docs/models#gpt-4o',
        release_date=datetime.datetime(2024, 5, 13),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=15.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='chatgpt-4o-latest',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4o model ChatGPT version (latest).',
        url='https://platform.openai.com/docs/models#gpt-4o',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=16_384,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=15.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=2_000_000,
        ),
    ),
    # GPT-4 Turbo models.
    OpenAIModelInfo(
        model_id='gpt-4-turbo',
        alias_for='gpt-4-turbo-2024-04-09',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 Turbo model (latest).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=800_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-turbo-2024-04-09',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 Turbo model (04/09/2024).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2024, 4, 9),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=800_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-turbo-preview',
        alias_for='gpt-4-0125-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 Turbo preview model (latest).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=800_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-0125-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 Turbo preview model (01/25/2024).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2024, 1, 25),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=800_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-1106-preview',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 Turbo preview model (11/06/2024).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2024, 11, 6),
        input_modalities=OpenAIModelInfo.INPUT_IMAGE_TYPES,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=30.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=800_000,
        ),
    ),
    # GPT-4 models.
    OpenAIModelInfo(
        model_id='gpt-4',
        alias_for='gpt-4-0613',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 model (latest).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=8_192,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=300_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-0613',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 4 model (06/13/2023).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2023, 6, 13),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=8_192,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=300_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-0314',
        in_service=False,
        model_type='instruction-tuned',
        description='GPT 4 model (03/14/2023).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2023, 3, 14),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=8_192,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=300_000,
        ),
    ),
    # GPT-4 32K models.
    OpenAIModelInfo(
        model_id='gpt-4-32k',
        alias_for='gpt-4-32k-0613',
        in_service=False,
        model_type='instruction-tuned',
        description='GPT 4 32K model (latest).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=32_768,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=60.0,
            cost_per_1m_output_tokens=120.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=300_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-32k-0613',
        in_service=False,
        model_type='instruction-tuned',
        description='GPT 4 32K model (06/13/2023).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2023, 6, 13),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=32_768,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=60.0,
            cost_per_1m_output_tokens=120.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=300_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-4-32k-0314',
        in_service=False,
        model_type='instruction-tuned',
        description='GPT 4 32K model (03/14/2023).',
        url='https://platform.openai.com/docs/models#gpt-4-turbo-and-gpt-4',
        release_date=datetime.datetime(2023, 3, 14),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=32_768,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=60.0,
            cost_per_1m_output_tokens=120.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=300_000,
        ),
    ),
    # GPT 3.5 Turbo models.
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo',
        alias_for='gpt-3.5-turbo-0125',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo model (latest).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=0.5,
            cost_per_1m_output_tokens=1.5,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo-0125',
        in_service=True,
        release_date=datetime.datetime(2024, 1, 25),
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo model (01/25/2024).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=0.5,
            cost_per_1m_output_tokens=1.5,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo-1106',
        in_service=True,
        release_date=datetime.datetime(2023, 11, 6),
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo model (11/06/2023).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=1.0,
            cost_per_1m_output_tokens=2.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo-0613',
        in_service=False,
        release_date=datetime.datetime(2023, 6, 13),
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo model (06/13/2023).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=1.5,
            cost_per_1m_output_tokens=2.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    # GPT 3.5 Turbo 16K models.
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo-16k',
        alias_for='gpt-3.5-turbo-16k-0613',
        in_service=True,
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo 16K model (latest).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_385,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=4.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo-16k-0613',
        in_service=False,
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo 16K model (06/13/2023).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        release_date=datetime.datetime(2023, 6, 13),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_385,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=4.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='gpt-3.5-turbo-16k-0301',
        in_service=False,
        model_type='instruction-tuned',
        description='GPT 3.5 Turbo 16K model (03/01/2023).',
        url='https://platform.openai.com/docs/models#gpt-3-5-turbo',
        release_date=datetime.datetime(2023, 3, 1),
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_385,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=4.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=10_000,
            max_tokens_per_minute=10_000_000,
        ),
    ),
    # GPT 3.5 models.
    OpenAIModelInfo(
        model_id='text-davinci-003',
        in_service=False,
        model_type='instruction-tuned',
        description='ChatGPT 3.5 model.',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=3.0,
            cost_per_1m_output_tokens=3.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=None,
            max_tokens_per_minute=None,
        ),
    ),
    # GPT 3 models.
    OpenAIModelInfo(
        model_id='babbage-002',
        in_service=True,
        model_type='pretrained',
        description='GPT3 base model babagge-002',
        url='https://platform.openai.com/docs/models#gpt-base',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=2.0,
            cost_per_1m_output_tokens=2.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=3_000,
            max_tokens_per_minute=250_000,
        ),
    ),
    OpenAIModelInfo(
        model_id='davinci-002',
        in_service=True,
        model_type='pretrained',
        description='GPT3 base model Davinci-002 ',
        url='https://platform.openai.com/docs/models#gpt-base',
        input_modalities=lf.ModelInfo.TEXT_INPUT_ONLY,
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=16_384,
            max_output_tokens=4_096,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_cached_input_tokens=None,
            cost_per_1m_input_tokens=2.0,
            cost_per_1m_output_tokens=2.0,
        ),
        # Tier 5 rate limits.
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=3_000,
            max_tokens_per_minute=250_000,
        ),
    ),
]

_SUPPORTED_MODELS_BY_MODEL_ID = {m.model_id: m for m in SUPPORTED_MODELS}


@lf.use_init_args(['model'])
class OpenAI(openai_compatible.OpenAICompatible):
  """OpenAI model."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(s.model_id for s in SUPPORTED_MODELS)
      ),
      'The name of the model to use.',
  ]

  api_endpoint: str = 'https://api.openai.com/v1/chat/completions'

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
          "variable 'OPENAI_ORGANIZATION'. Based on the value, usages from "
          "these API requests will count against the organization's quota. "
      ),
  ] = None

  project: Annotated[
      str | None,
      (
          'Project. If None, the key will be read from environment '
          "variable 'OPENAI_PROJECT'. Based on the value, usages from "
          "these API requests will count against the project's quota. "
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    self._api_key = None
    self._organization = None
    self._project = None
    self.__dict__.pop('model_info', None)

  def _initialize(self):
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
    self._project = self.project or os.environ.get('OPENAI_PROJECT', None)

  @property
  def headers(self) -> dict[str, Any]:
    assert self._api_initialized
    headers = super().headers
    headers['Authorization'] = f'Bearer {self._api_key}'
    if self._organization:
      headers['OpenAI-Organization'] = self._organization
    if self._project:
      headers['OpenAI-Project'] = self._project
    return headers

  @functools.cached_property
  def model_info(self) -> OpenAIModelInfo:
    return _SUPPORTED_MODELS_BY_MODEL_ID[self.model]

  @classmethod
  def dir(cls):
    return [s.model_id for s in SUPPORTED_MODELS if s.in_service]

  def _request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    # Reasoning models (o1 series) does not support `logprobs` by 2024/09/12.
    if options.logprobs and self.model.startswith(('o1-', 'o3-')):
      raise RuntimeError('`logprobs` is not supported on {self.model!r}.')
    return super()._request_args(options)


class Gpt41(OpenAI):
  """GPT-4.1."""
  model = 'gpt-4.1'


class GptO4Mini(OpenAI):
  """GPT O4 mini."""

  model = 'o4-mini'
  timeout = None


class GptO3(OpenAI):
  """GPT O3."""

  model = 'o3'
  timeout = None


class Gpt45Preview_20250227(OpenAI):  # pylint: disable=invalid-name
  """Gpt-4.5 Preview 2025-02-27."""

  model = 'gpt-4.5-preview-2025-02-27'


class GptO3Mini(OpenAI):
  """GPT-O3-mini."""
  model = 'o3-mini'
  timeout = None


class GptO1(OpenAI):
  """GPT-O1."""
  model = 'o1'
  timeout = None


class GptO1Preview(OpenAI):
  """GPT-O1."""
  model = 'o1-preview'
  timeout = None


class GptO1Preview_20240912(OpenAI):   # pylint: disable=invalid-name
  """GPT O1."""
  model = 'o1-preview-2024-09-12'
  timeout = None


class GptO1Mini(OpenAI):
  """GPT O1-mini."""
  model = 'o1-mini'
  timeout = None


class GptO1Mini_20240912(OpenAI):   # pylint: disable=invalid-name
  """GPT O1-mini."""
  model = 'o1-mini-2024-09-12'
  timeout = None


class Gpt4oMini(OpenAI):
  """GPT-4o Mini."""
  model = 'gpt-4o-mini'


class Gpt4oMini_20240718(OpenAI):  # pylint:disable=invalid-name
  """GPT-4o Mini."""
  model = 'gpt-4o-mini-2024-07-18'


class Gpt4o(OpenAI):
  """GPT-4o."""
  model = 'gpt-4o'


class Gpt4o_20241120(OpenAI):     # pylint:disable=invalid-name
  """GPT-4o version 2024-11-20."""
  model = 'gpt-4o-2024-11-20'


class Gpt4o_20240806(OpenAI):     # pylint:disable=invalid-name
  """GPT-4o version 2024-08-06."""
  model = 'gpt-4o-2024-08-06'


class Gpt4o_20240513(OpenAI):     # pylint:disable=invalid-name
  """GPT-4o version 2024-05-13."""
  model = 'gpt-4o-2024-05-13'


class Gpt4(OpenAI):
  """GPT-4."""
  model = 'gpt-4'


class Gpt4Turbo(Gpt4):
  """GPT-4 Turbo with 128K context window. Knowledge up to Dec. 2023."""
  model = 'gpt-4-turbo'


class Gpt4Turbo_20240409(Gpt4Turbo):  # pylint:disable=invalid-name
  """GPT-4 Turbo with 128K context window. Knowledge up to Dec. 2023."""
  model = 'gpt-4-turbo-2024-04-09'


class Gpt4TurboPreview(Gpt4):
  """GPT-4 Turbo Preview with 128k context window. Knowledge up to Dec. 2023."""
  model = 'gpt-4-turbo-preview'


class Gpt4TurboPreview_20240125(Gpt4TurboPreview):  # pylint: disable=invalid-name
  """GPT-4 Turbo Preview with 128k context window. Knowledge up to Dec. 2023."""
  model = 'gpt-4-0125-preview'


class Gpt4TurboPreview_20231106(Gpt4TurboPreview):  # pylint: disable=invalid-name
  """GPT-4 Turbo Preview with 128k context window. Knowledge up to Apr. 2023."""
  model = 'gpt-4-1106-preview'


class Gpt4_20230613(Gpt4):    # pylint:disable=invalid-name
  """GPT-4 @20230613. 8K context window. Knowledge up to 9-2021."""
  model = 'gpt-4-0613'


class Gpt4_32K(Gpt4):       # pylint:disable=invalid-name
  """Latest GPT-4 with 32K context window."""
  model = 'gpt-4-32k'


class Gpt4_32K_20230613(Gpt4_32K):    # pylint:disable=invalid-name
  """GPT-4 @20230613. 32K context window. Knowledge up to 9-2021."""
  model = 'gpt-4-32k-0613'


class Gpt35(OpenAI):
  """GPT-3.5. 4K max tokens, trained up on data up to Sep, 2021."""
  model = 'text-davinci-003'


class Gpt35Turbo(Gpt35):
  """Most capable GPT-3.5 model, 10x cheaper than GPT35 (text-davinci-003)."""
  model = 'gpt-3.5-turbo'


class Gpt35Turbo_20240125(Gpt35Turbo):   # pylint:disable=invalid-name
  """GPT-3.5 Turbo @20240125. 16K context window. Knowledge up to 09/2021."""
  model = 'gpt-3.5-turbo-0125'


class Gpt35Turbo_20231106(Gpt35Turbo):   # pylint:disable=invalid-name
  """Gpt3.5 Turbo @20231106. 16K context window. Knowledge up to 09/2021."""
  model = 'gpt-3.5-turbo-1106'


class Gpt35Turbo_20230613(Gpt35Turbo):   # pylint:disable=invalid-name
  """Gpt3.5 Turbo snapshot at 2023/06/13, with 4K context window size."""
  model = 'gpt-3.5-turbo-0613'


class Gpt35Turbo16K(Gpt35Turbo):
  """Latest GPT-3.5 model with 16K context window size."""
  model = 'gpt-3.5-turbo-16k'


class Gpt35Turbo16K_20230613(Gpt35Turbo):   # pylint:disable=invalid-name
  """Gtp 3.5 Turbo 16K 0613."""
  model = 'gpt-3.5-turbo-16k-0613'


def _register_openai_models():
  """Registers OpenAI models."""
  for m in SUPPORTED_MODELS:
    lf.LanguageModel.register(m.model_id, OpenAI)

_register_openai_models()
