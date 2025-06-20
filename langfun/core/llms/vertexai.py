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
"""Vertex AI generative models."""

import datetime
import functools
import os
from typing import Annotated, Any, Literal

import langfun.core as lf
from langfun.core.llms import anthropic
from langfun.core.llms import gemini
from langfun.core.llms import openai_compatible
from langfun.core.llms import rest
import pyglove as pg

try:
  # pylint: disable=g-import-not-at-top
  from google import auth as google_auth
  from google.auth import exceptions as auth_exceptions
  from google.auth import credentials as credentials_lib
  from google.auth.transport import requests as auth_requests
  # pylint: enable=g-import-not-at-top

  Credentials = credentials_lib.Credentials
except ImportError:
  google_auth = None
  auth_exceptions = None
  credentials_lib = None
  auth_requests = None
  Credentials = Any


@pg.use_init_args(['api_endpoint'])
class VertexAI(rest.REST):
  """Base class for VertexAI models.

  This class handles the authentication of vertex AI models. Subclasses
  should implement `request` and `result` methods, as well as the `api_endpoint`
  property. Or let users to provide them as __init__ arguments.

  Please check out VertexAIGemini in `gemini.py` as an example.
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE,
          [
              m.model_id for m in gemini.SUPPORTED_MODELS
              if m.provider == 'VertexAI' or (
                  isinstance(m.provider, pg.hyper.OneOf)
                  and 'VertexAI' in m.provider.candidates
              )
          ]
      ),
      'The name of the model to use.',
  ]

  model: Annotated[
      str | None,
      'Model name.'
  ] = None

  project: Annotated[
      str | None,
      (
          'Vertex AI project ID. Or set from environment variable '
          'VERTEXAI_PROJECT.'
      ),
  ] = None

  location: Annotated[
      str | None,
      (
          'Vertex AI service location. Or set from environment variable '
          'VERTEXAI_LOCATION.'
      ),
  ] = None

  credentials: Annotated[
      Credentials | None,
      (
          'Credentials to use. If None, the default credentials to the '
          'environment will be used.'
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    if google_auth is None:
      raise ValueError(
          'Please install "langfun[llm-google-vertex]" to use Vertex AI models.'
      )
    self._project = None
    self._credentials = None

  def _initialize(self):
    project = self.project or os.environ.get('VERTEXAI_PROJECT', None)
    if not project:
      raise ValueError(
          'Please specify `project` during `__init__` or set environment '
          'variable `VERTEXAI_PROJECT` with your Vertex AI project ID.'
      )

    location = self.location or os.environ.get('VERTEXAI_LOCATION', None)
    if not location:
      raise ValueError(
          'Please specify `location` during `__init__` or set environment '
          'variable `VERTEXAI_LOCATION` with your Vertex AI service location.'
      )

    self._project = project
    self._location = location

    credentials = self.credentials
    if credentials is None:
      # Use default credentials.
      credentials, _ = google_auth.default(
          scopes=['https://www.googleapis.com/auth/cloud-platform']
      )
    self._credentials = credentials

  def _session(self):
    assert self._credentials is not None
    assert auth_requests is not None
    return auth_requests.AuthorizedSession(self._credentials)

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    assert auth_exceptions is not None
    try:
      return super()._sample_single(prompt)
    except (
        auth_exceptions.RefreshError,
    ) as e:
      raise lf.TemporaryLMError(
          f'Failed to refresh Google authentication credentials: {e}'
      ) from e

#
# Gemini models served by Vertex AI.
#


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAIGemini(VertexAI, gemini.Gemini):
  """Gemini models served by Vertex AI.."""

  # Set default location to us-central1.
  location = 'us-central1'

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:generateContent'
    )

  @functools.cached_property
  def model_info(self) -> gemini.GeminiModelInfo:
    return super().model_info.clone(override=dict(provider='VertexAI'))


#
# Production models.
#
class VertexAIGemini25Pro(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Pro GA model launched on 06/17/2025."""

  model = 'gemini-2.5-pro'
  location = 'global'


class VertexAIGemini25Flash(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Flash GA model launched on 06/17/2025."""

  model = 'gemini-2.5-flash'
  location = 'global'


class VertexAIGemini25ProPreview_20250605(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Pro model launched on 06/05/2025."""
  model = 'gemini-2.5-pro-preview-06-05'
  location = 'global'


class VertexAIGemini25FlashPreview_20250520(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Flash model launched on 05/20/2025."""
  model = 'gemini-2.5-flash-preview-05-20'


class VertexAIGemini25ProPreview_20250506(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Pro model launched on 05/06/2025."""
  model = 'gemini-2.5-pro-preview-05-06'


class VertexAIGemini25FlashPreview_20250417(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Flash model launched on 04/17/2025."""
  model = 'gemini-2.5-flash-preview-04-17'


class VertexAIGemini25ProPreview_20250325(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Pro model launched on 03/25/2025."""
  model = 'gemini-2.5-pro-preview-03-25'


class VertexAIGemini25ProExp_20250325(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.5 Pro model launched on 03/25/2025."""
  model = 'gemini-2.5-pro-exp-03-25'


class VertexAIGemini2Flash(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini Flash 2.0 model (latest stable)."""
  model = 'gemini-2.0-flash'


class VertexAIGemini2Flash_001(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini Flash 2.0 model version 001."""
  model = 'gemini-2.0-flash-001'


class VertexAIGemini2FlashLitePreview_20250205(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini 2.0 Flash lite preview model launched on 02/05/2025."""
  model = 'gemini-2.0-flash-lite-preview-02-05'


class VertexAIGemini15Pro(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model (latest stable)."""
  model = 'gemini-1.5-pro'


class VertexAIGemini15Pro_002(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model (version 002)."""
  model = 'gemini-1.5-pro-002'


class VertexAIGemini15Pro_001(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model (version 001)."""
  model = 'gemini-1.5-pro-001'


class VertexAIGemini15Flash(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model (latest stable)."""
  model = 'gemini-1.5-flash'


class VertexAIGemini15Flash_002(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model (version 002)."""
  model = 'gemini-1.5-flash-002'


class VertexAIGemini15Flash_001(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model (version 001)."""
  model = 'gemini-1.5-flash-001'


class VertexAIGemini15Flash8B(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash 8B model (latest stable)."""
  model = 'gemini-1.5-flash-8b'


class VertexAIGemini15Flash8B_001(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash 8B model (version 001)."""
  model = 'gemini-1.5-flash-8b-001'

#
# Experimental models.
#


class VertexAIGemini2ProExp_20250205(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini Flash 2.0 Pro model launched on 02/05/2025."""
  model = 'gemini-2.0-pro-exp-02-05'


class VertexAIGemini2FlashThinkingExp_20250121(VertexAIGemini):  # pylint: disable=invalid-name
  """Gemini Flash 2.0 Thinking model launched on 01/21/2025."""
  model = 'gemini-2.0-flash-thinking-exp-01-21'
  timeout = None


class VertexAIGeminiFlash2_0Exp(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 2.0 Flash model."""
  model = 'gemini-2.0-flash-exp'


class VertexAIGeminiExp_20241206(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini Experimental model launched on 12/06/2024."""
  model = 'gemini-exp-1206'


#
# Anthropic models on Vertex AI.
#


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAIAnthropic(VertexAI, anthropic.Anthropic):
  """Anthropic models on VertexAI."""

  location: Annotated[
      Literal['us-east5', 'europe-west1'],
      'GCP location with Anthropic models hosted.'
  ] = 'us-east5'

  api_version = 'vertex-2023-10-16'

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    mi = anthropic._SUPPORTED_MODELS_BY_MODEL_ID[self.model]  # pylint: disable=protected-access
    if mi.provider != 'VertexAI':
      for m in anthropic.SUPPORTED_MODELS:
        if m.provider == 'VertexAI' and m.alias_for == m.model_id:
          mi = m
          self.rebind(model=mi.model_id, skip_notification=True)
          break
    return mi

  @property
  def headers(self):
    return {
        'Content-Type': 'application/json; charset=utf-8',
    }

  @property
  def api_endpoint(self) -> str:
    return (
        f'https://{self.location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self.location}/publishers/anthropic/'
        f'models/{self.model}:streamRawPredict'
    )

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ):
    request = super().request(prompt, sampling_options)
    request['anthropic_version'] = self.api_version
    del request['model']
    return request


# pylint: disable=invalid-name


class VertexAIClaude4Opus_20250514(VertexAIAnthropic):
  """Anthropic's Claude 4 Opus model on VertexAI."""
  model = 'claude-opus-4@20250514'
  location = 'us-east5'


class VertexAIClaude4Sonnet_20250514(VertexAIAnthropic):
  """Anthropic's Claude 4 Sonnet model on VertexAI."""
  model = 'claude-sonnet-4@20250514'
  location = 'us-east5'


class VertexAIClaude37Sonnet_20250219(VertexAIAnthropic):
  """Anthropic's Claude 3.7 model on VertexAI."""
  model = 'claude-3-7-sonnet@20250219'
  location = 'us-east5'


class VertexAIClaude35Sonnet_20241022(VertexAIAnthropic):
  """Anthropic's Claude 3.5 Sonnet model on VertexAI."""
  model = 'claude-3-5-sonnet-v2@20241022'


class VertexAIClaude35Haiku_20241022(VertexAIAnthropic):
  """Anthropic's Claude 3.5 Haiku model on VertexAI."""
  model = 'claude-3-5-haiku@20241022'


class VertexAIClaude3Opus_20240229(VertexAIAnthropic):
  """Anthropic's Claude 3 Opus model on VertexAI."""
  model = 'claude-3-opus@20240229'

# pylint: enable=invalid-name

#
# Llama models on Vertex AI.
# https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing#meta-models

LLAMA_MODELS = [
    lf.ModelInfo(
        model_id='llama-3.1-405b-instruct-maas',
        in_service=True,
        model_type='instruction-tuned',
        provider='VertexAI',
        description=(
            'Llama 3.2 405B vision instruct model on VertexAI (Preview)'
        ),
        url='https://huggingface.co/meta-llama/Llama-3.1-405B',
        release_date=datetime.datetime(2024, 7, 23),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=5.0,
            cost_per_1m_output_tokens=16.0,
        ),
        rate_limits=None,
    ),
    lf.ModelInfo(
        model_id='llama-3.2-90b-vision-instruct-maas',
        in_service=True,
        model_type='instruction-tuned',
        provider='VertexAI',
        description=(
            'Llama 3.2 90B vision instruct model on VertexAI (Preview)'
        ),
        release_date=datetime.datetime(2024, 7, 23),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=8_192,
        ),
        # Free during preview.
        pricing=None,
        rate_limits=None,
    ),
    lf.ModelInfo(
        model_id='llama-3.1-70b-instruct-maas',
        in_service=True,
        model_type='instruction-tuned',
        provider='VertexAI',
        description=(
            'Llama 3.2 70B vision instruct model on VertexAI (Preview)'
        ),
        release_date=datetime.datetime(2024, 7, 23),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=8_192,
        ),
        # Free during preview.
        pricing=None,
        rate_limits=None,
    ),
    lf.ModelInfo(
        model_id='llama-3.1-8b-instruct-maas',
        in_service=True,
        model_type='instruction-tuned',
        provider='VertexAI',
        description=(
            'Llama 3.2 8B vision instruct model on VertexAI (Preview)'
        ),
        release_date=datetime.datetime(2024, 7, 23),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=0,
            max_output_tokens=0,
        ),
        # Free during preview.
        pricing=None,
        rate_limits=None,
    ),
]

_LLAMA_MODELS_BY_MODEL_ID = {m.model_id: m for m in LLAMA_MODELS}


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAILlama(VertexAI, openai_compatible.OpenAICompatible):
  """Llama models on VertexAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(pg.MISSING_VALUE, [m.model_id for m in LLAMA_MODELS]),
      'Llama model ID.',
  ]

  locations: Annotated[
      Literal['us-central1'],
      (
          'GCP locations with Llama models hosted. '
          'See https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#regions-quotas'
      )
  ] = 'us-central1'

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    return _LLAMA_MODELS_BY_MODEL_ID[self.model]

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1beta1/projects/'
        f'{self._project}/locations/{self._location}/endpoints/'
        f'openapi/chat/completions'
    )

  def request(
      self,
      prompt: lf.Message,
      sampling_options: lf.LMSamplingOptions
  ):
    request = super().request(prompt, sampling_options)
    request['model'] = f'meta/{self.model}'
    return request


# pylint: disable=invalid-name
class VertexAILlama32_90B(VertexAILlama):
  """Llama 3.2 90B vision instruct model on VertexAI."""
  model = 'llama-3.2-90b-vision-instruct-maas'


class VertexAILlama31_405B(VertexAILlama):
  """Llama 3.1 405B vision instruct model on VertexAI."""
  model = 'llama-3.1-405b-instruct-maas'


class VertexAILlama31_70B(VertexAILlama):
  """Llama 3.1 70B vision instruct model on VertexAI."""
  model = 'llama-3.1-70b-instruct-maas'


class VertexAILlama31_8B(VertexAILlama):
  """Llama 3.1 8B vision instruct model on VertexAI."""
  model = 'llama-3.1-8b-instruct-maas'


# pylint: enable=invalid-name

#
# Mistral models on Vertex AI.
# pylint: disable=line-too-long
# Models: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing#mistral-models
# pylint: enable=line-too-long

MISTRAL_MODELS = [
    lf.ModelInfo(
        model_id='mistral-large-2411',
        in_service=True,
        model_type='instruction-tuned',
        provider='VertexAI',
        description='Mistral Large model on VertexAI (GA) version 11/21/2024',
        release_date=datetime.datetime(2024, 11, 21),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=2.0,
            cost_per_1m_output_tokens=6.0,
        ),
        rate_limits=None,
    ),
    lf.ModelInfo(
        model_id='codestral-2501',
        in_service=True,
        model_type='instruction-tuned',
        provider='VertexAI',
        description=(
            'Mistral Codestral model on VertexAI (GA) (version 01/13/2025)'
        ),
        release_date=datetime.datetime(2025, 1, 13),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=128_000,
            max_output_tokens=8_192,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.3,
            cost_per_1m_output_tokens=0.9,
        ),
        rate_limits=None,
    ),
]

_MISTRAL_MODELS_BY_MODEL_ID = {m.model_id: m for m in MISTRAL_MODELS}


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAIMistral(VertexAI, openai_compatible.OpenAICompatible):
  """Mistral AI models on VertexAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(pg.MISSING_VALUE, [m.model_id for m in MISTRAL_MODELS]),
      'Mistral model ID.',
  ]

  locations: Annotated[
      Literal['us-central1', 'europe-west4'],
      (
          'GCP locations with Mistral models hosted. '
          'See https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral#regions-quotas'
      )
  ] = 'us-central1'

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    return _MISTRAL_MODELS_BY_MODEL_ID[self.model]

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/mistralai/'
        f'models/{self.model}:rawPredict'
    )


# pylint: disable=invalid-name
class VertexAIMistralLarge_20241121(VertexAIMistral):
  """Mistral Large model on VertexAI released on 2024/11/21."""
  model = 'mistral-large-2411'


class VertexAICodestral_20250113(VertexAIMistral):
  """Mistral Nemo model on VertexAI released on 2024/07/24."""
  model = 'codestral-2501'

# pylint: enable=invalid-name


#
# Register Vertex AI models so they can be retrieved with LanguageModel.get().
#


def _register_vertexai_models():
  """Register Vertex AI models."""
  for m in gemini.SUPPORTED_MODELS:
    if m.provider == 'VertexAI' or (
        isinstance(m.provider, pg.hyper.OneOf)
        and 'VertexAI' in m.provider.candidates
    ):
      lf.LanguageModel.register(m.model_id, VertexAIGemini)

  for m in anthropic.SUPPORTED_MODELS:
    if m.provider == 'VertexAI':
      lf.LanguageModel.register(m.model_id, VertexAIAnthropic)

  for m in LLAMA_MODELS:
    lf.LanguageModel.register(m.model_id, VertexAILlama)

  for m in MISTRAL_MODELS:
    lf.LanguageModel.register(m.model_id, VertexAIMistral)

_register_vertexai_models()
