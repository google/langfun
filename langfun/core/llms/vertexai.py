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
  from google.auth import credentials as credentials_lib
  from google.auth.transport import requests as auth_requests
  # pylint: enable=g-import-not-at-top

  Credentials = credentials_lib.Credentials
except ImportError:
  google_auth = None
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

  model: Annotated[
      str | None,
      'Model ID.'
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

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'VertexAI({self.model})'

  @functools.cached_property
  def _session(self):
    assert self._api_initialized
    assert self._credentials is not None
    assert auth_requests is not None
    s = auth_requests.AuthorizedSession(self._credentials)
    s.headers.update(self.headers or {})
    return s


#
# Gemini models served by Vertex AI.
#


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAIGemini(VertexAI, gemini.Gemini):
  """Gemini models served by Vertex AI.."""

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:generateContent'
    )


class VertexAIGeminiFlash2_0ThinkingExp_20241219(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini Flash 2.0 Thinking model launched on 12/19/2024."""

  api_version = 'v1alpha'
  model = 'gemini-2.0-flash-thinking-exp-1219'
  timeout = None


class VertexAIGeminiFlash2_0Exp(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 2.0 Flash model."""

  model = 'gemini-2.0-flash-exp'


class VertexAIGeminiExp_20241206(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini Experimental model launched on 12/06/2024."""

  model = 'gemini-exp-1206'


class VertexAIGeminiExp_20241114(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini Experimental model launched on 11/14/2024."""

  model = 'gemini-exp-1114'


class VertexAIGeminiPro1_5(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-latest'


class VertexAIGeminiPro1_5_002(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-002'


class VertexAIGeminiPro1_5_001(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-001'


class VertexAIGeminiFlash1_5(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash'


class VertexAIGeminiFlash1_5_002(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-002'


class VertexAIGeminiFlash1_5_001(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-001'


class VertexAIGeminiPro1(VertexAIGemini):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.0 Pro model."""

  model = 'gemini-1.0-pro'


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


class VertexAIClaude3_Opus_20240229(VertexAIAnthropic):
  """Anthropic's Claude 3 Opus model on VertexAI."""
  model = 'claude-3-opus@20240229'


class VertexAIClaude3_5_Sonnet_20241022(VertexAIAnthropic):
  """Anthropic's Claude 3.5 Sonnet model on VertexAI."""
  model = 'claude-3-5-sonnet-v2@20241022'


class VertexAIClaude3_5_Sonnet_20240620(VertexAIAnthropic):
  """Anthropic's Claude 3.5 Sonnet model on VertexAI."""
  model = 'claude-3-5-sonnet@20240620'


class VertexAIClaude3_5_Haiku_20241022(VertexAIAnthropic):
  """Anthropic's Claude 3.5 Haiku model on VertexAI."""
  model = 'claude-3-5-haiku@20241022'

# pylint: enable=invalid-name

#
# Llama models on Vertex AI.
# pylint: disable=line-too-long
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing?_gl=1*ukuk6u*_ga*MjEzMjc4NjM2My4xNzMzODg4OTg3*_ga_WH2QY8WWF5*MTczNzEzNDU1Mi4xMjQuMS4xNzM3MTM0NzczLjU5LjAuMA..#meta-models
# pylint: enable=line-too-long

LLAMA_MODELS = {
    'llama-3.2-90b-vision-instruct-maas': pg.Dict(
        latest_update='2024-09-25',
        in_service=True,
        rpm=0,
        tpm=0,
        # Free during preview.
        cost_per_1m_input_tokens=None,
        cost_per_1m_output_tokens=None,
    ),
    'llama-3.1-405b-instruct-maas': pg.Dict(
        latest_update='2024-09-25',
        in_service=True,
        rpm=0,
        tpm=0,
        # GA.
        cost_per_1m_input_tokens=5,
        cost_per_1m_output_tokens=16,
    ),
    'llama-3.1-70b-instruct-maas': pg.Dict(
        latest_update='2024-09-25',
        in_service=True,
        rpm=0,
        tpm=0,
        # Free during preview.
        cost_per_1m_input_tokens=None,
        cost_per_1m_output_tokens=None,
    ),
    'llama-3.1-8b-instruct-maas': pg.Dict(
        latest_update='2024-09-25',
        in_service=True,
        rpm=0,
        tpm=0,
        # Free during preview.
        cost_per_1m_input_tokens=None,
        cost_per_1m_output_tokens=None,
    )
}


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAILlama(VertexAI, openai_compatible.OpenAICompatible):
  """Llama models on VertexAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(pg.MISSING_VALUE, list(LLAMA_MODELS.keys())),
      'Llama model ID.',
  ]

  locations: Annotated[
      Literal['us-central1'],
      (
          'GCP locations with Llama models hosted. '
          'See https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#regions-quotas'
      )
  ] = 'us-central1'

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

  @property
  def max_concurrency(self) -> int:
    rpm = LLAMA_MODELS[self.model].get('rpm', 0)
    tpm = LLAMA_MODELS[self.model].get('tpm', 0)
    return self.rate_to_max_concurrency(
        requests_per_min=rpm, tokens_per_min=tpm
    )

  def estimate_cost(
      self,
      num_input_tokens: int,
      num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    cost_per_1m_input_tokens = LLAMA_MODELS[self.model].get(
        'cost_per_1m_input_tokens', None
    )
    cost_per_1m_output_tokens = LLAMA_MODELS[self.model].get(
        'cost_per_1m_output_tokens', None
    )
    if cost_per_1m_output_tokens is None or cost_per_1m_input_tokens is None:
      return None
    return (
        cost_per_1m_input_tokens * num_input_tokens
        + cost_per_1m_output_tokens * num_output_tokens
    ) / 1000_000


# pylint: disable=invalid-name
class VertexAILlama3_2_90B(VertexAILlama):
  """Llama 3.2 90B vision instruct model on VertexAI."""

  model = 'llama-3.2-90b-vision-instruct-maas'


class VertexAILlama3_1_405B(VertexAILlama):
  """Llama 3.1 405B vision instruct model on VertexAI."""

  model = 'llama-3.1-405b-instruct-maas'


class VertexAILlama3_1_70B(VertexAILlama):
  """Llama 3.1 70B vision instruct model on VertexAI."""

  model = 'llama-3.1-70b-instruct-maas'


class VertexAILlama3_1_8B(VertexAILlama):
  """Llama 3.1 8B vision instruct model on VertexAI."""

  model = 'llama-3.1-8b-instruct-maas'
# pylint: enable=invalid-name

#
# Mistral models on Vertex AI.
# pylint: disable=line-too-long
# Pricing: https://cloud.google.com/vertex-ai/generative-ai/pricing?_gl=1*ukuk6u*_ga*MjEzMjc4NjM2My4xNzMzODg4OTg3*_ga_WH2QY8WWF5*MTczNzEzNDU1Mi4xMjQuMS4xNzM3MTM0NzczLjU5LjAuMA..#mistral-models
# pylint: enable=line-too-long


MISTRAL_MODELS = {
    'mistral-large-2411': pg.Dict(
        latest_update='2024-11-21',
        in_service=True,
        rpm=0,
        tpm=0,
        # GA.
        cost_per_1m_input_tokens=2,
        cost_per_1m_output_tokens=6,
    ),
    'mistral-large@2407': pg.Dict(
        latest_update='2024-07-24',
        in_service=True,
        rpm=0,
        tpm=0,
        # GA.
        cost_per_1m_input_tokens=2,
        cost_per_1m_output_tokens=6,
    ),
    'mistral-nemo@2407': pg.Dict(
        latest_update='2024-07-24',
        in_service=True,
        rpm=0,
        tpm=0,
        # GA.
        cost_per_1m_input_tokens=0.15,
        cost_per_1m_output_tokens=0.15,
    ),
    'codestral-2501': pg.Dict(
        latest_update='2025-01-13',
        in_service=True,
        rpm=0,
        tpm=0,
        # GA.
        cost_per_1m_input_tokens=0.3,
        cost_per_1m_output_tokens=0.9,
    ),
    'codestral@2405': pg.Dict(
        latest_update='2024-05-29',
        in_service=True,
        rpm=0,
        tpm=0,
        # GA.
        cost_per_1m_input_tokens=0.2,
        cost_per_1m_output_tokens=0.6,
    ),
}


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAIMistral(VertexAI, openai_compatible.OpenAICompatible):
  """Mistral AI models on VertexAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(pg.MISSING_VALUE, list(MISTRAL_MODELS.keys())),
      'Mistral model ID.',
  ]

  locations: Annotated[
      Literal['us-central1', 'europe-west4'],
      (
          'GCP locations with Mistral models hosted. '
          'See https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral#regions-quotas'
      )
  ] = 'us-central1'

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/mistralai/'
        f'models/{self.model}:rawPredict'
    )

  @property
  def max_concurrency(self) -> int:
    rpm = MISTRAL_MODELS[self.model].get('rpm', 0)
    tpm = MISTRAL_MODELS[self.model].get('tpm', 0)
    return self.rate_to_max_concurrency(
        requests_per_min=rpm, tokens_per_min=tpm
    )

  def estimate_cost(
      self,
      num_input_tokens: int,
      num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    cost_per_1m_input_tokens = MISTRAL_MODELS[self.model].get(
        'cost_per_1m_input_tokens', None
    )
    cost_per_1m_output_tokens = MISTRAL_MODELS[self.model].get(
        'cost_per_1m_output_tokens', None
    )
    if cost_per_1m_output_tokens is None or cost_per_1m_input_tokens is None:
      return None
    return (
        cost_per_1m_input_tokens * num_input_tokens
        + cost_per_1m_output_tokens * num_output_tokens
    ) / 1000_000


# pylint: disable=invalid-name
class VertexAIMistralLarge_20241121(VertexAIMistral):
  """Mistral Large model on VertexAI released on 2024/11/21."""

  model = 'mistral-large-2411'


class VertexAIMistralLarge_20240724(VertexAIMistral):
  """Mistral Large model on VertexAI released on 2024/07/24."""

  model = 'mistral-large@2407'


class VertexAIMistralNemo_20240724(VertexAIMistral):
  """Mistral Nemo model on VertexAI released on 2024/07/24."""

  model = 'mistral-nemo@2407'


class VertexAICodestral_20250113(VertexAIMistral):
  """Mistral Nemo model on VertexAI released on 2024/07/24."""

  model = 'codestral-2501'


class VertexAICodestral_20240529(VertexAIMistral):
  """Mistral Nemo model on VertexAI released on 2024/05/29."""

  model = 'codestral@2405'
# pylint: enable=invalid-name
