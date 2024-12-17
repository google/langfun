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
"""Vertex AI generative models."""

import base64
import functools
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
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


# https://cloud.google.com/vertex-ai/generative-ai/pricing
# describes that the average number of characters per token is about 4.
AVGERAGE_CHARS_PER_TOKEN = 4


# Price in US dollars,
# from https://cloud.google.com/vertex-ai/generative-ai/pricing
# as of 2024-10-10.
SUPPORTED_MODELS_AND_SETTINGS = {
    'gemini-1.5-pro-001': pg.Dict(
        rpm=100,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-pro-002': pg.Dict(
        rpm=100,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash-002': pg.Dict(
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-flash-001': pg.Dict(
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-pro': pg.Dict(
        rpm=100,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash': pg.Dict(
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-pro-preview-0514': pg.Dict(
        rpm=50,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-pro-preview-0409': pg.Dict(
        rpm=50,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash-preview-0514': pg.Dict(
        rpm=200,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.0-pro': pg.Dict(
        rpm=300,
        cost_per_1k_input_chars=0.000125,
        cost_per_1k_output_chars=0.000375,
    ),
    'gemini-1.0-pro-vision': pg.Dict(
        rpm=100,
        cost_per_1k_input_chars=0.000125,
        cost_per_1k_output_chars=0.000375,
    ),
    # TODO(sharatsharat): Update costs when published
    'gemini-exp-1206': pg.Dict(
        rpm=20,
        cost_per_1k_input_chars=0.000,
        cost_per_1k_output_chars=0.000,
    ),
    # TODO(sharatsharat): Update costs when published
    'gemini-2.0-flash-exp': pg.Dict(
        rpm=20,
        cost_per_1k_input_chars=0.000,
        cost_per_1k_output_chars=0.000,
    ),
    # TODO(chengrun): Set a more appropriate rpm for endpoint.
    'vertexai-endpoint': pg.Dict(
        rpm=20,
        cost_per_1k_input_chars=0.0000125,
        cost_per_1k_output_chars=0.0000375,
    ),
}


@lf.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class VertexAI(rest.REST):
  """Language model served on VertexAI with REST API."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      (
          'Vertex AI model name with REST API support. See '
          'https://cloud.google.com/vertex-ai/generative-ai/docs/'
          'model-reference/inference#supported-models'
          ' for details.'
      ),
  ]

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

  supported_modalities: Annotated[
      list[str],
      'A list of MIME types for supported modalities'
  ] = []

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
    credentials = self.credentials
    if credentials is None:
      # Use default credentials.
      credentials = google_auth.default(
          scopes=['https://www.googleapis.com/auth/cloud-platform']
      )
    self._credentials = credentials

  @property
  def max_concurrency(self) -> int:
    """Returns the maximum number of concurrent requests."""
    return self.rate_to_max_concurrency(
        requests_per_min=SUPPORTED_MODELS_AND_SETTINGS[self.model].rpm,
        tokens_per_min=0,
    )

  def estimate_cost(
      self,
      num_input_tokens: int,
      num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    cost_per_1k_input_chars = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_input_chars', None
    )
    cost_per_1k_output_chars = SUPPORTED_MODELS_AND_SETTINGS[self.model].get(
        'cost_per_1k_output_chars', None
    )
    if cost_per_1k_output_chars is None or cost_per_1k_input_chars is None:
      return None
    return (
        cost_per_1k_input_chars * num_input_tokens
        + cost_per_1k_output_chars * num_output_tokens
    ) * AVGERAGE_CHARS_PER_TOKEN / 1000

  @functools.cached_property
  def _session(self):
    assert self._api_initialized
    assert self._credentials is not None
    assert auth_requests is not None
    s = auth_requests.AuthorizedSession(self._credentials)
    s.headers.update(self.headers or {})
    return s

  @property
  def headers(self):
    return {
        'Content-Type': 'application/json; charset=utf-8',
    }

  @property
  def api_endpoint(self) -> str:
    return (
        f'https://{self.location}-aiplatform.googleapis.com/v1/projects/'
        f'{self.project}/locations/{self.location}/publishers/google/'
        f'models/{self.model}:generateContent'
    )

  def request(
      self, prompt: lf.Message, sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    request = dict(
        generationConfig=self._generation_config(prompt, sampling_options)
    )
    request['contents'] = [self._content_from_message(prompt)]
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
    return config

  def _content_from_message(self, prompt: lf.Message) -> dict[str, Any]:
    """Gets generation content from langfun message."""
    parts = []
    for lf_chunk in prompt.chunk():
      if isinstance(lf_chunk, str):
        parts.append({'text': lf_chunk})
      elif isinstance(lf_chunk, lf_modalities.Mime):
        try:
          modalities = lf_chunk.make_compatible(
              self.supported_modalities + ['text/plain']
          )
          if isinstance(modalities, lf_modalities.Mime):
            modalities = [modalities]
          for modality in modalities:
            if modality.is_text:
              parts.append({'text': modality.to_text()})
            else:
              parts.append({
                  'inlineData': {
                      'data': base64.b64encode(modality.to_bytes()).decode(),
                      'mimeType': modality.mime_type,
                  }
              })
        except lf.ModalityError as e:
          raise lf.ModalityError(f'Unsupported modality: {lf_chunk!r}') from e
      else:
        raise lf.ModalityError(f'Unsupported modality: {lf_chunk!r}')
    return dict(role='user', parts=parts)

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    messages = [
        self._message_from_content_parts(candidate['content']['parts'])
        for candidate in json['candidates']
    ]
    usage = json['usageMetadata']
    input_tokens = usage['promptTokenCount']
    output_tokens = usage['candidatesTokenCount']
    return lf.LMSamplingResult(
        [lf.LMSample(message) for message in messages],
        usage=lf.LMSamplingUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost=self.estimate_cost(
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
            ),
        ),
    )

  def _message_from_content_parts(
      self, parts: list[dict[str, Any]]
  ) -> lf.Message:
    """Converts Vertex AI's content parts protocol to message."""
    chunks = []
    for part in parts:
      if text_part := part.get('text'):
        chunks.append(text_part)
      else:
        raise ValueError(f'Unsupported part: {part}')
    return lf.AIMessage.from_chunks(chunks)


IMAGE_TYPES = [
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/heic',
    'image/heif',
]

AUDIO_TYPES = [
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

VIDEO_TYPES = [
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

DOCUMENT_TYPES = [
    'application/pdf',
    'text/plain',
    'text/csv',
    'text/html',
    'text/xml',
    'text/x-script.python',
    'application/json',
]


class VertexAIGemini2_0(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 2.0 model."""

  supported_modalities: pg.typing.List(str).freeze(  # pytype: disable=invalid-annotation
      DOCUMENT_TYPES + IMAGE_TYPES + AUDIO_TYPES + VIDEO_TYPES
  )


class VertexAIGeminiFlash2_0Exp(VertexAIGemini2_0):  # pylint: disable=invalid-name
  """Vertex AI Gemini 2.0 Flash model."""

  model = 'gemini-2.0-flash-exp'


class VertexAIGemini1_5(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 model."""

  supported_modalities: pg.typing.List(str).freeze(  # pytype: disable=invalid-annotation
      DOCUMENT_TYPES + IMAGE_TYPES + AUDIO_TYPES + VIDEO_TYPES
  )


class VertexAIGeminiPro1_5(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro'


class VertexAIGeminiPro1_5_002(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-002'


class VertexAIGeminiPro1_5_001(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-001'


class VertexAIGeminiPro1_5_0514(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro preview model."""

  model = 'gemini-1.5-pro-preview-0514'


class VertexAIGeminiPro1_5_0409(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro preview model."""

  model = 'gemini-1.5-pro-preview-0409'


class VertexAIGeminiFlash1_5(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash'


class VertexAIGeminiFlash1_5_002(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-002'


class VertexAIGeminiFlash1_5_001(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-001'


class VertexAIGeminiFlash1_5_0514(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash preview model."""

  model = 'gemini-1.5-flash-preview-0514'


class VertexAIGeminiPro1(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.0 Pro model."""

  model = 'gemini-1.0-pro'


class VertexAIGeminiPro1Vision(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.0 Pro Vision model."""

  model = 'gemini-1.0-pro-vision'
  supported_modalities: pg.typing.List(str).freeze(  # pytype: disable=invalid-annotation
      IMAGE_TYPES + VIDEO_TYPES
  )


class VertexAIEndpoint(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Endpoint model."""

  model = 'vertexai-endpoint'

  endpoint: Annotated[str, 'Vertex AI Endpoint ID.']

  @property
  def api_endpoint(self) -> str:
    return (
        f'https://{self.location}-aiplatform.googleapis.com/v1/projects/'
        f'{self.project}/locations/{self.location}/'
        f'endpoints/{self.endpoint}:generateContent'
    )
