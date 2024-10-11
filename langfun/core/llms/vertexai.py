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

import functools
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
import pyglove as pg

try:
  # pylint: disable=g-import-not-at-top
  from google.auth import credentials as credentials_lib
  import vertexai
  from google.cloud.aiplatform import models as aiplatform_models
  from vertexai import generative_models
  from vertexai import language_models
  # pylint: enable=g-import-not-at-top

  Credentials = credentials_lib.Credentials
except ImportError:
  credentials_lib = None   # pylint: disable=invalid-name
  vertexai = None
  generative_models = None
  language_models = None
  aiplatform_models = None
  Credentials = Any


# https://cloud.google.com/vertex-ai/generative-ai/pricing
# describes that the average number of characters per token is about 4.
AVGERAGE_CHARS_PER_TOEKN = 4


# Price in US dollars,
# from https://cloud.google.com/vertex-ai/generative-ai/pricing
# as of 2024-10-10.
SUPPORTED_MODELS_AND_SETTINGS = {
    'gemini-1.5-pro-001': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-pro-002': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash-002': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-flash-001': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-pro': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-pro-latest': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash-latest': pg.Dict(
        api='gemini',
        rpm=500,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.5-pro-preview-0514': pg.Dict(
        api='gemini',
        rpm=50,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-pro-preview-0409': pg.Dict(
        api='gemini',
        rpm=50,
        cost_per_1k_input_chars=0.0003125,
        cost_per_1k_output_chars=0.00125,
    ),
    'gemini-1.5-flash-preview-0514': pg.Dict(
        api='gemini',
        rpm=200,
        cost_per_1k_input_chars=0.00001875,
        cost_per_1k_output_chars=0.000075,
    ),
    'gemini-1.0-pro': pg.Dict(
        api='gemini',
        rpm=300,
        cost_per_1k_input_chars=0.000125,
        cost_per_1k_output_chars=0.000375,
    ),
    'gemini-1.0-pro-vision': pg.Dict(
        api='gemini',
        rpm=100,
        cost_per_1k_input_chars=0.000125,
        cost_per_1k_output_chars=0.000375,
    ),
    # PaLM APIs.
    'text-bison': pg.Dict(
        api='palm',
        rpm=1600
    ),
    'text-bison-32k': pg.Dict(
        api='palm',
        rpm=300
    ),
    'text-unicorn': pg.Dict(
        api='palm',
        rpm=100
    ),
    # Endpoint
    # TODO(chengrun): Set a more appropriate rpm for endpoint.
    'custom': pg.Dict(api='endpoint', rpm=20),
}


@lf.use_init_args(['model'])
class VertexAI(lf.LanguageModel):
  """Language model served on VertexAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      (
          'Vertex AI model name. See '
          'https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models '
          'for details.'
      ),
  ]

  endpoint_name: pg.typing.Annotated[
      str | None,
      'Vertex Endpoint name or ID.',
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
    self.__dict__.pop('_api_initialized', None)
    if generative_models is None:
      raise RuntimeError(
          'Please install "langfun[llm-google-vertex]" to use Vertex AI models.'
      )

  @functools.cached_property
  def _api_initialized(self):
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

    credentials = self.credentials
    # Placeholder for Google-internal credentials.
    assert vertexai is not None
    vertexai.init(project=project, location=location, credentials=credentials)
    return True

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'VertexAI({self.model})'

  @property
  def resource_id(self) -> str:
    """Returns a string to identify the resource for rate control."""
    return self.model_id

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
    ) * AVGERAGE_CHARS_PER_TOEKN / 1000

  def _generation_config(
      self, prompt: lf.Message, options: lf.LMSamplingOptions
  ) -> Any:  # generative_models.GenerationConfig
    """Creates generation config from langfun sampling options."""
    assert generative_models is not None
    # Users could use `metadata_json_schema` to pass additional
    # request arguments.
    json_schema = prompt.metadata.get('json_schema')
    response_mime_type = None
    if json_schema is not None:
      if not isinstance(json_schema, dict):
        raise ValueError(
            f'`json_schema` must be a dict, got {json_schema!r}.'
        )
      response_mime_type = 'application/json'
      prompt.metadata.formatted_text = (
          prompt.text
          + '\n\n [RESPONSE FORMAT (not part of prompt)]\n'
          + pg.to_json_str(json_schema, json_indent=2)
      )

    return generative_models.GenerationConfig(
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        max_output_tokens=options.max_tokens,
        stop_sequences=options.stop,
        response_mime_type=response_mime_type,
        response_schema=json_schema,
    )

  def _content_from_message(
      self, prompt: lf.Message
  ) -> list[str | Any]:
    """Gets generation input from langfun message."""
    assert generative_models is not None
    chunks = []

    for lf_chunk in prompt.chunk():
      if isinstance(lf_chunk, str):
        chunks.append(lf_chunk)
      elif isinstance(lf_chunk, lf_modalities.Mime):
        try:
          modalities = lf_chunk.make_compatible(
              self.supported_modalities + ['text/plain']
          )
          if isinstance(modalities, lf_modalities.Mime):
            modalities = [modalities]
          for modality in modalities:
            if modality.is_text:
              chunk = modality.to_text()
            else:
              chunk = generative_models.Part.from_data(
                  modality.to_bytes(), modality.mime_type
              )
            chunks.append(chunk)
        except lf.ModalityError as e:
          raise lf.ModalityError(f'Unsupported modality: {lf_chunk!r}') from e
      else:
        raise lf.ModalityError(f'Unsupported modality: {lf_chunk!r}')
    return chunks

  def _generation_response_to_message(
      self,
      response: Any,  # generative_models.GenerationResponse
  ) -> lf.Message:
    """Parses generative response into message."""
    return lf.AIMessage(response.text)

  def _generation_endpoint_response_to_message(
      self,
      response: Any,  # google.cloud.aiplatform.aiplatform.models.Prediction
  ) -> lf.Message:
    """Parses Endpoint response into message."""
    return lf.AIMessage(response.predictions[0])

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    assert self._api_initialized, 'Vertex AI API is not initialized.'
    # TODO(yifenglu): It seems this exception is due to the instability of the
    # API. We should revisit this later.
    retry_on_errors = [
        (Exception, 'InternalServerError'),
        (Exception, 'ResourceExhausted'),
        (Exception, '_InactiveRpcError'),
        (Exception, 'ValueError'),
    ]

    return self._parallel_execute_with_currency_control(
        self._sample_single,
        prompts,
        retry_on_errors=retry_on_errors,
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    if self.sampling_options.n > 1:
      raise ValueError(
          f'`n` greater than 1 is not supported: {self.sampling_options.n}.'
      )
    api = SUPPORTED_MODELS_AND_SETTINGS[self.model].api
    match api:
      case 'gemini':
        return self._sample_generative_model(prompt)
      case 'palm':
        return self._sample_text_generation_model(prompt)
      case 'endpoint':
        return self._sample_endpoint_model(prompt)
      case _:
        raise ValueError(f'Unsupported API: {api}')

  def _sample_generative_model(self, prompt: lf.Message) -> lf.LMSamplingResult:
    """Samples a generative model."""
    model = _VERTEXAI_MODEL_HUB.get_generative_model(self.model)
    input_content = self._content_from_message(prompt)
    response = model.generate_content(
        input_content,
        generation_config=self._generation_config(
            prompt, self.sampling_options
        ),
    )
    usage_metadata = response.usage_metadata
    usage = lf.LMSamplingUsage(
        prompt_tokens=usage_metadata.prompt_token_count,
        completion_tokens=usage_metadata.candidates_token_count,
        total_tokens=usage_metadata.total_token_count,
        estimated_cost=self.estimate_cost(
            num_input_tokens=usage_metadata.prompt_token_count,
            num_output_tokens=usage_metadata.candidates_token_count,
        ),
    )
    return lf.LMSamplingResult(
        [
            # Scoring is not supported.
            lf.LMSample(
                self._generation_response_to_message(response), score=0.0
            ),
        ],
        usage=usage,
    )

  def _sample_text_generation_model(
      self, prompt: lf.Message
  ) -> lf.LMSamplingResult:
    """Samples a text generation model."""
    model = _VERTEXAI_MODEL_HUB.get_text_generation_model(self.model)
    predict_options = dict(
        temperature=self.sampling_options.temperature,
        top_k=self.sampling_options.top_k,
        top_p=self.sampling_options.top_p,
        max_output_tokens=self.sampling_options.max_tokens,
        stop_sequences=self.sampling_options.stop,
    )
    response = model.predict(prompt.text, **predict_options)
    return lf.LMSamplingResult([
        # Scoring is not supported.
        lf.LMSample(lf.AIMessage(response.text), score=0.0)
    ])

  def _sample_endpoint_model(self, prompt: lf.Message) -> lf.LMSamplingResult:
    """Samples a text generation model."""
    assert aiplatform_models is not None
    model = aiplatform_models.Endpoint(self.endpoint_name)
    # TODO(chengrun): Add support for stop_sequences.
    predict_options = dict(
        temperature=self.sampling_options.temperature
        if self.sampling_options.temperature is not None
        else 1.0,
        top_k=self.sampling_options.top_k
        if self.sampling_options.top_k is not None
        else 32,
        top_p=self.sampling_options.top_p
        if self.sampling_options.top_p is not None
        else 1,
        max_tokens=self.sampling_options.max_tokens
        if self.sampling_options.max_tokens is not None
        else 8192,
    )
    instances = [{'prompt': prompt.text, **predict_options}]
    response = model.predict(instances=instances)

    return lf.LMSamplingResult([
        # Scoring is not supported.
        lf.LMSample(
            self._generation_endpoint_response_to_message(response), score=0.0
        )
    ])


class _ModelHub:
  """Vertex AI model hub."""

  def __init__(self):
    self._generative_model_cache = {}
    self._text_generation_model_cache = {}

  def get_generative_model(
      self, model_id: str
  ) -> Any:  # generative_models.GenerativeModel:
    """Gets a generative model by model id."""
    model = self._generative_model_cache.get(model_id, None)
    if model is None:
      assert generative_models is not None
      model = generative_models.GenerativeModel(model_id)
      self._generative_model_cache[model_id] = model
    return model

  def get_text_generation_model(
      self, model_id: str
  ) -> Any:  # language_models.TextGenerationModel
    """Gets a text generation model by model id."""
    model = self._text_generation_model_cache.get(model_id, None)
    if model is None:
      assert language_models is not None
      model = language_models.TextGenerationModel.from_pretrained(model_id)
      self._text_generation_model_cache[model_id] = model
    return model


_VERTEXAI_MODEL_HUB = _ModelHub()


_IMAGE_TYPES = [
    'image/png',
    'image/jpeg',
    'image/webp',
    'image/heic',
    'image/heif',
]

_AUDIO_TYPES = [
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
    'audio/webm'
]

_VIDEO_TYPES = [
    'video/mov',
    'video/mpeg',
    'video/mpegps',
    'video/mpg',
    'video/mp4',
    'video/webm',
    'video/wmv',
    'video/x-flv',
    'video/3gpp',
]

_DOCUMENT_TYPES = [
    'application/pdf',
    'text/plain',
    'text/csv',
    'text/html',
    'text/xml',
    'text/x-script.python',
    'application/json',
]


class VertexAIGemini1_5(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 model."""

  supported_modalities = (
      _DOCUMENT_TYPES + _IMAGE_TYPES + _AUDIO_TYPES + _VIDEO_TYPES
  )


class VertexAIGeminiPro1_5_Latest(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Pro model."""

  model = 'gemini-1.5-pro-latest'


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


class VertexAIGeminiFlash1_5_Latest(VertexAIGemini1_5):  # pylint: disable=invalid-name
  """Vertex AI Gemini 1.5 Flash model."""

  model = 'gemini-1.5-flash-latest'


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
  """Vertex AI Gemini 1.0 Pro model."""

  model = 'gemini-1.0-pro-vision'
  supported_modalities = _IMAGE_TYPES + _VIDEO_TYPES


class VertexAIPalm2(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI PaLM2 text generation model."""

  model = 'text-bison'


class VertexAIPalm2_32K(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI PaLM2 text generation model (32K context length)."""

  model = 'text-bison-32k'


class VertexAICustom(VertexAI):  # pylint: disable=invalid-name
  """Vertex AI Custom model (Endpoint)."""

  model = 'custom'
