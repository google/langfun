# Copyright 2024 The Langfun Authors
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
"""Gemini models exposed through Google Generative AI APIs."""

import abc
import functools
import os
from typing import Annotated, Any, Literal

import google.generativeai as genai
import langfun.core as lf
from langfun.core import modalities as lf_modalities
import pyglove as pg


@lf.use_init_args(['model'])
class GenAI(lf.LanguageModel):
  """Language models provided by Google GenAI."""

  model: Annotated[
      Literal[
          'gemini-pro',
          'gemini-pro-vision',
          'text-bison-001',
          'chat-bison-001',
          'gemini-1.5-pro-latest',
      ],
      'Model name.',
  ]

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'GOOGLE_API_KEY'. "
          'Get an API key at https://ai.google.dev/tutorials/setup'
      ),
  ] = None

  multimodal: Annotated[bool, 'Whether this model has multimodal support.'] = (
      False
  )

  # Set the default max concurrency to 8 workers.
  max_concurrency = 8

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('_api_initialized', None)

  @functools.cached_property
  def _api_initialized(self):
    api_key = self.api_key or os.environ.get('GOOGLE_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` during `__init__` or set environment '
          'variable `GOOGLE_API_KEY` with your Google Cloud API key. '
          'Check out '
          'https://cloud.google.com/api-keys/docs/create-manage-api-keys '
          'for more details.'
      )
    genai.configure(api_key=api_key)
    return True

  @classmethod
  def dir(cls) -> list[str]:
    """Lists generative models."""
    return [
        m.name.lstrip('models/')
        for m in genai.list_models()
        if (
            'generateContent' in m.supported_generation_methods
            or 'generateText' in m.supported_generation_methods
            or 'generateMessage' in m.supported_generation_methods
        )
    ]

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @property
  def resource_id(self) -> str:
    """Returns a string to identify the resource for rate control."""
    return self.model_id

  def _generation_config(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Creates generation config from langfun sampling options."""
    return genai.GenerationConfig(
        candidate_count=options.n,
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        max_output_tokens=options.max_tokens,
        stop_sequences=options.stop,
    )

  def _content_from_message(
      self, prompt: lf.Message
  ) -> list[str | genai.types.BlobDict]:
    """Gets Evergreen formatted content from langfun message."""
    formatted = lf.UserMessage(prompt.text)
    formatted.source = prompt

    chunks = []
    for lf_chunk in formatted.chunk():
      if isinstance(lf_chunk, str):
        chunk = lf_chunk
      elif self.multimodal and isinstance(lf_chunk, lf_modalities.MimeType):
        chunk = genai.types.BlobDict(
            data=lf_chunk.to_bytes(), mime_type=lf_chunk.mime_type
        )
      else:
        raise ValueError(f'Unsupported modality: {lf_chunk!r}')
      chunks.append(chunk)
    return chunks

  def _response_to_result(
      self, response: genai.types.GenerateContentResponse | pg.Dict
  ) -> lf.LMSamplingResult:
    """Parses generative response into message."""
    samples = []
    for candidate in response.candidates:
      chunks = []
      for part in candidate.content.parts:
        # TODO(daiyip): support multi-modal parts when they are available via
        # Gemini API.
        if hasattr(part, 'text'):
          chunks.append(part.text)
      samples.append(lf.LMSample(lf.AIMessage.from_chunks(chunks), score=0.0))
    return lf.LMSamplingResult(samples)

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    assert self._api_initialized, 'Vertex AI API is not initialized.'
    return self._parallel_execute_with_currency_control(
        self._sample_single,
        prompts,
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    """Samples a single prompt."""
    model = _GOOGLE_GENAI_MODEL_HUB.get(self.model)
    input_content = self._content_from_message(prompt)
    response = model.generate_content(
        input_content,
        generation_config=self._generation_config(self.sampling_options),
    )
    return self._response_to_result(response)


class _LegacyGenerativeModel(pg.Object):
  """Base for legacy GenAI generative model."""

  model: str

  def generate_content(
      self,
      input_content: list[str | genai.types.BlobDict],
      generation_config: genai.GenerationConfig,
  ) -> pg.Dict:
    """Generate content."""
    segments = []
    for s in input_content:
      if not isinstance(s, str):
        raise ValueError(f'Unsupported modality: {s!r}')
      segments.append(s)
    return self.generate(' '.join(segments), generation_config)

  @abc.abstractmethod
  def generate(
      self, prompt: str, generation_config: genai.GenerationConfig) -> pg.Dict:
    """Generate response based on prompt."""


class _LegacyCompletionModel(_LegacyGenerativeModel):
  """Legacy GenAI completion model."""

  def generate(
      self, prompt: str, generation_config: genai.GenerationConfig
  ) -> pg.Dict:
    completion: genai.types.Completion = genai.generate_text(
        model=f'models/{self.model}',
        prompt=prompt,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        candidate_count=generation_config.candidate_count,
        max_output_tokens=generation_config.max_output_tokens,
        stop_sequences=generation_config.stop_sequences,
    )
    return pg.Dict(
        candidates=[
            pg.Dict(content=pg.Dict(parts=[pg.Dict(text=c['output'])]))
            for c in completion.candidates
        ]
    )


class _LegacyChatModel(_LegacyGenerativeModel):
  """Legacy GenAI chat model."""

  def generate(
      self, prompt: str, generation_config: genai.GenerationConfig
  ) -> pg.Dict:
    response: genai.types.ChatResponse = genai.chat(
        model=f'models/{self.model}',
        messages=prompt,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        candidate_count=generation_config.candidate_count,
    )
    return pg.Dict(
        candidates=[
            pg.Dict(content=pg.Dict(parts=[pg.Dict(text=c['content'])]))
            for c in response.candidates
        ]
    )


class _ModelHub:
  """Google Generative AI model hub."""

  def __init__(self):
    self._model_cache = {}

  def get(
      self, model_name: str
  ) -> genai.GenerativeModel | _LegacyGenerativeModel:
    """Gets a generative model by model id."""
    model = self._model_cache.get(model_name, None)
    if model is None:
      model_info = genai.get_model(f'models/{model_name}')
      if 'generateContent' in model_info.supported_generation_methods:
        model = genai.GenerativeModel(model_name)
      elif 'generateText' in model_info.supported_generation_methods:
        model = _LegacyCompletionModel(model_name)
      elif 'generateMessage' in model_info.supported_generation_methods:
        model = _LegacyChatModel(model_name)
      else:
        raise ValueError(f'Unsupported model: {model_name!r}')
      self._model_cache[model_name] = model
    return model


_GOOGLE_GENAI_MODEL_HUB = _ModelHub()


#
# Public Gemini models.
#


class GeminiPro1_5(GenAI):  # pylint: disable=invalid-name
  """Gemini Pro latest model."""

  model = 'gemini-1.5-pro-latest'
  multimodal = True


class GeminiPro(GenAI):
  """Gemini Pro model."""

  model = 'gemini-pro'


class GeminiProVision(GenAI):
  """Gemini Pro vision model."""

  model = 'gemini-pro-vision'
  multimodal = True


class Palm2(GenAI):
  """PaLM2 model."""

  model = 'text-bison-001'


class Palm2_IT(GenAI):  # pylint: disable=invalid-name
  """PaLM2 instruction-tuned model."""

  model = 'chat-bison-001'
