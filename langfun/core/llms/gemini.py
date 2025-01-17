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

import base64
from typing import Any

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import rest
import pyglove as pg

# Supported modalities.

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

TEXT_ONLY = []

ALL_MODALITIES = (
    IMAGE_TYPES + AUDIO_TYPES + VIDEO_TYPES + DOCUMENT_TYPES
)

SUPPORTED_MODELS_AND_SETTINGS = {
    # For automatically rate control and cost estimation, we explicitly register
    # supported models here. This may be inconvenient for new models, but it
    # helps us to keep track of the models and their pricing.
    # Models and RPM are from
    # https://ai.google.dev/gemini-api/docs/models/gemini?_gl=1*114hbho*_up*MQ..&gclid=Cj0KCQiAst67BhCEARIsAKKdWOljBY5aQdNQ41zOPkXFCwymUfMNFl_7ukm1veAf75ZTD9qWFrFr11IaApL3EALw_wcB
    # Pricing in US dollars, from https://ai.google.dev/pricing
    # as of 2025-01-03.
    # NOTE: Please update google_genai.py, vertexai.py, __init__.py when
    # adding new models.
    # !!! PLEASE KEEP MODELS SORTED BY RELEASE DATE !!!
    'gemini-2.0-flash-thinking-exp-1219': pg.Dict(
        latest_update='2024-12-19',
        experimental=True,
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=10,
        tpm_free=4_000_000,
        rpm_paid=0,
        tpm_paid=0,
        cost_per_1m_input_tokens_up_to_128k=0,
        cost_per_1m_output_tokens_up_to_128k=0,
        cost_per_1m_cached_tokens_up_to_128k=0,
        cost_per_1m_input_tokens_longer_than_128k=0,
        cost_per_1m_output_tokens_longer_than_128k=0,
        cost_per_1m_cached_tokens_longer_than_128k=0,
    ),
    'gemini-2.0-flash-exp': pg.Dict(
        latest_update='2024-12-11',
        experimental=True,
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=10,
        tpm_free=4_000_000,
        rpm_paid=0,
        tpm_paid=0,
        cost_per_1m_input_tokens_up_to_128k=0,
        cost_per_1m_output_tokens_up_to_128k=0,
        cost_per_1m_cached_tokens_up_to_128k=0,
        cost_per_1m_input_tokens_longer_than_128k=0,
        cost_per_1m_output_tokens_longer_than_128k=0,
        cost_per_1m_cached_tokens_longer_than_128k=0,
    ),
    'gemini-exp-1206': pg.Dict(
        latest_update='2024-12-06',
        experimental=True,
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=10,
        tpm_free=4_000_000,
        rpm_paid=0,
        tpm_paid=0,
        cost_per_1m_input_tokens_up_to_128k=0,
        cost_per_1m_output_tokens_up_to_128k=0,
        cost_per_1m_cached_tokens_up_to_128k=0,
        cost_per_1m_input_tokens_longer_than_128k=0,
        cost_per_1m_output_tokens_longer_than_128k=0,
        cost_per_1m_cached_tokens_longer_than_128k=0,
    ),
    'learnlm-1.5-pro-experimental': pg.Dict(
        latest_update='2024-11-19',
        experimental=True,
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=10,
        tpm_free=4_000_000,
        rpm_paid=0,
        tpm_paid=0,
        cost_per_1m_input_tokens_up_to_128k=0,
        cost_per_1m_output_tokens_up_to_128k=0,
        cost_per_1m_cached_tokens_up_to_128k=0,
        cost_per_1m_input_tokens_longer_than_128k=0,
        cost_per_1m_output_tokens_longer_than_128k=0,
        cost_per_1m_cached_tokens_longer_than_128k=0,
    ),
    'gemini-exp-1114': pg.Dict(
        latest_update='2024-11-14',
        experimental=True,
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=10,
        tpm_free=4_000_000,
        rpm_paid=0,
        tpm_paid=0,
        cost_per_1m_input_tokens_up_to_128k=0,
        cost_per_1m_output_tokens_up_to_128k=0,
        cost_per_1m_cached_tokens_up_to_128k=0,
        cost_per_1m_input_tokens_longer_than_128k=0,
        cost_per_1m_output_tokens_longer_than_128k=0,
        cost_per_1m_cached_tokens_longer_than_128k=0,
    ),
    'gemini-1.5-flash-latest': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=15,
        tpm_free=1_000_000,
        rpm_paid=2000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=0.075,
        cost_per_1m_output_tokens_up_to_128k=0.3,
        cost_per_1m_cached_tokens_up_to_128k=0.01875,
        cost_per_1m_input_tokens_longer_than_128k=0.15,
        cost_per_1m_output_tokens_longer_than_128k=0.6,
        cost_per_1m_cached_tokens_longer_than_128k=0.0375,
    ),
    'gemini-1.5-flash': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=15,
        tpm_free=1_000_000,
        rpm_paid=2000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=0.075,
        cost_per_1m_output_tokens_up_to_128k=0.3,
        cost_per_1m_cached_tokens_up_to_128k=0.01875,
        cost_per_1m_input_tokens_longer_than_128k=0.15,
        cost_per_1m_output_tokens_longer_than_128k=0.6,
        cost_per_1m_cached_tokens_longer_than_128k=0.0375,
    ),
    'gemini-1.5-flash-001': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=15,
        tpm_free=1_000_000,
        rpm_paid=2000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=0.075,
        cost_per_1m_output_tokens_up_to_128k=0.3,
        cost_per_1m_cached_tokens_up_to_128k=0.01875,
        cost_per_1m_input_tokens_longer_than_128k=0.15,
        cost_per_1m_output_tokens_longer_than_128k=0.6,
        cost_per_1m_cached_tokens_longer_than_128k=0.0375,
    ),
    'gemini-1.5-flash-002': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=15,
        tpm_free=1_000_000,
        rpm_paid=2000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=0.075,
        cost_per_1m_output_tokens_up_to_128k=0.3,
        cost_per_1m_cached_tokens_up_to_128k=0.01875,
        cost_per_1m_input_tokens_longer_than_128k=0.15,
        cost_per_1m_output_tokens_longer_than_128k=0.6,
        cost_per_1m_cached_tokens_longer_than_128k=0.0375,
    ),
    'gemini-1.5-flash-8b': pg.Dict(
        latest_update='2024-10-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=15,
        tpm_free=1_000_000,
        rpm_paid=4000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=0.0375,
        cost_per_1m_output_tokens_up_to_128k=0.15,
        cost_per_1m_cached_tokens_up_to_128k=0.01,
        cost_per_1m_input_tokens_longer_than_128k=0.075,
        cost_per_1m_output_tokens_longer_than_128k=0.3,
        cost_per_1m_cached_tokens_longer_than_128k=0.02,
    ),
    'gemini-1.5-flash-8b-001': pg.Dict(
        latest_update='2024-10-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=15,
        tpm_free=1_000_000,
        rpm_paid=4000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=0.0375,
        cost_per_1m_output_tokens_up_to_128k=0.15,
        cost_per_1m_cached_tokens_up_to_128k=0.01,
        cost_per_1m_input_tokens_longer_than_128k=0.075,
        cost_per_1m_output_tokens_longer_than_128k=0.3,
        cost_per_1m_cached_tokens_longer_than_128k=0.02,
    ),
    'gemini-1.5-pro-latest': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=2,
        tpm_free=32_000,
        rpm_paid=1000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=1.25,
        cost_per_1m_output_tokens_up_to_128k=5.00,
        cost_per_1m_cached_tokens_up_to_128k=0.3125,
        cost_per_1m_input_tokens_longer_than_128k=2.5,
        cost_per_1m_output_tokens_longer_than_128k=10.00,
        cost_per_1m_cached_tokens_longer_than_128k=0.625,
    ),
    'gemini-1.5-pro': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=2,
        tpm_free=32_000,
        rpm_paid=1000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=1.25,
        cost_per_1m_output_tokens_up_to_128k=5.00,
        cost_per_1m_cached_tokens_up_to_128k=0.3125,
        cost_per_1m_input_tokens_longer_than_128k=2.5,
        cost_per_1m_output_tokens_longer_than_128k=10.00,
        cost_per_1m_cached_tokens_longer_than_128k=0.625,
    ),
    'gemini-1.5-pro-001': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=2,
        tpm_free=32_000,
        rpm_paid=1000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=1.25,
        cost_per_1m_output_tokens_up_to_128k=5.00,
        cost_per_1m_cached_tokens_up_to_128k=0.3125,
        cost_per_1m_input_tokens_longer_than_128k=2.5,
        cost_per_1m_output_tokens_longer_than_128k=10.00,
        cost_per_1m_cached_tokens_longer_than_128k=0.625,
    ),
    'gemini-1.5-pro-002': pg.Dict(
        latest_update='2024-09-30',
        in_service=True,
        supported_modalities=ALL_MODALITIES,
        rpm_free=2,
        tpm_free=32_000,
        rpm_paid=1000,
        tpm_paid=4_000_000,
        cost_per_1m_input_tokens_up_to_128k=1.25,
        cost_per_1m_output_tokens_up_to_128k=5.00,
        cost_per_1m_cached_tokens_up_to_128k=0.3125,
        cost_per_1m_input_tokens_longer_than_128k=2.5,
        cost_per_1m_output_tokens_longer_than_128k=10.00,
        cost_per_1m_cached_tokens_longer_than_128k=0.625,
    ),
    'gemini-1.0-pro': pg.Dict(
        in_service=False,
        supported_modalities=TEXT_ONLY,
        rpm_free=15,
        tpm_free=32_000,
        rpm_paid=360,
        tpm_paid=120_000,
        cost_per_1m_input_tokens_up_to_128k=0.5,
        cost_per_1m_output_tokens_up_to_128k=1.5,
        cost_per_1m_cached_tokens_up_to_128k=0,
        cost_per_1m_input_tokens_longer_than_128k=0.5,
        cost_per_1m_output_tokens_longer_than_128k=1.5,
        cost_per_1m_cached_tokens_longer_than_128k=0,
    ),
}


@pg.use_init_args(['model'])
class Gemini(rest.REST):
  """Language models provided by Google GenAI."""

  model: pg.typing.Annotated[
      pg.typing.Enum(
          pg.MISSING_VALUE, list(SUPPORTED_MODELS_AND_SETTINGS.keys())
      ),
      'The name of the model to use.',
  ]

  @property
  def supported_modalities(self) -> list[str]:
    """Returns the list of supported modalities."""
    return SUPPORTED_MODELS_AND_SETTINGS[self.model].supported_modalities

  @property
  def max_concurrency(self) -> int:
    """Returns the maximum number of concurrent requests."""
    return self.rate_to_max_concurrency(
        requests_per_min=max(
            SUPPORTED_MODELS_AND_SETTINGS[self.model].rpm_free,
            SUPPORTED_MODELS_AND_SETTINGS[self.model].rpm_paid
        ),
        tokens_per_min=max(
            SUPPORTED_MODELS_AND_SETTINGS[self.model].tpm_free,
            SUPPORTED_MODELS_AND_SETTINGS[self.model].tpm_paid,
        ),
    )

  def estimate_cost(
      self,
      num_input_tokens: int,
      num_output_tokens: int
  ) -> float | None:
    """Estimate the cost based on usage."""
    entry = SUPPORTED_MODELS_AND_SETTINGS[self.model]
    if num_input_tokens < 128_000:
      cost_per_1m_input_tokens = entry.cost_per_1m_input_tokens_up_to_128k
      cost_per_1m_output_tokens = entry.cost_per_1m_output_tokens_up_to_128k
    else:
      cost_per_1m_input_tokens = entry.cost_per_1m_input_tokens_longer_than_128k
      cost_per_1m_output_tokens = (
          entry.cost_per_1m_output_tokens_longer_than_128k
      )
    return (
        cost_per_1m_input_tokens * num_input_tokens
        + cost_per_1m_output_tokens * num_output_tokens
    ) / 1000_000

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return self.model

  @classmethod
  def dir(cls):
    return [k for k, v in SUPPORTED_MODELS_AND_SETTINGS.items() if v.in_service]

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
    thought_chunks = []
    for part in parts:
      if text_part := part.get('text'):
        if part.get('thought'):
          thought_chunks.append(text_part)
        else:
          chunks.append(text_part)
      else:
        raise ValueError(f'Unsupported part: {part}')
    message = lf.AIMessage.from_chunks(chunks)
    if thought_chunks:
      message.set('thought', lf.AIMessage.from_chunks(thought_chunks))
    return message
