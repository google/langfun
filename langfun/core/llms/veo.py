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
"""Veo video generation models on Vertex AI."""

import base64
import datetime
import functools
import time
from typing import Annotated, Any, Literal

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import vertexai
import pyglove as pg


class VeoModelInfo(lf.ModelInfo):
  """Veo model info."""

  LINKS = dict(
      models='https://cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos',
      pricing='https://cloud.google.com/vertex-ai/generative-ai/pricing',
  )


SUPPORTED_MODELS = [
    VeoModelInfo(
        model_id='veo-3.1-generate-001',
        in_service=True,
        provider='VertexAI',
        model_type='unknown',
        description='Veo 3.1 stable video generation model.',
        release_date=datetime.datetime(2025, 1, 1),
        context_length=lf.ModelInfo.ContextLength(
            max_input_tokens=1024,
            max_output_tokens=0,
        ),
        pricing=lf.ModelInfo.Pricing(
            cost_per_1m_input_tokens=0.0,
            cost_per_1m_output_tokens=0.0,
        ),
        rate_limits=lf.ModelInfo.RateLimits(
            max_requests_per_minute=50,
            max_tokens_per_minute=None,
        ),
    ),
]

_SUPPORTED_MODELS_BY_ID = {m.model_id: m for m in SUPPORTED_MODELS}


class VeoSafetyError(lf.LMError):
  """Raised when video generation is blocked by safety filters."""

  def __init__(self, filtered_count: int, reasons: list[str]):
    self.filtered_count = filtered_count
    self.reasons = reasons
    super().__init__(
        f'Video blocked by safety filters. Count: {filtered_count}, '
        f'Reason codes: {reasons}'
    )


@pg.use_init_args(['model'])
@pg.members([('api_endpoint', pg.typing.Str().freeze(''))])
class Veo(vertexai.VertexAI):
  """Base class for Veo video generation models on Vertex AI.

  Veo models use Long-Running Operations (LRO) for video generation.
  Unlike synchronous LLM calls, video generation requires polling
  until the operation completes.

  **Text-to-Video:**

  veo = lf.llms.VertexAIVeo31(project='my-project', location='us-central1')
  result = lf.query('A cinematic drone shot over mountains', lm=veo)
  print(result.videos[0])

  **Image-to-Video (First Frame):**

  image = lf_modalities.Image.from_uri('<image_url>')
  result = lf.query(image, lm=veo)
  print(result.videos[0])

  **Image-to-Video (First + Last Frame):**

  first_frame = lf_modalities.Image.from_uri('<image_url>')
  last_frame = lf_modalities.Image.from_uri('<image_url>')
  result = lf.query(
      [first_frame, last_frame],
      lm=veo,
      first_frame=first_frame,
      last_frame=last_frame,
  )
  print(result.videos[0])
  """

  model: pg.typing.Annotated[
      pg.typing.Enum(pg.MISSING_VALUE, [m.model_id for m in SUPPORTED_MODELS]),
      'The name of the model to use.',
  ]

  duration_seconds: Annotated[
      Literal[4, 6, 8],
      'Video duration in seconds. Must be 4, 6, or 8.',
  ] = 8

  aspect_ratio: Annotated[
      Literal['16:9', '9:16'],
      'Video aspect ratio.',
  ] = '16:9'

  resolution: Annotated[
      Literal['720p', '1080p'],
      'Video resolution.',
  ] = '720p'

  generate_audio: Annotated[
      bool,
      'Whether to generate audio from the text prompt.',
  ] = False

  person_generation: Annotated[
      Literal['allow_adult', 'dont_allow', 'allow_all'],
      'Person generation policy.',
  ] = 'allow_adult'

  storage_uri: Annotated[
      str | None,
      (
          'GCS URI directory to save output (e.g., gs://bucket/output/). '
          'If None, video is returned as Base64.'
      ),
  ] = None

  sample_count: Annotated[
      int,
      'Number of video variants to generate (1-4).',
  ] = 1

  negative_prompt: Annotated[
      str | None,
      'Content to exclude from the video (e.g., "blurry, distorted").',
  ] = None

  seed: Annotated[
      int | None,
      'Fixed seed for deterministic generation.',
  ] = None

  poll_interval_seconds: Annotated[
      float,
      'Seconds to wait between polling operation status.',
  ] = 10.0

  location = 'us-central1'

  @functools.cached_property
  def model_info(self) -> VeoModelInfo:
    return _SUPPORTED_MODELS_BY_ID[self.model]

  @property
  def headers(self):
    return {
        'Content-Type': 'application/json; charset=utf-8',
    }

  @property
  def api_endpoint(self) -> str:
    assert self._api_initialized
    return (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:predictLongRunning'
    )

  def request(
      self, prompt: lf.Message, sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    instance: dict[str, Any] = {'prompt': prompt.text}

    # Image-to-Video support:
    # images[0] = first frame (start), images[1] = last frame (end)
    # Veo supports at most 2 images.
    if prompt.images:
      if len(prompt.images) > 2:
        raise ValueError(
            'Veo supports at most 2 images (first frame and last frame), '
            f'got {len(prompt.images)}.'
        )
      first_frame = prompt.images[0]
      instance['image'] = self._encode_image(first_frame)

      if len(prompt.images) > 1:
        last_frame = prompt.images[1]
        instance['lastFrame'] = self._encode_image(last_frame)

    parameters: dict[str, Any] = {
        'durationSeconds': self.duration_seconds,
        'aspectRatio': self.aspect_ratio,
        'resolution': self.resolution,
        'generateAudio': self.generate_audio,
        'personGeneration': self.person_generation,
        'sampleCount': self.sample_count,
    }

    if self.storage_uri:
      parameters['storageUri'] = self.storage_uri

    if self.negative_prompt:
      parameters['negativePrompt'] = self.negative_prompt

    if self.seed is not None:
      parameters['seed'] = self.seed

    return {
        'instances': [instance],
        'parameters': parameters,
    }

  def _encode_image(self, image: lf_modalities.Image) -> dict[str, str]:
    """Encode an image for the Veo API."""
    if image.uri and image.uri.startswith('gs://'):
      return {
          'gcsUri': image.uri,
          'mimeType': image.mime_type,
      }
    return {
        'bytesBase64Encoded': base64.b64encode(image.to_bytes()).decode(),
        'mimeType': image.mime_type,
    }

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    if json.get('raiMediaFilteredCount', 0) > 0:
      raise VeoSafetyError(
          filtered_count=json['raiMediaFilteredCount'],
          reasons=json.get('raiMediaFilteredReasons', []),
      )

    videos = json.get('videos', [])
    if not videos:
      raise lf.TemporaryLMError('No videos returned in response.')

    chunks = []
    for video_data in videos:
      if 'gcsUri' in video_data:
        chunks.append(lf_modalities.Video.from_uri(video_data['gcsUri']))
      elif 'bytesBase64Encoded' in video_data:
        video_bytes = base64.b64decode(video_data['bytesBase64Encoded'])
        chunks.append(lf_modalities.Video.from_bytes(video_bytes))

    message = lf.AIMessage.from_chunks(chunks)
    return lf.LMSamplingResult(
        [lf.LMSample(message)],
        usage=lf.LMSamplingUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    with self.session() as session:
      response = session.post(
          self.api_endpoint,
          json=self.request(prompt, self.sampling_options),
          timeout=self.timeout,
      )

      if response.status_code != 200:
        raise self._error(response.status_code, response.content)

      operation = response.json()
      operation_name = operation.get('name')
      if not operation_name:
        raise lf.LMError('No operation name returned from predictLongRunning.')

      return self._poll_operation(session, operation_name)

  def _poll_operation(
      self, session: Any, operation_name: str
  ) -> lf.LMSamplingResult:
    fetch_endpoint = (
        f'https://{self._location}-aiplatform.googleapis.com/v1/projects/'
        f'{self._project}/locations/{self._location}/publishers/google/'
        f'models/{self.model}:fetchPredictOperation'
    )

    while True:
      response = session.post(
          fetch_endpoint,
          json={'operationName': operation_name},
          timeout=self.timeout,
      )

      if response.status_code != 200:
        raise self._error(response.status_code, response.content)

      status = response.json()

      if status.get('done') or status.get('state') == 'JOB_STATE_SUCCEEDED':
        if 'error' in status:
          error = status['error']
          raise lf.LMError(
              f"Operation failed: {error.get('message', str(error))}"
          )
        return self.result(status.get('response', status))

      if status.get('state') == 'JOB_STATE_FAILED':
        raise lf.LMError(f'Video generation failed: {status}')

      time.sleep(self.poll_interval_seconds)

  def _error(self, status_code: int, content: str) -> lf.LMError:
    if status_code == 429:
      return lf.RateLimitError(f'{status_code}: {content}')
    elif status_code in (500, 502, 503, 529, 499):
      return lf.TemporaryLMError(f'{status_code}: {content}')
    return lf.LMError(f'{status_code}: {content}')


class VertexAIVeo31(Veo):
  """Veo 3.1 stable video generation model on Vertex AI."""

  model = 'veo-3.1-generate-001'


def _register_veo_models():
  """Register Veo models."""
  for m in SUPPORTED_MODELS:
    lf.LanguageModel.register(m.model_id, Veo)


_register_veo_models()
