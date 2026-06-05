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
"""Tests for Veo video generation models."""

import unittest
from unittest import mock

import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.llms import veo
from langfun.core.llms import vertexai


class VeoModelInfoTest(unittest.TestCase):
  """Tests for VeoModelInfo."""

  def test_supported_models(self):
    self.assertEqual(len(veo.SUPPORTED_MODELS), 1)
    model = veo.SUPPORTED_MODELS[0]
    self.assertEqual(model.model_id, 'veo-3.1-generate-001')
    self.assertEqual(model.provider, 'VertexAI')
    self.assertEqual(model.rate_limits.max_requests_per_minute, 50)


class VeoTest(unittest.TestCase):
  """Tests for Veo model."""

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_required(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = veo.VertexAIVeo31()._api_initialized

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_default_location(self):
    model = veo.VertexAIVeo31(project='test-project')
    self.assertEqual(model.location, 'us-central1')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_api_endpoint(self):
    model = veo.VertexAIVeo31(project='test-project', location='us-central1')
    model._initialize()
    self.assertEqual(
        model.api_endpoint,
        'https://us-central1-aiplatform.googleapis.com/v1/projects/'
        'test-project/locations/us-central1/publishers/google/'
        'models/veo-3.1-generate-001:predictLongRunning',
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_request_text_to_video(self):
    model = veo.VertexAIVeo31(
        project='test-project',
        duration_seconds=6,
        aspect_ratio='16:9',
        resolution='1080p',
        generate_audio=True,
        storage_uri='gs://bucket/output/',
    )
    request = model.request(
        lf.UserMessage('A cinematic drone shot'),
        lf.LMSamplingOptions(),
    )
    self.assertEqual(
        request,
        {
            'instances': [{'prompt': 'A cinematic drone shot'}],
            'parameters': {
                'durationSeconds': 6,
                'aspectRatio': '16:9',
                'resolution': '1080p',
                'generateAudio': True,
                'personGeneration': 'allow_adult',
                'storageUri': 'gs://bucket/output/',
                'sampleCount': 1,
            },
        },
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  @mock.patch.object(lf_modalities.Image, 'mime_type', 'image/png')
  def test_request_image_to_video_first_frame(self):
    model = veo.VertexAIVeo31(project='test-project')
    image = lf_modalities.Image.from_uri('gs://bucket/input.png')
    msg = lf.UserMessage(
        f'Animate this image <<[[{image.id}]]>>',
        referred_modalities=[image],
    )
    request = model.request(msg, lf.LMSamplingOptions())
    self.assertIn('image', request['instances'][0])
    self.assertEqual(
        request['instances'][0]['image']['gcsUri'],
        'gs://bucket/input.png',
    )
    self.assertNotIn('lastFrame', request['instances'][0])

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  @mock.patch.object(lf_modalities.Image, 'mime_type', 'image/png')
  def test_request_image_to_video_first_and_last_frame(self):
    model = veo.VertexAIVeo31(project='test-project')
    first = lf_modalities.Image.from_uri('gs://bucket/first.png')
    last = lf_modalities.Image.from_uri('gs://bucket/last.png')
    msg = lf.UserMessage(
        f'Generate video between <<[[{first.id}]]>> and <<[[{last.id}]]>>',
        referred_modalities=[first, last],
    )
    request = model.request(msg, lf.LMSamplingOptions())
    self.assertEqual(
        request['instances'][0]['image']['gcsUri'],
        'gs://bucket/first.png',
    )
    self.assertEqual(
        request['instances'][0]['lastFrame']['gcsUri'],
        'gs://bucket/last.png',
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  @mock.patch.object(lf_modalities.Image, 'mime_type', 'image/png')
  def test_request_too_many_images_error(self):
    model = veo.VertexAIVeo31(project='test-project')
    img1 = lf_modalities.Image.from_uri('gs://bucket/1.png')
    img2 = lf_modalities.Image.from_uri('gs://bucket/2.png')
    img3 = lf_modalities.Image.from_uri('gs://bucket/3.png')
    msg = lf.UserMessage(
        f'Too many <<[[{img1.id}]]>> <<[[{img2.id}]]>> <<[[{img3.id}]]>>',
        referred_modalities=[img1, img2, img3],
    )
    with self.assertRaisesRegex(ValueError, 'at most 2 images'):
      model.request(msg, lf.LMSamplingOptions())

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_request_with_negative_prompt_and_seed(self):
    model = veo.VertexAIVeo31(
        project='test-project',
        negative_prompt='blurry, distorted',
        seed=42,
    )
    request = model.request(
        lf.UserMessage('A sunset over mountains'),
        lf.LMSamplingOptions(),
    )
    self.assertEqual(
        request['parameters']['negativePrompt'], 'blurry, distorted'
    )
    self.assertEqual(request['parameters']['seed'], 42)

  def test_result_with_gcs_uri(self):
    model = veo.VertexAIVeo31.__new__(veo.VertexAIVeo31)
    result = model.result({
        'videos': [
            {'gcsUri': 'gs://bucket/output/video1.mp4'},
            {'gcsUri': 'gs://bucket/output/video2.mp4'},
        ]
    })
    self.assertEqual(len(result.samples), 1)
    response = result.samples[0].response
    self.assertEqual(len(response.videos), 2)
    for video in response.videos:
      self.assertIsInstance(video, lf_modalities.Video)

  def test_result_safety_error(self):
    model = veo.VertexAIVeo31.__new__(veo.VertexAIVeo31)
    with self.assertRaises(veo.VeoSafetyError) as ctx:
      model.result({
          'raiMediaFilteredCount': 1,
          'raiMediaFilteredReasons': ['58061214', '90789179'],
      })
    self.assertEqual(ctx.exception.filtered_count, 1)
    self.assertEqual(ctx.exception.reasons, ['58061214', '90789179'])

  def test_result_empty_videos(self):
    model = veo.VertexAIVeo31.__new__(veo.VertexAIVeo31)
    with self.assertRaises(lf.TemporaryLMError):
      model.result({'videos': []})

  def test_model_registration(self):
    model = lf.LanguageModel.get('veo-3.1-generate-001')
    self.assertIsInstance(model, veo.Veo)


class VeoSafetyErrorTest(unittest.TestCase):
  """Tests for VeoSafetyError."""

  def test_error_message(self):
    error = veo.VeoSafetyError(
        filtered_count=2,
        reasons=['58061214', '90789179'],
    )
    self.assertIn('Count: 2', str(error))
    self.assertIn('58061214', str(error))
    self.assertIn('90789179', str(error))


if __name__ == '__main__':
  unittest.main()
