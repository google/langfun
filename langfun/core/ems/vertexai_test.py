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
"""Tests for Vertex AI embedding models."""

import os
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core.ems import vertexai


class VertexAIEmbeddingTest(unittest.TestCase):

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = vertexai.VertexAI(model='test-model')._api_initialized

    with self.assertRaisesRegex(ValueError, 'Please specify `location`'):
      _ = vertexai.VertexAI(
          model='test-model', project='abc', location=None
      )._api_initialized

    self.assertTrue(
        vertexai.VertexAI(
            model='test-model', project='abc', location='us-central1'
        )._api_initialized
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_from_env(self):
    os.environ['VERTEXAI_PROJECT'] = 'env-project'
    model = vertexai.VertexAI(
        model='test-model',
    )
    self.assertTrue(model._api_initialized)
    self.assertIn('us-central1', model.api_endpoint)
    self.assertIn('env-project', model.api_endpoint)
    del os.environ['VERTEXAI_PROJECT']

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_multi_project_support(self):
    model = vertexai.VertexAI(
        model='test-model',
        project=['project-a', 'project-b'],
        location='us-central1',
    )
    self.assertTrue(model._api_initialized)
    endpoint = model.api_endpoint
    self.assertTrue(
        'project-a' in endpoint or 'project-b' in endpoint
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_comma_separated_projects(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='project-a, project-b',
        location='us-central1',
    )
    self.assertTrue(model._api_initialized)
    self.assertEqual(model._projects, ['project-a', 'project-b'])

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_api_endpoint(self):
    model = vertexai.VertexAI(
        model='gemini-embedding-2-preview',
        project='my-project',
        location='us-central1',
    )
    self.assertTrue(model._api_initialized)
    self.assertEqual(
        model.api_endpoint,
        'https://us-central1-aiplatform.googleapis.com/v1/projects/'
        'my-project/locations/us-central1/publishers/google/'
        'models/gemini-embedding-2-preview:embedContent',
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_request_text_only(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
    )
    msg = lf.UserMessage('hello world')
    req = model.request(msg)
    self.assertEqual(req, {'content': {'parts': [{'text': 'hello world'}]}})

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_request_with_task_type(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
        task_type='RETRIEVAL_QUERY',
    )
    req = model.request(lf.UserMessage('hello'))
    self.assertEqual(
        req['taskType'],
        'RETRIEVAL_QUERY',
    )

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_request_with_output_dimensionality(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
        output_dimensionality=256,
    )
    req = model.request(lf.UserMessage('hello'))
    self.assertEqual(req['outputDimensionality'], 256)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_result_parsing(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
    )
    json_response = {'embedding': {'values': [0.1, 0.2]}}
    result = model.result(json_response)
    self.assertEqual(result.embedding, [0.1, 0.2])

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_result_parsing_with_embedding_key(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
    )
    json_response = {'embedding': {'values': [0.4, 0.5]}}
    result = model.result(json_response)
    self.assertEqual(result.embedding, [0.4, 0.5])

  def test_gemini_embedding2_defaults(self):
    model = vertexai.VertexAIGeminiEmbedding2.__schema__.get_field('model')
    self.assertEqual(model.default_value, 'gemini-embedding-2-preview')

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_embedding_options_passthrough(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
        task_type='CLUSTERING',
        output_dimensionality=512,
    )
    self.assertEqual(model.embedding_options.task_type, 'CLUSTERING')
    self.assertEqual(model.embedding_options.output_dimensionality, 512)
    req = model.request(lf.UserMessage('hello'))
    self.assertEqual(req['taskType'], 'CLUSTERING')
    self.assertEqual(req['outputDimensionality'], 512)


class VertexAIPredictAPITest(unittest.TestCase):

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_predict_api_endpoint(self):
    model = vertexai.VertexAIPredictAPI(
        model='text-embedding-005',
        project='my-project',
        location='us-central1',
    )
    self.assertIn(':predict', model.api_endpoint)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_predict_request_text(self):
    model = vertexai.VertexAIPredictAPI(
        model='text-embedding-005',
        project='my-project',
        location='us-central1',
    )
    req = model.request(lf.UserMessage('hello'))
    self.assertEqual(req, {'instances': [{'content': 'hello'}]})

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_predict_request_with_params(self):
    model = vertexai.VertexAIPredictAPI(
        model='text-embedding-005',
        project='my-project',
        location='us-central1',
        task_type='RETRIEVAL_QUERY',
        output_dimensionality=256,
    )
    req = model.request(lf.UserMessage('hello'))
    self.assertEqual(req['parameters']['task_type'], 'RETRIEVAL_QUERY')
    self.assertEqual(req['parameters']['outputDimensionality'], 256)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_predict_rejects_multimodal(self):
    model = vertexai.VertexAIPredictAPI(
        model='text-embedding-005',
        project='my-project',
        location='us-central1',
    )
    msg = lf.UserMessage('hello')
    with mock.patch.object(
        type(msg), 'referred_modalities',
        new_callable=lambda: property(lambda self: {'image': mock.MagicMock()})
    ):
      with self.assertRaisesRegex(ValueError, 'does not support multimodal'):
        model.request(msg)

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_predict_result_parsing(self):
    model = vertexai.VertexAIPredictAPI(
        model='text-embedding-005',
        project='my-project',
        location='us-central1',
    )
    json_response = {
        'predictions': [{
            'embeddings': {
                'values': [0.1, 0.2, 0.3],
                'statistics': {'truncated': False, 'token_count': 3},
            }
        }]
    }
    result = model.result(json_response)
    self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
    self.assertEqual(result.usage.prompt_tokens, 3)

  def test_gemini_embedding1_defaults(self):
    model = vertexai.VertexAIGeminiEmbedding1.__schema__.get_field('model')
    self.assertEqual(model.default_value, 'gemini-embedding-001')

  def test_text_embedding_005_defaults(self):
    model = vertexai.VertexAITextEmbedding005.__schema__.get_field('model')
    self.assertEqual(model.default_value, 'text-embedding-005')

  def test_text_multilingual_embedding_002_defaults(self):
    model = vertexai.VertexAITextMultilingualEmbedding002.__schema__.get_field(
        'model'
    )
    self.assertEqual(model.default_value, 'text-multilingual-embedding-002')


class VertexAIGoogleAuthMissingTest(unittest.TestCase):

  def test_on_bound_raises_when_google_auth_is_none(self):
    with mock.patch.object(vertexai, 'google_auth', None):
      with self.assertRaisesRegex(
          ValueError, 'Please install'
      ):
        vertexai.VertexAI(model='test-model', project='p', location='l')


class VertexAISessionTest(unittest.TestCase):

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_session_returns_authorized_session(self):
    model = vertexai.VertexAI(
        model='test-model',
        project='my-project',
        location='us-central1',
    )
    _ = model._api_initialized
    mock_session = mock.MagicMock()
    with mock.patch.object(
        vertexai.auth_requests,
        'AuthorizedSession',
        return_value=mock_session,
    ):
      s = model.session()
      self.assertEqual(s, mock_session)


class VertexAIDefaultCredentialsTest(unittest.TestCase):

  def test_uses_default_credentials_when_none_provided(self):
    import sys  # pylint: disable=g-import-not-at-top
    mock_creds = mock.MagicMock()
    mock_google_auth = mock.MagicMock()
    mock_google_auth.default.return_value = (mock_creds, 'project')
    mock_g3_auth = mock.MagicMock()
    mock_g3_auth.gcp_corp_access.return_value = mock_creds
    with mock.patch.dict(sys.modules, {
        'langfun.core.google.auth': mock_g3_auth,
    }), mock.patch.object(vertexai, 'google_auth', mock_google_auth):
      model = vertexai.VertexAI(
          model='test-model',
          project='my-project',
          location='us-central1',
          credentials=None,
      )
      _ = model._api_initialized
      self.assertIsNotNone(model._credentials)


class VertexAIMultimodalRequestTest(unittest.TestCase):

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_request_with_image(self):
    model = vertexai.VertexAI(
        model='gemini-embedding-2-preview',
        project='my-project',
        location='us-central1',
    )
    mock_modality = mock.MagicMock()
    mock_modality.to_bytes.return_value = b'fake-image-data'
    msg = lf.UserMessage('describe this image')
    with mock.patch.object(
        type(msg), 'referred_modalities',
        new_callable=lambda: property(
            lambda self: {'image': mock_modality}
        ),
    ):
      req = model.request(msg)
    parts = req['content']['parts']
    self.assertEqual(len(parts), 2)
    self.assertEqual(parts[0], {'text': 'describe this image'})
    self.assertIn('inline_data', parts[1])
    self.assertEqual(parts[1]['inline_data']['mime_type'], 'image/png')


if __name__ == '__main__':
  unittest.main()
