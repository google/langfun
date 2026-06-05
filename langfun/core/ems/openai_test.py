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
"""Tests for OpenAI embedding models."""

import os
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core.ems import openai


class OpenAIEmbeddingTest(unittest.TestCase):

  def test_api_key_required(self):
    model = openai.OpenAI(model='text-embedding-3-small')
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      _ = model._api_initialized

  def test_api_key_from_env(self):
    os.environ['OPENAI_API_KEY'] = 'test-key'
    model = openai.OpenAI(model='text-embedding-3-small')
    self.assertTrue(model._api_initialized)
    self.assertEqual(model._api_key, 'test-key')
    del os.environ['OPENAI_API_KEY']

  def test_api_key_from_init(self):
    model = openai.OpenAI(
        model='text-embedding-3-small', api_key='my-key'
    )
    self.assertTrue(model._api_initialized)
    self.assertEqual(model._api_key, 'my-key')

  def test_api_endpoint(self):
    model = openai.OpenAI(
        model='text-embedding-3-small', api_key='test'
    )
    self.assertEqual(
        model.api_endpoint, 'https://api.openai.com/v1/embeddings'
    )

  def test_headers(self):
    model = openai.OpenAI(
        model='text-embedding-3-small', api_key='test-key'
    )
    headers = model.headers
    self.assertEqual(headers['Authorization'], 'Bearer test-key')
    self.assertNotIn('OpenAI-Organization', headers)

  def test_headers_with_organization(self):
    model = openai.OpenAI(
        model='text-embedding-3-small',
        api_key='test-key',
        organization='my-org',
    )
    headers = model.headers
    self.assertEqual(headers['OpenAI-Organization'], 'my-org')

  def test_request_text(self):
    model = openai.OpenAI(
        model='text-embedding-3-small', api_key='test'
    )
    req = model.request(lf.UserMessage('hello world'))
    self.assertEqual(req, {
        'model': 'text-embedding-3-small',
        'input': 'hello world',
    })

  def test_request_with_dimensions(self):
    model = openai.OpenAI(
        model='text-embedding-3-small',
        api_key='test',
        output_dimensionality=256,
    )
    req = model.request(lf.UserMessage('hello'))
    self.assertEqual(req['dimensions'], 256)

  def test_request_rejects_multimodal(self):
    model = openai.OpenAI(
        model='text-embedding-3-small', api_key='test'
    )
    msg = lf.UserMessage('hello')
    with mock.patch.object(
        type(msg), 'referred_modalities',
        new_callable=lambda: property(lambda self: {'image': mock.MagicMock()})
    ):
      with self.assertRaisesRegex(ValueError, 'does not support multimodal'):
        model.request(msg)

  def test_result_parsing(self):
    model = openai.OpenAI(
        model='text-embedding-3-small', api_key='test'
    )
    json_response = {
        'data': [{'embedding': [0.1, 0.2, 0.3], 'index': 0}],
        'usage': {'prompt_tokens': 3, 'total_tokens': 3},
    }
    result = model.result(json_response)
    self.assertEqual(result.embedding, [0.1, 0.2, 0.3])
    self.assertEqual(result.usage.prompt_tokens, 3)
    self.assertEqual(result.usage.total_tokens, 3)

  def test_text_embedding_3_small_defaults(self):
    model = openai.TextEmbedding3Small.__schema__.get_field('model')
    self.assertEqual(model.default_value, 'text-embedding-3-small')

  def test_text_embedding_3_large_defaults(self):
    model = openai.TextEmbedding3Large.__schema__.get_field('model')
    self.assertEqual(model.default_value, 'text-embedding-3-large')

  def test_text_embedding_ada_002_defaults(self):
    model = openai.TextEmbeddingAda002.__schema__.get_field('model')
    self.assertEqual(model.default_value, 'text-embedding-ada-002')


if __name__ == '__main__':
  unittest.main()
