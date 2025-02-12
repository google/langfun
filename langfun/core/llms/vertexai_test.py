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
"""Tests for VertexAI models."""

import os
import unittest
from unittest import mock

from google.auth import exceptions
import langfun.core as lf
from langfun.core.llms import vertexai
import pyglove as pg


class VertexAITest(unittest.TestCase):
  """Tests for Vertex model with REST API."""

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = vertexai.VertexAIGemini15Pro()._api_initialized

    with self.assertRaisesRegex(ValueError, 'Please specify `location`'):
      _ = vertexai.VertexAIGemini15Pro(
          project='abc', location=None)._api_initialized

    self.assertTrue(
        vertexai.VertexAIGemini15Pro(
            project='abc', location='us-central1'
        )._api_initialized
    )

    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGemini15Pro(location=pg.MISSING_VALUE)
    self.assertEqual(model.resource_id, 'vertexai://gemini-1.5-pro-002')
    self.assertIn('us-central1', model.api_endpoint)
    self.assertTrue(model._api_initialized)
    self.assertIsNotNone(model.session())
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']


class VertexAIAnthropicTest(unittest.TestCase):
  """Tests for VertexAI Anthropic models."""

  def test_basics(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      lm = vertexai.VertexAIClaude35Sonnet_20241022()
      lm('hi')

    model = vertexai.VertexAIClaude35Sonnet_20241022(project='langfun')
    self.assertEqual(model.resource_id, 'vertexai://claude-3-5-sonnet-20241022')
    # Map a Anthropic model back to VertexAI model.
    self.assertTrue(
        vertexai.VertexAIAnthropic(
            'claude-3-5-sonnet-20241022', project='langfun'
        ).model,
        'claude-3-5-sonnet-v2@20241022',
    )
    # NOTE(daiyip): For OSS users, default credentials are not available unless
    # users have already set up their GCP project. Therefore we ignore the
    # exception here.
    try:
      model._initialize()
    except exceptions.DefaultCredentialsError:
      pass

    self.assertEqual(
        model.api_endpoint,
        (
            'https://us-east5-aiplatform.googleapis.com/v1/projects/'
            'langfun/locations/us-east5/publishers/anthropic/'
            'models/claude-3-5-sonnet-v2@20241022:streamRawPredict'
        )
    )
    self.assertEqual(
        model.headers,
        {
            'Content-Type': 'application/json; charset=utf-8',
        },
    )
    request = model.request(
        lf.UserMessage('hi'), lf.LMSamplingOptions(temperature=0.0),
    )
    self.assertEqual(
        request,
        {
            'anthropic_version': 'vertex-2023-10-16',
            'max_tokens': 8192,
            'messages': [
                {'content': [{'text': 'hi', 'type': 'text'}], 'role': 'user'}
            ],
            'stream': False,
            'temperature': 0.0,
            'top_k': 40,
        },
    )

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('gemini-2.0-flash'),
        vertexai.VertexAIGemini,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('claude-3-5-sonnet-v2@20241022'),
        vertexai.VertexAIAnthropic,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('llama-3.1-405b-instruct-maas'),
        vertexai.VertexAILlama,
    )
    self.assertIsInstance(
        lf.LanguageModel.get('mistral-large-2411'),
        vertexai.VertexAIMistral,
    )


if __name__ == '__main__':
  unittest.main()
