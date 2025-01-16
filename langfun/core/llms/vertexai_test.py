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


class VertexAITest(unittest.TestCase):
  """Tests for Vertex model with REST API."""

  @mock.patch.object(vertexai.VertexAI, 'credentials', new=True)
  def test_project_and_location_check(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      _ = vertexai.VertexAIGeminiPro1()._api_initialized

    with self.assertRaisesRegex(ValueError, 'Please specify `location`'):
      _ = vertexai.VertexAIGeminiPro1(project='abc')._api_initialized

    self.assertTrue(
        vertexai.VertexAIGeminiPro1(
            project='abc', location='us-central1'
        )._api_initialized
    )

    os.environ['VERTEXAI_PROJECT'] = 'abc'
    os.environ['VERTEXAI_LOCATION'] = 'us-central1'
    model = vertexai.VertexAIGeminiPro1()
    self.assertTrue(model.model_id.startswith('VertexAI('))
    self.assertIn('us-central1', model.api_endpoint)
    self.assertTrue(model._api_initialized)
    self.assertIsNotNone(model._session)
    del os.environ['VERTEXAI_PROJECT']
    del os.environ['VERTEXAI_LOCATION']


class VertexAIAnthropicTest(unittest.TestCase):
  """Tests for VertexAI Anthropic models."""

  def test_basics(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `project`'):
      lm = vertexai.VertexAIClaude3_5_Sonnet_20241022()
      lm('hi')

    model = vertexai.VertexAIClaude3_5_Sonnet_20241022(project='langfun')

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


if __name__ == '__main__':
  unittest.main()
