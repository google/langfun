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


if __name__ == '__main__':
  unittest.main()
