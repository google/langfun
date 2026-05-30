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
"""Tests for Google GenAI models."""

import os
import unittest
import langfun.core as lf
from langfun.core.llms import google_genai


class GenAITest(unittest.TestCase):
  """Tests for GenAI model."""

  def test_basics(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      _ = google_genai.Gemini15Pro().api_endpoint

    self.assertIsNotNone(google_genai.Gemini15Pro(api_key='abc').api_endpoint)

    os.environ['GOOGLE_API_KEY'] = 'abc'
    lm = google_genai.Gemini15Pro_001()
    self.assertIsNotNone(lm.api_endpoint)
    self.assertEqual(lm.model_id, 'gemini-1.5-pro-001')
    self.assertEqual(lm.resource_id, 'google_genai://gemini-1.5-pro-001')
    del os.environ['GOOGLE_API_KEY']

  def test_gemini_31_flash_lite_preview(self):
    os.environ['GOOGLE_API_KEY'] = 'abc'
    lm = google_genai.Gemini31FlashLitePreview()
    self.assertEqual(lm.model_id, 'gemini-3.1-flash-lite-preview')
    self.assertEqual(
        lm.resource_id, 'google_genai://gemini-3.1-flash-lite-preview'
    )
    del os.environ['GOOGLE_API_KEY']

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('google_genai://gemini-1.5-pro'),
        google_genai.GenAI,
    )

  def test_gemini_31_pro_preview_tools_enabled_via_extras(self):
    # Subclass smoke test: Gemini31ProPreview must inherit the AUTO-on-tools
    # behavior from Gemini.request().
    os.environ['GOOGLE_API_KEY'] = 'abc'
    try:
      lm = google_genai.Gemini31ProPreview()
      tools = [{'functionDeclarations': [{'name': 'fn', 'description': 'd'}]}]
      request = lm.request(
          lf.UserMessage('hi'),
          lf.LMSamplingOptions(extras={'tools': tools}),
      )
      self.assertEqual(request.get('tools'), tools)
      self.assertEqual(
          request.get('toolConfig'),
          {'functionCallingConfig': {'mode': 'AUTO'}},
      )
      # And verify backward-compat: no tools => NONE.
      request_no_tools = lm.request(
          lf.UserMessage('hi'),
          lf.LMSamplingOptions(),
      )
      self.assertEqual(
          request_no_tools.get('toolConfig'),
          {'functionCallingConfig': {'mode': 'NONE'}},
      )
    finally:
      del os.environ['GOOGLE_API_KEY']

if __name__ == '__main__':
  unittest.main()
