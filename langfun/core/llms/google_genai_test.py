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

import datetime
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

  def test_gemini_37_flash(self):
    os.environ['GOOGLE_API_KEY'] = 'abc'
    lm = google_genai.Gemini37Flash()
    self.assertEqual(lm.model_id, 'gemini-3.7-flash')
    self.assertEqual(lm.resource_id, 'google_genai://gemini-3.7-flash')
    del os.environ['GOOGLE_API_KEY']

  def test_gemini_38_flash(self):
    os.environ['GOOGLE_API_KEY'] = 'abc'
    lm = google_genai.Gemini38Flash()
    self.assertEqual(lm.model_id, 'gemini-3.8-flash')
    self.assertEqual(lm.resource_id, 'google_genai://gemini-3.8-flash')
    del os.environ['GOOGLE_API_KEY']

  def test_gemini_31_flash_lite_preview(self):
    os.environ['GOOGLE_API_KEY'] = 'abc'
    lm = google_genai.Gemini31FlashLitePreview()
    self.assertEqual(lm.model_id, 'gemini-3.1-flash-lite-preview')
    self.assertEqual(
        lm.resource_id, 'google_genai://gemini-3.1-flash-lite-preview'
    )
    del os.environ['GOOGLE_API_KEY']

  def test_gemini_31_flash_lite(self):
    os.environ['GOOGLE_API_KEY'] = 'abc'
    lm = google_genai.Gemini31FlashLite()
    self.assertEqual(lm.model_id, 'gemini-3.1-flash-lite')
    self.assertEqual(lm.resource_id, 'google_genai://gemini-3.1-flash-lite')
    del os.environ['GOOGLE_API_KEY']

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('google_genai://gemini-1.5-pro'),
        google_genai.GenAI,
    )

  def test_gemini_25_pro_billed_as_gemini_38_flash_from_december(self):
    pricing = google_genai.Gemini25Pro(api_key='abc').pricing
    gemini_25_pro = pricing.effective_on(datetime.date(2026, 11, 30))
    self.assertEqual(gemini_25_pro.cost_per_1m_input_tokens, 1.25)
    self.assertEqual(gemini_25_pro.cost_per_1m_cached_input_tokens, 0.125)
    self.assertEqual(gemini_25_pro.cost_per_1m_output_tokens, 10.0)
    self.assertEqual(
        gemini_25_pro.cost_per_1m_input_tokens_with_prompt_longer_than_128k,
        2.5,
    )
    introductory = pricing.effective_on(datetime.date(2026, 12, 1))
    self.assertEqual(introductory.cost_per_1m_input_tokens, 0.75)
    self.assertEqual(introductory.cost_per_1m_cached_input_tokens, 0.075)
    self.assertEqual(introductory.cost_per_1m_output_tokens, 3.75)
    self.assertIsNone(
        introductory.cost_per_1m_input_tokens_with_prompt_longer_than_128k
    )
    standard = pricing.effective_on(datetime.date(2027, 1, 1))
    self.assertEqual(standard.cost_per_1m_input_tokens, 1.5)
    self.assertEqual(standard.cost_per_1m_output_tokens, 7.5)

  def test_rerouted_model_without_pricing_is_unchanged(self):
    model_info = lf.ModelInfo(model_id='gemini-2.5-pro')
    self.assertIs(google_genai._with_reroute_pricing(model_info), model_info)

  def test_other_models_keep_their_rates(self):
    pricing = google_genai.Gemini25Flash(api_key='abc').pricing
    self.assertEqual(
        pricing.effective_on(datetime.date(2027, 1, 1)),
        pricing,
    )
    self.assertEqual(pricing.cost_per_1m_input_tokens, 0.3)


if __name__ == '__main__':
  unittest.main()
