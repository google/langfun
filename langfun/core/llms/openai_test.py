# Copyright 2023 The Langfun Authors
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
"""Tests for OpenAI models."""

import unittest
import langfun.core as lf
from langfun.core.llms import openai


class OpenAITest(unittest.TestCase):
  """Tests for OpenAI language model."""

  def test_dir(self):
    self.assertIn('gpt-4-turbo', openai.OpenAI.dir())

  def test_key(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      openai.Gpt4()('hi')

  def test_model_id(self):
    self.assertEqual(
        openai.Gpt35(api_key='test_key').model_id, 'text-davinci-003')

  def test_resource_id(self):
    self.assertEqual(
        openai.Gpt35(api_key='test_key').resource_id,
        'openai://text-davinci-003'
    )

  def test_headers(self):
    self.assertEqual(
        openai.Gpt35(api_key='test_key').headers,
        {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test_key',
        },
    )

  def test_max_concurrency(self):
    self.assertGreater(
        openai.Gpt4o(api_key='test_key').max_concurrency, 0
    )

  def test_request_args(self):
    self.assertEqual(
        openai.Gpt4(api_key='test_key')._request_args(
            lf.LMSamplingOptions(
                temperature=1.0, stop=['\n'], n=1, random_seed=123
            )
        ),
        dict(
            model='gpt-4',
            top_logprobs=None,
            n=1,
            temperature=1.0,
            stop=['\n'],
            seed=123,
        ),
    )
    with self.assertRaisesRegex(RuntimeError, '`logprobs` is not supported.*'):
      openai.GptO1Preview(api_key='test_key')._request_args(
          lf.LMSamplingOptions(
              temperature=1.0, logprobs=True
          )
      )

  def test_estimate_cost(self):
    self.assertEqual(
        openai.Gpt4(api_key='test_key').estimate_cost(
            lf.LMSamplingUsage(
                total_tokens=200,
                prompt_tokens=100,
                completion_tokens=100,
            )
        ),
        0.009
    )

  def test_lm_get(self):
    self.assertIsInstance(lf.LanguageModel.get('gpt-4o'), openai.OpenAI)


if __name__ == '__main__':
  unittest.main()
