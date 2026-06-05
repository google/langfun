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
import os
import unittest
import langfun.core as lf
from langfun.core.llms import groq


class GroqTest(unittest.TestCase):

  def test_basics(self):
    self.assertEqual(groq.GroqMistral_8x7B().model_id, 'mixtral-8x7b-32768')
    self.assertEqual(
        groq.GroqMistral_8x7B().resource_id, 'groq://mixtral-8x7b-32768'
    )

  def test_request_args(self):
    args = groq.GroqMistral_8x7B()._request_args(
        lf.LMSamplingOptions(
            temperature=1.0, stop=['\n'], n=1, random_seed=123,
            logprobs=True, top_logprobs=True
        )
    )
    self.assertNotIn('logprobs', args)
    self.assertNotIn('top_logprobs', args)

  def test_api_key(self):
    lm = groq.GroqMistral_8x7B()
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      _ = lm.headers

    lm = groq.GroqMistral_8x7B(api_key='fake key')
    self.assertEqual(
        lm.headers,
        {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer fake key',
        }
    )

    os.environ['GROQ_API_KEY'] = 'abc'
    lm = groq.GroqMistral_8x7B()
    self.assertEqual(
        lm.headers,
        {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer abc',
        }
    )
    del os.environ['GROQ_API_KEY']

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('groq://gemma2-9b-it'),
        groq.Groq,
    )

if __name__ == '__main__':
  unittest.main()
