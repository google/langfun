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
import unittest
import langfun.core as lf
from langfun.core.llms import deepseek


class DeepSeekTest(unittest.TestCase):
  """Tests for DeepSeek language model."""

  def test_dir(self):
    self.assertIn('deepseek-v3', deepseek.DeepSeek.dir())

  def test_key(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      _ = deepseek.DeepSeekV3().headers
    self.assertEqual(
        deepseek.DeepSeekV3(api_key='test_key').headers,
        {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test_key',
        }
    )

  def test_model_id(self):
    self.assertEqual(
        deepseek.DeepSeekV3(api_key='test_key').model_id,
        'deepseek-v3',
    )

  def test_resource_id(self):
    self.assertEqual(
        deepseek.DeepSeekV3(api_key='test_key').resource_id,
        'deepseek://deepseek-v3',
    )

  def test_request(self):
    request = deepseek.DeepSeekV3(api_key='test_key').request(
        lf.UserMessage('hi'), lf.LMSamplingOptions(temperature=0.0),
    )
    self.assertEqual(request['model'], 'deepseek-chat')

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('deepseek-v3'), deepseek.DeepSeek
    )

if __name__ == '__main__':
  unittest.main()
