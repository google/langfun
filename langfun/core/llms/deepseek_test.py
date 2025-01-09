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
from langfun.core.llms import deepseek


class DeepSeekTest(unittest.TestCase):
  """Tests for DeepSeek language model."""

  def test_dir(self):
    self.assertIn('deepseek-chat', deepseek.DeepSeek.dir())

  def test_key(self):
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      _ = deepseek.DeepSeekChat().headers
    self.assertEqual(
        deepseek.DeepSeekChat(api_key='test_key').headers,
        {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test_key',
        }
    )

  def test_model_id(self):
    self.assertEqual(
        deepseek.DeepSeekChat(api_key='test_key').model_id,
        'DeepSeek(deepseek-chat)',
    )

  def test_resource_id(self):
    self.assertEqual(
        deepseek.DeepSeekChat(api_key='test_key').resource_id,
        'DeepSeek(deepseek-chat)',
    )

  def test_max_concurrency(self):
    self.assertGreater(
        deepseek.DeepSeekChat(api_key='test_key').max_concurrency, 0
    )

  def test_estimate_cost(self):
    self.assertEqual(
        deepseek.DeepSeekChat(api_key='test_key').estimate_cost(
            num_input_tokens=100, num_output_tokens=100
        ),
        4.2e-5
    )

if __name__ == '__main__':
  unittest.main()
