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
import os
import unittest
import langfun.core as lf
from langfun.core.llms import minimax


class MiniMaxTest(unittest.TestCase):
  """Tests for MiniMax language model."""

  def test_dir(self):
    self.assertIn('MiniMax-M2.7', minimax.MiniMax.dir())
    self.assertIn('MiniMax-M2.7-highspeed', minimax.MiniMax.dir())

  def test_key(self):
    old_key = os.environ.pop('MINIMAX_API_KEY', None)
    try:
      with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
        _ = minimax.MiniMaxM27().headers
      self.assertEqual(
          minimax.MiniMaxM27(api_key='test_key').headers,
          {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer test_key',
          }
      )
    finally:
      if old_key is not None:
        os.environ['MINIMAX_API_KEY'] = old_key

  def test_key_from_env(self):
    os.environ['MINIMAX_API_KEY'] = 'env_key'
    try:
      self.assertEqual(
          minimax.MiniMaxM27().headers,
          {
              'Content-Type': 'application/json',
              'Authorization': 'Bearer env_key',
          }
      )
    finally:
      del os.environ['MINIMAX_API_KEY']

  def test_model_id(self):
    self.assertEqual(
        minimax.MiniMaxM27(api_key='test_key').model_id,
        'MiniMax-M2.7',
    )

  def test_resource_id(self):
    self.assertEqual(
        minimax.MiniMaxM27(api_key='test_key').resource_id,
        'minimax://MiniMax-M2.7',
    )

  def test_model_info(self):
    lm = minimax.MiniMaxM27(api_key='test_key')
    self.assertEqual(lm.model_info.provider, 'MiniMax')
    self.assertEqual(lm.model_info.model_id, 'MiniMax-M2.7')
    self.assertTrue(lm.model_info.in_service)

  def test_request(self):
    request = minimax.MiniMaxM27(api_key='test_key').request(
        lf.UserMessage('hi'), lf.LMSamplingOptions(temperature=0.5),
    )
    self.assertEqual(request['model'], 'MiniMax-M2.7')
    self.assertEqual(request['temperature'], 0.5)

  def test_temperature_clamping(self):
    request = minimax.MiniMaxM27(api_key='test_key').request(
        lf.UserMessage('hi'), lf.LMSamplingOptions(temperature=0.0),
    )
    self.assertEqual(request['temperature'], 0.01)

  def test_highspeed_model(self):
    self.assertEqual(
        minimax.MiniMaxM27Highspeed(api_key='test_key').model_id,
        'MiniMax-M2.7-highspeed',
    )

  def test_lm_get(self):
    self.assertIsInstance(
        lf.LanguageModel.get('MiniMax-M2.7'), minimax.MiniMax
    )
    self.assertIsInstance(
        lf.LanguageModel.get('MiniMax-M2.7-highspeed'), minimax.MiniMax
    )


if __name__ == '__main__':
  unittest.main()
