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
"""Tests for langfun.core.templates.Completion."""

import unittest
import langfun.core as lf
from langfun.core.llms import fake as lf_llms
from langfun.core.templates.completion import Completion


class CompletionTest(unittest.TestCase):

  def test_prompt_only(self):
    l = Completion(prompt='Yesterday is 2023-06-04, today is ')
    self.assertIsNone(l.response)
    self.assertEqual(l.render(), 'Yesterday is 2023-06-04, today is')

  def test_prompt_and_response(self):
    l = Completion(
        prompt='Yesterday is 2023-06-04, today is ', lm_response='2023-06-05'
    )
    self.assertEqual(
        l.render(), 'Yesterday is 2023-06-04, today is\n2023-06-05')

  def test_call_and_cache(self):
    l = Completion(
        prompt='Yesterday is 2023-06-04, today is ', cache_response=True)

    with lf.context(
        lm=lf_llms.StaticMapping(
            mapping={'Yesterday is 2023-06-04, today is': '2023-06-05'}
        )
    ):
      l()
      self.assertEqual(l.response, '2023-06-05')


if __name__ == '__main__':
  unittest.main()
