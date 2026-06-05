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
"""Tests for demonstration."""

import inspect
import unittest
from langfun.core.templates.completion import Completion
from langfun.core.templates.demonstration import Demonstration


class DemonstrationTest(unittest.TestCase):

  def test_bool(self):
    self.assertTrue(Demonstration([
        Completion(prompt='hi', response='hello')
    ]))
    self.assertFalse(Demonstration(examples=[]))

  def test_examples_only(self):
    self.assertEqual(
        Demonstration([
            Completion(prompt='hi', lm_response='hello'),
            Completion(prompt='goodbye', lm_response='bye'),
        ]).render(),
        inspect.cleandoc("""
            Here are the examples for demonstrating this:

              Example 1:
                hi
                hello

              Example 2:
                goodbye
                bye
            """),
    )

  def test_examples_with_description(self):
    self.assertEqual(
        Demonstration(
            [
                Completion(prompt='hi', lm_response='hello'),
                Completion(prompt='goodbye', lm_response='bye'),
            ],
            'Here are a few examples to greet people:',
        ).render(),
        inspect.cleandoc("""
            Here are a few examples to greet people:

              Example 1:
                hi
                hello

              Example 2:
                goodbye
                bye
            """),
    )

  def test_call(self):
    with self.assertRaises(ValueError):
      _ = Demonstration(examples=[])()


if __name__ == '__main__':
  unittest.main()
