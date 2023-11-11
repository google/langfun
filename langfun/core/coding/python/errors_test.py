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
"""Tests for code errors."""

import unittest

from langfun.core.coding.python import errors
from langfun.core.coding.python import execution


def code_error(code: str) -> errors.CodeError | None:
  try:
    execution.run(code, timeout=2)
    return None
  except errors.CodeError as e:
    return e


class CodeErrorsTest(unittest.TestCase):

  def test_format(self):
    e = code_error(
        """
        x = y + 1
        """
    )
    self.assertIn('[Generated Code]', str(e))
    self.assertNotIn(
        '[Generated Code]', e.format(include_complete_code=False))

  def test_lineno(self):
    self.assertEqual(
        code_error(
            """
            x = y + 1
            """
        ).lineno, 1)
    self.assertEqual(
        code_error(
            """
            x = 1
            for i of x:
              y = i
            """
        ).lineno, 2)
    self.assertEqual(
        code_error(
            """
            x = 1
            y = 2
            raise ValueError
            """
        ).lineno, 3)

  def test_lineno_in_error_message(self):
    def assert_lineno(code):
      e = code_error(code)
      self.assertIn('line', e.format(include_complete_code=False))

    assert_lineno(
        """
        x = y + 1
        """
    )
    assert_lineno(
        """
        x = 1
          y = 2
        """
    )
    assert_lineno(
        """
        raise ValueError()
        """
    )


class SerializationErrorTest(unittest.TestCase):

  def test_str(self):
    e = errors.SerializationError(
        'Output cannot be serialized.', ValueError('abc'))
    self.assertIn('Output cannot be serialized', str(e))
    self.assertIn('ValueError: abc', str(e))


if __name__ == '__main__':
  unittest.main()
