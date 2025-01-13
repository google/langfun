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
"""Tests for Python code handling."""

import inspect
import unittest
from langfun.core.coding.python import parsing


class CleanTest(unittest.TestCase):

  def assert_clean(self, code: str, cleaned_code: str, clean: bool = True):
    if clean:
      cleaned_code = inspect.cleandoc(cleaned_code)
    self.assertEqual(
        parsing.clean(code), cleaned_code
    )

  def test_clean(self):
    self.assert_clean(
        """
        x = y + 1
        if x > 0:
          print(x)
        """,
        'x = y + 1\nif x > 0:\n  print(x)',
        clean=False
    )
    self.assert_clean(
        """
        def foo(x):
          return x + 1
        """,
        'def foo(x):\n  return x + 1',
        clean=False
    )
    self.assert_clean(
        """
        Here is the code:

        ```
        x = 'abc\\\''
        if len(x) > 10:
          print(x)
        ```
        """,
        """
        x = 'abc\\\''
        if len(x) > 10:
          print(x)
        """
    )
    self.assert_clean(
        """
        Here's the code:

        ```
        x = 'abc'  # Comment with '
        if len(x) > 10:
          print(x)
        ```
        """,
        """
        x = 'abc'  # Comment with '
        if len(x) > 10:
          print(x)
        """
    )
    self.assert_clean(
        """
        The code looks as below:

        ```python
        x = y + 1  # ```
        z = x * y  # "
        ```
        """,
        """
        x = y + 1  # ```
        z = x * y  # "
        """
    )
    self.assert_clean(
        """

        ```python
        x = y + 1
        '''
        Example:
          ```
          p = q + 1
          ```
        '''
        z = x * y
        ```
        """,
        """
        x = y + 1
        '''
        Example:
          ```
          p = q + 1
          ```
        '''
        z = x * y
        """
    )
    self.assert_clean(
        """
        ```python
        x = y + 1
        ```
        And another one:
        ```python
        y = z + 1
        ```
        """,
        """
        x = y + 1
        """
    )
    self.assert_clean(
        """
        ```python
        class A:
          x: int
          y: str
        ```
        """,
        """
        class A:
          x: int
          y: str
        """
    )
    self.assert_clean(
        """
        ```tool-code
        class A:
          x: int
          y: str
        ```
        """,
        """
        class A:
          x: int
          y: str
        """
    )
    self.assert_clean(
        """
        ```
        class A:
          '''Class a.

          Examples:
            ```
            A(1, 2)
            ```
          '''
          x: int
          y: str
        ```
        """,
        """
        class A:
          '''Class a.

          Examples:
            ```
            A(1, 2)
            ```
          '''
          x: int
          y: str
        """
    )

  def test_clean_with_auto_escape(self):
    self.assert_clean(
        """
        ```python
        x = 'John's home'
        ```
        """,
        """
        x = 'John\\'s home'
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'Girls' home'
        ```
        """,
        """
        x = 'Girls\\' home'
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'These are the girls'.'
        ```
        """,
        """
        x = 'These are the girls\\'.'
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'girls'.split('')
        ```
        """,
        """
        x = 'girls'.split('')
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'girls' + 'boys'
        ```
        """,
        """
        x = 'girls' + 'boys'
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'girls' in ['girls', 'boys']
        ```
        """,
        """
        x = 'girls' in ['girls', 'boys']
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'girls' not in ['girls', 'boys']
        ```
        """,
        """
        x = 'girls' not in ['girls', 'boys']
        """
    )
    self.assert_clean(
        """
        ```python
        x = 'Hello
        World'
        ```
        """,
        """
        x = 'Hello\\n        World'
        """
    )


if __name__ == '__main__':
  unittest.main()
