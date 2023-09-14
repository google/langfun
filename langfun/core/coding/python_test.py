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
from typing import Any
import unittest
from langfun.core.coding import python
import pyglove as pg


class CodePermissionTest(unittest.TestCase):

  def assert_set(self,
                 permission: python.CodePermission,
                 flag: python.CodePermission):
    self.assertEqual(permission & flag, flag)

  def assert_not_set(self,
                     permission: python.CodePermission,
                     flag: python.CodePermission):
    self.assertFalse(permission & flag)

  def test_all(self):
    self.assert_set(
        python.CodePermission.ALL, python.CodePermission.BASIC)
    self.assert_set(
        python.CodePermission.ALL, python.CodePermission.CONDITION)
    self.assert_set(
        python.CodePermission.ALL, python.CodePermission.LOOP)
    self.assert_set(
        python.CodePermission.ALL, python.CodePermission.EXCEPTION)
    self.assert_set(
        python.CodePermission.ALL, python.CodePermission.CLASS_DEFINITION)
    self.assert_set(
        python.CodePermission.ALL,
        python.CodePermission.FUNCTION_DEFINITION)
    self.assert_set(
        python.CodePermission.ALL, python.CodePermission.IMPORT)

  def test_xor(self):
    self.assert_not_set(
        python.CodePermission.ALL ^ python.CodePermission.BASIC,
        python.CodePermission.BASIC)
    self.assert_set(
        python.CodePermission.ALL ^ python.CodePermission.BASIC,
        python.CodePermission.CONDITION)

  def test_permission_control(self):
    self.assertEqual(python.get_permission(), python.CodePermission.ALL)
    with python.permission(python.CodePermission.BASIC):
      self.assertEqual(python.get_permission(), python.CodePermission.BASIC)
      with python.permission(python.CodePermission.ALL):
        self.assertEqual(python.get_permission(), python.CodePermission.BASIC)


class PythonCodeParserTest(unittest.TestCase):

  def assert_clean(self, code: str, cleaned_code: str):
    self.assertEqual(
        python.PythonCodeParser().clean(code),
        inspect.cleandoc(cleaned_code))

  def test_clean(self):
    self.assert_clean(
        """
        x = y + 1
        if x > 0:
          print(x)
        """,
        """
        x = y + 1
        if x > 0:
          print(x)
        """
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
        Here is the code:

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
        x = y + 1
        z = x * y
        ```
        """,
        """
        x = y + 1
        z = x * y
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

  def assert_allowed(self, code: str, permission: python.CodePermission):
    self.assertIsNotNone(python.PythonCodeParser().parse(code, permission))

  def assert_not_allowed(self, code: str, permission: python.CodePermission):
    with self.assertRaisesRegex(SyntaxError, '.* is not allowed'):
      python.PythonCodeParser().parse(code, permission)

  def test_parse_with_allowed_code(self):
    self.assert_allowed(
        """
        x = y + 1
        z = x + y
        """,
        python.CodePermission.BASIC,
    )
    self.assert_allowed(
        """
        if x > 0:
          print(x)
        """,
        python.CodePermission.CONDITION,
    )
    self.assert_allowed(
        """
        for i in range(5):
          print(i)
        """,
        python.CodePermission.LOOP,
    )
    self.assert_allowed(
        """
        assert x > 1
        """,
        python.CodePermission.EXCEPTION,
    )
    self.assert_allowed(
        """
        class A:
          pass
        """,
        python.CodePermission.CLASS_DEFINITION,
    )
    self.assert_allowed(
        """
        def foo(x, y):
          return x + y
        """,
        python.CodePermission.FUNCTION_DEFINITION,
    )
    self.assert_allowed(
        """
        import re
        """,
        python.CodePermission.IMPORT,
    )

  def test_parse_with_not_allowed_code(self):
    self.assert_not_allowed(
        """
        if x > 0:
          print(x)
        """,
        python.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        for i in range(5):
          print(i)
        """,
        python.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        assert x > 1
        """,
        python.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        class A:
          pass
        """,
        python.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        def foo(x, y):
          return x + y
        """,
        python.CodePermission.BASIC,
    )
    self.assert_not_allowed(
        """
        import re
        """,
        python.CodePermission.BASIC,
    )


class RunTest(unittest.TestCase):

  def assert_run(
      self,
      code: str,
      expected_result: dict[str, Any],
      **kwargs):
    self.assertEqual(
        python.run(code, permission=python.CodePermission.ALL, **kwargs),
        expected_result)

  def test_with_context(self):
    with python.context(x=1, y=0):
      with python.context(x=2, z=2):
        self.assert_run(
            """
            p = x + y + z
            """,
            dict(p=2 + 0 + 3, __result__=2 + 0 + 3),
            z=3  # Override value from the context.
        )

  def test_basics(self):
    self.assert_run(
        """
        x = 1
        y = x + 1
        z = x + y
        """,
        dict(x=1, y=2, z=3, __result__=3)
    )

  def test_class_def(self):
    ret = python.run(
        """
        class A(pg.Object):
          x: int
          y: int
          def __call__(self):
            return self.x + self.y
        """,
        python.CodePermission.ALL, pg=pg
    )
    self.assertEqual(list(ret.keys()), ['A', '__result__'])
    self.assertTrue(issubclass(ret['A'], pg.Object))
    self.assertIs(ret['__result__'], ret['A'])

  def test_function_def(self):
    ret = python.run(
        """
        def foo(x, y):
          return x + y
        """,
        python.CodePermission.ALL,
    )
    self.assertEqual(list(ret.keys()), ['foo', '__result__'])
    self.assertTrue(inspect.isfunction(ret['foo']))
    self.assertIs(ret['__result__'], ret['foo'])

  def test_complex(self):
    ret = python.run(
        """
        class A(pg.Object):
          x: int
          y: int
          def __call__(self, z):
            return self.x + self.y + z

        def foo(x, y):
          return x + y
        k = A(1, 2)
        k(foo(3, 4))
        """,
        python.CodePermission.ALL, pg=pg
    )
    self.assertEqual(list(ret.keys()), ['A', 'foo', 'k', '__result__'])
    self.assertTrue(issubclass(ret['A'], pg.Object))
    self.assertTrue(inspect.isfunction(ret['foo']))
    self.assertIsInstance(ret['k'], pg.Object)
    self.assertEqual(ret['__result__'], 10)


class PythonCodeTest(unittest.TestCase):

  def test_auto_run(self):
    with pg.auto_call_functors():
      self.assertEqual(
          python.PythonCode(
              """
              x = 1
              y = x + 1
              z = x + y
              """,
              ),
          3
      )

  def test_eval(self):
    self.assertEqual(
        python.PythonCode(
            """
            x = 1
            y = x + 1
            z = x + y
            """
            ).eval(),
        dict(x=1, y=2, z=3, __result__=3))

  def test_call(self):
    self.assertEqual(python.PythonCode(
        """
        x = 1
        y = x + 1
        z = x + y
        """
        )(), 3)

  def test_call_class_def(self):
    with python.permission(python.CodePermission.CLASS_DEFINITION):
      v = python.PythonCode(
          """
          class A:
            pass
          """
          )()
      self.assertTrue(inspect.isclass(v))


if __name__ == '__main__':
  unittest.main()
