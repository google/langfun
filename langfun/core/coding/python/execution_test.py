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
"""Tests for Python code execution."""

import inspect
import time
from typing import Any
import unittest
from langfun.core.coding.python import errors
from langfun.core.coding.python import execution
from langfun.core.coding.python import permissions
import pyglove as pg


class RunTest(unittest.TestCase):

  def assert_run(
      self,
      code: str,
      expected_result: dict[str, Any],
      **kwargs):
    self.assertEqual(
        execution.run(
            code, permission=permissions.CodePermission.ALL, **kwargs
        ),
        expected_result,
    )

  def test_with_context(self):
    with execution.context(x=1, y=0):
      with execution.context(x=2, z=2):
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
    ret = execution.run(
        """
        class A(pg.Object):
          x: int
          y: int
          def __call__(self):
            return self.x + self.y
        """,
        permissions.CodePermission.ALL,
        pg=pg,
    )
    self.assertEqual(list(ret.keys()), ['A', '__result__'])
    self.assertTrue(issubclass(ret['A'], pg.Object))
    self.assertIs(ret['__result__'], ret['A'])

  def test_function_def(self):
    ret = execution.run(
        """
        def foo(x, y):
          return x + y
        """,
        permissions.CodePermission.ALL,
    )
    self.assertEqual(list(ret.keys()), ['foo', '__result__'])
    self.assertTrue(inspect.isfunction(ret['foo']))
    self.assertIs(ret['__result__'], ret['foo'])

  def test_complex(self):
    ret = execution.run(
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
        permissions.CodePermission.ALL,
        pg=pg,
    )
    self.assertEqual(list(ret.keys()), ['A', 'foo', 'k', '__result__'])
    self.assertTrue(issubclass(ret['A'], pg.Object))
    self.assertTrue(inspect.isfunction(ret['foo']))
    self.assertIsInstance(ret['k'], pg.Object)
    self.assertEqual(ret['__result__'], 10)

  def test_run_with_error(self):
    with self.assertRaisesRegex(
        errors.CodeError, 'NameError: name .* is not defined'
    ):
      execution.run(
          """
          x = 1
          y = x + z
          """,
          permissions.CodePermission.ALL,
      )
    with self.assertRaisesRegex(errors.CodeError, 'ValueError'):
      execution.run('raise ValueError()', permissions.CodePermission.ALL)


class Foo(pg.Object):
  x: int
  y: int


class SandboxTest(unittest.TestCase):

  def test_basics(self):
    def f(x, y):
      return x + y
    self.assertEqual(execution.sandbox_call(f, 1, y=2), 3)

  def test_complex_type(self):
    def f(x, y):
      return Foo(x, y)

    self.assertEqual(execution.sandbox_call(f, 1, 2), Foo(1, 2))

  def test_timeout(self):
    def f(x):
      time.sleep(x)

    self.assertIsNone(execution.sandbox_call(f, 0, timeout=1))
    with self.assertRaises(TimeoutError):
      execution.sandbox_call(f, 2, timeout=1)

  def test_raise(self):
    def f(x):
      if x == 0:
        raise ValueError()

    self.assertIsNone(execution.sandbox_call(f, 1))
    with self.assertRaises(ValueError):
      execution.sandbox_call(f, 0)

  def test_sandbox_run(self):
    code = inspect.cleandoc("""
        x = Foo(1, 2)
        y = Foo(2, 3)
        """)
    self.assertEqual(
        execution.sandbox_run(code, Foo=Foo),
        {'x': Foo(1, 2), 'y': Foo(2, 3), '__result__': Foo(2, 3)},
    )


if __name__ == '__main__':
  unittest.main()
