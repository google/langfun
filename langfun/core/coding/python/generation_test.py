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
"""Tests for structures for Python code generation."""

import inspect
import unittest
from langfun.core.coding.python import generation
from langfun.core.coding.python import permissions


class PythonCodeTest(unittest.TestCase):

  def test_auto_run(self):
    with generation.PythonCode.auto_run():
      self.assertEqual(
          generation.PythonCode(
              """
              x = 1
              y = x + 1
              z = x + y
              """,
          ),
          3,
      )
      with generation.PythonCode.auto_run(False):
        self.assertIsInstance(generation.PythonCode('1'), generation.PythonCode)

  def test_eval(self):
    self.assertEqual(
        generation.PythonCode("""
            x = 1
            y = x + 1
            z = x + y
            """).eval(),
        dict(x=1, y=2, z=3, __result__=3),
    )

  def test_call(self):
    self.assertEqual(
        generation.PythonCode("""
        x = 1
        y = x + 1
        z = x + y
        """)(),
        3,
    )

  def test_call_class_def(self):
    with permissions.permission(permissions.CodePermission.CLASS_DEFINITION):
      v = generation.PythonCode("""
          class A:
            pass
          """)(sandbox=False)
      self.assertTrue(inspect.isclass(v))


class PythonFunctionTest(unittest.TestCase):

  def test_basic(self):
    f = generation.PythonFunction(
        name='sum',
        args=dict(x='int', y='int'),
        returns='int',
        source=("""
            def sum(x: int, y: int):
              return x + y
            """),
    )
    self.assertEqual(f(1, y=2), 3)
    self.assertEqual(f(1, y=2, sandbox=False), 3)

  def test_bad_code(self):
    f = generation.PythonFunction(
        name='sum',
        args=dict(x='int', y='int'),
        returns='int',
        source=("""
            def sum(x: int, y: int):
              s = 0
              for _ in range(x):
                s += 1
              for _ in range(y):
                s += 1
              return s
            """),
    )
    with self.assertRaises(TimeoutError):
      f(100000000, y=10000000, timeout=1)


if __name__ == '__main__':
  unittest.main()
