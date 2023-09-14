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
"""Tests for structured mapping example."""

import inspect
import unittest

from langfun.core.structured import mapping
import pyglove as pg


class MappingErrorTest(unittest.TestCase):

  def test_eq(self):
    self.assertEqual(
        mapping.MappingError('Parse failed.'),
        mapping.MappingError('Parse failed.')
    )


class MappingExampleTest(unittest.TestCase):

  def test_basics(self):
    m = mapping.MappingExample('Compute 1 + 1', '2')
    self.assertEqual(m.schema_str(), '')

    m = mapping.MappingExample('Compute 1 + 1', '2', schema=int)
    self.assertEqual(m.schema_str('python'), 'int')
    self.assertEqual(m.schema_str('json'), '{"result": int}')

  def test_str(self):
    self.assertEqual(
        str(mapping.MappingExample(
            'Give the answer.',
            '1 + 1 = 2',
            2,
            int)),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[TEXT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31m{"result": int}\x1b[0m

            \x1b[1m[VALUE]
            \x1b[0m\x1b[34m{"result": 2}\x1b[0m
            """),
    )

  def test_str_no_context(self):
    self.assertEqual(
        str(mapping.MappingExample(None, '1 + 1 = 2', 2, int)),
        inspect.cleandoc("""
            \x1b[1m[TEXT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31m{"result": int}\x1b[0m

            \x1b[1m[VALUE]
            \x1b[0m\x1b[34m{"result": 2}\x1b[0m
            """),
    )

  def test_str_no_text(self):
    self.assertEqual(
        str(mapping.MappingExample(
            'Give the answer.',
            None,
            2,
            int)),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31m{"result": int}\x1b[0m

            \x1b[1m[VALUE]
            \x1b[0m\x1b[34m{"result": 2}\x1b[0m
            """),
    )

  def test_str_no_schema(self):
    self.assertEqual(
        str(mapping.MappingExample(
            'Give the answer.',
            '1 + 1 = 2',
            2,
            None)),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[TEXT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[VALUE]
            \x1b[0m\x1b[34m{"result": 2}\x1b[0m
            """),
    )

  def test_str_no_value(self):
    self.assertEqual(
        str(mapping.MappingExample(
            'Give the answer.',
            '1 + 1 = 2',
            mapping.MappingExample.MISSING_VALUE,
            int)),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[TEXT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31m{"result": int}\x1b[0m
            """),
    )

  def test_serialization(self):
    example = mapping.MappingExample(
        'compute 1 + 1', 'the answer is 2', 2, int
    )
    self.assertTrue(
        pg.eq(pg.from_json_str(example.to_json_str()), example)
    )


if __name__ == '__main__':
  unittest.main()

