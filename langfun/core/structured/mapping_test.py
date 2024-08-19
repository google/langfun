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

import langfun.core as lf
from langfun.core.structured import mapping
import pyglove as pg


class MappingErrorTest(unittest.TestCase):

  def test_format(self):
    error = mapping.MappingError(
        lf.AIMessage('hi'), ValueError('Cannot parse message.')
    )
    self.assertEqual(
        lf.text_formatting.decolored(str(error)),
        'ValueError: Cannot parse message.\n\n[LM Response]\nhi',
    )
    self.assertEqual(
        lf.text_formatting.decolored(error.format(include_lm_response=False)),
        'ValueError: Cannot parse message.',
    )


class MappingExampleTest(unittest.TestCase):

  def test_basics(self):
    m = mapping.MappingExample('Compute 1 + 1', '2')
    self.assertEqual(m.schema_repr(), '')

    m = mapping.MappingExample('Compute 1 + 1', '2', schema=int)
    self.assertEqual(m.schema_repr('python'), 'int')
    self.assertEqual(m.schema_repr('json'), '{"result": int}')

  def test_str(self):
    self.assertEqual(
        str(
            mapping.MappingExample(
                input='1 + 1 = 2',
                output=2,
                context='Give the answer.',
                schema=int,
            )
        ),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[INPUT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31mint\x1b[0m

            \x1b[1m[OUTPUT]
            \x1b[0m\x1b[34m```python
            2
            ```\x1b[0m
            """),
    )

  def test_str_no_context(self):
    self.assertEqual(
        str(mapping.MappingExample('1 + 1 = 2', 2, schema=int)),
        inspect.cleandoc("""
            \x1b[1m[INPUT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31mint\x1b[0m

            \x1b[1m[OUTPUT]
            \x1b[0m\x1b[34m```python
            2
            ```\x1b[0m
            """),
    )

  def test_str_no_schema(self):
    self.assertEqual(
        str(mapping.MappingExample('1 + 1 = 2', 2, context='Give the answer.')),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[INPUT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[OUTPUT]
            \x1b[0m\x1b[34m```python
            2
            ```\x1b[0m
            """),
    )

  def test_str_no_output(self):
    self.assertEqual(
        str(
            mapping.MappingExample(
                '1 + 1 = 2',
                schema=int,
                context='Give the answer.',
            )
        ),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[INPUT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31mint\x1b[0m
            """),
    )

  def test_str_with_metadata(self):
    self.assertEqual(
        str(
            mapping.MappingExample(
                '1 + 1 = 2',
                schema=int,
                context='Give the answer.',
                metadata={'foo': 'bar'},
            )
        ),
        inspect.cleandoc("""
            \x1b[1m[CONTEXT]
            \x1b[0m\x1b[35mGive the answer.\x1b[0m

            \x1b[1m[INPUT]
            \x1b[0m\x1b[32m1 + 1 = 2\x1b[0m

            \x1b[1m[SCHEMA]
            \x1b[0m\x1b[31mint\x1b[0m

            \x1b[1m[METADATA]
            \x1b[0m\x1b[36m{
              foo = 'bar'
            }\x1b[0m
            """),
    )

  def test_serialization(self):
    example = mapping.MappingExample(
        'the answer is 2', 2, int, context='compute 1 + 1'
    )
    self.assertTrue(
        pg.eq(pg.from_json_str(example.to_json_str()), example)
    )


if __name__ == '__main__':
  unittest.main()
