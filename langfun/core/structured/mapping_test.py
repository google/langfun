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
from typing import Any
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

  def assert_html_content(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_html(self):

    class Answer(pg.Object):
      answer: int

    class Addition(lf.Template):
      """Template Addition.

      {{x}} + {{y}} = ?
      """
      x: Any
      y: Any

    example = mapping.MappingExample(
        input=Addition(x=1, y=2),
        schema=Answer,
        context='compute 1 + 1',
        output=Answer(answer=3),
        metadata={'foo': 'bar'},
    )
    self.assert_html_content(
        example.to_html(
            enable_summary_tooltip=False, include_message_metadata=False
        ),
        """
        <details open class="pyglove mapping-example"><summary><div class="summary_title">MappingExample(...)</div></summary><div class="complex_value mapping-example"><div class="special_value"><details open class="pyglove user-message lf-message"><summary><div class="summary_name">input</div><div class="summary_title">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>rendered</span></div><div class="message-text">1 + 2 = ?</div></div></details></div><div class="special_value"><details open class="pyglove answer"><summary><div class="summary_name">output</div><div class="summary_title">Answer(...)</div></summary><div class="complex_value answer"><table><tr><td><span class="object_key str">answer</span><span class="tooltip key-path">output.answer</span></td><td><div><span class="simple_value int">3</span></div></td></tr></table></div></details></div><div class="special_value"><details open class="pyglove str"><summary><div class="summary_name">context</div><div class="summary_title">&#x27;compute 1 + 1&#x27;</div></summary><span class="simple_value str">&#x27;compute 1 + 1&#x27;</span></details></div><div class="special_value"><details open class="pyglove schema"><summary><div class="summary_name">schema</div><div class="summary_title">Schema(...)</div></summary><div class="lf-schema-definition">Answer

        ```python
        class Answer:
          answer: int
        ```</div></details></div><div class="special_value"><details open class="pyglove dict"><summary><div class="summary_name">metadata</div><div class="summary_title">Dict(...)</div></summary><div class="complex_value dict"><table><tr><td><span class="object_key str">foo</span><span class="tooltip key-path">metadata.foo</span></td><td><div><span class="simple_value str">&#x27;bar&#x27;</span></div></td></tr></table></div></details></div></div></details>
        """
    )

    example = mapping.MappingExample(
        input=Addition(x=1, y=2),
        output=Answer(answer=3),
    )
    self.assert_html_content(
        example.to_html(
            enable_summary_tooltip=False, include_message_metadata=False
        ),
        """
        <details open class="pyglove mapping-example"><summary><div class="summary_title">MappingExample(...)</div></summary><div class="complex_value mapping-example"><div class="special_value"><details open class="pyglove user-message lf-message"><summary><div class="summary_name">input</div><div class="summary_title">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>rendered</span></div><div class="message-text">1 + 2 = ?</div></div></details></div><div class="special_value"><details open class="pyglove answer"><summary><div class="summary_name">output</div><div class="summary_title">Answer(...)</div></summary><div class="complex_value answer"><table><tr><td><span class="object_key str">answer</span><span class="tooltip key-path">output.answer</span></td><td><div><span class="simple_value int">3</span></div></td></tr></table></div></details></div></div></details>
        """
    )


if __name__ == '__main__':
  unittest.main()
