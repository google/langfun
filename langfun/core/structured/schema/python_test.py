# Copyright 2025 The Langfun Authors
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
import dataclasses
import inspect
import typing
import unittest

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.structured.schema import base
from langfun.core.structured.schema import python
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  """A travel itinerary for a day."""

  day: pg.typing.Int[1, None]
  type: pg.typing.Enum['daytime', 'nighttime']
  activities: list[Activity]
  hotel: pg.typing.Annotated[
      pg.typing.Str['.*Hotel'] | None,
      'Hotel to stay if applicable.'
  ]


class PlaceOfInterest(pg.Object):
  """The name of a place of interest.

  A place of interest is a place that people ususally visit during their
  travels.
  """

  name: str


Itinerary.__serialization_key__ = 'Itinerary'


class Node(pg.Object):
  children: list['Node']


class SchemaReprTest(unittest.TestCase):

  def test_class_definition(self):
    self.assertEqual(
        python.class_definition(Activity, allowed_dependencies=set()),
        'class Activity:\n  description: str\n',
    )
    self.assertEqual(
        python.class_definition(Itinerary),
        inspect.cleandoc("""
            class Itinerary(Object):
              \"\"\"A travel itinerary for a day.\"\"\"
              day: int(min=1)
              type: Literal['daytime', 'nighttime']
              activities: list[Activity]
              # Hotel to stay if applicable.
              hotel: str(regex='.*Hotel') | None
            """) + '\n',
    )
    self.assertEqual(
        python.class_definition(
            PlaceOfInterest, allowed_dependencies=set()
        ),
        inspect.cleandoc("""
            class PlaceOfInterest:
              \"\"\"The name of a place of interest.

              A place of interest is a place that people ususally visit during their
              travels.
              \"\"\"
              name: str
            """) + '\n',
    )

    class A(pg.Object):
      pass

    self.assertEqual(
        python.class_definition(A, allowed_dependencies=set()),
        'class A:\n  pass\n',
    )
    self.assertEqual(
        python.class_definition(A),
        'class A(Object):\n  pass\n',
    )

    class C(pg.Object):
      x: str
      __kwargs__: typing.Any

    self.assertEqual(
        python.class_definition(C), 'class C(Object):\n  x: str\n'
    )

    class D(pg.Object):
      x: str
      @python.include_method_in_prompt
      def __call__(self, y: int) -> int:
        return len(self.x) + y

    self.assertEqual(
        python.class_definition(D),
        inspect.cleandoc(
            """
            class D(Object):
              x: str

              def __call__(self, y: int) -> int:
                return len(self.x) + y
            """) + '\n'
    )

    class E(pg.Object):
      x: str
      y: typing.Annotated[int, 'y', dict(exclude_from_prompt=True)]

    self.assertEqual(
        python.class_definition(E),
        inspect.cleandoc(
            """
            class E(Object):
              x: str
            """) + '\n'
    )

  def test_select_object_code(self):
    text = (
        '```python\nclass A:\n  x: int\n```\n\n'
        '```python\noutput = A(x=1)\n```\n'
    )
    self.assertEqual(python.select_object_code(text), 'output = A(x=1)\n')

  def test_select_object_code_passes_through_a_single_block(self):
    text = '```python\noutput = A(x=1)\n```\n'
    self.assertIs(python.select_object_code(text), text)

  def test_select_object_code_prefers_the_last_plausible_block(self):
    text = '```python\nimport os\n```\n\n```python\nA(x=1)\n```\n'
    self.assertEqual(python.select_object_code(text), 'A(x=1)\n')

  def test_select_object_code_scores_blocks(self):
    def select(*blocks):
      return python.select_object_code(
          ''.join(f'```python\n{b}\n```\n' for b in blocks)
      )

    # An `output` assignment wins, then a non-definition, then a call.
    self.assertEqual(select('output = A(x=1)', 'A(x=2)'), 'output = A(x=1)\n')
    self.assertEqual(select('y = 1', 'class B:\n  y: int'), 'y = 1\n')
    self.assertEqual(select('A(x=1)', 'y = 2'), 'A(x=1)\n')
    # Ties go to the last block.
    self.assertEqual(select('A(x=1)', 'A(x=2)'), 'A(x=2)\n')

  def test_select_object_code_prefers_the_block_that_builds_the_object(self):
    text = (
        '```python\noutput = A(x=1)\n```\n'
        '```python\noutput = summarize(rows)\n```\n'
    )
    self.assertEqual(
        python.select_object_code(text), 'output = summarize(rows)\n'
    )
    self.assertEqual(
        python.select_object_code(text, class_names=['A']), 'output = A(x=1)\n'
    )

  def test_select_object_code_rewrites_a_class_body_of_assignments(self):
    text = '```python\nclass A:\n  x = 1\n  y: int = 2\n```\n'
    self.assertEqual(python.select_object_code(text), 'A(x=1, y=2)')

  def test_select_object_code_drops_definitions_around_the_object(self):
    text = '```python\nimport os\noutput = A(x=1)\n```\n'
    self.assertEqual(python.select_object_code(text), 'output = A(x=1)')

  def test_select_object_code_leaves_other_code_alone(self):
    for code in (
        'output = A(x=1',  # Not valid Python.
        'import os',  # Nothing but definitions.
        'class A:\n  x: int',  # A schema: no values.
        'class A:\n  x = 1\n  def f(self):\n    pass',
        'class A:\n  x = y = 1',
        'class A:\n  x.y = 1',
    ):
      with self.subTest(code=code):
        text = f'```python\n{code}\n```\n'
        self.assertIs(python.select_object_code(text), text)

  def test_class_definition_with_field_hint(self):
    class F(pg.Object):
      x: int
      y: str

    self.assertEqual(
        python.class_definition(
            F, field_hint=lambda f: 'Hint.' if str(f.key) == 'y' else None
        ),
        'class F(Object):\n  x: int\n  # Hint.\n  y: str\n',
    )

  def test_supports_autofix(self):
    self.assertTrue(base.PromptingProtocol.supports_autofix('python'))

  def test_structure_from_python_returns_code(self):
    self.assertEqual(
        python.structure_from_python('1', returns_code=True), (1, '1')
    )

  def test_repr(self):
    class Foo(pg.Object):
      x: int

    @dataclasses.dataclass(frozen=True)
    class Bar:
      """Class Bar."""
      y: str

    @dataclasses.dataclass(frozen=True)
    class Baz(Bar):  # pylint: disable=unused-variable
      pass

    class A(pg.Object):
      foo: Foo

      @python.include_method_in_prompt
      def foo_value(self) -> int:
        return self.foo.x

      def baz_value(self) -> str:
        return 'baz'

    class B(A):
      bar: Bar
      foo2: Foo

      @python.include_method_in_prompt
      def bar_value(self) -> str:
        return self.bar.y

    schema = base.Schema([B])
    self.assertEqual(
        python.PythonPromptingProtocol().class_definitions(schema),
        inspect.cleandoc('''
            class Foo:
              x: int

            class Bar:
              """Class Bar."""
              y: str

            class Baz(Bar):
              """Baz(y: str)"""
              y: str

            class B:
              foo: Foo
              bar: Bar
              foo2: Foo

              def bar_value(self) -> str:
                return self.bar.y

              def foo_value(self) -> int:
                return self.foo.x
            ''') + '\n',
    )

    self.assertEqual(
        python.PythonPromptingProtocol().result_definition(schema), 'list[B]'
    )

    self.assertEqual(
        base.schema_repr(schema),
        inspect.cleandoc('''
            list[B]

            ```python
            class Foo:
              x: int

            class Bar:
              """Class Bar."""
              y: str

            class Baz(Bar):
              """Baz(y: str)"""
              y: str

            class B:
              foo: Foo
              bar: Bar
              foo2: Foo

              def bar_value(self) -> str:
                return self.bar.y

              def foo_value(self) -> int:
                return self.foo.x
            ```
            '''),
    )
    self.assertEqual(
        python.PythonPromptingProtocol().schema_repr(
            schema,
            include_result_definition=False,
            markdown=False,
        ),
        inspect.cleandoc('''
            class Foo:
              x: int

            class Bar:
              """Class Bar."""
              y: str

            class Baz(Bar):
              """Baz(y: str)"""
              y: str

            class B:
              foo: Foo
              bar: Bar
              foo2: Foo

              def bar_value(self) -> str:
                return self.bar.y

              def foo_value(self) -> int:
                return self.foo.x
            '''),
    )


class ValuePythonReprTest(unittest.TestCase):

  def test_repr(self):
    class Foo(pg.Object):
      x: int

    class A(pg.Object):
      foo: list[Foo]
      y: str | None

    self.assertEqual(
        base.value_repr(1, base.Schema(int)),
        '```python\n1\n```'
    )
    self.assertEqual(
        base.value_repr(lf.Template('hi, {{a}}', a='foo')),
        'hi, foo'
    )
    self.assertEqual(
        base.value_repr(
            A([Foo(1), Foo(2)], 'bar'), base.Schema(A), markdown=False,
        ),
        "A(foo=[Foo(x=1), Foo(x=2)], y='bar')",
    )
    self.assertEqual(
        base.value_repr(
            A([Foo(1), Foo(2)], 'bar'),
            base.Schema(A),
            markdown=True,
            compact=False,
            assign_to_var='output',
        ),
        inspect.cleandoc("""
            ```python
            output = A(
              foo=[
                Foo(
                  x=1
                ),
                Foo(
                  x=2
                )
              ],
              y='bar'
            )
            ```
            """),
    )
    self.assertEqual(
        base.value_repr(A),
        inspect.cleandoc("""
            ```python
            class Foo(Object):
              x: int

            class A(Object):
              foo: list[Foo]
              y: str | None
            ```
            """),
    )
    self.assertEqual(python.source_form(int), 'int')

  def test_parse(self):
    class Foo(pg.Object):
      x: int

    class A(pg.Object):
      foo: list[Foo]
      y: str | None

    self.assertEqual(
        base.parse_value(
            "A(foo=[Foo(x=1), Foo(x=2)], y='bar')", base.Schema(A)
        ),
        A([Foo(1), Foo(2)], y='bar'),
    )

  def test_parse_with_correction(self):
    class Foo(pg.Object):
      x: int

    class A(pg.Object):
      foo: list[Foo]
      y: str | None

    self.assertEqual(
        base.parse_value(
            "A(foo=[Foo(x=1), Foo(x=2)], y='bar'",
            base.Schema(A),
            autofix=1,
            autofix_lm=fake.StaticResponse(
                inspect.cleandoc(
                    """
                    CorrectedCode(
                        corrected_code='A(foo=[Foo(x=1), Foo(x=2)], y=\\\'bar\\\')',
                    )
                    """
                )
            ),
        ),
        A([Foo(1), Foo(2)], y='bar'),
    )

  def test_parse_class_def(self):
    self.assertTrue(
        inspect.isclass(
            base.parse_value(
                """
                class A:
                  x: Dict[str, Any]
                  y: Optional[Sequence[str]]
                  z: Union[int, List[int], Tuple[int]]
                """,
                permission=pg.coding.CodePermission.ALL,
            )
        )
    )


class StructureFromPythonTest(unittest.TestCase):

  def test_parse_class_def(self):

    class B:
      pass

    schema = base.Schema([B])
    v = python.structure_from_python(
        """
        class C(B):
          pass
        """,
        global_vars=dict(B=B),
        permission=pg.coding.CodePermission.ALL,
    )
    self.assertEqual(v.__module__, 'builtins')
    self.assertEqual(schema.class_dependencies(), [B])


if __name__ == '__main__':
  unittest.main()
