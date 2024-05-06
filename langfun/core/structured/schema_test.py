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
"""Tests for structured parsing."""

import dataclasses
import inspect
import typing
import unittest

from langfun.core.llms import fake
from langfun.core.structured import schema as schema_lib
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


class SchemaTest(unittest.TestCase):

  def assert_schema(self, annotation, spec):
    self.assertEqual(schema_lib.Schema(annotation).spec, spec)

  def assert_unsupported_annotation(self, annotation):
    with self.assertRaises(ValueError):
      schema_lib.Schema(annotation)

  def test_init(self):
    self.assert_schema(int, pg.typing.Int())
    self.assert_schema(float, pg.typing.Float())
    self.assert_schema(str, pg.typing.Str())
    self.assert_schema(bool, pg.typing.Bool())
    self.assert_schema(bool | None, pg.typing.Bool().noneable())

    # Top-level dictionary with 'result' as the only key is flattened.
    self.assert_schema(dict(result=int), pg.typing.Int())

    self.assert_schema(list[str], pg.typing.List(pg.typing.Str()))
    self.assert_schema([str], pg.typing.List(pg.typing.Str()))

    with self.assertRaisesRegex(
        ValueError, 'Annotation with list must be a list of a single element.'
    ):
      schema_lib.Schema([str, int])

    self.assert_schema(
        dict[str, int], pg.typing.Dict([(pg.typing.StrKey(), pg.typing.Int())])
    )

    self.assert_schema(
        {
            'x': int,
            'y': [str],
        },
        pg.typing.Dict([
            ('x', int),
            ('y', pg.typing.List(pg.typing.Str())),
        ]),
    )

    self.assert_schema(Itinerary, pg.typing.Object(Itinerary))

    self.assert_unsupported_annotation(typing.Type[int])
    self.assert_unsupported_annotation(typing.Union[int, str, bool])
    self.assert_unsupported_annotation(typing.Any)

  def test_schema_dict(self):
    schema = schema_lib.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema.schema_dict(),
        {
            'result': [
                {
                    'x': {
                        '_type': 'Itinerary',
                        'day': pg.typing.Int(min_value=1),
                        'activities': [{
                            '_type': Activity.__type_name__,
                            'description': pg.typing.Str(),
                        }],
                        'hotel': pg.typing.Str(regex='.*Hotel').noneable(),
                        'type': pg.typing.Enum['daytime', 'nighttime'],
                    }
                }
            ]
        },
    )

  def test_class_dependencies(self):
    class Foo(pg.Object):
      x: int

    class Bar(pg.Object):
      y: str

    class A(pg.Object):
      foo: tuple[Foo, int]

    class X(pg.Object):
      k: int

    class B(A):
      bar: Bar
      foo2: Foo | X

    schema = schema_lib.Schema([B])
    self.assertEqual(schema.class_dependencies(), [Foo, A, Bar, X, B])

  def test_class_dependencies_non_pyglove(self):
    class Baz:
      def __init__(self, x: int):
        pass

    @dataclasses.dataclass(frozen=True)
    class AA:
      foo: tuple[Baz, int]

    class XX(pg.Object):
      pass

    @dataclasses.dataclass(frozen=True)
    class BB(AA):
      foo2: Baz | XX

    schema = schema_lib.Schema([AA])
    self.assertEqual(schema.class_dependencies(), [Baz, AA, XX, BB])

  def test_schema_repr(self):
    schema = schema_lib.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema.schema_str(protocol='json'),
        (
            '{"result": [{"x": {"_type": "Itinerary", "day":'
            ' int(min=1), "type": "daytime" | "nighttime", "activities":'
            ' [{"_type": "%s", "description": str}], "hotel":'
            ' str(regex=.*Hotel) | None}}]}' % (
                Activity.__type_name__,
            )
        ),
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema.schema_str(protocol='text')

  def test_value_repr(self):
    schema = schema_lib.Schema(int)
    self.assertEqual(schema.value_str(1, protocol='json'), '{"result": 1}')
    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema.value_str(1, protocol='text')

  def test_parse(self):
    schema = schema_lib.Schema(int)
    self.assertEqual(schema.parse('{"result": 1}'), 1)
    schema = schema_lib.Schema(dict[str, int])
    self.assertEqual(
        schema.parse('{"result": {"x": 1}}}'),
        dict(x=1)
    )
    with self.assertRaisesRegex(
        schema_lib.SchemaError, 'Expect .* but encountered .*'):
      schema.parse('{"result": "def"}')

    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema.parse('1', protocol='text')


class ClassDependenciesTest(unittest.TestCase):

  def test_class_dependencies_from_specs(self):
    class Foo(pg.Object):
      x: int

    class Bar(pg.Object):
      y: str

    class A(pg.Object):
      foo: tuple[Foo, int]

    class X(pg.Object):
      k: int

    class B(A):
      bar: Bar
      foo2: Foo | X

    self.assertEqual(schema_lib.class_dependencies(Foo), [Foo])

    self.assertEqual(
        schema_lib.class_dependencies((A,), include_subclasses=False), [Foo, A]
    )

    self.assertEqual(
        schema_lib.class_dependencies(A, include_subclasses=True),
        [Foo, A, Bar, X, B],
    )

    self.assertEqual(
        schema_lib.class_dependencies(schema_lib.Schema(A)), [Foo, A, Bar, X, B]
    )

    self.assertEqual(
        schema_lib.class_dependencies(pg.typing.Object(A)), [Foo, A, Bar, X, B]
    )

    with self.assertRaisesRegex(TypeError, 'Unsupported spec type'):
      schema_lib.class_dependencies((Foo, 1))

  def test_class_dependencies_recursive(self):
    self.assertEqual(
        schema_lib.class_dependencies(Node),
        [Node]
    )

  def test_class_dependencies_from_value(self):
    class Foo(pg.Object):
      x: int

    class Bar(pg.Object):
      y: str

    class A(pg.Object):
      foo: tuple[Foo, int]

    class X(pg.Object):
      k: int

    class B(A):
      bar: Bar
      foo2: Foo | X

    a = A(foo=(Foo(1), 0))
    self.assertEqual(schema_lib.class_dependencies(a), [Foo, A, Bar, X, B])

    self.assertEqual(schema_lib.class_dependencies(1), [])


class SchemaPythonReprTest(unittest.TestCase):

  def assert_annotation(
      self,
      value_spec: pg.typing.ValueSpec,
      expected_annotation: str,
      strict: bool = False,
  ) -> None:
    self.assertEqual(
        schema_lib.annotation(value_spec, strict=strict),
        expected_annotation,
    )

  def test_annotation(self):
    # Bool.
    self.assert_annotation(pg.typing.Bool(), 'bool')
    self.assert_annotation(pg.typing.Bool().noneable(), 'bool | None')

    # Str.
    self.assert_annotation(pg.typing.Str(), 'str')
    self.assert_annotation(pg.typing.Str().noneable(), 'str | None')
    self.assert_annotation(pg.typing.Str(regex='a.*'), "str(regex='a.*')")
    self.assert_annotation(pg.typing.Str(regex='a.*'), "str(regex='a.*')")
    self.assert_annotation(
        pg.typing.Str(regex='a.*'), "pg.typing.Str(regex='a.*')", strict=True
    )

    # Int.
    self.assert_annotation(pg.typing.Int(), 'int')
    self.assert_annotation(pg.typing.Int().noneable(), 'int | None')
    self.assert_annotation(pg.typing.Int(min_value=0), 'int(min=0)')
    self.assert_annotation(pg.typing.Int(max_value=1), 'int(max=1)')
    self.assert_annotation(
        pg.typing.Int(min_value=0, max_value=1), 'int(min=0, max=1)'
    )

    self.assert_annotation(pg.typing.Int(), 'int', strict=True)
    self.assert_annotation(
        pg.typing.Int(min_value=0), 'pg.typing.Int(min_value=0)', strict=True
    )
    self.assert_annotation(
        pg.typing.Int(max_value=1), 'pg.typing.Int(max_value=1)', strict=True
    )
    self.assert_annotation(
        pg.typing.Int(min_value=0, max_value=1),
        'pg.typing.Int(min_value=0, max_value=1)',
        strict=True,
    )

    # Float.
    self.assert_annotation(pg.typing.Float(), 'float')
    self.assert_annotation(pg.typing.Float().noneable(), 'float | None')
    self.assert_annotation(pg.typing.Float(min_value=0), 'float(min=0)')
    self.assert_annotation(pg.typing.Float(max_value=1), 'float(max=1)')
    self.assert_annotation(
        pg.typing.Float(min_value=0, max_value=1), 'float(min=0, max=1)'
    )

    self.assert_annotation(pg.typing.Float(), 'float', strict=True)
    self.assert_annotation(
        pg.typing.Float(min_value=0),
        'pg.typing.Float(min_value=0)',
        strict=True,
    )
    self.assert_annotation(
        pg.typing.Float(max_value=1),
        'pg.typing.Float(max_value=1)',
        strict=True,
    )
    self.assert_annotation(
        pg.typing.Float(min_value=0, max_value=1),
        'pg.typing.Float(min_value=0, max_value=1)',
        strict=True,
    )

    # Enum
    self.assert_annotation(
        pg.typing.Enum[1, 'foo'].noneable(), "Literal[1, 'foo', None]"
    )

    # Object.
    self.assert_annotation(pg.typing.Object(Activity), 'Activity')
    self.assert_annotation(
        pg.typing.Object(Activity).noneable(), 'Activity | None'
    )

    # List.
    self.assert_annotation(
        pg.typing.List(pg.typing.Object(Activity)), 'list[Activity]'
    )
    self.assert_annotation(
        pg.typing.List(pg.typing.Object(Activity)).noneable(),
        'list[Activity] | None',
    )
    self.assert_annotation(
        pg.typing.List(pg.typing.Object(Activity).noneable()),
        'list[Activity | None]',
    )

    # Tuple.
    self.assert_annotation(
        pg.typing.Tuple([pg.typing.Int(), pg.typing.Str()]), 'tuple[int, str]'
    )
    self.assert_annotation(
        pg.typing.Tuple([pg.typing.Int(), pg.typing.Str()]).noneable(),
        'tuple[int, str] | None',
    )

    # Dict.
    self.assert_annotation(
        pg.typing.Dict({'x': int, 'y': str}), '{\'x\': int, \'y\': str}'
    )
    self.assert_annotation(
        pg.typing.Dict({'x': int, 'y': str}),
        'pg.typing.Dict({\'x\': int, \'y\': str})',
        strict=True,
    )
    self.assert_annotation(
        pg.typing.Dict(),
        'dict[str, Any]',
        strict=False,
    )
    self.assert_annotation(
        pg.typing.Dict(),
        'dict[str, Any]',
        strict=True,
    )

    # Union.
    self.assert_annotation(
        pg.typing.Union(
            [pg.typing.Object(Activity), pg.typing.Object(Itinerary)]
        ).noneable(),
        'Union[Activity, Itinerary, None]',
    )

    # Any.
    self.assert_annotation(pg.typing.Any(), 'Any')
    self.assert_annotation(pg.typing.Any().noneable(), 'Any')

  def test_class_definition(self):
    self.assertEqual(
        schema_lib.class_definition(Activity),
        'class Activity:\n  description: str\n',
    )
    self.assertEqual(
        schema_lib.class_definition(Itinerary),
        inspect.cleandoc("""
            class Itinerary:
              \"\"\"A travel itinerary for a day.\"\"\"
              day: int(min=1)
              type: Literal['daytime', 'nighttime']
              activities: list[Activity]
              # Hotel to stay if applicable.
              hotel: str(regex='.*Hotel') | None
            """) + '\n',
    )
    self.assertEqual(
        schema_lib.class_definition(PlaceOfInterest),
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
        schema_lib.class_definition(A),
        'class A:\n  pass\n',
    )
    self.assertEqual(
        schema_lib.class_definition(A, include_pg_object_as_base=True),
        'class A(Object):\n  pass\n',
    )

    class C(pg.Object):
      x: str
      __kwargs__: typing.Any

    self.assertEqual(schema_lib.class_definition(C), 'class C:\n  x: str\n')

    class D(pg.Object):
      x: str
      def __call__(self, y: int) -> int:
        return len(self.x) + y

    self.assertEqual(
        schema_lib.class_definition(D, include_methods=True),
        inspect.cleandoc(
            """
            class D:
              x: str

              def __call__(self, y: int) -> int:
                return len(self.x) + y
            """) + '\n'
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

      def foo_value(self) -> int:
        return self.foo.x

    class B(A):
      bar: Bar
      foo2: Foo

      def bar_value(self) -> str:
        return self.bar.y

    schema = schema_lib.Schema([B])
    self.assertEqual(
        schema_lib.SchemaPythonRepr().class_definitions(
            schema, include_methods=True
        ),
        inspect.cleandoc('''
            class Foo:
              x: int

            class A:
              foo: Foo

              def foo_value(self) -> int:
                return self.foo.x

            class Bar:
              """Class Bar."""
              y: str

            class Baz(Bar):
              """Baz(y: str)"""
              y: str

            class B(A):
              foo: Foo
              bar: Bar
              foo2: Foo

              def bar_value(self) -> str:
                return self.bar.y
            ''') + '\n',
    )

    self.assertEqual(
        schema_lib.SchemaPythonRepr().result_definition(schema), 'list[B]'
    )

    self.assertEqual(
        schema_lib.SchemaPythonRepr().repr(schema),
        inspect.cleandoc('''
            list[B]

            ```python
            class Foo:
              x: int

            class A:
              foo: Foo

            class Bar:
              """Class Bar."""
              y: str

            class Baz(Bar):
              """Baz(y: str)"""
              y: str

            class B(A):
              foo: Foo
              bar: Bar
              foo2: Foo
            ```
            '''),
    )
    self.assertEqual(
        schema_lib.SchemaPythonRepr().repr(
            schema,
            include_result_definition=False,
            include_pg_object_as_base=True,
            markdown=False,
        ),
        inspect.cleandoc('''
            class Foo(Object):
              x: int

            class A(Object):
              foo: Foo

            class Bar:
              """Class Bar."""
              y: str

            class Baz(Bar):
              """Baz(y: str)"""
              y: str

            class B(A):
              foo: Foo
              bar: Bar
              foo2: Foo
            '''),
    )


class SchemaJsonReprTest(unittest.TestCase):

  def test_repr(self):
    schema = schema_lib.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema_lib.SchemaJsonRepr().repr(schema),
        (
            '{"result": [{"x": {"_type": "Itinerary", "day":'
            ' int(min=1), "type": "daytime" | "nighttime", "activities":'
            ' [{"_type": "%s", "description": str}], "hotel":'
            ' str(regex=.*Hotel) | None}}]}' % (
                Activity.__type_name__,
            )
        ),
    )


class ValuePythonReprTest(unittest.TestCase):

  def test_repr(self):
    class Foo(pg.Object):
      x: int

    class A(pg.Object):
      foo: list[Foo]
      y: str | None

    self.assertEqual(
        schema_lib.ValuePythonRepr().repr(1, schema_lib.Schema(int)),
        '```python\n1\n```'
    )
    self.assertEqual(
        schema_lib.ValuePythonRepr().repr(
            A([Foo(1), Foo(2)], 'bar'), schema_lib.Schema(A), markdown=False,
        ),
        "A(foo=[Foo(x=1), Foo(x=2)], y='bar')",
    )
    self.assertEqual(
        schema_lib.ValuePythonRepr().repr(A),
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
    self.assertEqual(schema_lib.source_form(int), 'int')

  def test_parse(self):
    class Foo(pg.Object):
      x: int

    class A(pg.Object):
      foo: list[Foo]
      y: str | None

    self.assertEqual(
        schema_lib.ValuePythonRepr().parse(
            "A(foo=[Foo(x=1), Foo(x=2)], y='bar')", schema_lib.Schema(A)
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
        schema_lib.ValuePythonRepr().parse(
            "A(foo=[Foo(x=1), Foo(x=2)], y='bar'",
            schema_lib.Schema(A),
            autofix=1,
            autofix_lm=fake.StaticResponse(inspect.cleandoc("""
                    CorrectedCode(
                        corrected_code='A(foo=[Foo(x=1), Foo(x=2)], y=\\\'bar\\\')',
                    )
                    """)),
        ),
        A([Foo(1), Foo(2)], y='bar'),
    )

  def test_parse_class_def(self):
    self.assertTrue(
        inspect.isclass(
            schema_lib.ValuePythonRepr().parse(
                """
                class A:
                  x: Dict[str, Any]
                  y: Optional[Sequence[str]]
                  z: Union[int, List[int], Tuple[int]]
                """
            )
        )
    )


class ValueJsonReprTest(unittest.TestCase):

  def test_repr(self):
    self.assertEqual(schema_lib.ValueJsonRepr().repr(1), '{"result": 1}')

  def assert_parse(self, inputs, output) -> None:
    self.assertEqual(schema_lib.ValueJsonRepr().parse(inputs), output)

  def test_parse_basics(self):
    self.assert_parse('{"result": 1}', 1)
    self.assert_parse('{"result": "\\"}ab{"}', '"}ab{')
    self.assert_parse(
        '{"result": {"x": true, "y": null}}',
        {'x': True, 'y': None},
    )
    self.assert_parse(
        (
            '{"result": {"_type": "%s", "description": "play"}}'
            % Activity.__type_name__
        ),
        Activity('play'),
    )
    with self.assertRaisesRegex(
        schema_lib.JsonError, 'JSONDecodeError'
    ):
      schema_lib.ValueJsonRepr().parse('{"abc", 1}')

    with self.assertRaisesRegex(
        schema_lib.JsonError,
        'The root node of the JSON must be a dict with key `result`'
    ):
      schema_lib.ValueJsonRepr().parse('{"abc": 1}')

  def test_parse_with_surrounding_texts(self):
    self.assert_parse('The answer is {"result": 1}.', 1)

  def test_parse_with_new_lines(self):
    self.assert_parse(
        """
        {
            "result": [
"foo
bar"]
        }
        """,
        ['foo\nbar'])

  def test_parse_with_malformated_json(self):
    with self.assertRaisesRegex(
        schema_lib.JsonError, 'No JSON dict in the output'
    ):
      schema_lib.ValueJsonRepr().parse('The answer is 1.')

    with self.assertRaisesRegex(
        schema_lib.JsonError,
        'Malformated JSON: missing .* closing curly braces'
    ):
      schema_lib.ValueJsonRepr().parse('{"result": 1')


class ProtocolTest(unittest.TestCase):

  def test_schema_repr(self):
    self.assertIsInstance(
        schema_lib.schema_repr('json'), schema_lib.SchemaJsonRepr)
    self.assertIsInstance(
        schema_lib.schema_repr('python'), schema_lib.SchemaPythonRepr)
    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema_lib.schema_repr('text')

  def test_value_repr(self):
    self.assertIsInstance(
        schema_lib.value_repr('json'), schema_lib.ValueJsonRepr)
    self.assertIsInstance(
        schema_lib.value_repr('python'), schema_lib.ValuePythonRepr)
    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema_lib.value_repr('text')


class MissingTest(unittest.TestCase):

  def test_basics(self):
    a = Itinerary(
        day=1,
        type=schema_lib.Missing(),
        activities=schema_lib.Missing(),
        hotel=schema_lib.Missing(),
    )
    self.assertFalse(a.is_partial)
    self.assertEqual(str(schema_lib.Missing()), 'MISSING')
    self.assertEqual(str(a.type), "MISSING(Literal['daytime', 'nighttime'])")
    self.assertEqual(str(a.activities), 'MISSING(list[Activity])')
    self.assertEqual(str(a.hotel), "MISSING(str(regex='.*Hotel') | None)")

  def assert_missing(self, value, expected_missing):
    value = schema_lib.mark_missing(value)
    self.assertEqual(schema_lib.Missing.find_missing(value), expected_missing)

  def test_find_missing(self):
    self.assert_missing(
        Itinerary.partial(),
        {
            'day': schema_lib.MISSING,
            'type': schema_lib.MISSING,
            'activities': schema_lib.MISSING,
        },
    )

    self.assert_missing(
        Itinerary.partial(
            day=1, type='daytime', activities=[Activity.partial()]
        ),
        {
            'activities[0].description': schema_lib.MISSING,
        },
    )

  def test_mark_missing(self):
    class A(pg.Object):
      x: typing.Any

    self.assertEqual(schema_lib.mark_missing(1), 1)
    self.assertEqual(
        schema_lib.mark_missing(pg.MISSING_VALUE), pg.MISSING_VALUE
    )
    self.assertEqual(
        schema_lib.mark_missing(A.partial(A.partial(A.partial()))),
        A(A(A(schema_lib.MISSING))),
    )
    self.assertEqual(
        schema_lib.mark_missing(dict(a=A.partial())),
        dict(a=A(schema_lib.MISSING)),
    )
    self.assertEqual(
        schema_lib.mark_missing([1, dict(a=A.partial())]),
        [1, dict(a=A(schema_lib.MISSING))],
    )


class UnknownTest(unittest.TestCase):

  def test_basics(self):
    class A(pg.Object):
      x: int

    a = A(x=schema_lib.Unknown())
    self.assertFalse(a.is_partial)
    self.assertEqual(a.x, schema_lib.UNKNOWN)
    self.assertEqual(schema_lib.UNKNOWN, schema_lib.Unknown())


if __name__ == '__main__':
  unittest.main()
