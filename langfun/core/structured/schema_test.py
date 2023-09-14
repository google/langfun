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

import inspect
import typing
import unittest

from langfun.core.structured import schema as schema_lib
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  day: pg.typing.Int[1, None]
  type: pg.typing.Enum['daytime', 'nighttime']
  activities: list[Activity]
  hotel: pg.typing.Str['.*Hotel'] | None


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

    class X:
      pass

    # X must be a symbolic type to be parsable.
    self.assert_unsupported_annotation(X)

  def test_schema_dict(self):
    schema = schema_lib.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema.schema_dict(),
        {
            'result': [
                {
                    'x': {
                        '_type': Itinerary.__type_name__,
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

  def test_schema_repr(self):
    schema = schema_lib.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema.schema_str(protocol='json'),
        (
            '{"result": [{"x": {"_type": "%s", "day":'
            ' int(min=1), "type": "daytime" | "nighttime", "activities":'
            ' [{"_type": "%s", "description": str}], "hotel":'
            ' str(regex=.*Hotel) | None}}]}' % (
                Itinerary.__type_name__,
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
    with self.assertRaisesRegex(TypeError, 'Expect .* but encountered .*'):
      schema.parse('{"result": "def"}')

    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema.parse('1', protocol='text')


class SchemaPythonReprTest(unittest.TestCase):

  def assert_annotation(
      self,
      value_spec: pg.typing.ValueSpec,
      expected_annotation: str,
      strict: bool = False,
  ) -> None:
    self.assertEqual(
        schema_lib.SchemaPythonRepr().annotate(value_spec, strict=strict),
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

  def test_class_def(self):
    self.assertEqual(
        schema_lib.SchemaPythonRepr().class_def(Activity),
        'class Activity:\n  description: str\n',
    )
    self.assertEqual(
        schema_lib.SchemaPythonRepr().class_def(Itinerary),
        inspect.cleandoc("""
            class Itinerary:
              day: int(min=1)
              type: Literal['daytime', 'nighttime']
              activities: list[Activity]
              hotel: str(regex='.*Hotel') | None
            """) + '\n',
    )

    class A(pg.Object):
      pass

    self.assertEqual(
        schema_lib.SchemaPythonRepr().class_def(A),
        'class A:\n  pass\n',
    )

    class B:
      pass

    with self.assertRaisesRegex(
        TypeError, 'Classes must be `pg.Object` subclasses.*'):
      schema_lib.SchemaPythonRepr().class_def(B)

    class C(pg.Object):
      x: str
      __kwargs__: typing.Any

    with self.assertRaisesRegex(
        TypeError, 'Variable-length keyword arguments is not supported'):
      schema_lib.SchemaPythonRepr().class_def(C)

  def test_repr(self):
    class Foo(pg.Object):
      x: int

    class Bar(pg.Object):
      y: str

    class Baz(Bar):  # pylint: disable=unused-variable
      pass

    class A(pg.Object):
      foo: Foo

    class B(A):
      bar: Bar
      foo2: Foo

    schema = schema_lib.Schema([B])
    self.assertEqual(
        schema_lib.SchemaPythonRepr().class_definitions(schema),
        inspect.cleandoc("""
            class Foo:
              x: int

            class A:
              foo: Foo

            class Bar:
              y: str

            class Baz(Bar):
              y: str

            class B(A):
              foo: Foo
              bar: Bar
              foo2: Foo
            """) + '\n',
    )

    self.assertEqual(
        schema_lib.SchemaPythonRepr().result_definition(schema), 'list[B]'
    )

    self.assertEqual(
        schema_lib.SchemaPythonRepr().repr(schema),
        inspect.cleandoc("""
            list[B]

            ```python
            class Foo:
              x: int

            class A:
              foo: Foo

            class Bar:
              y: str

            class Baz(Bar):
              y: str

            class B(A):
              foo: Foo
              bar: Bar
              foo2: Foo
            ```
            """),
    )


class SchemaJsonReprTest(unittest.TestCase):

  def test_repr(self):
    schema = schema_lib.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema_lib.SchemaJsonRepr().repr(schema),
        (
            '{"result": [{"x": {"_type": "%s", "day":'
            ' int(min=1), "type": "daytime" | "nighttime", "activities":'
            ' [{"_type": "%s", "description": str}], "hotel":'
            ' str(regex=.*Hotel) | None}}]}' % (
                Itinerary.__type_name__,
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
        ValueError, 'The root node of the JSON must be a dict with key `result`'
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
        ValueError, 'No JSON dict in the output'
    ):
      schema_lib.ValueJsonRepr().parse('The answer is 1.')

    with self.assertRaisesRegex(
        ValueError, 'Malformated JSON: missing .* closing curly braces'
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


if __name__ == '__main__':
  unittest.main()
