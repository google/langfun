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
import typing
import unittest

from langfun.core.structured.schema import base
from langfun.core.structured.schema import json  # pylint: disable=unused-import
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


Itinerary.__serialization_key__ = 'Itinerary'


class Node(pg.Object):
  children: list['Node']


class SchemaTest(unittest.TestCase):

  def assert_schema(self, annotation, spec):
    self.assertEqual(base.Schema(annotation).spec, spec)

  def assert_unsupported_annotation(self, annotation):
    with self.assertRaises(ValueError):
      base.Schema(annotation)

  def test_init(self):
    self.assert_schema(int, pg.typing.Int())
    self.assert_schema(float, pg.typing.Float())
    self.assert_schema(str, pg.typing.Str())
    self.assert_schema(bool, pg.typing.Bool())

    # Top-level dictionary with 'result' as the only key is flattened.
    self.assert_schema(dict(result=int), pg.typing.Int())

    self.assert_schema(list[str], pg.typing.List(pg.typing.Str()))
    self.assert_schema([str], pg.typing.List(pg.typing.Str()))

    with self.assertRaisesRegex(
        ValueError, 'Annotation with list must be a list of a single element.'
    ):
      base.Schema([str, int])

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
    schema = base.Schema([{'x': Itinerary}])
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
      k: int | bytes

    class B(A):
      bar: Bar
      foo2: Foo | X

    schema = base.Schema([B])
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

    schema = base.Schema([AA])
    self.assertEqual(schema.class_dependencies(), [Baz, AA, XX, BB])

  def test_schema_repr(self):
    schema = base.Schema([{'x': Itinerary}])
    self.assertEqual(
        schema.schema_repr(protocol='json'),
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
      schema.schema_repr(protocol='text')

  def test_value_repr(self):
    schema = base.Schema(int)
    self.assertEqual(schema.value_repr(1, protocol='json'), '{"result": 1}')
    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema.value_repr(1, protocol='text')

  def test_parse(self):
    schema = base.Schema(int)
    self.assertEqual(schema.parse_value('{"result": 1}', protocol='json'), 1)
    schema = base.Schema(dict[str, int])
    self.assertEqual(
        schema.parse_value('{"result": {"x": 1}}}', protocol='json'),
        dict(x=1)
    )
    with self.assertRaisesRegex(
        base.SchemaError, 'Expect .* but encountered .*'):
      schema.parse_value('{"result": "def"}', protocol='json')

    with self.assertRaisesRegex(ValueError, 'Unsupported protocol'):
      schema.parse_value('1', protocol='text')


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

    self.assertEqual(base.class_dependencies(Foo), [Foo])

    self.assertEqual(
        base.class_dependencies((A,), include_subclasses=False), [Foo, A]
    )

    self.assertEqual(
        base.class_dependencies(A, include_subclasses=True),
        [Foo, A, Bar, X, B],
    )

    self.assertEqual(
        base.class_dependencies(base.Schema(A)), [Foo, A, Bar, X, B]
    )

    self.assertEqual(
        base.class_dependencies(pg.typing.Object(A)), [Foo, A, Bar, X, B]
    )

    with self.assertRaisesRegex(TypeError, 'Unsupported spec type'):
      base.class_dependencies((Foo, 1))

  def test_class_dependencies_recursive(self):
    self.assertEqual(
        base.class_dependencies(Node),
        [Node]
    )

  def test_class_dependencies_from_value(self):
    class Foo(pg.Object):
      x: int

    class Bar(pg.Object):
      y: str

    class A(pg.Object):
      foo: tuple[Foo, int]

    class B(pg.Object):
      pass

    class X(pg.Object):
      k: dict[str, B]

    class C(A):
      bar: Bar
      foo2: Foo | X

    a = A(foo=(Foo(1), 0))
    self.assertEqual(base.class_dependencies(a), [Foo, A, Bar, B, X, C])

    self.assertEqual(base.class_dependencies(1), [])


class AnnotationTest(unittest.TestCase):

  def assert_annotation(
      self,
      value_spec: pg.typing.ValueSpec,
      expected_annotation: str,
      strict: bool = False,
      **kwargs,
  ) -> None:
    self.assertEqual(
        base.annotation(value_spec, strict=strict, **kwargs),
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
    self.assert_annotation(
        pg.typing.Object(Activity).noneable(), 'Activity | None',
        allowed_dependencies=set([Activity]),
    )
    self.assert_annotation(
        pg.typing.Object(Activity).noneable(), 'Any | None',
        allowed_dependencies=set(),
    )

    # List.
    self.assert_annotation(
        pg.typing.List(pg.typing.Object(Activity)), 'list[Activity]'
    )
    self.assert_annotation(
        pg.typing.List(pg.typing.Object(Activity)), 'list[Activity]',
        allowed_dependencies=set([Activity]),
    )
    self.assert_annotation(
        pg.typing.List(pg.typing.Object(Activity)), 'list[Any]',
        allowed_dependencies=set(),
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
        pg.typing.Tuple([Activity, pg.typing.Str()]), 'tuple[Activity, str]'
    )
    self.assert_annotation(
        pg.typing.Tuple([Activity, pg.typing.Str()]), 'tuple[Activity, str]',
        allowed_dependencies=set([Activity]),
    )
    self.assert_annotation(
        pg.typing.Tuple([Activity, pg.typing.Str()]), 'tuple[Any, str]',
        allowed_dependencies=set(),
    )
    self.assert_annotation(
        pg.typing.Tuple([Activity, pg.typing.Str()]).noneable(),
        'tuple[Activity, str] | None',
    )

    # Dict.
    self.assert_annotation(
        pg.typing.Dict({'x': Activity, 'y': str}),
        '{\'x\': Activity, \'y\': str}'
    )
    self.assert_annotation(
        pg.typing.Dict({'x': Activity, 'y': str}),
        '{\'x\': Activity, \'y\': str}',
        allowed_dependencies=set([Activity]),
    )
    self.assert_annotation(
        pg.typing.Dict({'x': Activity, 'y': str}),
        '{\'x\': Any, \'y\': str}',
        allowed_dependencies=set(),
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

    class DictValue(pg.Object):
      pass

    self.assert_annotation(
        pg.typing.Dict([(pg.typing.StrKey(), DictValue)]),
        'dict[str, DictValue]',
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
    self.assert_annotation(
        pg.typing.Union(
            [pg.typing.Object(Activity), pg.typing.Object(Itinerary)]
        ).noneable(),
        'Union[Activity, Any, None]',
        allowed_dependencies=set([Activity]),
    )

    # Any.
    self.assert_annotation(pg.typing.Any(), 'Any')
    self.assert_annotation(pg.typing.Any().noneable(), 'Any')


class MissingTest(unittest.TestCase):

  def test_basics(self):
    a = Itinerary(
        day=1,
        type=base.Missing(),
        activities=base.Missing(),
        hotel=base.Missing(),
    )
    self.assertFalse(a.is_partial)
    self.assertEqual(str(base.Missing()), 'MISSING')
    self.assertEqual(str(a.type), "MISSING(Literal['daytime', 'nighttime'])")
    self.assertEqual(str(a.activities), 'MISSING(list[Activity])')
    self.assertEqual(str(a.hotel), "MISSING(str(regex='.*Hotel') | None)")

  def assert_missing(self, value, expected_missing):
    value = base.mark_missing(value)
    self.assertEqual(base.Missing.find_missing(value), expected_missing)

  def test_find_missing(self):
    self.assert_missing(
        Itinerary.partial(),
        {
            'day': base.MISSING,
            'type': base.MISSING,
            'activities': base.MISSING,
        },
    )

    self.assert_missing(
        Itinerary.partial(
            day=1, type='daytime', activities=[Activity.partial()]
        ),
        {
            'activities[0].description': base.MISSING,
        },
    )

  def test_mark_missing(self):
    class A(pg.Object):
      x: typing.Any

    self.assertEqual(base.mark_missing(1), 1)
    self.assertEqual(
        base.mark_missing(pg.MISSING_VALUE), pg.MISSING_VALUE
    )
    self.assertEqual(
        base.mark_missing(A.partial(A.partial(A.partial()))),
        A(A(A(base.MISSING))),
    )
    self.assertEqual(
        base.mark_missing(dict(a=A.partial())),
        dict(a=A(base.MISSING)),
    )
    self.assertEqual(
        base.mark_missing([1, dict(a=A.partial())]),
        [1, dict(a=A(base.MISSING))],
    )


class UnknownTest(unittest.TestCase):

  def test_basics(self):
    class A(pg.Object):
      x: int

    a = A(x=base.Unknown())
    self.assertFalse(a.is_partial)
    self.assertEqual(a.x, base.UNKNOWN)
    self.assertEqual(base.UNKNOWN, base.Unknown())


if __name__ == '__main__':
  unittest.main()
