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
from unittest import mock

from langfun.core.structured.schema import base
from langfun.core.structured.schema import json
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

  def test_class_dependencies_missing_module(self):
    class Meta(type):
      @property
      def __module__(cls):
        raise AttributeError('Testing AttributeError')

    class Base:
      pass

    class Sub(Base, metaclass=Meta):
      pass

    self.assertEqual(
        base.class_dependencies(Base, include_generated_subclasses=True),
        [Base, Sub],
    )
    self.assertEqual(
        base.class_dependencies(Base, include_generated_subclasses=False),
        [Base],
    )


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


# Test features. `upper` comes first by rank, although it is registered after
# `strict` and sorts after it by name.
class _Strict(base._ProtocolFeature):  # pylint: disable=protected-access
  NAME = 'strict'
  VERSION = '1.0'
  BASES = {'json': ('1.0',)}


class _Upper(base._ProtocolFeature):  # pylint: disable=protected-access
  NAME = 'upper'
  VERSION = '1.0'
  BASES = {'json': None}
  RANK = -1


class _TwiceV1(base._ProtocolFeature):  # pylint: disable=protected-access
  NAME = 'twice'
  VERSION = '1.0'
  BASES = {'json': None}
  DEFAULT = True


class _TwiceV2(base._ProtocolFeature):  # pylint: disable=protected-access
  NAME = 'twice'
  VERSION = '2.0'
  BASES = {'json': None}


class _Preamble(base._ProtocolFeature):  # pylint: disable=protected-access
  NAME = 'preamble'
  VERSION = '1.0'
  BASES = {'json': None}

  def applies_to(self, schema):
    return schema is None or not isinstance(schema.spec, pg.typing.Int)

  def query_preamble(self, base_preamble):
    return f'{base_preamble} Preamble.'

  def parse_instructions(self):
    return 'Instructions.'


class _More(base._ProtocolFeature):  # pylint: disable=protected-access
  NAME = 'more'
  VERSION = '1.0'
  BASES = {'json': None}

  def parse_instructions(self):
    return 'More.'


class ProtocolNameTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # A feature that a test defines is registered for that test only.
    self.enterContext(
        mock.patch.dict(base._ProtocolFeature._REGISTRY)  # pylint: disable=protected-access
    )

  def assert_name(self, name, normalized, **kwargs):
    self.assertEqual(str(base.ProtocolName.parse(name, **kwargs)), normalized)

  def test_names_without_features(self):
    self.assertEqual(
        base.ProtocolName.parse('json'), base.ProtocolName('json')
    )
    self.assertEqual(
        base.ProtocolName.parse('python:2.0'),
        base.ProtocolName('python', '2.0'),
    )
    self.assert_name('python', 'python')
    self.assert_name('python', 'python:2.0', default_version='2.0')
    self.assert_name('python:1.0', 'python:1.0', default_version='2.0')
    self.assert_name('python-verbatim:1.0', 'python-verbatim:1.0')

  def test_features_are_versioned_completed_and_ordered(self):
    self.assertEqual(
        base.ProtocolName.parse('json:1.0+strict+upper'),
        base.ProtocolName('json', '1.0', (('upper', '1.0'), ('strict', '1.0'))),
    )
    self.assert_name('json+strict+upper', 'json+upper:1.0+strict:1.0')
    # Features of equal rank sort by name.
    self.assert_name('json+twice+strict', 'json+strict:1.0+twice:1.0')
    self.assert_name(
        'json+strict:1.0+upper',
        'json:1.0+upper:1.0+strict:1.0',
        default_version='1.0',
    )
    self.assert_name('json+twice:2.0+upper:1.0', 'json+upper:1.0+twice:2.0')

  def test_malformed_names(self):
    for name in (
        '',
        'json:',
        ':1.0',
        'json+',
        'json++upper',
        'json+upper:',
        'json:1.0:2.0',
        'json+upper:1.0:2.0',
        None,
    ):
      with self.subTest(name=name):
        with self.assertRaisesRegex(ValueError, 'Invalid protocol'):
          base.ProtocolName.parse(name)

  def test_bad_features(self):
    for name, error in [
        ('json+nope', "Unknown feature 'nope' in protocol 'json\\+nope'"),
        ('json+upper+upper:1.0', "Feature 'upper' appears twice"),
        ('json+upper:2.0', "Version '2.0' is not supported for feature"),
        ('python+upper', 'upper:1.0 .* does not support python:\\*'),
        ('json:2.0+strict', 'strict:1.0 .* does not support json:2.0'),
    ]:
      with self.subTest(name=name):
        with self.assertRaisesRegex(ValueError, error):
          base.ProtocolName.parse(name)
    # A feature that requires a base version also checks the default one.
    self.assert_name('json+strict', 'json+strict:1.0')
    with self.assertRaisesRegex(ValueError, 'does not support json:3.0'):
      base.ProtocolName.parse('json+strict', default_version='3.0')

  def test_feature_versions(self):
    # A feature that the name gives without a version takes its default one.
    self.assert_name('json+twice', 'json+twice:1.0')
    self.assert_name('json+twice:2.0', 'json+twice:2.0')

  def test_a_feature_version_registers_once(self):
    with self.assertRaisesRegex(ValueError, 'upper:1.0 is already registered'):

      class _Another(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
        NAME = 'upper'
        VERSION = '1.0'
        BASES = {}

  def test_adding_a_version_keeps_unpinned_names(self):

    class _GrowV1(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
      NAME = 'grow'
      VERSION = '1.0'
      BASES = {'json': None}
      DEFAULT = True

    names = ['json+grow', 'json:1.0+grow']
    normalized = ['json+grow:1.0', 'json:1.0+grow:1.0']
    self.assertEqual(
        [str(base.ProtocolName.parse(n)) for n in names], normalized
    )

    class _GrowV2(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
      NAME = 'grow'
      VERSION = '2.0'
      BASES = {'json': None}

    self.assertEqual(
        [str(base.ProtocolName.parse(n)) for n in names], normalized
    )
    self.assert_name('json+grow:2.0', 'json+grow:2.0')

  def test_a_feature_with_several_versions_has_one_default(self):

    class _OnceV1(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
      NAME = 'once'
      VERSION = '1.0'
      BASES = {'json': None}

    with self.assertRaisesRegex(
        ValueError,
        "Feature 'once' has versions \\['1.0', '2.0'\\], so exactly one of "
        'them must set `DEFAULT = True`; found \\[\\]',
    ):

      class _OnceV2(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
        NAME = 'once'
        VERSION = '2.0'
        BASES = {'json': None}

    with self.assertRaisesRegex(ValueError, "found \\['1.0', '3.0'\\]"):

      class _TwiceV3(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
        NAME = 'twice'
        VERSION = '3.0'
        BASES = {'json': None}
        DEFAULT = True

    # A rejected version is not registered.
    self.assert_name('json+once', 'json+once:1.0')
    self.assert_name('json+twice', 'json+twice:1.0')
    with self.assertRaisesRegex(ValueError, "Version '3.0' is not supported"):
      base.ProtocolName.parse('json+twice:3.0')

    # A new version becomes the default only by saying so.
    class _OnceV3(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
      NAME = 'once'
      VERSION = '3.0'
      BASES = {'json': None}
      DEFAULT = True

    self.assert_name('json+once', 'json+once:3.0')
    self.assert_name('json+once:1.0', 'json+once:1.0')

  def test_versions_of_a_feature_share_its_rank(self):
    with self.assertRaisesRegex(
        ValueError, "Versions of feature 'twice' have ranks \\[0, 1\\]"
    ):

      class _TwiceV4(base._ProtocolFeature):  # pylint: disable=protected-access,unused-variable
        NAME = 'twice'
        VERSION = '4.0'
        BASES = {'json': None}
        RANK = 1

    with self.assertRaisesRegex(ValueError, "Version '4.0' is not supported"):
      base.ProtocolName.parse('json+twice:4.0')

  def test_names_and_versions_exclude_separators(self):
    for name, version, error in [
        ('a:b', '1.0', "Feature name 'a:b' must be non-empty"),
        ('', '1.0', "Feature name '' must be non-empty"),
        ('ab', '1+0', "Feature version '1\\+0' must be non-empty"),
    ]:
      with self.subTest(name=name, version=version):
        with self.assertRaisesRegex(ValueError, error):
          type(
              '_Bad',
              (base._ProtocolFeature,),  # pylint: disable=protected-access
              dict(NAME=name, VERSION=version, BASES={}),
          )

  def test_registration_keeps_the_init_subclass_chain(self):

    class _Recorder:
      subclasses = []

      def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _Recorder.subclasses.append(cls)

    class _Recorded(base._ProtocolFeature, _Recorder):  # pylint: disable=protected-access
      NAME = 'recorded'
      VERSION = '1.0'
      BASES = {}

    self.assertEqual(_Recorder.subclasses, [_Recorded])


class PromptingProtocolTest(unittest.TestCase):

  def test_from_name(self):
    protocol = base.PromptingProtocol.from_name('json')
    self.assertIsInstance(protocol, json.JsonPromptingProtocol)
    self.assertEqual(protocol.name, 'json')
    self.assertEqual(protocol.features, ())

  def test_from_name_with_version_and_features(self):
    protocol = base.PromptingProtocol.from_name('json:1.0+strict+upper')
    self.assertIsInstance(protocol, json.JsonPromptingProtocol)
    self.assertEqual(protocol.version, '1.0')
    self.assertEqual(
        [type(f) for f in protocol.features], [_Upper, _Strict]
    )
    self.assertEqual(protocol.name, 'json:1.0+upper:1.0+strict:1.0')
    # The version and features leave the representation as is.
    self.assertEqual(
        base.Schema(int).value_repr(1, protocol='json:1.0+strict'),
        '{"result": 1}',
    )

  def test_from_name_builds_a_subclass_without_arguments(self):

    class _Indented(json.JsonPromptingProtocol):
      NAME = 'json-indented'

      def __init__(self, indent: int = 2):
        super().__init__()
        self.indent = indent

    for name, version in [
        ('json-indented', None),
        ('json-indented:1.0', '1.0'),
    ]:
      with self.subTest(name=name):
        protocol = base.PromptingProtocol.from_name(name)
        self.assertIsInstance(protocol, _Indented)
        self.assertEqual(protocol.indent, 2)
        self.assertEqual(protocol.version, version)
        self.assertEqual(protocol.name, name)

  def test_protocol_names_exclude_separators(self):
    with self.assertRaisesRegex(ValueError, "Protocol name 'json\\+x'"):
      type('_Bad', (json.JsonPromptingProtocol,), dict(NAME='json+x'))

  def test_from_name_rejects_unknown_protocols_and_features(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported protocol: text:1.0'):
      base.PromptingProtocol.from_name('text:1.0')
    with self.assertRaisesRegex(ValueError, 'Unknown feature'):
      base.PromptingProtocol.from_name('json+nope')

  def test_supports_autofix(self):
    self.assertFalse(base.PromptingProtocol.supports_autofix('json'))
    self.assertFalse(base.PromptingProtocol.supports_autofix('json:1.0+upper'))
    self.assertFalse(
        base.PromptingProtocol.supports_autofix('no-such-protocol')
    )


class FeatureHooksTest(unittest.TestCase):

  def test_for_schema(self):
    protocol = base.PromptingProtocol.from_name('json:1.0+upper+preamble')
    self.assertEqual(
        protocol.for_schema(base.Schema(str)).name,
        'json:1.0+upper:1.0+preamble:1.0',
    )
    self.assertEqual(
        protocol.for_schema(base.Schema(int)).name, 'json:1.0+upper:1.0'
    )

  def test_query_preamble(self):
    self.assertIsNone(
        base.PromptingProtocol.from_name('json').query_preamble('Base.')
    )
    self.assertIsNone(
        base.PromptingProtocol.from_name('json+upper').query_preamble('Base.')
    )
    self.assertEqual(
        base.PromptingProtocol.from_name('json+upper+preamble').query_preamble(
            'Base.'
        ),
        'Base. Preamble.',
    )

  def test_parse_instructions(self):
    self.assertIsNone(
        base.PromptingProtocol.from_name('json+upper').parse_instructions()
    )
    # In feature order: `more` and `preamble` have the same rank.
    self.assertEqual(
        base.PromptingProtocol.from_name(
            'json+preamble+more'
        ).parse_instructions(),
        'More.\n\nInstructions.',
    )

  def test_add_instructions(self):
    self.assertEqual(
        base.PromptingProtocol.from_name('json+preamble+more').add_instructions(
            '\n  Be brief.\n  Really.\n'
        ),
        'Be brief.\nReally.\n\nMore.\n\nInstructions.',
    )
    preamble = 'Be brief.'
    self.assertIs(
        base.PromptingProtocol.from_name('json+upper').add_instructions(
            preamble
        ),
        preamble,
    )

  def test_default_hooks(self):
    feature = _Upper()
    self.assertTrue(feature.applies_to(None))
    self.assertIsNone(feature.query_preamble('Base.'))
    self.assertIsNone(feature.parse_instructions())
    self.assertIsNone(
        feature.field_hint(
            pg.schema(Activity).get_field('description'),
            base.Schema.from_value(Activity),
        )
    )
    self.assertEqual(feature.render_value(1, None, str), '1')
    hooks = feature.parse_hooks(None)
    self.assertEqual(hooks.before_exec('x = 1'), 'x = 1')
    global_vars = {}
    hooks.check_code('x = 1', global_vars)
    self.assertEqual(global_vars, {})
    self.assertEqual(hooks.after_exec(1, 'x = 1'), 1)


if __name__ == '__main__':
  unittest.main()
