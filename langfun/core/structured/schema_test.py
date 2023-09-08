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


class JsonSchemaTest(unittest.TestCase):

  def assert_schema(self, annotation, spec):
    self.assertEqual(schema_lib.JsonSchema(annotation).spec, spec)

  def assert_unsupported_annotation(self, annotation):
    with self.assertRaises(ValueError):
      schema_lib.JsonSchema(annotation)

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
      schema_lib.JsonSchema([str, int])

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
    schema = schema_lib.JsonSchema([{'x': Itinerary}])
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

  def test_schema_repr(self):
    schema = schema_lib.JsonSchema([{'x': Itinerary}])
    self.assertEqual(
        schema.schema_repr(),
        ('{"result": [{"x": {"_type": "__main__.Itinerary", "day": int(min=1), '
         '"type": "daytime" | "nighttime", "activities": [{"_type": '
         '"__main__.Activity", "description": str}], '
         '"hotel": str(regex=.*Hotel) | None}}]}'
         ))

  def test_value_repr(self):
    schema = schema_lib.JsonSchema(int)
    self.assertEqual(
        schema.value_repr(1),
        '{"result": 1}'
    )

  def assert_parse(self, annotation, inputs, output) -> None:
    schema = schema_lib.JsonSchema(annotation)
    self.assertEqual(schema.parse(inputs), output)

  def test_parse_basics(self):
    self.assert_parse(int, '{"result": 1}', 1)
    self.assert_parse(str, '{"result": "\\"}ab{"}', '"}ab{')
    self.assert_parse(
        {'x': bool, 'y': str | None},
        '{"result": {"x": true, "y": null}}',
        {'x': True, 'y': None},
    )
    self.assert_parse(
        Activity,
        (
            '{"result": {"_type": "%s", "description": "play"}}'
            % Activity.__type_name__
        ),
        Activity('play'),
    )
    with self.assertRaisesRegex(
        ValueError, 'The root node of the JSON must be a dict with key `result`'
    ):
      schema_lib.JsonSchema(int).parse('{"abc": 1}')

  def test_parse_with_surrounding_texts(self):
    self.assert_parse(int, 'The answer is {"result": 1}.', 1)

  def test_parse_with_new_lines(self):
    self.assert_parse(
        [str],
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
      schema_lib.JsonSchema(
          int).parse('The answer is 1.')

    with self.assertRaisesRegex(
        ValueError, 'Malformated JSON: missing .* closing curly braces'
    ):
      schema_lib.JsonSchema(int).parse('{"result": 1')


if __name__ == '__main__':
  unittest.main()
