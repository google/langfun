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
import unittest
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


class SchemaReprTest(unittest.TestCase):

  def test_repr(self):
    schema = base.Schema([{'x': Itinerary}])
    self.assertEqual(
        base.schema_repr(schema, protocol='json'),
        (
            '{"result": [{"x": {"_type": "Itinerary", "day":'
            ' int(min=1), "type": "daytime" | "nighttime", "activities":'
            ' [{"_type": "%s", "description": str}], "hotel":'
            ' str(regex=.*Hotel) | None}}]}' % (
                Activity.__type_name__,
            )
        ),
    )


class ValueReprTest(unittest.TestCase):

  def test_value_repr(self):
    self.assertEqual(base.value_repr(1, protocol='json'), '{"result": 1}')

  def assert_parse_value(self, inputs, output) -> None:
    self.assertEqual(base.parse_value(inputs, protocol='json'), output)

  def test_parse_basics(self):
    self.assert_parse_value('{"result": 1}', 1)
    self.assert_parse_value('{"result": "\\"}ab{"}', '"}ab{')
    self.assert_parse_value(
        '{"result": {"x": true, "y": null}}',
        {'x': True, 'y': None},
    )
    self.assert_parse_value(
        (
            '{"result": {"_type": "%s", "description": "play"}}'
            % Activity.__type_name__
        ),
        Activity('play'),
    )
    with self.assertRaisesRegex(
        json.JsonError, 'JSONDecodeError'
    ):
      base.parse_value('{"abc", 1}', protocol='json')

    with self.assertRaisesRegex(
        json.JsonError,
        'The root node of the JSON must be a dict with key `result`'
    ):
      base.parse_value('{"abc": 1}', protocol='json')

  def test_parse_with_surrounding_texts(self):
    self.assert_parse_value('The answer is {"result": 1}.', 1)

  def test_parse_with_new_lines(self):
    self.assert_parse_value(
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
        json.JsonError, 'No JSON dict in the output'
    ):
      base.parse_value('The answer is 1.', protocol='json')

    with self.assertRaisesRegex(
        json.JsonError,
        'Malformated JSON: missing .* closing curly braces'
    ):
      base.parse_value('{"result": 1', protocol='json')


if __name__ == '__main__':
  unittest.main()
