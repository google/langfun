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

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.transforms import parse_structured
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  day: pg.typing.Int[1, None]
  type: pg.typing.Enum['daytime', 'nighttime']
  activities: list[Activity]
  hotel: pg.typing.Str['.*Hotel'] | None


class ParsingSchemaTest(unittest.TestCase):

  def assert_from_annotation(self, annotation, spec):
    t = parse_structured.ParsingSchema.from_annotation(annotation)
    self.assertEqual(t.spec, spec)

  def assert_unsupported_annotation(self, annotation):
    with self.assertRaises(ValueError):
      parse_structured.ParsingSchema.from_annotation(annotation)

  def test_from_annotation(self):
    self.assert_from_annotation(int, pg.typing.Int())
    self.assert_from_annotation(float, pg.typing.Float())
    self.assert_from_annotation(str, pg.typing.Str())
    self.assert_from_annotation(bool, pg.typing.Bool())
    self.assert_from_annotation(bool | None, pg.typing.Bool().noneable())

    # Top-level dictionary with 'result' as the only key is flattened.
    self.assert_from_annotation(dict(result=int), pg.typing.Int())

    self.assert_from_annotation(list[str], pg.typing.List(pg.typing.Str()))
    self.assert_from_annotation([str], pg.typing.List(pg.typing.Str()))

    with self.assertRaisesRegex(
        ValueError, 'Annotation with list must be a list of a single element.'
    ):
      parse_structured.ParsingSchema.from_annotation([str, int])

    self.assert_from_annotation(
        dict[str, int], pg.typing.Dict([(pg.typing.StrKey(), pg.typing.Int())])
    )

    self.assert_from_annotation(
        {
            'x': int,
            'y': [str],
        },
        pg.typing.Dict([
            ('x', int),
            ('y', pg.typing.List(pg.typing.Str())),
        ]),
    )

    self.assert_from_annotation(Itinerary, pg.typing.Object(Itinerary))

    self.assert_unsupported_annotation(typing.Type[int])
    self.assert_unsupported_annotation(typing.Union[int, str, bool])

    class X:
      pass

    # X must be a symbolic type to be parsable.
    self.assert_unsupported_annotation(X)

  def test_schema_dict(self):
    schema = parse_structured.ParsingSchema.from_annotation([{'x': Itinerary}])
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

  def test_json_repr(self):
    schema = parse_structured.ParsingSchema.from_annotation([{'x': Itinerary}])
    self.assertEqual(
        schema.json_repr(),
        ('{"result": [{"x": {"_type": "__main__.Itinerary", "day": int(min=1), '
         '"type": "daytime" | "nighttime", "activities": [{"_type": '
         '"__main__.Activity", "description": str}], '
         '"hotel": str(regex=.*Hotel) | None}}]}'
         ))

  def assert_parse(self, annotation, inputs, output) -> None:
    schema = parse_structured.ParsingSchema.from_annotation(annotation)
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
      parse_structured.ParsingSchema.from_annotation(int).parse('{"abc": 1}')

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
      parse_structured.ParsingSchema.from_annotation(
          int).parse('The answer is 1.')

    with self.assertRaisesRegex(
        ValueError, 'Malformated JSON: missing .* closing curly braces'
    ):
      parse_structured.ParsingSchema.from_annotation(int).parse('{"result": 1')


class ParsingExampleTest(unittest.TestCase):
  """Tests for ParsingExample."""

  def test_render(self):
    self.assertEqual(
        parse_structured.ParsingExample(
            'Give the answer.',
            '1 + 1 = 2',
            2,
            int).render(),
        inspect.cleandoc("""
            USER_REQUEST:
              Give the answer.

            LM_RESPONSE:
              1 + 1 = 2

            SCHEMA:
              {"result": int}

            JSON:
              {"result": 2}
            """),
    )
    # No user request.
    self.assertEqual(
        parse_structured.ParsingExample(None, '1 + 1 = 2', 2, int).render(),
        inspect.cleandoc("""
            LM_RESPONSE:
              1 + 1 = 2

            SCHEMA:
              {"result": int}

            JSON:
              {"result": 2}
            """),
    )

  def test_format(self):
    self.assertEqual(
        repr(parse_structured.ParsingExample(
            'Give the answer.',
            '1 + 1 = 2',
            2,
            int)),
        "ParsingExample(request='Give the answer.', response='1 + 1 = 2', "
        'result_schema=ParsingSchema(spec=Int()), result=2)')


class FewshotJsonifyTest(unittest.TestCase):
  """Tests for FewshotJsonify."""

  def test_render_no_examples(self):
    l = parse_structured.FewshotJsonify(
        result_schema=parse_structured.ParsingSchema.from_annotation(int),
    )
    m = lf.AIMessage('Bla bla bla 12 / 6 + 2 = 4.', result='12 / 6 + 2 = 4')
    m.source = lf.UserMessage('Compute 12 / 6 + 2.', tags=['lm-input'])

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help transform the LM response into JSON based on the request and the schema:

            INSTRUCTIONS:
              1. If the schema has `_type`, carry it over to the JSON output.
              2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

            USER_REQUEST:
              Compute 12 / 6 + 2.

            LM_RESPONSE:
              12 / 6 + 2 = 4

            SCHEMA:
              {"result": int}

            JSON:
            """),
    )

  def test_render_no_request(self):
    l = parse_structured.FewshotJsonify(
        result_schema=parse_structured.ParsingSchema.from_annotation(int),
    )
    m = lf.AIMessage('Bla bla bla 12 / 6 + 2 = 4.', result='12 / 6 + 2 = 4')

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help transform the LM response into JSON based on the request and the schema:

            INSTRUCTIONS:
              1. If the schema has `_type`, carry it over to the JSON output.
              2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

            LM_RESPONSE:
              12 / 6 + 2 = 4

            SCHEMA:
              {"result": int}

            JSON:
            """),
    )

  def test_render(self):
    l = parse_structured.FewshotJsonify(
        examples=[
            parse_structured.ParsingExample(
                'What is the answer of 1 plus 1?', '1 + 1 = 2', 2),
            parse_structured.ParsingExample(
                'Compute the value of 3 + (2 * 6).', 'fifteen', 15),
        ],
        result_schema=parse_structured.ParsingSchema.from_annotation(int),
    )
    self.assertEqual(
        l.render(message=lf.AIMessage('Compute 12 / 6 + 2.')).text,
        inspect.cleandoc("""
            Please help transform the LM response into JSON based on the request and the schema:

            INSTRUCTIONS:
              1. If the schema has `_type`, carry it over to the JSON output.
              2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

            USER_REQUEST:
              What is the answer of 1 plus 1?

            LM_RESPONSE:
              1 + 1 = 2

            SCHEMA:
              {"result": int}

            JSON:
              {"result": 2}

            USER_REQUEST:
              Compute the value of 3 + (2 * 6).

            LM_RESPONSE:
              fifteen

            SCHEMA:
              {"result": int}

            JSON:
              {"result": 15}


            LM_RESPONSE:
              Compute 12 / 6 + 2.

            SCHEMA:
              {"result": int}

            JSON:
            """),
    )


class ParseStructuredTest(unittest.TestCase):
  """Tests for ParseStructured."""

  def test_transform(self):
    lm_input = '3-day itineraries to San Francisco'
    lm_response = inspect.cleandoc("""
        **Day 1:**

        * Arrive in San Francisco and check into your hotel.
        * Take a walk around Fisherman's Wharf and have dinner at one of the
          many seafood restaurants.
        * Visit Pier 39 and see the sea lions.

        **Day 2:**

        * Take a ferry to Alcatraz Island and tour the infamous prison.
        * Take a walk across the Golden Gate Bridge.
        * Visit the Japanese Tea Garden in Golden Gate Park.

        **Day 3:**

        * Visit the de Young Museum and see the collection of American art.
        * Visit the San Francisco Museum of Modern Art.
        * Take a cable car ride.
        """)

    parse_structured_response = (
        lf.LangFunc(
            """
        {"result": [
          {
            "_type": {{itinerary_type}},
            "day": 1,
            "type": "daytime",
            "activities": [
              {
                "_type": {{activity_type}},
                "description": "Arrive in San Francisco and check into your hotel."
              },
              {
                "_type": {{activity_type}},
                "description": "Take a walk around Fisherman's Wharf and have dinner at one of the many seafood restaurants."
              },
              {
                "_type": {{activity_type}},
                "description": "Visit Pier 39 and see the sea lions."
              }
            ],
            "hotel": null
          },
          {
              "_type": {{itinerary_type}},
              "day": 2,
              "type": "daytime",
              "activities": [
                {
                  "_type": {{activity_type}},
                  "description": "Take a ferry to Alcatraz Island and tour the infamous prison."
                },
                {
                  "_type": {{activity_type}},
                  "description": "Take a walk across the Golden Gate Bridge."
                },
                {
                  "_type": {{activity_type}},
                  "description": "Visit the Japanese Tea Garden in Golden Gate Park."
                }
              ], 
              "hotel": null
           },
           {
              "_type": {{itinerary_type}},
              "day": 3,
              "type": "daytime",
              "activities": [
                {
                  "_type": {{activity_type}},
                  "description": "Visit the de Young Museum and see the collection of American art."
                },
                {
                  "_type": {{activity_type}},
                  "description": "Visit the San Francisco Museum of Modern Art."
                },
                {
                  "_type": {{activity_type}},
                  "description": "Take a cable car ride."
                }
              ],
              "hotel": null
            }
          ]}
        """,
            itinerary_type=f'"{Itinerary.__type_name__}"',
            activity_type=f'"{Activity.__type_name__}"',
        )
        .render()
        .text
    )
    with lf.context(
        lm=fake.StaticSequence(
            [lm_response, parse_structured_response],
            debug=True,
        ),
        override_attrs=True,
    ):
      l = lf.LangFunc(lm_input).as_structured(
          [Itinerary],
          examples=[
              parse_structured.ParsingExample(
                  request=inspect.cleandoc("""
                      Find the alternatives of expressing \"feeling great\".
                      """),
                  response=inspect.cleandoc("""
                      Here are some words for expressing \"feeling great\".
                      * Ecstatic
                      * Delighted
                      * Wonderful
                      * Enjoyable
                      * Fantastic"""),
                  result_schema={'expression': str, 'words': list[str]},
                  result={
                      'expression': 'feeling great',
                      'words': [
                          'Ecstatic',
                          'Delighted',
                          'Wonderful',
                          'Enjoyable',
                          'Fantastic',
                      ],
                  },
              )
          ],
      )
      r = l()
      self.assertEqual(len(r.result), 3)
      self.assertIsInstance(r.result[0], Itinerary)
      self.assertEqual(len(r.result[0].activities), 3)
      self.assertIsNone(r.result[0].hotel)

  def test_bad_transform(self):
    with lf.context(
        lm=fake.StaticSequence(['three', '3'], debug=True),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          parse_structured.ParsingError,
          'Cannot parse message text into structured output',
      ):
        lf.LangFunc('Compute 1 + 2').as_structured(int)()

  def test_parse(self):
    lm = fake.StaticSequence(['{"result": 1}'])
    self.assertEqual(parse_structured.parse('the answer is 1', int, lm=lm), 1)
    self.assertEqual(
        parse_structured.parse(
            'the answer is 1', int, user_prompt='what is 0 + 1?', lm=lm), 1)


if __name__ == '__main__':
  unittest.main()
