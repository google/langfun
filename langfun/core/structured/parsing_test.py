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
import unittest

import langfun.core as lf
from langfun.core import coding
from langfun.core.llms import fake

from langfun.core.structured import mapping
from langfun.core.structured import parsing
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  day: pg.typing.Int[1, None]
  type: pg.typing.Enum['daytime', 'nighttime']
  activities: list[Activity]
  hotel: pg.typing.Str['.*Hotel'] | None


class ParseStructurePythonTest(unittest.TestCase):

  def test_render_no_examples(self):
    l = parsing.ParseStructurePython(int)
    m = lf.AIMessage('Bla bla bla 12 / 6 + 2 = 4.', result='12 / 6 + 2 = 4')
    m.source = lf.UserMessage('Compute 12 / 6 + 2.', tags=['lm-input'])

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help translate LM_RESPONSE into RESULT_OBJECT based on RESULT_TYPE.
            Both RESULT_TYPE and RESULT_OBJECT are described in Python.

            USER_REQUEST:
              Compute 12 / 6 + 2.

            LM_RESPONSE:
              12 / 6 + 2 = 4

            RESULT_TYPE:
              int

            RESULT_OBJECT:
            """),
    )

  def test_render_no_context(self):
    l = parsing.ParseStructurePython(int)
    m = lf.AIMessage('Bla bla bla 12 / 6 + 2 = 4.', result='12 / 6 + 2 = 4')

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help translate LM_RESPONSE into RESULT_OBJECT based on RESULT_TYPE.
            Both RESULT_TYPE and RESULT_OBJECT are described in Python.

            LM_RESPONSE:
              12 / 6 + 2 = 4

            RESULT_TYPE:
              int

            RESULT_OBJECT:
            """),
    )

  def test_render(self):
    l = parsing.ParseStructurePython(
        int,
        examples=[
            mapping.MappingExample(
                'What is the answer of 1 plus 1?', '1 + 1 = 2', 2
            ),
            mapping.MappingExample(
                'Compute the value of 3 + (2 * 6).', 'fifteen', 15
            ),
        ],
    )
    self.assertEqual(
        l.render(message=lf.AIMessage('Compute 12 / 6 + 2.')).text,
        inspect.cleandoc("""
            Please help translate LM_RESPONSE into RESULT_OBJECT based on RESULT_TYPE.
            Both RESULT_TYPE and RESULT_OBJECT are described in Python.

            USER_REQUEST:
              What is the answer of 1 plus 1?

            LM_RESPONSE:
              1 + 1 = 2

            RESULT_TYPE:
              int

            RESULT_OBJECT:
              ```python
              2
              ```

            USER_REQUEST:
              Compute the value of 3 + (2 * 6).

            LM_RESPONSE:
              fifteen

            RESULT_TYPE:
              int

            RESULT_OBJECT:
              ```python
              15
              ```


            LM_RESPONSE:
              Compute 12 / 6 + 2.

            RESULT_TYPE:
              int

            RESULT_OBJECT:
            """),
    )

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

    parse_structured_response = inspect.cleandoc(
        """
        ```python
        [
            Itinerary(
                day=1,
                type='daytime',
                activities=[
                    Activity(description='Arrive in San Francisco and check into your hotel.'),
                    Activity(description='Take a walk around Fisherman\\'s Wharf and have dinner at one of the many seafood restaurants.'),
                    Activity(description='Visit Pier 39 and see the sea lions.'),
                ], 
                hotel=None),
            Itinerary(
                day=2,
                type='daytime',
                activities=[
                    Activity(description='Take a ferry to Alcatraz Island and tour the infamous prison.'),
                    Activity(description='Take a walk across the Golden Gate Bridge.'),
                    Activity(description='Visit the Japanese Tea Garden in Golden Gate Park.'),
                ], 
                hotel=None),
            Itinerary(
                day=3,
                type='daytime',
                activities=[
                    Activity(description='Visit the de Young Museum and see the collection of American art.'),
                    Activity(description='Visit the San Francisco Museum of Modern Art.'),
                    Activity(description='Take a cable car ride.'),
                ], 
                hotel=None),
        ]
        ```
        """)

    with lf.context(
        lm=fake.StaticSequence(
            [lm_response, parse_structured_response],
        ),
        override_attrs=True,
    ):
      l = lf.LangFunc(lm_input) >> parsing.ParseStructurePython(
          [Itinerary],
          examples=[
              mapping.MappingExample(
                  nl_context=inspect.cleandoc("""
                      Find the alternatives of expressing \"feeling great\".
                      """),
                  nl_text=inspect.cleandoc("""
                      Here are some words for expressing \"feeling great\".
                      * Ecstatic
                      * Delighted
                      * Wonderful
                      * Enjoyable
                      * Fantastic"""),
                  schema={'expression': str, 'words': list[str]},
                  value={
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
        lm=fake.StaticSequence(['three', 'a3']),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          coding.CodeError,
          'name .* is not defined',
      ):
        lf.LangFunc('Compute 1 + 2').as_structured(int)()

  def test_parse(self):
    lm = fake.StaticSequence(['1'])
    self.assertEqual(parsing.parse('the answer is 1', int, lm=lm), 1)
    self.assertEqual(
        parsing.parse(
            'the answer is 1', int, user_prompt='what is 0 + 1?', lm=lm
        ),
        1,
    )


class ParseStructureJsonTest(unittest.TestCase):

  def test_render_no_examples(self):
    l = parsing.ParseStructureJson(int)
    m = lf.AIMessage('Bla bla bla 12 / 6 + 2 = 4.', result='12 / 6 + 2 = 4')
    m.source = lf.UserMessage('Compute 12 / 6 + 2.', tags=['lm-input'])

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help translate the LM response into JSON based on the request and the schema:

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

  def test_render_no_context(self):
    l = parsing.ParseStructureJson(int)
    m = lf.AIMessage('Bla bla bla 12 / 6 + 2 = 4.', result='12 / 6 + 2 = 4')

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help translate the LM response into JSON based on the request and the schema:

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
    l = parsing.ParseStructureJson(
        int,
        examples=[
            mapping.MappingExample(
                'What is the answer of 1 plus 1?', '1 + 1 = 2', 2
            ),
            mapping.MappingExample(
                'Compute the value of 3 + (2 * 6).', 'fifteen', 15
            ),
        ],
    )
    self.assertEqual(
        l.render(message=lf.AIMessage('Compute 12 / 6 + 2.')).text,
        inspect.cleandoc("""
            Please help translate the LM response into JSON based on the request and the schema:

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
        ),
        override_attrs=True,
    ):
      l = lf.LangFunc(lm_input) >> parsing.ParseStructureJson(
          [Itinerary],
          examples=[
              mapping.MappingExample(
                  nl_context=inspect.cleandoc("""
                      Find the alternatives of expressing \"feeling great\".
                      """),
                  nl_text=inspect.cleandoc("""
                      Here are some words for expressing \"feeling great\".
                      * Ecstatic
                      * Delighted
                      * Wonderful
                      * Enjoyable
                      * Fantastic"""),
                  schema={'expression': str, 'words': list[str]},
                  value={
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
        lm=fake.StaticSequence(['three', '`3`']),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          coding.CodeError,
          'invalid syntax',
      ):
        lf.LangFunc('Compute 1 + 2').as_structured(int)()

    with lf.context(
        lm=fake.StaticSequence(['three', '`3`']),
        override_attrs=True,
    ):
      # Test default.
      self.assertIsNone(
          lf.LangFunc('Compute 1 + 2').as_structured(int, default=None)().result
      )

  def test_parse(self):
    lm = fake.StaticSequence(['{"result": 1}'])
    self.assertEqual(
        parsing.parse('the answer is 1', int, lm=lm, protocol='json'), 1
    )
    self.assertEqual(
        parsing.parse(
            'the answer is 1',
            int,
            user_prompt='what is 0 + 1?',
            lm=lm,
            protocol='json',
        ),
        1,
    )


if __name__ == '__main__':
  unittest.main()
