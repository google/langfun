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
"""Tests for structured prompting."""

import inspect
import unittest

import langfun.core as lf
from langfun.core import modalities
from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import prompting
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  day: pg.typing.Int[1, None]
  type: pg.typing.Enum['daytime', 'nighttime']
  activities: list[Activity]
  hotel: pg.typing.Str['.*Hotel'] | None


class QueryTest(unittest.TestCase):

  def assert_render(
      self,
      prompt,
      schema,
      examples: list[mapping.MappingExample] | None = None,
      *,
      expected_snippet: str,
      exact_match: bool = False,
      expected_modalities: int = 0,
      **kwargs,
  ):
    m = prompting.query(
        prompt, schema=schema, examples=examples,
        **kwargs, returns_message=True
    )
    self.assertIsNotNone(m.lm_input)
    if exact_match:
      self.assertEqual(expected_snippet, m.lm_input.text)
    else:
      self.assertIn(expected_snippet, m.lm_input.text)
    self.assertEqual(
        len([c for c in m.lm_input.chunk() if isinstance(c, lf.Modality)]),
        expected_modalities,
    )

  def test_call(self):
    lm = fake.StaticSequence(['1'])
    self.assertEqual(prompting.query('what is 1 + 0', int, lm=lm), 1)

    # Testing calling the same `lm` without copy.
    with self.assertRaises(IndexError):
      prompting.query('what is 1 + 2', int, lm=lm)

    self.assertEqual(
        prompting.query(
            'what is 1 + 0', int, lm=lm.clone(), returns_message=True
        ),
        lf.AIMessage(
            '1',
            result=1,
            score=1.0,
            logprobs=None,
            usage=lf.LMSamplingUsage(323, 1, 324),
            tags=['lm-response', 'lm-output', 'transformed'],
        ),
    )
    self.assertEqual(
        prompting.query(
            lf.Template('what is {{x}} + {{y}}'), int, x=1, y=0, lm=lm.clone()
        ),
        1,
    )
    self.assertEqual(
        prompting.query('what is {{x}} + {{y}}', int, x=1, y=0, lm=lm.clone()),
        1,
    )
    self.assertEqual(
        prompting.query(
            'what is {{x}} + {{y}}',
            x=1,
            y=0,
            lm=fake.StaticResponse('The answer is one.'),
        ),
        'The answer is one.',
    )
    self.assertEqual(
        prompting.query(
            Activity.partial(),
            lm=fake.StaticResponse('Activity(description="hello")'),
        ),
        Activity(description='hello'),
    )

  def test_str_to_structure_render(self):
    lm = fake.StaticResponse('1')
    self.assert_render(
        'What is {{x}} + {{y}}?',
        int,
        x=1,
        y=2,
        lm=lm.clone(),
        expected_snippet=(
            'Please respond to the last INPUT_OBJECT with OUTPUT_OBJECT '
            'according to OUTPUT_TYPE.\n\n'
            'INPUT_OBJECT:\n  1 + 1 =\n\n'
            'OUTPUT_TYPE:\n'
            '  Answer\n\n'
            '  ```python\n'
            '  class Answer:\n'
            '    final_answer: int\n'
            '  ```\n\n'
            'OUTPUT_OBJECT:\n'
            '  ```python\n'
            '  Answer(\n'
            '    final_answer=2\n'
            '  )\n'
            '  ```\n\n'
            'INPUT_OBJECT:\n'
            '  What is 1 + 2?\n\n'
            'OUTPUT_TYPE:\n'
            '  int\n\n'
            'OUTPUT_OBJECT:'
        ),
    )

  def test_str_to_structure_render_custom_template(self):
    lm = fake.StaticResponse('1')
    self.assert_render(
        'What is {{x}} + {{y}}?',
        int,
        x=1,
        y=2,
        lm=lm.clone(),
        template_str='!!{{ DEFAULT }}!!',
        expected_snippet=(
            '!!Please respond to the last INPUT_OBJECT with OUTPUT_OBJECT '
            'according to OUTPUT_TYPE.\n\n'
            'INPUT_OBJECT:\n  1 + 1 =\n\n'
            'OUTPUT_TYPE:\n'
            '  Answer\n\n'
            '  ```python\n'
            '  class Answer:\n'
            '    final_answer: int\n'
            '  ```\n\n'
            'OUTPUT_OBJECT:\n'
            '  ```python\n'
            '  Answer(\n'
            '    final_answer=2\n'
            '  )\n'
            '  ```\n\n'
            'INPUT_OBJECT:\n'
            '  What is 1 + 2?\n\n'
            'OUTPUT_TYPE:\n'
            '  int\n\n'
            'OUTPUT_OBJECT:!!'
        ),
    )

  def test_str_to_str_render(self):
    lm = fake.StaticResponse('1')
    self.assert_render(
        'What is {{x}} + {{y}}?',
        None,
        x=1,
        y=2,
        lm=lm.clone(),
        expected_snippet='What is 1 + 2?',
        exact_match=True,
    )

  def test_structure_to_structure_render(self):
    lm = fake.StaticResponse('[1]')
    self.assert_render(
        [1],
        list[int],
        x=1,
        y=2,
        lm=lm.clone(),
        expected_snippet=(
            '\n\nINPUT_OBJECT:\n  ```python\n  [\n    1\n  ]\n  ```\n\n'
        ),
    )

  def test_structure_to_str_render(self):
    lm = fake.StaticResponse('[1]')
    self.assert_render(
        [1], None, x=1, y=2, lm=lm, expected_snippet='`[1]`', exact_match=True
    )

  def test_root_modality_to_structure_render(self):
    lm = fake.StaticResponse('1')
    self.assert_render(
        modalities.Image.from_bytes(b'mock_image'),
        int,
        lm=lm,
        expected_snippet='\n\nINPUT_OBJECT:\n  <<[[input]]>>\n\n',
        expected_modalities=1,
    )

  def test_root_modality_to_str_render(self):
    lm = fake.StaticResponse('1')
    self.assert_render(
        modalities.Image.from_bytes(b'mock_image'),
        None,
        lm=lm,
        expected_snippet='<<[[input]]>>',
        exact_match=True,
        expected_modalities=1,
    )

  def test_str_with_modality_to_str_render(self):
    lm = fake.StaticResponse('A cat and a mouse.')
    self.assert_render(
        'What are these? {{this_image}} and {{that_image}}',
        None,
        this_image=modalities.Image.from_bytes(b'cat_image'),
        that_image=modalities.Image.from_bytes(b'mouse_image'),
        lm=lm,
        expected_snippet=(
            'What are these? <<[[this_image]]>> and <<[[that_image]]>>'
        ),
        exact_match=True,
        expected_modalities=2,
    )

  def test_structure_with_modality_to_str_render(self):
    lm = fake.StaticResponse('A cat and a mouse.')
    self.assert_render(
        [
            modalities.Image.from_bytes(b'cat_image'),
            modalities.Image.from_bytes(b'mouse_image'),
        ],
        None,
        lm=lm,
        expected_snippet='`[<<[[input[0]]]>>, <<[[input[1]]]>>]`',
        exact_match=True,
        expected_modalities=2,
    )

  def test_structure_with_modality_to_structure_render(self):
    lm = fake.StaticResponse('["cat", "mouse"]')
    self.assert_render(
        [
            modalities.Image.from_bytes(b'cat_image'),
            modalities.Image.from_bytes(b'mouse_image'),
        ],
        list[str],
        lm=lm,
        expected_snippet=inspect.cleandoc("""
            INPUT_OBJECT:
              ```python
              [
                <<[[input[0]]]>>,
                <<[[input[1]]]>>
              ]
              ```
            """),
        expected_modalities=2,
    )

  def test_structure_with_modality_and_examples_to_structure_render(self):
    lm = fake.StaticResponse('["cat", "mouse"]')
    self.assert_render(
        [
            modalities.Image.from_bytes(b'cat_image'),
            modalities.Image.from_bytes(b'mouse_image'),
        ],
        list[str],
        examples=[
            mapping.MappingExample(
                input=[modalities.Image.from_bytes(b'dog_image')],
                schema=list[str],
                output=['dog'],
            ),
        ],
        lm=lm,
        expected_snippet=inspect.cleandoc("""
            INPUT_OBJECT:
              ```python
              [
                <<[[examples[0].input[0]]]>>
              ]
              ```

            OUTPUT_TYPE:
              list[str]

            OUTPUT_OBJECT:
              ```python
              [
                'dog'
              ]
              ```


            INPUT_OBJECT:
              ```python
              [
                <<[[input[0]]]>>,
                <<[[input[1]]]>>
              ]
              ```
            """),
        expected_modalities=3,
    )

  def test_bad_protocol(self):
    with self.assertRaisesRegex(ValueError, 'Unknown protocol'):
      prompting.query('what is 1 + 1', int, protocol='text')

  def test_query_prompt(self):
    self.assertEqual(
        prompting.query_prompt('what is this?', int),
        inspect.cleandoc("""
            Please respond to the last INPUT_OBJECT with OUTPUT_OBJECT according to OUTPUT_TYPE.

            INPUT_OBJECT:
              1 + 1 =

            OUTPUT_TYPE:
              Answer

              ```python
              class Answer:
                final_answer: int
              ```

            OUTPUT_OBJECT:
              ```python
              Answer(
                final_answer=2
              )
              ```

            INPUT_OBJECT:
              what is this?

            OUTPUT_TYPE:
              int

            OUTPUT_OBJECT:
            """),
    )

  def test_query_output(self):
    self.assertEqual(
        prompting.query_output(
            lf.AIMessage('1'),
            int,
        ),
        1,
    )


class QueryStructurePythonTest(unittest.TestCase):

  def test_render_no_examples(self):
    l = prompting.QueryStructurePython(
        input=lf.AIMessage('Compute 12 / 6 + 2.'), schema=int
    )
    self.assertEqual(
        l.render().text,
        inspect.cleandoc("""
            Please respond to the last INPUT_OBJECT with OUTPUT_OBJECT according to OUTPUT_TYPE.

            INPUT_OBJECT:
              1 + 1 =

            OUTPUT_TYPE:
              Answer

              ```python
              class Answer:
                final_answer: int
              ```

            OUTPUT_OBJECT:
              ```python
              Answer(
                final_answer=2
              )
              ```

            INPUT_OBJECT:
              Compute 12 / 6 + 2.

            OUTPUT_TYPE:
              int

            OUTPUT_OBJECT:
            """),
    )

  def test_render(self):
    l = prompting.QueryStructurePython(
        input=lf.AIMessage('Compute 12 / 6 + 2.'),
        schema=int,
        examples=[
            mapping.MappingExample(
                input='What is the answer of 1 plus 1?', output=2
            ),
            mapping.MappingExample(
                input='Compute the value of 3 + (2 * 6).', output=15
            ),
        ],
    )
    self.assertEqual(
        l.render().text,
        inspect.cleandoc("""
            Please respond to the last INPUT_OBJECT with OUTPUT_OBJECT according to OUTPUT_TYPE.

            INPUT_OBJECT:
              1 + 1 =

            OUTPUT_TYPE:
              Answer

              ```python
              class Answer:
                final_answer: int
              ```

            OUTPUT_OBJECT:
              ```python
              Answer(
                final_answer=2
              )
              ```

            INPUT_OBJECT:
              What is the answer of 1 plus 1?

            OUTPUT_TYPE:
              int

            OUTPUT_OBJECT:
              ```python
              2
              ```

            INPUT_OBJECT:
              Compute the value of 3 + (2 * 6).

            OUTPUT_TYPE:
              int

            OUTPUT_OBJECT:
              ```python
              15
              ```


            INPUT_OBJECT:
              Compute 12 / 6 + 2.

            OUTPUT_TYPE:
              int

            OUTPUT_OBJECT:
            """),
    )

  def test_invocation(self):
    lm_input = lf.UserMessage('3-day itineraries to San Francisco')
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
            [parse_structured_response],
        ),
        override_attrs=True,
    ):
      l = prompting.QueryStructurePython(
          input=lm_input,
          schema=[Itinerary],
          examples=[
              mapping.MappingExample(
                  input=inspect.cleandoc("""
                      Find the alternatives of expressing \"feeling great\".
                      """),
                  schema={'expression': str, 'words': list[str]},
                  output={
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

  def test_bad_response(self):
    with lf.context(
        lm=fake.StaticSequence(['a2']),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          mapping.MappingError,
          'name .* is not defined',
      ):
        prompting.query('Compute 1 + 2', int)

  def test_autofix(self):
    lm = fake.StaticSequence([
        '=1',
        inspect.cleandoc("""
            CorrectedCode(
                corrected_code='1',
            )
            """),
    ])
    self.assertEqual(prompting.query('what is 1 + 0', int, lm=lm, autofix=3), 1)

  def test_response_postprocess(self):
    with lf.context(
        lm=fake.StaticResponse('<!-- some comment-->\n3'),
        override_attrs=True,
    ):
      self.assertEqual(
          prompting.query(
              'Compute 1 + 2', response_postprocess=lambda x: x.split('\n')[1]),
          '3'
      )
      self.assertEqual(
          prompting.query(
              'Compute 1 + 2', int,
              response_postprocess=lambda x: x.split('\n')[1]),
          3
      )


class QueryStructureJsonTest(unittest.TestCase):

  def test_render_no_examples(self):
    l = prompting.QueryStructureJson(
        input=lf.AIMessage('Compute 12 / 6 + 2.'), schema=int
    )
    self.assertEqual(
        l.render().text,
        inspect.cleandoc("""
            Please respond to the last INPUT_OBJECT with JSON according to SCHEMA:

            INSTRUCTIONS:
              1. If the schema has `_type`, carry it over to the JSON output.
              2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

            INPUT_OBJECT:
              1 + 1 =

            SCHEMA:
              {"result": {"_type": "langfun.core.structured.prompting.Answer", "final_answer": int}}

            JSON:
              {"result": {"_type": "langfun.core.structured.prompting.Answer", "final_answer": 2}}

            INPUT_OBJECT:
              Compute 12 / 6 + 2.

            SCHEMA:
              {"result": int}

            JSON:
            """),
    )

  def test_render(self):
    l = prompting.QueryStructureJson(
        input=lf.AIMessage('Compute 12 / 6 + 2.'),
        schema=int,
        examples=[
            mapping.MappingExample('What is the answer of 1 plus 1?', 2),
            mapping.MappingExample('Compute the value of 3 + (2 * 6).', 15),
        ],
    )
    self.assertEqual(
        l.render().text,
        inspect.cleandoc("""
            Please respond to the last INPUT_OBJECT with JSON according to SCHEMA:

            INSTRUCTIONS:
              1. If the schema has `_type`, carry it over to the JSON output.
              2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

            INPUT_OBJECT:
              1 + 1 =

            SCHEMA:
              {"result": {"_type": "langfun.core.structured.prompting.Answer", "final_answer": int}}

            JSON:
              {"result": {"_type": "langfun.core.structured.prompting.Answer", "final_answer": 2}}

            INPUT_OBJECT:
              What is the answer of 1 plus 1?

            SCHEMA:
              {"result": int}

            JSON:
              {"result": 2}

            INPUT_OBJECT:
              Compute the value of 3 + (2 * 6).

            SCHEMA:
              {"result": int}

            JSON:
              {"result": 15}


            INPUT_OBJECT:
              Compute 12 / 6 + 2.

            SCHEMA:
              {"result": int}

            JSON:
            """),
    )

  def test_invocation(self):
    lm_input = lf.UserMessage('3-day itineraries to San Francisco')
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
            [parse_structured_response],
        ),
        override_attrs=True,
    ):
      l = prompting.QueryStructureJson(
          input=lm_input,
          schema=[Itinerary],
          examples=[
              mapping.MappingExample(
                  input=inspect.cleandoc("""
                      Find the alternatives of expressing \"feeling great\".
                      """),
                  schema={'expression': str, 'words': list[str]},
                  output={
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
        lm=fake.StaticSequence(['3']),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          mapping.MappingError,
          'No JSON dict in the output',
      ):
        prompting.query('Compute 1 + 2', int, protocol='json')

  def test_query(self):
    lm = fake.StaticSequence(['{"result": 1}'])
    self.assertEqual(
        prompting.query('what is 1 + 0', int, lm=lm, protocol='json'), 1
    )


if __name__ == '__main__':
  unittest.main()
