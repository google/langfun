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
"""Tests for langfun.core.structured.completion."""

import inspect
import unittest

import langfun.core as lf
from langfun.core import coding
from langfun.core import modalities
from langfun.core.llms import fake
from langfun.core.structured import completion
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  day: pg.typing.Int[1, None]
  type: pg.typing.Annotated[
      pg.typing.Enum['daytime', 'nighttime'],
      'Type of itinerary.'
  ]
  activities: list[Activity]


class TripPlan(pg.Object):
  place: str
  itineraries: list[Itinerary]


class CompleteStructureTest(unittest.TestCase):

  def test_render_no_examples(self):
    l = completion.CompleteStructure()
    input_value = schema_lib.mark_missing(
        TripPlan.partial(
            place='San Francisco',
            itineraries=[
                Itinerary.partial(day=1),
                Itinerary.partial(day=2),
                Itinerary.partial(day=3),
            ],
        )
    )
    self.assertEqual(
        l.render(input_value=input_value).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              TripPlan(
                place='San Francisco',
                itineraries=[
                  Itinerary(
                    day=1,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity])
                  ),
                  Itinerary(
                    day=2,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity])
                  ),
                  Itinerary(
                    day=3,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity])
                  )
                ]
              )
              ```

            CLASS_DEFINITIONS:
              ```python
              class Activity:
                description: str
              ```

            OUTPUT_OBJECT:
            """),
    )

  def test_render_no_class_definitions(self):
    l = completion.CompleteStructure()
    input_value = schema_lib.mark_missing(
        TripPlan.partial(
            place='San Francisco',
            itineraries=[
                Itinerary.partial(day=1, activities=[Activity.partial()]),
                Itinerary.partial(day=2, activities=[Activity.partial()]),
                Itinerary.partial(day=3, activities=[Activity.partial()]),
            ],
        )
    )
    self.assertEqual(
        l.render(input_value=input_value).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              TripPlan(
                place='San Francisco',
                itineraries=[
                  Itinerary(
                    day=1,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=[
                      Activity(
                        description=MISSING(str)
                      )
                    ]
                  ),
                  Itinerary(
                    day=2,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=[
                      Activity(
                        description=MISSING(str)
                      )
                    ]
                  ),
                  Itinerary(
                    day=3,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=[
                      Activity(
                        description=MISSING(str)
                      )
                    ]
                  )
                ]
              )
              ```

            OUTPUT_OBJECT:
            """),
    )

  def test_render_with_examples(self):
    l = completion.CompleteStructure(
        examples=completion.DEFAULT_COMPLETE_EXAMPLES
    )
    input_value = schema_lib.mark_missing(
        TripPlan.partial(
            place='San Francisco',
            itineraries=[
                Itinerary.partial(day=1),
                Itinerary.partial(day=2),
                Itinerary.partial(day=3),
            ],
        )
    )
    self.assertEqual(
        l.render(input_value=input_value).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              _Country(
                name='United States of America',
                founding_date=MISSING(_Date),
                continent=MISSING(Literal['Africa', 'Asia', 'Europe', 'Oceania', 'North America', 'South America']),
                population=MISSING(int)
              )
              ```

            CLASS_DEFINITIONS:
              ```python
              class _Date:
                year: int
                month: int
                day: int
              ```

            OUTPUT_OBJECT:
              ```python
              _Country(
                name='United States of America',
                founding_date=_Date(
                  year=1776,
                  month=7,
                  day=4
                ),
                continent='North America',
                population=33000000
              )
              ```


            INPUT_OBJECT:
              ```python
              TripPlan(
                place='San Francisco',
                itineraries=[
                  Itinerary(
                    day=1,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity])
                  ),
                  Itinerary(
                    day=2,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity])
                  ),
                  Itinerary(
                    day=3,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity])
                  )
                ]
              )
              ```

            CLASS_DEFINITIONS:
              ```python
              class Activity:
                description: str
              ```

            OUTPUT_OBJECT:
            """),
    )

  def test_invocation(self):
    structured_response = inspect.cleandoc("""
        ```python
        TripPlan(
          place='San Francisco',
          itineraries=[
            Itinerary(
                day=1,
                type='daytime',
                activities=[
                    Activity(description='Arrive in San Francisco and check into your hotel.'),
                    Activity(description='Take a walk around Fisherman\\'s Wharf and have dinner at one of the many seafood restaurants.'),
                    Activity(description='Visit Pier 39 and see the sea lions.'),
                ], 
                ),
            Itinerary(
                day=2,
                type='daytime',
                activities=[
                    Activity(description='Take a ferry to Alcatraz Island and tour the infamous prison.'),
                    Activity(description='Take a walk across the Golden Gate Bridge.'),
                    Activity(description='Visit the Japanese Tea Garden in Golden Gate Park.'),
                ], 
                ),
            Itinerary(
                day=3,
                type='daytime',
                activities=[
                    Activity(description='Visit the de Young Museum and see the collection of American art.'),
                    Activity(description='Visit the San Francisco Museum of Modern Art.'),
                    Activity(description='Take a cable car ride.'),
                ], 
                ),
          ]
        )
        ```
        """)

    with lf.context(
        lm=fake.StaticSequence(
            [structured_response],
        ),
        override_attrs=True,
    ):
      r = completion.complete(
          TripPlan.partial(
              place='San Francisco',
              itineraries=[
                  Itinerary.partial(day=1),
                  Itinerary.partial(day=2),
                  Itinerary.partial(day=3),
              ],
          )
      )
      # pylint: disable=line-too-long
      self.assertEqual(
          r,
          TripPlan(
              place='San Francisco',
              itineraries=[
                  Itinerary(
                      day=1,
                      type='daytime',
                      activities=[
                          Activity(
                              description=(
                                  'Arrive in San Francisco and check into your'
                                  ' hotel.'
                              )
                          ),
                          Activity(
                              description=(
                                  "Take a walk around Fisherman's Wharf and"
                                  ' have dinner at one of the many seafood'
                                  ' restaurants.'
                              )
                          ),
                          Activity(
                              description='Visit Pier 39 and see the sea lions.'
                          ),
                      ],
                  ),
                  Itinerary(
                      day=2,
                      type='daytime',
                      activities=[
                          Activity(
                              description=(
                                  'Take a ferry to Alcatraz Island and tour the'
                                  ' infamous prison.'
                              )
                          ),
                          Activity(
                              description=(
                                  'Take a walk across the Golden Gate Bridge.'
                              )
                          ),
                          Activity(
                              description=(
                                  'Visit the Japanese Tea Garden in Golden Gate'
                                  ' Park.'
                              )
                          ),
                      ],
                  ),
                  Itinerary(
                      day=3,
                      type='daytime',
                      activities=[
                          Activity(
                              description=(
                                  'Visit the de Young Museum and see the'
                                  ' collection of American art.'
                              )
                          ),
                          Activity(
                              description=(
                                  'Visit the San Francisco Museum of Modern'
                                  ' Art.'
                              )
                          ),
                          Activity(description='Take a cable car ride.'),
                      ],
                  ),
              ],
          ),
      )
      # pylint: enable=line-too-long

  def test_invocation_with_modality(self):
    class Animal(pg.Object):
      image: modalities.Image
      name: str

    l = completion.CompleteStructure(
        examples=[
            mapping.MappingExample(
                value=mapping.Pair(
                    left=Animal.partial(
                        modalities.Image.from_bytes(b'image_of_rabbit')
                    ),
                    right=Animal(
                        modalities.Image.from_bytes(b'image_of_rabbit'),
                        'rabbit',
                    ),
                )
            )
        ]
    )
    input_value = schema_lib.mark_missing(
        Animal.partial(
            modalities.Image.from_bytes(b'image_of_elephant'),
        )
    )
    lm_input = l.render(input_value=input_value)
    self.assertEqual(
        lm_input.text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              Animal(
                image=ModalityRef(
                  name='image'
                ),
                name=MISSING(str)
              )
              ```

            MODALITY_REFS:
              {
                'image': {{examples[0].value.left.image}}
              }

            OUTPUT_OBJECT:
              ```python
              Animal(
                image=ModalityRef(
                  name='image'
                ),
                name='rabbit'
              )
              ```


            INPUT_OBJECT:
              ```python
              Animal(
                image=ModalityRef(
                  name='image'
                ),
                name=MISSING(str)
              )
              ```

            MODALITY_REFS:
              {
                'image': {{image}}
              }

            OUTPUT_OBJECT:
            """),
    )
    self.assertTrue(
        pg.eq(
            {
                'examples': lm_input.get('examples'),
                'image': lm_input.get('image'),
            },
            {
                'examples': [
                    {
                        'value': {
                            'left': {
                                'image': modalities.Image.from_bytes(
                                    b'image_of_rabbit'
                                )
                            },
                            'right': {
                                'image': modalities.Image.from_bytes(
                                    b'image_of_rabbit'
                                )
                            },
                        }
                    }
                ],
                'image': modalities.Image.from_bytes(b'image_of_elephant'),
            },
        )
    )
    lm_output = l(
        input_value=input_value,
        lm=fake.StaticResponse(inspect.cleandoc("""
            ```python
            Animal(
              image=ModalityRef(
                name='image'
              ),
              name='elephant'
            )
            ```
            """)),
    )
    self.assertTrue(
        pg.eq(
            lm_output.result,
            Animal(
                image=modalities.Image.from_bytes(b'image_of_elephant'),
                name='elephant',
            ),
        )
    )

  def test_autofix(self):
    class Solution(pg.Object):
      question: str
      answer: int

    lm = fake.StaticSequence(
        [
            "Solution(question='Compute 1 + 1', answer=2",
            inspect.cleandoc("""
            CodeCorrection(
                latest_code=CodeWithError(
                    code='Solution(question=\\\'Compute 1 + 1\\\', answer=2',
                    error='SyntaxError: incomplete input (<unknown> line 1)\\n: Solution(question=\\\'Compute 1 + 1\\\', answer=2'
                ),
                correction_history=[],
                corrected_code='Solution(question=\\\'Compute 1 + 1\\\', answer=2)',
            )
            """),
        ],
        debug=True,
    )
    self.assertEqual(
        completion.complete(Solution.partial('Compute 1 + 1'), lm=lm),
        Solution(question='Compute 1 + 1', answer=2),
    )

  def test_returns_message(self):
    self.assertEqual(
        completion.complete(
            Activity.partial(),
            lm=fake.StaticSequence(['Activity(description="foo")']),
            returns_message=True),
        lf.AIMessage(
            text='Activity(description="foo")',
            result=Activity(description='foo'),
            score=1.0,
            tags=['lm-response', 'lm-output', 'transformed']
        )
    )

  def test_using_the_same_lm_instance(self):
    lm = fake.StaticSequence(['Activity(description="foo")'])
    self.assertEqual(
        completion.complete(Activity.partial(), lm=lm), Activity('foo')
    )
    with self.assertRaises(IndexError):
      completion.complete(Activity.partial(), lm=lm)

  def test_bad_init(self):
    with self.assertRaisesRegex(ValueError, '.*must be.*Pair'):
      completion.CompleteStructure(examples=[mapping.MappingExample(value=1)])

  def test_bad_transform(self):
    with lf.context(
        lm=fake.StaticSequence(['Activity(description=1)']),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          coding.CodeError,
          'Expect .* but encountered .*',
      ):
        completion.complete(Activity.partial(), autofix=0)

  def test_default(self):
    with lf.context(
        lm=fake.StaticSequence(['Activity(description=1)']),
        override_attrs=True,
    ):
      self.assertIsNone(completion.complete(Activity.partial(), None))


if __name__ == '__main__':
  unittest.main()
