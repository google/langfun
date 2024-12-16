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
    l = completion._CompleteStructure()
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
        l.render(input=input_value).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              Answer(
                question='1 + 1 =',
                answer=MISSING(int)
              )
              ```

            OUTPUT_OBJECT:
              ```python
              Answer(
                question='1 + 1 =',
                answer=2
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

  def test_render_no_class_definitions(self):
    l = completion._CompleteStructure()
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
        l.render(input=input_value).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              Answer(
                question='1 + 1 =',
                answer=MISSING(int)
              )
              ```

            OUTPUT_OBJECT:
              ```python
              Answer(
                question='1 + 1 =',
                answer=2
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
    l = completion._CompleteStructure()
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
        l.render(input=input_value).text,
        inspect.cleandoc("""
          Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

          INSTRUCTIONS:
          1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
          2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

          INPUT_OBJECT:
            ```python
            Answer(
              question='1 + 1 =',
              answer=MISSING(int)
            )
            ```

          OUTPUT_OBJECT:
            ```python
            Answer(
              question='1 + 1 =',
              answer=2
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

          OUTPUT_OBJECT:"""),
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

    input_value = schema_lib.mark_missing(
        Animal.partial(
            modalities.Image.from_bytes(b'image_of_elephant'),
        )
    )
    l = completion._CompleteStructure(
        input=input_value,
        examples=[
            mapping.MappingExample(
                input=Animal.partial(
                    modalities.Image.from_bytes(b'image_of_rabbit')
                ),
                output=Animal(
                    modalities.Image.from_bytes(b'image_of_rabbit'),
                    'rabbit',
                ),
            )
        ],
    )
    lm_input = l.render()
    self.maxDiff = None
    self.assertEqual(
        lm_input.text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the last INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.

            INPUT_OBJECT:
              ```python
              Answer(
                question='1 + 1 =',
                answer=MISSING(int)
              )
              ```

            OUTPUT_OBJECT:
              ```python
              Answer(
                question='1 + 1 =',
                answer=2
              )
              ```

            INPUT_OBJECT:
              ```python
              Animal(
                image=ModalityRef(
                  name='examples[0].input.image'
                ),
                name=MISSING(str)
              )
              ```

            MODALITY_REFERENCES:
              {
                'examples[0].input.image': <<[[examples[0].input.image]]>>
              }

            OUTPUT_OBJECT:
              ```python
              Animal(
                image=ModalityRef(
                  name='examples[0].output.image'
                ),
                name='rabbit'
              )
              ```


            INPUT_OBJECT:
              ```python
              Animal(
                image=ModalityRef(
                  name='input.image'
                ),
                name=MISSING(str)
              )
              ```

            MODALITY_REFERENCES:
              {
                'input.image': <<[[input.image]]>>
              }

            OUTPUT_OBJECT:
            """),
    )
    self.assertTrue(
        pg.eq(
            {
                'examples': lm_input.get('examples'),
                'input': lm_input.get('input'),
            },
            {
                'examples': [
                    mapping.MappingExample(
                        input=Animal.partial(
                            image=modalities.Image.from_bytes(
                                b'image_of_rabbit'
                            )
                        ),
                        output=Animal.partial(
                            image=modalities.Image.from_bytes(
                                b'image_of_rabbit'
                            ),
                            name='rabbit',
                        ),
                    )
                ],
                'input': Animal(
                    image=modalities.Image.from_bytes(b'image_of_elephant'),
                    name=schema_lib.MISSING,
                ),
            },
        )
    )
    lm_output = l(
        input=input_value,
        lm=fake.StaticResponse(inspect.cleandoc("""
            ```python
            Animal(
              image=ModalityRef(
                name='input.image'
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
            CorrectedCode(
                corrected_code='Solution(question=\\\'Compute 1 + 1\\\', answer=2)',
            )
            """),
        ],
        debug=True,
    )
    self.assertEqual(
        completion.complete(
            Solution.partial('Compute 1 + 1'), lm=lm, autofix=3
        ),
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
            is_cached=False,
            logprobs=None,
            usage=lf.LMSamplingUsage(553, 27, 580),
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

  def test_bad_call(self):
    with self.assertRaisesRegex(
        ValueError, '.*must contain a least .* missing'
    ):
      completion.complete(Activity('foo'), lm=fake.StaticResponse(''))

  def test_bad_transform(self):
    with lf.context(
        lm=fake.StaticSequence(['Activity(description=1)']),
        override_attrs=True,
    ):
      with self.assertRaisesRegex(
          mapping.MappingError,
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
