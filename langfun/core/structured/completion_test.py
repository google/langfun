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
from langfun.core.llms import fake
from langfun.core.structured import completion
from langfun.core.structured import mapping
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
  hotel: pg.typing.Str['.*Hotel'] | None


class TripPlan(pg.Object):
  place: str
  itineraries: list[Itinerary]


class CompleteStructureTest(unittest.TestCase):

  def test_render_no_examples(self):
    l = completion.CompleteStructure()
    m = lf.UserMessage(
        '',
        result=TripPlan.partial(
            place='San Francisco',
            itineraries=[
                Itinerary.partial(day=1),
                Itinerary.partial(day=2),
                Itinerary.partial(day=3),
            ],
        ),
    )

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.
            3. If a MISSING field could not be filled, mark it as `UNKNOWN`.

            INPUT_OBJECT:
              ```python
              TripPlan(
                place='San Francisco',
                itineraries=[
                  Itinerary(
                    day=1,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity]),
                    hotel=None
                  ),
                  Itinerary(
                    day=2,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity]),
                    hotel=None
                  ),
                  Itinerary(
                    day=3,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity]),
                    hotel=None
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
    m = lf.UserMessage(
        '',
        result=TripPlan.partial(
            place='San Francisco',
            itineraries=[
                Itinerary.partial(day=1, activities=[Activity.partial()]),
                Itinerary.partial(day=2, activities=[Activity.partial()]),
                Itinerary.partial(day=3, activities=[Activity.partial()]),
            ],
        ),
    )

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.
            3. If a MISSING field could not be filled, mark it as `UNKNOWN`.

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
                    ],
                    hotel=None
                  ),
                  Itinerary(
                    day=2,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=[
                      Activity(
                        description=MISSING(str)
                      )
                    ],
                    hotel=None
                  ),
                  Itinerary(
                    day=3,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=[
                      Activity(
                        description=MISSING(str)
                      )
                    ],
                    hotel=None
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
    m = lf.UserMessage(
        '',
        result=TripPlan.partial(
            place='San Francisco',
            itineraries=[
                Itinerary.partial(day=1),
                Itinerary.partial(day=2),
                Itinerary.partial(day=3),
            ],
        ),
    )

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please generate the OUTPUT_OBJECT by completing the MISSING fields from the INPUT_OBJECT.

            INSTRUCTIONS:
            1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
            2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.
            3. If a MISSING field could not be filled, mark it as `UNKNOWN`.

            INPUT_OBJECT:
              ```python
              _Country(
                name='United States of America',
                founding_date=MISSING(_Date),
                continent=MISSING(Literal['Africa', 'Asia', 'Europe', 'Oceania', 'North America', 'South America']),
                population=MISSING(int),
                hobby=MISSING(str)
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
                population=33000000,
                hobby=UNKNOWN
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
                    activities=MISSING(list[Activity]),
                    hotel=None
                  ),
                  Itinerary(
                    day=2,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity]),
                    hotel=None
                  ),
                  Itinerary(
                    day=3,
                    # Type of itinerary.
                    type=MISSING(Literal['daytime', 'nighttime']),
                    activities=MISSING(list[Activity]),
                    hotel=None
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

  def test_transform(self):
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
                      hotel=None,
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
                      hotel=None,
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
                      hotel=None,
                  ),
              ],
          ),
      )
      # pylint: enable=line-too-long

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
        completion.complete(Activity.partial())

  def test_default(self):
    with lf.context(
        lm=fake.StaticSequence(['Activity(description=1)']),
        override_attrs=True,
    ):
      self.assertIsNone(completion.complete(Activity.partial(), None))


if __name__ == '__main__':
  unittest.main()
