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
"""Tests for structured description."""

import inspect
import unittest

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import structure2nl
import pyglove as pg


class Activity(pg.Object):
  description: str


class Itinerary(pg.Object):
  day: pg.typing.Int[1, None]
  type: pg.typing.Enum['daytime', 'nighttime']
  activities: list[Activity]
  hotel: pg.typing.Str['.*Hotel'] | None


class DescribeStructureTest(unittest.TestCase):

  def test_render(self):
    l = structure2nl.DescribeStructure(
        examples=[
            mapping.MappingExample(
                nl_context='Compute 1 + 2',
                nl_text='The result of 1 + 2 is 3',
                value=3,
            ),
            mapping.MappingExample(
                nl_context='Best activity to do in New York city.',
                nl_text=(
                    'The best thing to do in New York city is to watch shows '
                    'in Broadway threatres.'
                ),
                value=Activity('Visit Broadway threatres for shows'),
            ),
        ]
    )
    m = lf.UserMessage(
        '1 day itinerary to SF',
        result=Itinerary(
            day=1,
            type='daytime',
            activities=[
                Activity('Visit Golden Gate Bridge.'),
                Activity("Visit Fisherman's Wharf."),
                Activity('Visit Alcatraz Island.'),
            ],
            hotel=None,
        ),
    )

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help describe PYTHON_OBJECT in natural language.

            INSTRUCTIONS:
              1. Do not add details which are not present in the object.
              2. If a field in the object has None as its value, do not mention it.

            CONTEXT_FOR_DESCRIPTION:
              Compute 1 + 2

            PYTHON_OBJECT:
              3

            NATURAL_LANGUAGE_TEXT:
              The result of 1 + 2 is 3

            CONTEXT_FOR_DESCRIPTION:
              Best activity to do in New York city.

            PYTHON_OBJECT:
              Activity(
                description='Visit Broadway threatres for shows'
              )

            NATURAL_LANGUAGE_TEXT:
              The best thing to do in New York city is to watch shows in Broadway threatres.


            CONTEXT_FOR_DESCRIPTION:
              1 day itinerary to SF

            PYTHON_OBJECT:
              Itinerary(
                day=1,
                type='daytime',
                activities=[
                  Activity(
                    description='Visit Golden Gate Bridge.'
                  ),
                  Activity(
                    description='Visit Fisherman's Wharf.'
                  ),
                  Activity(
                    description='Visit Alcatraz Island.'
                  )
                ],
                hotel=None
              )

            NATURAL_LANGUAGE_TEXT:
            """),
    )

  def test_render_no_examples(self):
    l = structure2nl.DescribeStructure()
    m = lf.UserMessage(
        '1 day itinerary to SF',
        result=Itinerary(
            day=1,
            type='daytime',
            activities=[
                Activity('Visit Golden Gate Bridge.'),
                Activity("Visit Fisherman's Wharf."),
                Activity('Visit Alcatraz Island.'),
            ],
            hotel=None,
        ),
    )

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help describe PYTHON_OBJECT in natural language.

            INSTRUCTIONS:
              1. Do not add details which are not present in the object.
              2. If a field in the object has None as its value, do not mention it.

            CONTEXT_FOR_DESCRIPTION:
              1 day itinerary to SF

            PYTHON_OBJECT:
              Itinerary(
                day=1,
                type='daytime',
                activities=[
                  Activity(
                    description='Visit Golden Gate Bridge.'
                  ),
                  Activity(
                    description='Visit Fisherman's Wharf.'
                  ),
                  Activity(
                    description='Visit Alcatraz Island.'
                  )
                ],
                hotel=None
              )

            NATURAL_LANGUAGE_TEXT:
            """),
    )

  def test_render_no_context(self):
    l = structure2nl.DescribeStructure()
    m = lf.UserMessage(
        '',
        result=Itinerary(
            day=1,
            type='daytime',
            activities=[
                Activity('Visit Golden Gate Bridge.'),
                Activity("Visit Fisherman's Wharf."),
                Activity('Visit Alcatraz Island.'),
            ],
            hotel=None,
        ),
    )

    self.assertEqual(
        l.render(message=m).text,
        inspect.cleandoc("""
            Please help describe PYTHON_OBJECT in natural language.

            INSTRUCTIONS:
              1. Do not add details which are not present in the object.
              2. If a field in the object has None as its value, do not mention it.

            PYTHON_OBJECT:
              Itinerary(
                day=1,
                type='daytime',
                activities=[
                  Activity(
                    description='Visit Golden Gate Bridge.'
                  ),
                  Activity(
                    description='Visit Fisherman's Wharf.'
                  ),
                  Activity(
                    description='Visit Alcatraz Island.'
                  )
                ],
                hotel=None
              )

            NATURAL_LANGUAGE_TEXT:
            """),
    )

  def test_describe(self):
    lm = fake.StaticSequence(
        [
            (
                "On the first day, visit the Golden Gate Bridge, Fisherman's"
                ' Wharf, and Alcatraz Island.'
            )
        ]
    )
    self.assertEqual(
        structure2nl.describe(
            Itinerary(
                day=1,
                type='daytime',
                activities=[
                    Activity('Visit Golden Gate Bridge.'),
                    Activity("Visit Fisherman's Wharf."),
                    Activity('Visit Alcatraz Island.'),
                ],
                hotel=None,
            ),
            '1 day itinerary to SF',
            lm=lm,
        ),
        (
            "On the first day, visit the Golden Gate Bridge, Fisherman's Wharf,"
            ' and Alcatraz Island.'
        ),
    )


if __name__ == '__main__':
  unittest.main()
