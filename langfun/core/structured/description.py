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
"""Symbolic description."""

import inspect
from typing import Any, Literal

import langfun.core as lf
from langfun.core.structured import mapping
import pyglove as pg


@pg.use_init_args(['examples'])
class DescribeStructure(mapping.Mapping):
  """Describe a structured value in natural language."""

  input_title = 'PYTHON_OBJECT'
  context_title = 'CONTEXT_FOR_DESCRIPTION'
  output_title = 'NATURAL_LANGUAGE_TEXT'

  preamble = """
      Please help describe {{ input_title }} in natural language.

      INSTRUCTIONS:
        1. Do not add details which are not present in the object.
        2. If a field in the object has None as its value, do not mention it.
      """


def describe(
    value: Any,
    context: str | None = None,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    cache_seed: int | None = 0,
    **kwargs,
) -> str:
  """Describes a structured value using natural language.

  Examples:

    ```
    class FlightDuration(pg.Object):
      hours: int
      minutes: int

    class Flight(pg.Object):
      airline: str
      flight_number: str
      departure_airport: str
      arrival_airport: str
      departure_time: str
      arrival_time: str
      duration: FlightDuration
      stops: int
      price: float

    text = lf.describe(
        Flight(
            airline='United Airlines',
            flight_number='UA2631',
            depature_airport: 'SFO',
            arrival_airport: 'JFK',
            depature_time: '2023-09-07T05:15:00',
            arrival_time: '2023-09-07T12:12:00',
            duration: FlightDuration(
                hours=7,
                minutes=57
            ),
            stops=1,
            price=227,
        ))
    print(text)

    >> The flight is operated by United Airlines, has the flight number UA2631,
    >> departs from San Francisco International Airport (SFO), arrives at John
    >> F. Kennedy International Airport (JFK), It departs at
    >> 2023-09-07T05:15:00, arrives at 2023-09-07T12:12:00, has a duration of 7
    >> hours and 57 minutes, makes 1 stop, and costs $227.
    ```

  Args:
    value: A structured value to be mapped.
    context: The context information for describing the structured value.
    lm: The language model to use. If not specified, the language model from
      `lf.context` context manager will be used.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    cache_seed: Seed for computing cache key. The cache key is determined by a
      tuple of (lm, prompt, cache seed). If None, cache will be disabled for
      the query even cache is configured by the LM.
    **kwargs: Keyword arguments passed to the `lf.structured.DescribeStructure`.

  Returns:
    The parsed result based on the schema.
  """
  return DescribeStructure(
      input=value,
      context=context,
      examples=examples or default_describe_examples(),
      **kwargs,
  )(lm=lm, cache_seed=cache_seed).text


def default_describe_examples() -> list[mapping.MappingExample]:
  """Default describe examples."""

  class Country(pg.Object):
    """A example dataclass for structured mapping."""

    name: str
    continents: list[
        Literal[
            'Africa',
            'Asia',
            'Europe',
            'Oceania',
            'North America',
            'South America',
        ]
    ]
    num_states: int
    neighbor_countries: list[str]
    population: int
    capital: str | None
    president: str | None

  return [
      mapping.MappingExample(
          context='Brief intro to United States',
          input=Country(
              name='The United States of America',
              continents=['North America'],
              num_states=50,
              neighbor_countries=[
                  'Canada',
                  'Mexico',
                  'Bahamas',
                  'Cuba',
                  'Russia',
              ],
              population=333000000,
              capital='Washington, D.C',
              president=None,
          ),
          output=inspect.cleandoc("""
              The United States of America is a country primarily located in North America
              consisting of fifty states. It shares land borders with Canada to its north
              and with Mexico to its south and has maritime borders with the Bahamas, Cuba,
              Russia, and other nations. With a population of over 333 million. The national
              capital of the United States is Washington, D.C.
              """),
      ),
  ]
