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
"""Structure-to-structure mappings."""

from typing import Any, Literal

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class CompleteStructure(mapping.StructureToStructure):
  """Complete structure by filling the missing fields."""

  preamble = lf.LangFunc("""
      Please generate the OUTPUT_OBJECT by completing the MISSING fields from the INPUT_OBJECT.

      INSTRUCTIONS:
      1. Each MISSING field contains a Python annotation, please fill the value based on the annotation.
      2. Classes for the MISSING fields are defined under CLASS_DEFINITIONS.
      3. If a MISSING field could not be filled, mark it as `UNKNOWN`.
      """)

  # NOTE(daiyip): Set the input path of the transform to root, so this transform
  # could access the input via the `message.result` field.
  input_path = ''

  input_value_title = 'INPUT_OBJECT'
  output_value_title = 'OUTPUT_OBJECT'


class _Date(pg.Object):
  year: int
  month: int
  day: int


class _Country(pg.Object):
  """Country."""

  name: str
  founding_date: _Date
  continent: Literal[
      'Africa', 'Asia', 'Europe', 'Oceania', 'North America', 'South America'
  ]
  population: int
  hobby: str


DEFAULT_COMPLETE_EXAMPLES: list[mapping.MappingExample] = [
    mapping.MappingExample(
        value=mapping.Pair(
            left=_Country.partial(name='United States of America'),
            right=_Country(
                name='United States of America',
                founding_date=_Date(year=1776, month=7, day=4),
                continent='North America',
                population=33_000_000,
                hobby=schema_lib.UNKNOWN,
            ),
        )
    ),
]


def complete(
    value: pg.Symbolic,
    default: Any = lf.message_transform.RAISE_IF_HAS_ERROR,
    *,
    examples: list[mapping.MappingExample] | None = None,
    **kwargs,
) -> Any:
  """Complete a symbolic value by filling its missing fields.

  Examples:

    ```
    class FlightDuration:
      hours: int
      minutes: int

    class Flight(pg.Object):
      airline: str
      flight_number: str
      departure_airport_code: str
      arrival_airport_code: str
      departure_time: str
      arrival_time: str
      duration: FlightDuration
      stops: int
      price: float

    prompt = '''
      Information about flight UA2631.
      '''

    r = lf.query(prompt, Flight)
    assert isinstance(r, Flight)
    assert r.airline == 'United Airlines'
    assert r.departure_airport_code == 'SFO'
    assert r.duration.hour = 7
    ```

  Args:
    value: A symbolic value that may contain missing values.
    default: The default value if parsing failed. If not specified, error will
      be raised.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    **kwargs: Keyword arguments passed to the
      `lf.structured.NaturalLanguageToStructureed` transform, e.g. `lm` for
      specifying the language model for structured parsing.

  Returns:
    The result based on the schema.
  """
  if examples is None:
    examples = DEFAULT_COMPLETE_EXAMPLES
  t = CompleteStructure(default=default, examples=examples, **kwargs)
  return t.transform(message=lf.UserMessage(text='', result=value)).result
