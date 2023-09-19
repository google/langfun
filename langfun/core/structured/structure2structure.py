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

from typing import Annotated, Any, Literal, Type

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


class Pair(pg.Object):
  """Value pair used for expressing structure-to-structure mapping."""

  left: pg.typing.Annotated[
      pg.typing.Any(transform=schema_lib.mark_missing), 'The left-side value.'
  ]
  right: pg.typing.Annotated[
      pg.typing.Any(transform=schema_lib.mark_missing), 'The right-side value.'
  ]


class StructureToStructure(mapping.Mapping):
  """Base class for structure-to-structure mapping.

  {{ preamble }}

  {% if examples -%}
  {% for example in examples -%}
  {{ input_value_title }}:
  {{ value_str(example.value.left) | indent(2, True) }}

  {%- if missing_type_dependencies(example.value) %}

  {{ type_definitions_title }}:
  {{ type_definitions_str(example.value) | indent(2, True) }}
  {%- endif %}

  {{ output_value_title }}:
  {{ value_str(example.value.right) | indent(2, True) }}

  {% endfor %}
  {% endif -%}
  {{ input_value_title }}:
  {{ value_str(input_value) | indent(2, True) }}
  {%- if missing_type_dependencies(input_value) %}

  {{ type_definitions_title }}:
  {{ type_definitions_str(input_value) | indent(2, True) }}
  {%- endif %}

  {{ output_value_title }}:
  """

  default: Annotated[
      Any,
      (
          'The default value to use if mapping failed. '
          'If unspecified, error will be raisen.'
      ),
  ] = lf.message_transform.RAISE_IF_HAS_ERROR

  preamble: Annotated[
      lf.LangFunc,
      'Preamble used for structure-to-structure mapping.',
  ]

  type_definitions_title: Annotated[
      str, 'The section title for type definitions.'
  ] = 'CLASS_DEFINITIONS'

  input_value_title: Annotated[str, 'The section title for input value.']
  output_value_title: Annotated[str, 'The section title for output value.']

  def _on_bound(self):
    super()._on_bound()
    if self.examples:
      for example in self.examples:
        if not isinstance(example.value, Pair):
          raise ValueError(
              'The value of example must be a `lf.structured.Pair` object. '
              f'Encountered: { example.value }.'
          )

  @property
  def input_value(self) -> Any:
    return schema_lib.mark_missing(self.message.result)

  def value_str(self, value: Any) -> str:
    return schema_lib.value_repr('python').repr(value, compact=False)

  def missing_type_dependencies(self, value: Any) -> list[Type[Any]]:
    value_specs = tuple(
        [v.value_spec for v in schema_lib.Missing.find_missing(value).values()]
    )
    return schema_lib.class_dependencies(value_specs, include_subclasses=True)

  def type_definitions_str(self, value: Any) -> str | None:
    return schema_lib.class_definitions(
        self.missing_type_dependencies(value), markdown=True
    )

  def _value_context(self):
    classes = schema_lib.class_dependencies(self.input_value)
    return {cls.__name__: cls for cls in classes}

  def transform_output(self, lm_output: lf.Message) -> lf.Message:
    try:
      result = schema_lib.value_repr('python').parse(
          lm_output.text, additional_context=self._value_context()
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      if self.default == lf.message_transform.RAISE_IF_HAS_ERROR:
        raise mapping.MappingError(
            'Cannot parse message text into structured output. '
            f'Error={e}. Text={lm_output.text!r}.'
        ) from e
      result = self.default
    lm_output.result = result
    return lm_output


class CompleteStructure(StructureToStructure):
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


def _default_complete_examples() -> list[mapping.MappingExample]:
  return [
      mapping.MappingExample(
          value=Pair(
              left=_Country.partial(name='United States of America'),
              right=_Country(
                  name='United States of America',
                  founding_date=_Date(year=1776, month=7, day=4),
                  continent='North America',
                  population=33_000_000,
                  hobby=schema_lib.UNKNOWN,
              ),
          )
      )
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
    examples = _default_complete_examples()
  t = CompleteStructure(default=default, examples=examples, **kwargs)
  return t.transform(message=lf.UserMessage(text='', result=value)).result
