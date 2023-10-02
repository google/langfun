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
"""Natural language text to structured value."""

import inspect
from typing import Annotated, Any, Callable, Literal, Type, Union

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


@lf.use_init_args(['schema', 'default', 'examples'])
class ParseStructure(mapping.NaturalLanguageToStructure):
  """Parse an object out from a natural language text."""

  # Customize the source of the text to be mapped and its context.
  nl_context_getter: Annotated[
      Callable[[lf.Message], str] | None,
      (
          'A callable object to get the request text from the message. '
          'If None, it returns the entire LM input message ('
          '`message.lm_input.text`).'
      ),
  ] = None

  nl_text_getter: Annotated[
      Callable[[lf.Message], str] | None,
      (
          'A callable object to get the text to be mapped from the message. '
          'If None, it returns the entire LM output message (`message.text`).'
      ),
  ] = None

  @property
  def nl_context(self) -> str | None:
    """Returns the user request."""
    if self.nl_context_getter is None:
      return self.message.lm_input.text if self.message.lm_input else None
    return self.nl_context_getter(self.message)    # pylint: disable=not-callable

  @property
  def nl_text(self) -> str:
    """Returns the LM response."""
    if self.nl_text_getter is None:
      # This allows external output transform chaining.
      if isinstance(self.message.result, str):
        return self.message.result
      return self.message.text
    return self.response_getter(self.message)    # pylint: disable=not-callable


class ParseStructureJson(ParseStructure):
  """Parse an object out from a NL text using JSON as the protocol."""

  preamble = """
      Please help translate the LM response into JSON based on the request and the schema:

      INSTRUCTIONS:
        1. If the schema has `_type`, carry it over to the JSON output.
        2. If a field from the schema cannot be extracted from the response, use null as the JSON value.
      """

  protocol = 'json'
  schema_title = 'SCHEMA'
  value_title = 'JSON'


class ParseStructurePython(ParseStructure):
  """Parse an object out from a NL text using Python as the protocol."""

  preamble = """
      Please help translate {{ nl_text_title }} into {{ value_title}} based on {{ schema_title }}.
      Both {{ schema_title }} and {{ value_title }} are described in Python.
      """

  protocol = 'python'
  schema_title = 'RESULT_TYPE'
  value_title = 'RESULT_OBJECT'


def parse(
    message: Union[lf.Message, str],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any]
    ],
    default: Any = lf.message_transform.RAISE_IF_HAS_ERROR,
    *,
    user_prompt: str | None = None,
    examples: list[mapping.MappingExample] | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    **kwargs,
) -> Any:
  """Parse a natural langugage message based on schema.

  Examples:

    ```
    class FlightDuration(pg.Object):
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

    input = '''
      The flight is operated by United Airlines, has the flight number UA2631,
      departs from San Francisco International Airport (SFO), arrives at John
      F. Kennedy International Airport (JFK), It departs at 2023-09-07T05:15:00,
      arrives at 2023-09-07T12:12:00, has a duration of 7 hours and 57 minutes,
      makes 1 stop, and costs $227.
      '''

    r = lf.parse(input, Flight)
    assert isinstance(r, Flight)
    assert r.airline == 'United Airlines'
    assert r.departure_airport_code == 'SFO'
    assert r.duration.hour = 7
    ```

  Args:
    message: A `lf.Message` object  or a string as the natural language input.
    schema: A `lf.transforms.ParsingSchema` object or equivalent annotations.
    default: The default value if parsing failed. If not specified, error will
      be raised.
    user_prompt: An optional user prompt as the description or ask for the
      message, which provide more context for parsing.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    protocol: The protocol for schema/value representation. Applicable values
      are 'json' and 'python'. By default 'python' will be used.`
    **kwargs: Keyword arguments passed to the `lf.structured.ParseStructure`
      transform, e.g. `lm` for specifying the language model.

  Returns:
    The parsed result based on the schema.
  """
  if examples is None:
    examples = DEFAULT_PARSE_EXAMPLES

  t = _parse_structure_cls(protocol)(
      schema, default=default, examples=examples, **kwargs
  )
  if isinstance(message, str):
    message = lf.AIMessage(message)

  if message.source is None and user_prompt is not None:
    message.source = lf.UserMessage(user_prompt, tags=['lm-input'])
  return t.transform(message=message).result


def _parse_structure_cls(
    protocol: schema_lib.SchemaProtocol,
) -> Type[ParseStructure]:
  if protocol == 'json':
    return ParseStructureJson
  elif protocol == 'python':
    return ParseStructurePython
  else:
    raise ValueError(f'Unknown protocol: {protocol!r}.')


def as_structured(
    self,
    annotation: Union[Type[Any], list[Type[Any]], dict[str, Any], None],
    default: Any = lf.message_transform.RAISE_IF_HAS_ERROR,
    examples: list[mapping.Mapping] | None = None,
    *,
    protocol: schema_lib.SchemaProtocol = 'python',
    **kwargs,
):
  """Returns the structured representation of the message text.

  Args:
    self: The Message transform object.
    annotation: The annotation used for representing the structured output. E.g.
      int, list[int], {'x': int, 'y': str}, A. If None, the return value will be
      the original LM response (str).
    default: The default value to use if parsing failed. If not specified, error
      will be raised.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    protocol: The protocol for schema/value representation. Applicable values
      are 'json' and 'python'. By default 'python' will be used.
    **kwargs: Additional keyword arguments that will be passed to
      `lf.structured.NaturalLanguageToStructure`.

  Returns:
    The structured output according to the annotation.
  """
  if annotation is None:
    return self
  if examples is None:
    examples = DEFAULT_PARSE_EXAMPLES
  return self >> _parse_structure_cls(protocol)(
      schema=annotation,
      default=default,
      examples=examples,
      **kwargs,
  )


class _Country(pg.Object):
  """A example dataclass for structured parsing."""
  name: str
  continents: list[Literal[
      'Africa',
      'Asia',
      'Europe',
      'Oceania',
      'North America',
      'South America'
  ]]
  num_states: int
  neighbor_countries: list[str]
  population: int
  capital: str | None
  president: str | None


DEFAULT_PARSE_EXAMPLES: list[mapping.MappingExample] = [
    mapping.MappingExample(
        nl_context='Brief introduction of the U.S.A.',
        nl_text=inspect.cleandoc("""
            The United States of America is a country primarily located in North America
            consisting of fifty states, a federal district, five major unincorporated territories,
            nine Minor Outlying Islands, and 326 Indian reservations. It shares land borders
            with Canada to its north and with Mexico to its south and has maritime borders
            with the Bahamas, Cuba, Russia, and other nations. With a population of over 333
            million. The national capital of the United States is Washington, D.C.
            """),
        schema=_Country,
        value=_Country(
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
    ),
]


lf.MessageTransform.as_structured = as_structured
