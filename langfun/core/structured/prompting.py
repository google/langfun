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

from typing import Annotated, Any, Literal, Type, Union

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


@lf.use_init_args(['schema', 'default', 'examples'])
class QueryStructure(mapping.NaturalLanguageToStructure):
  """Query an object out from a natural language text."""

  user_prompt: Annotated[
      lf.Message, 'Natural language prompt from the user to query the LM.'
  ] = lf.contextual()

  def transform_input(self, lm_input: lf.Message) -> lf.Message:
    lm_input.source = self.user_prompt
    return lm_input

  @property
  def nl_context(self) -> str:
    """Returns the user request."""
    return self.user_prompt.text

  @property
  def nl_text(self) -> None:
    """Returns the LM response."""
    return None


class QueryStructureJson(QueryStructure):
  """Query a structured value using JSON as the protocol."""

  preamble = """
      Please respond to the last {{ nl_context_title }} with {{ value_title}} according to {{ schema_title }}:

      INSTRUCTIONS:
        1. If the schema has `_type`, carry it over to the JSON output.
        2. If a field from the schema cannot be extracted from the response, use null as the JSON value.
      """

  protocol = 'json'
  schema_title = 'SCHEMA'
  value_title = 'JSON'


class QueryStructurePython(QueryStructure):
  """Query a structured value using Python as the protocol."""

  preamble = """
      Please respond to the last {{ nl_context_title }} with {{ value_title }} according to {{ schema_title }}.
      """
  protocol = 'python'
  schema_title = 'RESULT_TYPE'
  value_title = 'RESULT_OBJECT'


def _query_structure_cls(
    protocol: schema_lib.SchemaProtocol,
) -> Type[QueryStructure]:
  if protocol == 'json':
    return QueryStructureJson
  elif protocol == 'python':
    return QueryStructurePython
  else:
    raise ValueError(f'Unknown protocol: {protocol!r}.')


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


DEFAULT_QUERY_EXAMPLES: list[mapping.MappingExample] = [
    mapping.MappingExample(
        nl_context='Brief introduction of the U.S.A.',
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


def query(
    user_prompt: Union[str, lf.Template, lf.Modality],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any]
    ],
    default: Any = lf.RAISE_IF_HAS_ERROR,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    autofix: int = 3,
    autofix_lm: lf.LanguageModel | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    returns_message: bool = False,
    **kwargs,
) -> Any:
  """Parse a natural langugage message based on schema.

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
    user_prompt: A str, a `lf.Template` object or a `lf.Modality` object (e.g.
      `lf.Image`) as the prompt from the user.
    schema: A `lf.transforms.ParsingSchema` object or equivalent annotations.
    default: The default value if parsing failed. If not specified, error will
      be raised.
    lm: The language model to use. If not specified, the language model from
      `lf.context` context manager will be used.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    autofix: Number of attempts to auto fix the generated code. If 0, autofix is
      disabled. Auto-fix is not supported for 'json' protocol.
    autofix_lm: The language model to use for autofix. If not specified, the
      `autofix_lm` from `lf.context` context manager will be used. Otherwise it
      will use `lm`.
    protocol: The protocol for schema/value representation. Applicable values
      are 'json' and 'python'. By default `python` will be used.
    returns_message: If True, returns `lf.Message` as the output, instead of
      returning the structured `message.result`.
    **kwargs: Keyword arguments passed to the
      `lf.structured.NaturalLanguageToStructureed` transform.

  Returns:
    The result based on the schema.
  """
  # Autofix is not supported for JSON yet.
  if protocol == 'json':
    autofix = 0

  if examples is None:
    examples = DEFAULT_QUERY_EXAMPLES

  t = _query_structure_cls(protocol)(schema, default=default, examples=examples)
  if isinstance(user_prompt, lf.Template):
    user_prompt = user_prompt.render(**kwargs)
  user_prompt = lf.UserMessage.from_value(user_prompt)

  context = dict(autofix=autofix)
  if lm is not None:
    context['lm'] = lm
  autofix_lm = autofix_lm or lm
  if autofix_lm is not None:
    context['autofix_lm'] = autofix_lm
  context.update(kwargs)

  with t.override(**context):
    output = t(user_prompt=user_prompt)
  return output if returns_message else output.result
