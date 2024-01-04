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
"""Symbolic query."""

from typing import Any, Type, Union

import langfun.core as lf
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


@lf.use_init_args(['schema', 'default', 'examples'])
class QueryStructure(mapping.Mapping):
  """Query an object out from a natural language text."""

  context_title = 'CONTEXT'
  input_title = 'USER_REQUEST'

  # Mark schema as required.
  schema: pg.typing.Annotated[
      schema_lib.schema_spec(), 'Required schema for parsing.'
  ]


class QueryStructureJson(QueryStructure):
  """Query a structured value using JSON as the protocol."""

  preamble = """
      Please respond to the last {{ input_title }} with {{ output_title}} according to {{ schema_title }}:

      INSTRUCTIONS:
        1. If the schema has `_type`, carry it over to the JSON output.
        2. If a field from the schema cannot be extracted from the response, use null as the JSON value.

      {{ input_title }}:
        1 + 1 =

      {{ schema_title }}:
        {"result": {"_type": "langfun.core.structured.prompting.Answer", "final_answer": int}}

      {{ output_title}}:
        {"result": {"_type": "langfun.core.structured.prompting.Answer", "final_answer": 2}}
      """

  protocol = 'json'
  schema_title = 'SCHEMA'
  output_title = 'JSON'


class QueryStructurePython(QueryStructure):
  """Query a structured value using Python as the protocol."""

  preamble = """
      Please respond to the last {{ input_title }} with {{ output_title }} according to {{ schema_title }}.

      INSTRUCTIONS:
        1. Only response the required {{ output_title }} as illustrated by the given example.
        2. Don't add any comments in the response.
        3. {{ output_title }} must strictly follow the {{ schema_title }}.

      {{ input_title }}:
        1 + 1 =

      {{ schema_title }}:
        Answer

        ```python
        class Answer:
          final_answer: int
        ```

      {{ output_title }}:
        ```python
        Answer(final_answer=2)
        ```
      """
  protocol = 'python'
  schema_title = 'RESULT_TYPE'
  output_title = 'RESULT_OBJECT'


def _query_structure_cls(
    protocol: schema_lib.SchemaProtocol,
) -> Type[QueryStructure]:
  if protocol == 'json':
    return QueryStructureJson
  elif protocol == 'python':
    return QueryStructurePython
  else:
    raise ValueError(f'Unknown protocol: {protocol!r}.')


def query(
    prompt: Union[str, lf.Template, pg.Symbolic],
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
    prompt: A str or a `lf.Template` object as natural language input, or a 
      `pg.Symbolic` object as structured input as prompt to LLM.
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

  t = _query_structure_cls(protocol)(schema, default=default, examples=examples)
  if isinstance(prompt, lf.Template):
    prompt = prompt.render(**kwargs)

  if isinstance(prompt, (str, lf.Message)):
    prompt = lf.UserMessage.from_value(prompt)

  call_context = dict(autofix=autofix)
  if lm is not None:
    call_context['lm'] = lm
  autofix_lm = autofix_lm or lm
  if autofix_lm is not None:
    call_context['autofix_lm'] = autofix_lm
  call_context.update(kwargs)

  with t.override(**call_context):
    output = t(input=prompt)
  return output if returns_message else output.result
