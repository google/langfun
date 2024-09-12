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

import functools
from typing import Any, Callable, Type, Union

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import schema as schema_lib
import pyglove as pg


@lf.use_init_args(['schema', 'default', 'examples'])
class QueryStructure(mapping.Mapping):
  """Query an object out from a natural language text."""

  context_title = 'CONTEXT'
  input_title = 'INPUT_OBJECT'

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
        Answer(
          final_answer=2
        )
        ```
      """
  protocol = 'python'
  schema_title = 'OUTPUT_TYPE'
  output_title = 'OUTPUT_OBJECT'


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
    prompt: Union[str, pg.Symbolic],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    default: Any = lf.RAISE_IF_HAS_ERROR,
    *,
    lm: lf.LanguageModel | None = None,
    examples: list[mapping.MappingExample] | None = None,
    cache_seed: int | None = 0,
    response_postprocess: Callable[[str], str] | None = None,
    autofix: int = 0,
    autofix_lm: lf.LanguageModel | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    returns_message: bool = False,
    skip_lm: bool = False,
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
    prompt: A str (may contain {{}} as template) as natural language input, or a
      `pg.Symbolic` object as structured input as prompt to LLM.
    schema: A type annotation as the schema for output object. If str (default),
      the response will be a str in natural language.
    default: The default value if parsing failed. If not specified, error will
      be raised.
    lm: The language model to use. If not specified, the language model from
      `lf.context` context manager will be used.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    cache_seed: Seed for computing cache key. The cache key is determined by a
      tuple of (lm, prompt, cache seed). If None, cache will be disabled for the
      query even cache is configured by the LM.
    response_postprocess: An optional callable object to process the raw LM
      response before parsing it into the final output object. If None, the raw
      LM response will not be processed.
    autofix: Number of attempts to auto fix the generated code. If 0, autofix is
      disabled. Auto-fix is not supported for 'json' protocol.
    autofix_lm: The language model to use for autofix. If not specified, the
      `autofix_lm` from `lf.context` context manager will be used. Otherwise it
      will use `lm`.
    protocol: The protocol for schema/value representation. Applicable values
      are 'json' and 'python'. By default `python` will be used.
    returns_message: If True, returns `lf.Message` as the output, instead of
      returning the structured `message.result`.
    skip_lm: If True, returns the rendered prompt as a UserMessage object.
      otherwise return the LLM response based on the rendered prompt.
    **kwargs: Keyword arguments passed to render the prompt or configure the 
      `lf.structured.Mapping` class. Notable kwargs are:
      - template_str: Change the root template for query.
      - preamble: Change the preamble for query.
      - mapping_template: Change the template for each mapping examle.

  Returns:
    The result based on the schema.
  """
    # Internal usage logging.

  # When `lf.query` is used for symbolic completion, schema is automatically
  # inferred when it is None.
  if isinstance(prompt, pg.Symbolic) and prompt.sym_partial and schema is None:
    schema = prompt.__class__

  # Create a copy of the prompt if it has a parent object, so all child modality
  # objects could be referred by path relative to the prompt.
  if isinstance(prompt, lf.Template) and prompt.sym_parent:
    prompt = prompt.clone()

  if schema in (None, str):
    # Query with natural language output.
    output = lf.LangFunc.from_value(prompt, **kwargs)(
        lm=lm, cache_seed=cache_seed, skip_lm=skip_lm
    )
    if response_postprocess:
      processed_text = response_postprocess(output.text)
      if processed_text != output.text:
        output = lf.AIMessage(processed_text, source=output)
    return output if returns_message else output.text

  # Query with structured output.
  prompt_kwargs = kwargs.copy()

  # NOTE(daiyip): when `template_str` is passed in, it's intended to modify the
  # QueryStructure template string. Therefore, we pop out the argument for
  # prompt rendering.
  prompt_kwargs.pop('template_str', None)

  if isinstance(prompt, (str, lf.Message, lf.Template)):
    prompt = lf.Template.from_value(prompt, **prompt_kwargs).render(lm=lm)
  else:
    prompt = schema_lib.mark_missing(prompt)

  output = _query_structure_cls(protocol)(
      input=prompt,
      schema=schema,
      default=default,
      examples=examples,
      response_postprocess=response_postprocess,
      autofix=autofix if protocol == 'python' else 0,
      **kwargs,
  )(
      lm=lm,
      autofix_lm=autofix_lm or lm,
      cache_seed=cache_seed,
      skip_lm=skip_lm,
  )
  return output if returns_message else output.result


def query_prompt(
    prompt: Union[str, pg.Symbolic],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ] = None,
    **kwargs,
) -> lf.Message:
  """Returns the final prompt sent to LLM for `lf.query`."""
  kwargs.pop('returns_message', None)
  kwargs.pop('skip_lm', None)
  return query(prompt, schema, skip_lm=True, returns_message=True, **kwargs)


def query_output(
    response: Union[str, lf.Message],
    schema: Union[
        schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any], None
    ],
    **kwargs,
) -> Any:
  """Returns the final output of `lf.query` from a provided LLM response."""
  kwargs.pop('prompt', None)
  kwargs.pop('lm', None)
  return query(
      'Unused prompt', schema, lm=fake.StaticResponse(response), **kwargs
  )


def query_reward(
    mapping_example: Union[str, mapping.MappingExample],
    response: Union[str, lf.Message],
) -> float | None:
  """Returns the reward of an LLM response based on an mapping example."""
  if isinstance(mapping_example, str):
    mapping_example = pg.from_json_str(mapping_example)
    assert isinstance(mapping_example, mapping.MappingExample), mapping_example
  schema = mapping_example.schema

  if schema and isinstance(schema.spec, pg.typing.Object):
    output_cls = schema.spec.cls
  elif schema is None and isinstance(mapping_example.output, pg.Object):
    output_cls = mapping_example.output.__class__
  else:
    output_cls = None

  reward_fn = _reward_fn(output_cls)
  if reward_fn is None:
    return None

  return reward_fn(
      query_output(response, output_cls),
      mapping_example.input,
      mapping_example.output,
      mapping_example.metadata,
  )


@functools.cache
def _reward_fn(cls) -> Callable[
    [
        pg.Object,    # Actual output object.
        Any,          # Input object.
        pg.Object,    # Expected output object.
        pg.Dict       # User metadata.
    ], float] | None:
  """Returns the reward function for a class that is being queried."""
  if not callable(getattr(cls, '__reward__', None)):
    return None

  signature = pg.typing.signature(cls.__reward__)
  num_args = len(signature.args)
  if num_args < 2 or num_args > 4:
    raise TypeError(
        f'`{cls.__type_name__}.__reward__` should have signature: '
        '`__reward__(self, input, [expected_output], [expected_metadata])`.'
    )
  def _reward(self, input, expected_output, metadata):  # pylint: disable=redefined-builtin
    args = [self, input, expected_output, metadata]
    return cls.__reward__(*args[:num_args])
  return _reward
