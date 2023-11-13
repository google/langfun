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

  input_message: Annotated[lf.Message, 'The input message.'] = lf.contextual()

  def transform_input(self, lm_input: lf.Message) -> lf.Message:
    lm_input.source = self.input_message
    return lm_input

  @property
  def nl_context(self) -> str | None:
    """Returns the user request."""
    return getattr(self.input_message.lm_input, 'text', None)

  @property
  def nl_text(self) -> str:
    """Returns the LM response."""
    return self.input_message.text


class ParseStructureJson(ParseStructure):
  """Parse an object out from a NL text using JSON as the protocol."""

  preamble = """
      Please help translate the last LM response into JSON based on the request and the schema:

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
      Please help translate the last {{ nl_text_title }} into {{ value_title}} based on {{ schema_title }}.
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
    default: Any = lf.RAISE_IF_HAS_ERROR,
    *,
    user_prompt: str | None = None,
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
      are 'json' and 'python'. By default 'python' will be used.`
    returns_message: If True, returns `lf.Message` as the output, instead of
      returning the structured `message.result`.
    **kwargs: Keyword arguments passed to the `lf.structured.ParseStructure`
      transform.

  Returns:
    The parsed result based on the schema.
  """
  # Autofix is not supported for JSON yet.
  if protocol == 'json':
    autofix = 0

  if examples is None:
    examples = DEFAULT_PARSE_EXAMPLES

  t = _parse_structure_cls(protocol)(schema, default=default, examples=examples)
  message = lf.AIMessage.from_value(message)

  if message.source is None and user_prompt is not None:
    message.source = lf.UserMessage(user_prompt, tags=['lm-input'])

  # Setting up context.
  context = dict(autofix=autofix)
  if lm is not None:
    context['lm'] = lm
  autofix_lm = autofix_lm or lm
  if autofix_lm is not None:
    context['autofix_lm'] = autofix_lm
  context.update(kwargs)

  output = t(input_message=message, **context)
  return output if returns_message else output.result


def call(
    prompt: str | lf.Template,
    schema: Union[
        None, schema_lib.Schema, Type[Any], list[Type[Any]], dict[str, Any]
    ] = None,
    *,
    lm: lf.LanguageModel | None = None,
    parsing_lm: lf.LanguageModel | None = None,
    parsing_examples: list[mapping.MappingExample] | None = None,
    autofix: int = 3,
    autofix_lm: lf.LanguageModel | None = None,
    response_postprocess: Callable[[str], str] | None = None,
    protocol: schema_lib.SchemaProtocol = 'python',
    returns_message: bool = False,
    **kwargs,
) -> Any:
  """Call a language model with prompt and formulate response in return type.

  Examples::

    # Call with constant string-type prompt.
    lf.call('Compute one plus one', lm=lf.llms.Gpt35())
    >> "two"

    # Call with returning a structured (int) type.
    lf.call('Compute one plus one', int, lm=lf.llms.Gpt35())
    >> 2

    # Call with a template string with variables.
    lf.call('Compute {{x}} plus {{y}}', int,
            x='one', y='one', lm=lf.llms.Gpt35())
    >> 2

    # Call with an `lf.Template` object with variables.
    lf.call(lf.Template('Compute {{x}} plus {{y}}', x=1), int,
            y=1, lm=lf.llms.Gpt35())
    >> 2

  Args:
    prompt: User prompt that will be sent to LM, which could be a string or a
      string template whose variables are provided from **kwargs.
    schema: Type annotations for return type. If None, the raw LM response will
      be returned (str). Otherwise, the response will be parsed based on the
      return type.
    lm: Language model used to prompt to get a natural language-based response.
      If not specified, `lm` from `lf.context` context manager will be used.
    parsing_lm: Language model that will be used for parsing. If None, the `lm`
      for prompting the LM will be used.
    parsing_examples: Examples for parsing the output. If None,
      `lf.structured.DEFAULT_PARSE_EXAMPLES` will be used.
    autofix: Number of attempts to auto fix the generated code. If 0, autofix is
      disabled. Auto-fix is not supported for 'json' protocol.
    autofix_lm: The language model to use for autofix. If not specified, the
      `autofix_lm` from `lf.context` context manager will be used. Otherwise it
      will use `parsing_lm`.
    response_postprocess: A callback function to post process the text response
      before sending for parsing.
    protocol: The protocol for schema/value representation. Applicable values
      are 'json' and 'python'. By default 'python' will be used.`
    returns_message: If True, return a `lf.Message` object instead of its text
      or result.
    **kwargs: Keyword arguments. Including options that control the calling
      behavior, such as `lm`, `temperature`, etc. As well as variables that will
      be fed to the prompt if it's a string template.

  Returns:
    A string if `returns` is None or an instance of the return type.
  """
  if isinstance(prompt, str):
    prompt = lf.Template(prompt)

  if isinstance(prompt, lf.LangFunc):
    lfun = prompt
  elif isinstance(prompt, lf.Template):
    lfun = lf.LangFunc(prompt.template_str)
    lfun.sym_setparent(prompt)
  else:
    raise TypeError(
        '`prompt` should be a string or an `lf.Template` object. '
        f'Encountered {prompt!r}.'
    )

  # Call `lm` for natural response.
  call_kwargs = dict(kwargs)
  if lm is not None:
    call_kwargs['lm'] = lm

  lm_output = lfun(**call_kwargs)

  if response_postprocess is not None:
    lm_output.set('text', response_postprocess(lm_output.text))

  if schema is None:
    return lm_output if returns_message else lm_output.text

  # Call `parsing_lm` for structured parsing.
  parse_kwargs = dict(kwargs)
  parsing_lm = parsing_lm or lm
  if parsing_lm is not None:
    parse_kwargs['lm'] = parsing_lm
  if parsing_examples is not None:
    parse_kwargs['examples'] = parsing_examples
  return parse(
      lm_output,
      schema,
      autofix=autofix,
      autofix_lm=autofix_lm,
      protocol=protocol,
      returns_message=returns_message,
      **parse_kwargs,
  )


def _parse_structure_cls(
    protocol: schema_lib.SchemaProtocol,
) -> Type[ParseStructure]:
  if protocol == 'json':
    return ParseStructureJson
  elif protocol == 'python':
    return ParseStructurePython
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


DEFAULT_PARSE_EXAMPLES: list[mapping.MappingExample] = [
    mapping.MappingExample(
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
