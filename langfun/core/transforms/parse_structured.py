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
"""Natural language to structured data transform."""

import inspect
import io
from typing import Annotated, Any, Callable, Type, Union
import langfun.core as lf
from langfun.core import llms as lf_llms
import pyglove as pg


class ParsingSchema(lf.NaturalLanguageFormattable, pg.Object):
  """Schema for the JSON output."""

  spec: Annotated[
      pg.typing.ValueSpec,
      (
          'A PyGlove ValueSpec object representing the spec for the value '
          'to be parsed.'
      ),
  ]

  def parse(self, json_str: str) -> Any:
    """Parse a JSON string into structured output."""
    v = pg.from_json_str(self._cleanup_json(json_str))
    if not isinstance(v, dict) or 'result' not in v:
      raise ValueError(
          'The root node of the JSON must be a dict with key `result`. '
          f'Encountered: {v}'
      )
    return self.spec.apply(v['result'])

  def _cleanup_json(self, json_str: str) -> str:
    # Treatments:
    # 1. Extract the JSON string with a top-level dict from the response.
    #    This prevents the leading and trailing texts in the response to
    #    be counted as part of the JSON.
    # 2. Escape new lines in JSON values.

    curly_brackets = 0
    under_json = False
    under_str = False
    str_begin = -1

    cleaned = io.StringIO()
    for i, c in enumerate(json_str):
      if c == '{' and not under_str:
        cleaned.write(c)
        curly_brackets += 1
        under_json = True
        continue
      elif not under_json:
        continue

      if c == '}' and not under_str:
        cleaned.write(c)
        curly_brackets -= 1
        if curly_brackets == 0:
          break
      elif c == '"' and json_str[i - 1] != '\\':
        under_str = not under_str
        if under_str:
          str_begin = i
        else:
          assert str_begin > 0
          str_value = json_str[str_begin : i + 1].replace('\n', '\\n')
          cleaned.write(str_value)
          str_begin = -1
      elif not under_str:
        cleaned.write(c)

    if not under_json:
      raise ValueError(f'No JSON dict in the output: {json_str}')

    if curly_brackets > 0:
      raise ValueError(
          f'Malformated JSON: missing {curly_brackets} closing curly braces.'
      )

    return cleaned.getvalue()

  def schema_dict(self) -> dict[str, Any]:
    """Returns the dict representation of the schema."""

    def _node(vs: pg.typing.ValueSpec) -> Any:
      if isinstance(vs, pg.typing.PrimitiveType):
        return vs
      elif isinstance(vs, pg.typing.Dict):
        assert vs.schema is not None
        return {str(k): _node(f.value) for k, f in vs.schema.fields.items()}
      elif isinstance(vs, pg.typing.List):
        return [_node(vs.element.value)]
      elif isinstance(vs, pg.typing.Object):
        if issubclass(vs.cls, pg.Symbolic):
          d = {pg.JSONConvertible.TYPE_NAME_KEY: vs.cls.__type_name__}
          d.update(
              {
                  str(k): _node(f.value)
                  for k, f in vs.cls.__schema__.fields.items()
              }
          )
          return d
      raise TypeError(f'Unsupported value spec for structured parsing: {vs}.')

    return {'result': _node(self.spec)}

  def json_repr(self) -> str:
    out = io.StringIO()
    def _visit(node: Any) -> None:
      if isinstance(node, str):
        out.write(f'"{node}"')
      elif isinstance(node, list):
        assert len(node) == 1, node
        out.write('[')
        _visit(node[0])
        out.write(']')
      elif isinstance(node, dict):
        out.write('{')
        for i, (k, v) in enumerate(node.items()):
          if i != 0:
            out.write(', ')
          out.write(f'"{k}": ')
          _visit(v)
        out.write('}')
      elif isinstance(node, pg.typing.Enum):
        out.write(' | '.join(
            f'"{v}"' if isinstance(v, str) else repr(v)
            for v in node.values))
      elif isinstance(node, pg.typing.PrimitiveType):
        x = node.value_type.__name__
        if isinstance(node, pg.typing.Number):
          params = []
          if node.min_value is not None:
            params.append(f'min={node.min_value}')
          if node.max_value is not None:
            params.append(f'max={node.max_value}')
          if params:
            x += f'({", ".join(params)})'
        elif isinstance(node, pg.typing.Str):
          if node.regex is not None:
            x += f'(regex={node.regex.pattern})'
        if node.is_noneable:
          x = x + ' | None'
        out.write(x)
      else:
        raise ValueError(f'Unsupported schema node: {node}.')
    _visit(self.schema_dict())
    return out.getvalue()

  def natural_language_format(self) -> str:
    return self.json_repr()

  @classmethod
  def from_annotation(cls, annotation) -> 'ParsingSchema':
    """Creates a schema from a dict representation."""

    if (
        isinstance(annotation, dict)
        and len(annotation) == 1
        and 'result' in annotation
    ):
      annotation = annotation['result']

    def _parse_node(v) -> pg.typing.ValueSpec:
      if isinstance(v, dict):
        return pg.typing.Dict([(k, _parse_node(cv)) for k, cv in v.items()])
      elif isinstance(v, list):
        if len(v) != 1:
          raise ValueError(
              'Annotation with list must be a list of a single element. '
              f'Encountered: {v}'
          )
        return pg.typing.List(_parse_node(v[0]))
      else:
        spec = pg.typing.ValueSpec.from_annotation(v, auto_typing=True)
        if isinstance(
            spec,
            (
                pg.typing.Any,
                pg.typing.Callable,
                pg.typing.Tuple,
                pg.typing.Type,
                pg.typing.Union,
            ),
        ):
          raise ValueError(f'Unsupported schema specification: {v}')
        if isinstance(spec, pg.typing.Object) and not issubclass(
            spec.cls, pg.Symbolic
        ):
          raise ValueError(f'{v} must be a symbolic class to be parsable.')
        return spec

    return ParsingSchema(_parse_node(annotation))


#
# Interface for Jsonify LangFunc.
#


class Jsonify(lf.LangFunc):
  """Interface for LangFunc which converts a natural language to JSON."""

  message: Annotated[lf.Message, 'The input message.'] = lf.contextual()

  result_schema: Annotated[
      ParsingSchema,
      'A `ParsingSchema` object that constrains the structured output.',
  ] = lf.contextual()

  request_getter: Annotated[
      Callable[[lf.Message], str] | None,
      (
          'A callable object to get the request text from the message. '
          'If None, it returns the entire LM input message ('
          '`message.lm_input.text`).'
      )
  ] = None

  response_getter: Annotated[
      Callable[[lf.Message], str] | None,
      (
          'A callable object to get the response text from the message. '
          'If None, it returns the entire LM output message (`message.text`).'
      )
  ] = None

  @property
  def request(self) -> str | None:
    """Returns the user request."""
    if self.request_getter is None:
      return self.message.lm_input.text if self.message.lm_input else None
    return self.request_getter(self.message)    # pylint: disable=not-callable

  @property
  def response(self) -> str:
    """Returns the LM response."""
    if self.response_getter is None:
      # This allows external output transform chaining.
      if isinstance(self.message.result, str):
        return self.message.result
      return self.message.text
    return self.response_getter(self.message)    # pylint: disable=not-callable


#
# The default (fewshot-based) implementation of Jsonify.
#


class ParsingExample(lf.LangFunc):
  """Natural language to JSON example.

  {%- if request -%}
  USER_REQUEST:
  {{ request | indent(2, True)}}

  {% endif -%}
  LM_RESPONSE:
  {{ response | indent(2, True) }}

  SCHEMA:
  {{ result_schema.__str__() | indent(2, True) }}

  JSON:
  {{ json_str | indent(2, True) }}
  """

  request: Annotated[str | None, '(Optional) The user request.']

  response: Annotated[str, 'The LM response.']

  result_schema: Annotated[
      ParsingSchema,
      'A `ParsingSchema` object that constrains the structured output.',
  ] = lf.contextual()

  result: Annotated[Any, 'The converted JSON object.']

  @pg.explicit_method_override
  def __init__(
      self,  # pytype: disable=annotation-type-mismatch
      request: str | None,
      response: str,
      result: Any,  # pylint: disable=redefined-outer-name
      result_schema: Union[
          ParsingSchema, Type[Any], list[Type[Any]], dict[str, Any]
      ] = pg.MISSING_VALUE,
      **kwargs,
  ):
    if result_schema != pg.MISSING_VALUE and not isinstance(
        result_schema, ParsingSchema
    ):
      result_schema = ParsingSchema.from_annotation(result_schema)
    super().__init__(
        request=request, response=response,
        result=result, result_schema=result_schema, **kwargs
    )

  @property
  def json_str(self) -> str:
    return pg.to_json_str(dict(result=self.result))

  def format(self, *args, **kwargs) -> str:
    kwargs.pop('include_keys', None)
    return super().format(
        *args,
        include_keys=set(['request', 'response', 'result', 'result_schema']),
        **kwargs)


@pg.use_init_args(['examples'])
class FewshotJsonify(Jsonify):
  """Jsonify based on fewshot examples.

  {{ preamble }}

  {% if examples -%}
  {% for example in examples -%}
  {{ example }}

  {% endfor %}
  {% endif -%}
  {% if request -%}
  USER_REQUEST:
  {{ request | indent(2, True)}}

  {% endif -%}

  LM_RESPONSE:
  {{ response | indent(2, True) }}

  SCHEMA:
  {{ result_schema.__str__() | indent(2, True) }}

  JSON:
  """

  preamble: Annotated[
      lf.LangFunc,
      'Preamble used for zeroshot jsonify.',
  ] = lf.LangFunc("""
      Please help transform the LM response into JSON based on the request and the schema:

      INSTRUCTIONS:
        1. If the schema has `_type`, carry it over to the JSON output.
        2. If a field from the schema cannot be extracted from the response, use null as the JSON value.
      """)

  examples: Annotated[
      list[ParsingExample],
      (
          '(Optional) Fewshot examples from the user. If None, the examples '
          'from the `examples_path` will be loaded and used.'
      ),
  ] = lf.contextual(default=[])


#
# The parse structured transform.
#


class ParsingError(ValueError):
  """Parsing error."""

  def __eq__(self, other):
    return isinstance(other, ParsingError) and self.args == other.args

  def __ne__(self, other):
    return not self.__eq__(other)


class ParseStructured(lf.MessageTransform):
  """Parses a natural language-based text into a structured output."""

  # The default model for `jsonify` to use.
  lm = lf_llms.Gpt35(temperature=0.0)

  jsonify: Annotated[
      Jsonify | None,
      (
          'An LangFunc object for rewritting a natural language-based text '
          'into JSON.'
      ),
  ] = FewshotJsonify()

  result_schema: Annotated[
      ParsingSchema,
      'A `ParsingSchema` object that constrains the structured output.',
  ]

  default: Annotated[
      Any,
      (
          'The default value to use if parsing failed. '
          'If unspecified, error will be raisen.'
      ),
  ] = lf.message_transform.RAISE_IF_HAS_ERROR

  __kwargs__: Annotated[
      Any,
      (
          'Wildcard keyword arguments for `__init__` that can be referred in '
          'the `jsonify` object. This allows specifiying variables during '
          '`ParsedStructured.__init__` but used within `jsonify`.'
      ),
  ]

  @pg.explicit_method_override
  def __init__(
      self,
      result_schema: Union[
          ParsingSchema, Type[Any], list[Type[Any]], dict[str, Any]
      ],
      **kwargs,
  ):
    if not isinstance(result_schema, ParsingSchema):
      result_schema = ParsingSchema.from_annotation(result_schema)
    super().__init__(result_schema=result_schema, **kwargs)

  def _on_bound(self):
    super()._on_bound()

    # This allows the input path and output path to be automatically computed
    # by the composition layers.
    self.jsonify.rebind(
        input_path=self.input_path,
        output_path=self.output_path,
        raise_on_no_change=False,
        notify_parent=False,
    )

  def _transform_path(
      self, message: lf.Message, input_path: str, v: str
  ) -> lf.Message:
    try:
      output = self.jsonify(  # pylint: disable=not-callable
          result_schema=self.result_schema,
          message=message,
      )
      json_object = self.result_schema.parse(output.text)
      message.result = json_object
      return message
    except Exception as e:  # pylint: disable=broad-exception-caught
      if self.default == lf.message_transform.RAISE_IF_HAS_ERROR:
        raise ParsingError(
            'Cannot parse message text into structured output. '
            f'Error={e}. Text={message.text!r}.'
        ) from e
      return self.default


def parse(
    message: Union[lf.Message, str],
    schema: Union[ParsingSchema, Type[Any], list[Type[Any]], dict[str, Any]],
    default: Any = lf.message_transform.RAISE_IF_HAS_ERROR,
    *,
    user_prompt: str | None = None,
    examples: list[ParsingExample] | None = None,
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
    **kwargs: Keyword arguments passed to the `lf.ParseStructured` transform,
      e.g. `lm` for specifying the language model for structured parsing,
      `jsonify` for customizing the prompt for structured parsing, and etc.

  Returns:
    The parsed result based on the schema.
  """
  if examples is None:
    examples = _default_fewshot_examples()
  t = ParseStructured(schema, default=default, examples=examples, **kwargs)
  if isinstance(message, str):
    message = lf.AIMessage(message)

  if message.source is None and user_prompt is not None:
    message.source = lf.UserMessage(user_prompt, tags=['lm-input'])
  return t.transform(message=message).result


def as_structured(
    self,
    annotation: Union[Type[Any], list[Type[Any]], dict[str, Any]],
    default: Any = lf.message_transform.RAISE_IF_HAS_ERROR,
    examples: list[ParsingExample] | None = None,
    **kwargs,
):
  """Returns the structured representation of the message text.

  Args:
    self: The Message transform object.
    annotation: The annotation used for representing the structured output. E.g.
      int, list[int], {'x': int, 'y': str}, A.
    default: The default value to use if parsing failed. If not specified, error
      will be raised.
    examples: An optional list of fewshot examples for helping parsing. If None,
      the default one-shot example will be added.
    **kwargs: Additional keyword arguments that will be passed to
      `lf.transforms.ParseStructured`.

  Returns:
    The structured output according to the annotation.
  """
  if examples is None:
    examples = _default_fewshot_examples()
  return self >> ParseStructured(
      result_schema=annotation,
      default=default,
      examples=examples,
      **kwargs,
  )


class _Country(pg.Object):
  """A example dataclass for structured parsing."""
  name: str
  continents: list[pg.typing.Enum[
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


def _default_fewshot_examples() -> list[ParsingExample]:
  return [
      ParsingExample(
          request='Brief introduction of the U.S.A.',
          response=inspect.cleandoc("""
              The United States of America is a country primarily located in North America
              consisting of fifty states, a federal district, five major unincorporated territories,
              nine Minor Outlying Islands, and 326 Indian reservations. It shares land borders
              with Canada to its north and with Mexico to its south and has maritime borders
              with the Bahamas, Cuba, Russia, and other nations. With a population of over 333
              million. The national capital of the United States is Washington, D.C.
              """),
          result_schema=_Country,
          result=_Country(
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
      )
  ]


lf.MessageTransform.as_structured = as_structured
