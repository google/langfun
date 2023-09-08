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
"""Schema for structured data."""

import abc
import io
from typing import Any
import langfun.core as lf
import pyglove as pg


def parse_value_spec(value) -> pg.typing.ValueSpec:
  """Parses a PyGlove ValueSpec equivalence into a ValueSpec."""
  if isinstance(value, pg.typing.ValueSpec):
    return value

  if isinstance(value, dict) and len(value) == 1 and 'result' in value:
    value = value['result']

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

  return _parse_node(value)


class Schema(lf.NaturalLanguageFormattable, pg.Object):
  """Base class for structured data schema."""

  spec: pg.typing.Annotated[
      pg.typing.Object(pg.typing.ValueSpec, transform=parse_value_spec),
      (
          'A PyGlove ValueSpec object representing the spec for the value '
          'to be parsed.'
      ),
  ]

  @abc.abstractmethod
  def schema_repr(self) -> str:
    """Returns the representation of the schema."""

  @abc.abstractmethod
  def value_repr(self, value: Any) -> str:
    """Returns the representation of a structured value."""

  @abc.abstractmethod
  def parse(self, text: str) -> Any:
    """Parse a LM generated text into a structured value."""

  def natural_language_format(self) -> str:
    return self.schema_repr()

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

  @classmethod
  def from_value(cls, value) -> 'Schema':
    """Creates a schema from an equivalent representation."""

    if isinstance(value, Schema):
      return value
    return cls(parse_value_spec(value))


class JsonSchema(Schema):
  """JSON-represented schema."""

  def schema_repr(self) -> str:
    """Render the schema into a natural language string."""
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

  def value_repr(self, value: Any) -> str:
    return pg.to_json_str(dict(result=value))

  def parse(self, text: str) -> Any:
    """Parse a JSON string into a structured object."""
    v = pg.from_json_str(self._cleanup_json(text))
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
