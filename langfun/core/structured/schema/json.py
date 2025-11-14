# Copyright 2025 The Langfun Authors
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
"""JSON-based prompting protocol."""

import io
import textwrap
from typing import Any
from langfun.core.structured.schema import base
import pyglove as pg


class JsonError(Exception):  # pylint: disable=g-bad-exception-name
  """Json parsing error."""

  def __init__(self, json: str, cause: Exception):
    self.json = json
    self.cause = cause

  def __str__(self) -> str:
    r = io.StringIO()
    r.write(
        pg.colored(
            f'{self.cause.__class__.__name__}: {self.cause}', 'magenta'
        )
    )

    r.write('\n\n')
    r.write(pg.colored('JSON text:', 'red'))
    r.write('\n\n')
    r.write(textwrap.indent(pg.colored(self.json, 'magenta'), ' ' * 2))
    return r.getvalue()


class JsonPromptingProtocol(base.PromptingProtocol):
  """JSON-based prompting protocol."""

  NAME = 'json'

  def schema_repr(self, schema: base.Schema, **kwargs) -> str:
    del kwargs
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
        raise ValueError(
            f'Unsupported value spec being used as schema: {node}.')
    _visit(schema.schema_dict())
    return out.getvalue()

  def value_repr(
      self,
      value: Any,
      schema: base.Schema | None = None,
      **kwargs
  ) -> str:
    del schema, kwargs
    return pg.to_json_str(dict(result=value))

  def parse_value(
      self,
      text: str,
      schema: base.Schema | None = None,
      **kwargs
  ) -> Any:
    """Parses a JSON string into a structured object."""
    del schema
    try:
      text = cleanup_json(text)
      v = pg.from_json_str(text, **kwargs)
    except Exception as e:
      raise JsonError(text, e)  # pylint: disable=raise-missing-from

    if not isinstance(v, dict) or 'result' not in v:
      raise JsonError(text, ValueError(
          'The root node of the JSON must be a dict with key `result`. '
          f'Encountered: {v}'
      ))
    return v['result']


def cleanup_json(json_str: str) -> str:
  """Cleans up the LM responded JSON string."""
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
