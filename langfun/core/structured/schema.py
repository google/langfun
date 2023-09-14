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
import typing
from typing import Any, Literal, Type
import langfun.core as lf
from langfun.core import coding as lf_coding
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

  def schema_str(
      self,
      protocol: Literal['json', 'python'] = 'json',
      **kwargs) -> str:
    """Returns the representation of the schema."""
    return schema_repr(protocol).repr(self, **kwargs)

  def value_str(
      self,
      value: Any,
      protocol: Literal['json', 'python'] = 'json',
      **kwargs) -> str:
    """Returns the representation of a structured value."""
    return value_repr(protocol).repr(value, self, **kwargs)

  def parse(
      self,
      text: str,
      protocol: Literal['json', 'python'] = 'json',
      **kwargs) -> Any:
    """Parse a LM generated text into a structured value."""
    return self.spec.apply(value_repr(protocol).parse(text, self, **kwargs))

  def natural_language_format(self) -> str:
    return self.schema_str()

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
      raise TypeError(
          'Unsupported value spec being used as the schema for '
          f'structured data: {vs}.')

    return {'result': _node(self.spec)}

  def class_dependencies(
      self,
      include_subclasses: bool = True) -> list[Type[Any]]:
    """Returns a set of dependent classes for a ValueSpec."""
    seen = set()
    dependencies = []

    def _add_dependency(cls_or_classes):
      if isinstance(cls_or_classes, type):
        cls_or_classes = [cls_or_classes]
      for cls in cls_or_classes:
        if cls not in seen:
          dependencies.append(cls)
          seen.add(cls)

    def _fill_dependencies(vs: pg.typing.ValueSpec, include_subclasses: bool):
      if isinstance(vs, pg.typing.Object):
        if issubclass(vs.cls, pg.Object) and vs.cls not in seen:
          # Add base classes as dependencies.
          for base_cls in vs.cls.__bases__:
            # We only keep track of user-defined symbolic classes.
            if issubclass(
                base_cls, pg.Object
            ) and not base_cls.__module__.startswith('pyglove'):
              _fill_dependencies(
                  pg.typing.Object(base_cls), include_subclasses=False)

          # Add members as dependencies.
          if hasattr(vs.cls, '__schema__'):
            for field in vs.cls.__schema__.values():
              _fill_dependencies(field.value, include_subclasses)
        _add_dependency(vs.cls)

        # Check subclasses if available.
        if include_subclasses:
          for _, cls in pg.object_utils.registered_types():
            if issubclass(cls, vs.cls) and cls not in dependencies:
              _fill_dependencies(
                  pg.typing.Object(cls), include_subclasses=True)

      if isinstance(vs, pg.typing.List):
        _fill_dependencies(
            vs.element.value, include_subclasses)
      elif isinstance(vs, pg.typing.Tuple):
        for elem in vs.elements:
          _fill_dependencies(elem.value, include_subclasses)
      elif isinstance(vs, pg.typing.Dict) and vs.schema:
        for v in vs.schema.values():
          _fill_dependencies(v, include_subclasses)
      elif isinstance(vs, pg.typing.Union):
        for v in vs.candidates:
          _fill_dependencies(v, include_subclasses)

    _fill_dependencies(self.spec, include_subclasses)
    return dependencies

  @classmethod
  def from_value(cls, value) -> 'Schema':
    """Creates a schema from an equivalent representation."""
    if isinstance(value, Schema):
      return value
    return cls(parse_value_spec(value))


def schema_spec(noneable: bool = False) -> pg.typing.ValueSpec:  # pylint: disable=unused-argument
  if typing.TYPE_CHECKING:
    return Any
  return pg.typing.Object(
      Schema, transform=Schema.from_value, is_noneable=noneable
  )  # pylint: disable=unreachable-code


#
# Schema representations.
#


class SchemaRepr(metaclass=abc.ABCMeta):
  """Base class for schema representation."""

  @abc.abstractmethod
  def repr(self, schema: Schema) -> str:
    """Returns the representation of the schema."""


class SchemaPythonRepr(SchemaRepr):
  """Python-representation for a schema."""

  def repr(self, schema: Schema) -> str:
    ret = self.result_definition(schema)
    class_definition_str = self.class_definitions(schema)
    if class_definition_str:
      ret += f'\n\n```python\n{class_definition_str}```'
    return ret

  def class_definitions(
      self,
      schema: Schema,
      include_subclasses: bool = True) -> str | None:
    """Returns a str for class definitions from a schema."""
    class_dependencies = schema.class_dependencies(
        include_subclasses=include_subclasses)
    if not class_dependencies:
      return None
    def_str = io.StringIO()
    for i, cls in enumerate(class_dependencies):
      if i > 0:
        def_str.write('\n')
      def_str.write(self.class_def(cls))
    return def_str.getvalue()

  def result_definition(self, schema: Schema) -> str:
    return self.annotate(schema.spec)

  def class_def(self, cls, strict: bool = False) -> str:
    """Returns the Python class definition."""
    out = io.StringIO()
    if not issubclass(cls, pg.Object):
      raise TypeError(
          'Classes must be `pg.Object` subclasses to be used as schema. '
          f'Encountered: {cls}.'
      )
    schema = cls.__schema__
    eligible_bases = []
    for base_cls in cls.__bases__:
      if issubclass(
          base_cls, pg.Symbolic
      ) and not base_cls.__module__.startswith('pyglove'):
        eligible_bases.append(base_cls.__name__)
    if eligible_bases:
      base_cls_str = ', '.join(eligible_bases)
      out.write(f'class {cls.__name__}({base_cls_str}):\n')
    else:
      out.write(f'class {cls.__name__}:\n')

    if schema.fields:
      for key, field in schema.items():
        if not isinstance(key, pg.typing.ConstStrKey):
          raise TypeError(
              'Variable-length keyword arguments is not supported in '
              f'structured parsing or query. Encountered: {field}'
          )
        out.write(
            f'  {field.key}: {self.annotate(field.value, strict=strict)}\n')
    else:
      out.write('  pass\n')
    return out.getvalue()

  def annotate(
      self,
      vs: pg.typing.ValueSpec,
      annotate_optional: bool = True,
      strict: bool = False,
  ) -> str:
    """Returns the annotation string for a value spec."""
    if isinstance(vs, pg.typing.Any):
      return 'Any'
    elif isinstance(vs, pg.typing.Enum):
      candidate_str = ', '.join([repr(v) for v in vs.values])
      return f'Literal[{candidate_str}]'
    elif isinstance(vs, pg.typing.Union):
      candidate_str = ', '.join(
          [
              self.annotate(c, annotate_optional=False, strict=strict)
              for c in vs.candidates
          ]
      )
      if vs.is_noneable:
        candidate_str += ', None'
      return f'Union[{candidate_str}]'

    if isinstance(vs, pg.typing.Bool):
      x = 'bool'
    elif isinstance(vs, pg.typing.Str):
      if vs.regex is None:
        x = 'str'
      else:
        if strict:
          x = f"pg.typing.Str(regex='{vs.regex.pattern}')"
        else:
          x = f"str(regex='{vs.regex.pattern}')"
    elif isinstance(vs, pg.typing.Number):
      constraints = []
      min_label = 'min_value' if strict else 'min'
      max_label = 'max_value' if strict else 'max'
      if vs.min_value is not None:
        constraints.append(f'{min_label}={vs.min_value}')
      if vs.max_value is not None:
        constraints.append(f'{max_label}={vs.max_value}')
      x = 'int' if isinstance(vs, pg.typing.Int) else 'float'
      if constraints:
        if strict:
          x = (
              'pg.typing.Int'
              if isinstance(vs, pg.typing.Int)
              else 'pg.typing.Float'
          )
        x += '(' + ', '.join(constraints) + ')'
    elif isinstance(vs, pg.typing.Object):
      x = vs.cls.__name__
    elif isinstance(vs, pg.typing.List):
      item_str = self.annotate(vs.element.value, strict=strict)
      x = f'list[{item_str}]'
    elif isinstance(vs, pg.typing.Tuple):
      elem_str = ', '.join(
          [self.annotate(el.value, strict=strict) for el in vs.elements]
      )
      x = f'tuple[{elem_str}]'
    elif isinstance(vs, pg.typing.Dict):
      kv_pairs = None
      if vs.schema is not None:
        kv_pairs = [
            (k, self.annotate(f.value, strict=strict))
            for k, f in vs.schema.items()
            if isinstance(k, pg.typing.ConstStrKey)
        ]

      if kv_pairs:
        kv_str = ', '.join(f'\'{k}\': {v}' for k, v in kv_pairs)
        x = '{' + kv_str + '}'
        if strict:
          x = f'pg.typing.Dict({x})'
      else:
        x = 'dict[str, Any]'

    else:
      raise TypeError(f'Unsupported value spec being used as schema: {vs}.')

    if annotate_optional and vs.is_noneable:
      x += ' | None'
    return x


class SchemaJsonRepr(SchemaRepr):
  """JSON-representation for a schema."""

  def repr(self, schema: Schema) -> str:
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


#
# Value representations.
#


class ValueRepr(metaclass=abc.ABCMeta):
  """Base class for value representation."""

  @abc.abstractmethod
  def repr(self, value: Any, schema: Schema | None = None, **kwargs) -> str:
    """Returns the representation of a structured value."""

  @abc.abstractmethod
  def parse(self, text: str, schema: Schema | None = None, **kwargs) -> Any:
    """Parse a LM generated text into a structured value."""


class ValuePythonRepr(ValueRepr):
  """Python-representation for value."""

  def repr(self,
           value: Any,
           schema: Schema | None = None,
           *,
           compact: bool = True,
           verbose: bool = False,
           markdown: bool = True,
           **kwargs) -> str:
    del schema
    object_code = pg.format(
        value, compact=compact, verbose=verbose, python_format=True)
    if markdown:
      return f'```python\n{ object_code }\n```'
    return object_code

  def parse(self,
            text: str,
            schema: Schema | None = None,
            **kwargs) -> Any:
    context = {
        'pg': pg,
        'Any': typing.Any,
        'List': typing.List,
        'Tuple': typing.Tuple,
        'Dict': typing.Dict,
        'Sequence': typing.Sequence,
        'Optional': typing.Optional,
        'Union': typing.Union,
    }
    if schema is not None:
      dependencies = schema.class_dependencies()
      context.update({d.__name__: d for d in dependencies})
    with lf_coding.context(**context):
      return lf_coding.run(text)['__result__']


class ValueJsonRepr(ValueRepr):
  """JSON-representation for value."""

  def repr(self, value: Any, schema: Schema | None = None, **kwargs) -> str:
    del schema
    return pg.to_json_str(dict(result=value))

  def parse(self, text: str, schema: Schema | None = None, **kwargs) -> Any:
    """Parse a JSON string into a structured object."""
    del schema
    v = pg.from_json_str(self._cleanup_json(text))
    if not isinstance(v, dict) or 'result' not in v:
      raise ValueError(
          'The root node of the JSON must be a dict with key `result`. '
          f'Encountered: {v}'
      )
    return v['result']

  def _cleanup_json(self, json_str: str) -> str:
    """Clean up the LM responded JSON string."""
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


def schema_repr(protocol: Literal['json', 'python']) -> SchemaRepr:
  """Gets a SchemaRepr object from protocol."""
  if protocol == 'json':
    return SchemaJsonRepr()
  elif protocol == 'python':
    return SchemaPythonRepr()
  raise ValueError(f'Unsupported protocol: {protocol}.')


def value_repr(protocol: Literal['json', 'python']) -> ValueRepr:
  if protocol == 'json':
    return ValueJsonRepr()
  elif protocol == 'python':
    return ValuePythonRepr()
  raise ValueError(f'Unsupported protocol: {protocol}.')
