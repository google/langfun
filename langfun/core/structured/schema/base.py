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
"""Schema and Prompting Protocol for Structured Data."""

import abc
import copy
import dataclasses
import inspect
import io
import re
import textwrap
import typing
from typing import Any, Callable, ClassVar, Sequence, Type, Union
import langfun.core as lf
import pyglove as pg


def _parse_value_spec(value) -> pg.typing.ValueSpec:
  """Parses a PyGlove ValueSpec equivalent into a ValueSpec.

  Examples:
  ```
  _parse_value_spec(int) -> pg.typing.Int
  _parse_value_spec(list[int]) -> pg.typing.List(pg.typing.Int)
  _parse_value_spec(dict(a=int, b=str)) -> pg.typing.Dict(
      pg.typing.Int, pg.typing.Str
  )
  ```
  Args:
    value: The value to parse. It can be a PyGlove ValueSpec, a dict with a
      single 'result' key, or a Python type annotation.

  Returns:
    A PyGlove ValueSpec.
  """
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
          (  # pyrefly: ignore[invalid-argument]
              pg.typing.Any,
              pg.typing.Callable,
              pg.typing.Tuple,
              pg.typing.Type,
              pg.typing.Union,
          ),
      ):
        raise ValueError(f'Unsupported schema specification: {v}')
      return spec

  return _parse_node(value)


class SchemaError(Exception):   # pylint: disable=g-bad-exception-name
  """Schema error."""

  def __init__(
      self,
      schema: 'Schema',
      value: Any,
      protocol: str,
      cause: Exception
  ):
    self.schema = schema
    self.value = value
    self.protocol = protocol
    self.cause = cause

  def __str__(self):
    r = io.StringIO()
    r.write(
        pg.colored(
            f'{self.cause.__class__.__name__}: {self.cause}', 'magenta'
        )
    )

    r.write('\n')
    r.write(pg.colored('Schema:', 'red'))
    r.write('\n\n')
    r.write(textwrap.indent(
        pg.colored(
            schema_repr(self.schema, protocol=self.protocol), 'magenta'
        ),
        ' ' * 2
    ))
    r.write('\n\n')
    r.write(pg.colored('Generated value:', 'red'))
    r.write('\n\n')
    r.write(textwrap.indent(
        pg.colored(value_repr(self.value, protocol=self.protocol), 'magenta'),
        ' ' * 2
    ))
    return r.getvalue()


class Schema(
    lf.NaturalLanguageFormattable,
    pg.Object,
    pg.views.HtmlTreeView.Extension
):
  """Schema for structured inputs and outputs.

  `lf.Schema` provides a unified representation for defining the output schema
  used in Langfun's structured operations like `lf.query`, `lf.parse`,
  `lf.complete`, and `lf.describe`. It acts as an abstraction layer,
  allowing schemas to be defined using Python type annotations, `pg.Object`
  classes, or dictionaries, and then converting them into a format that
  language models can understand.

  `lf.Schema` can be created from various types using `lf.Schema.from_value`:
  *   Built-in types: `int`, `str`, `bool`, `float`
  *   Typing constructs: `list`, `dict`, `typing.Union`, `typing.Literal`,
      `typing.Optional`
  *   PyGlove classes: `pg.Object` subclasses

  **1. Creating a Schema:**

  ```python
  import langfun as lf
  import pyglove as pg
  from typing import Literal, Union

  # From a basic type
  int_schema = lf.Schema.from_value(int)

  # From a list type
  list_schema = lf.Schema.from_value(list[int])

  # From a dictionary
  dict_schema = lf.Schema.from_value(dict(a=int, b=str))

  # From pg.Object
  class Point(pg.Object):
    x: int
    y: int
  point_schema = lf.Schema.from_value(Point)

  # From Union or Literal
  union_schema = lf.Schema.from_value(Union[int, str])
  literal_schema = lf.Schema.from_value(Literal['A', 'B'])
  ```

  **2. Schema Representation:**
  Once created, a schema object can represent itself in different formats,
  such as Python-like syntax or JSON, which is used in prompts to LLMs.

  ```python
  print(point_schema.repr('python'))
  # Output:
  # class Point:
  #   x: int
  #   y: int

  print(dict_schema.repr('json'))
  # Output:
  # {
  #   "a": "int",
  #   "b": "str"
  # }
  ```
  """

  spec: pg.typing.Annotated[
      pg.typing.Object(pg.typing.ValueSpec, transform=_parse_value_spec),
      (
          'A PyGlove ValueSpec object representing the spec for the value '
          'to be parsed.'
      ),
  ]

  def schema_repr(self, protocol: str = 'python', **kwargs) -> str:
    """Returns the representation of the schema."""
    return schema_repr(self, protocol=protocol, **kwargs)

  def value_repr(
      self, value: Any, protocol: str = 'python', **kwargs
  ) -> str:
    """Returns the representation of a structured value."""
    return value_repr(value, schema=self, protocol=protocol, **kwargs)

  def parse_value(
      self, text: str, protocol: str = 'python', **kwargs
  ) -> Any:
    """Parses a LM generated text into a structured value."""
    value = parse_value(text, schema=self, protocol=protocol, **kwargs)

    # TODO(daiyip): support autofix for schema error.
    try:
      return self.spec.apply(value)
    except Exception as e:
      raise SchemaError(self, value, protocol, e)  # pylint: disable=raise-missing-from

  def natural_language_format(self) -> str:
    return self.schema_str()

  def schema_dict(self) -> dict[str, Any]:
    """Returns the dictionary representation of the schema."""

    def _node(vs: pg.typing.ValueSpec) -> Any:
      if isinstance(vs, pg.typing.PrimitiveType):
        return vs
      elif isinstance(vs, pg.typing.Dict):
        assert vs.schema is not None
        return {str(k): _node(f.value) for k, f in vs.schema.fields.items()}
      elif isinstance(vs, pg.typing.List):
        return [_node(vs.element.value)]
      elif isinstance(vs, pg.typing.Object):
        if issubclass(vs.cls, pg.Object):
          d = {pg.JSONConvertible.TYPE_NAME_KEY: vs.cls.__serialization_key__}
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
      include_base_classes: bool = True,
      include_subclasses: bool = True,
      include_generated_subclasses: bool = False) -> list[Type[Any]]:
    """Returns a list of class dependencies for current schema."""
    return class_dependencies(
        self.spec,
        include_base_classes,
        include_subclasses,
        include_generated_subclasses
    )

  @classmethod
  def from_value(cls, value) -> 'Schema':
    """Creates a schema from an equivalent representation."""
    if isinstance(value, Schema):
      return value
    return cls(_parse_value_spec(value))

  def _html_tree_view_content(
      self,
      *,
      view: pg.views.HtmlTreeView,
      **kwargs,
  ):
    return pg.Html.element(
        'div',
        [pg.Html.escape(self.schema_repr(protocol='python'))],
        css_classes=['lf-schema-definition']
    ).add_style(
        """
        .lf-schema-definition {
            color: blue;
            margin: 5px;
            white-space: pre-wrap;
        }
        """
    )


SchemaType = Union[Schema, Type[Any], list[Type[Any]], dict[str, Any]]


def _top_level_object_specs_from_value(value: pg.Symbolic) -> list[Type[Any]]:
  """Returns a list of top level value specs from a symbolic value."""
  top_level_object_specs = []

  def _collect_top_level_object_specs(k, v, p):
    del k, p
    if isinstance(v, pg.Object):
      top_level_object_specs.append(pg.typing.Object(v.__class__))
      return pg.TraverseAction.CONTINUE
    return pg.TraverseAction.ENTER

  pg.traverse(value, _collect_top_level_object_specs)
  return top_level_object_specs


def class_dependencies(
    value_or_spec: Union[
        pg.Symbolic,
        Schema,
        pg.typing.ValueSpec,
        Type[pg.Object],
        tuple[Union[pg.typing.ValueSpec, Type[pg.Object]], ...],
    ],
    include_base_classes: bool = True,
    include_subclasses: bool = True,
    include_generated_subclasses: bool = False,
) -> list[Type[Any]]:
  """Returns a list of class dependencies from a value or specs."""
  if isinstance(value_or_spec, Schema):
    value_or_spec = value_or_spec.spec

  if inspect.isclass(value_or_spec) or isinstance(
      value_or_spec, pg.typing.ValueSpec
  ):
    value_or_spec = (value_or_spec,)  # pyrefly: ignore[bad-assignment]

  if isinstance(value_or_spec, tuple):
    value_specs = []
    for v in value_or_spec:
      if isinstance(v, pg.typing.ValueSpec):
        value_specs.append(v)
      elif inspect.isclass(v):
        value_specs.append(pg.typing.Object(v))
      else:
        raise TypeError(f'Unsupported spec type: {v!r}')
  else:
    value_specs = _top_level_object_specs_from_value(value_or_spec)  # pyrefly: ignore[bad-argument-type]

  seen = set()
  dependencies = []

  def _add_dependency(cls_or_classes):
    if isinstance(cls_or_classes, type):
      cls_or_classes = [cls_or_classes]
    for cls in cls_or_classes:
      if cls not in dependencies:
        dependencies.append(cls)

  def _fill_dependencies(vs: pg.typing.ValueSpec, include_subclasses: bool):
    if isinstance(vs, pg.typing.Object):
      cls = vs.cls
      if (not include_generated_subclasses
          and getattr(cls, '__module__', 'builtins') == 'builtins'):
        return

      if cls not in seen:
        seen.add(cls)

        if include_base_classes:
          # Add base classes as dependencies.
          for base_cls in cls.__bases__:
            # We only keep track of user-defined symbolic classes.
            if base_cls is not object and base_cls is not pg.Object:
              _fill_dependencies(
                  pg.typing.Object(base_cls), include_subclasses=False
              )

        # Add members as dependencies.
        for field in pg.schema(cls).values():
          _fill_dependencies(field.value, include_subclasses)
      _add_dependency(cls)

      # Check subclasses if available.
      if include_subclasses:
        for subcls in cls.__subclasses__():
          # NOTE(daiyip): To prevent LLM-generated "hallucinated" classes from
          # polluting the generation space, classes dynamically created by
          # 'eval' (which have __module__ == 'builtins') are excluded from
          # dependencies by default.
          if ((include_generated_subclasses
               or getattr(subcls, '__module__', 'builtins') != 'builtins')
              and subcls not in dependencies):
            _fill_dependencies(
                pg.typing.Object(subcls), include_subclasses=True
            )

    if isinstance(vs, pg.typing.List):
      _fill_dependencies(vs.element.value, include_subclasses)
    elif isinstance(vs, pg.typing.Tuple):
      for elem in vs.elements:
        _fill_dependencies(elem.value, include_subclasses)
    elif isinstance(vs, pg.typing.Dict) and vs.schema:
      for v in vs.schema.values():
        _fill_dependencies(v.value, include_subclasses)
    elif isinstance(vs, pg.typing.Union):
      for v in vs.candidates:
        _fill_dependencies(v, include_subclasses)

  for value_spec in value_specs:
    _fill_dependencies(value_spec, include_subclasses)  # pyrefly: ignore[bad-argument-type]
  return dependencies


def schema_spec(noneable: bool = False) -> pg.typing.ValueSpec:  # pylint: disable=unused-argument
  if typing.TYPE_CHECKING:
    return Any  # pyrefly: ignore[bad-return]
  return pg.typing.Object(
      Schema, transform=Schema.from_value, is_noneable=noneable
  )  # pylint: disable=unreachable-code


def annotation(
    vs: pg.typing.ValueSpec,
    annotate_optional: bool = True,
    strict: bool = False,
    allowed_dependencies: set[Type[Any]] | None = None,
) -> str:
  """Returns the annotation string for a value spec."""
  child_annotation_kwargs = dict(
      strict=strict, allowed_dependencies=allowed_dependencies
  )
  if isinstance(vs, pg.typing.Any):  # pyrefly: ignore[invalid-argument]
    return 'Any'
  elif isinstance(vs, pg.typing.Enum):
    candidate_str = ', '.join([repr(v) for v in vs.values])
    return f'Literal[{candidate_str}]'
  elif isinstance(vs, pg.typing.Union):
    candidate_str = ', '.join(
        [
            annotation(c, annotate_optional=False, **child_annotation_kwargs)
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
    if allowed_dependencies is None or vs.cls in allowed_dependencies:
      x = vs.cls.__name__
    else:
      x = 'Any'
  elif isinstance(vs, pg.typing.List):
    item_str = annotation(vs.element.value, **child_annotation_kwargs)
    x = f'list[{item_str}]'
  elif isinstance(vs, pg.typing.Tuple):
    elem_str = ', '.join(
        [annotation(el.value, **child_annotation_kwargs) for el in vs.elements]
    )
    x = f'tuple[{elem_str}]'
  elif isinstance(vs, pg.typing.Dict):
    kv_pairs = None
    if vs.schema is not None:
      kv_pairs = [
          (k, annotation(f.value, **child_annotation_kwargs))
          for k, f in vs.schema.items()
          if isinstance(k, pg.typing.ConstStrKey)
      ]

    if kv_pairs:
      kv_str = ', '.join(f"'{k}': {v}" for k, v in kv_pairs)
      x = '{' + kv_str + '}'
      if strict:
        x = f'pg.typing.Dict({x})'
    elif vs.schema and vs.schema.dynamic_field:
      v = annotation(vs.schema.dynamic_field.value, **child_annotation_kwargs)
      x = f'dict[str, {v}]'
    else:
      x = 'dict[str, Any]'

  else:
    raise TypeError(f'Unsupported value spec being used as schema: {vs}.')

  if annotate_optional and vs.is_noneable:
    x += ' | None'
  return x

#
# Prompting protocols for structured data.
#


def _check_name_part(kind: str, value: str) -> None:
  """Raises ValueError if `value` cannot be a part of a protocol name."""
  if not re.fullmatch(r'[^:+]+', value):
    raise ValueError(f'{kind} {value!r} must be non-empty, without ":" or "+".')


class _ProtocolFeature:
  """A named, versioned addition to base prompting protocols.

  A protocol name lists the features it uses after its base, e.g.
  `python:2.0+feature:1.0` (see `ProtocolName`). Subclasses register on
  definition, and override the hooks they need; a base protocol decides which
  hooks it calls.
  """

  NAME: ClassVar[str]
  VERSION: ClassVar[str]

  # Base protocol name -> the base versions this feature supports, or None for
  # all of them.
  BASES: ClassVar[dict[str, tuple[str, ...] | None]]

  # Whether a protocol name that gives no version of this feature means this
  # version. A feature with several versions marks exactly one, so a new
  # version changes what such names mean only if it takes the mark.
  DEFAULT: ClassVar[bool] = False

  # Features are ordered by rank, then by name, in normalized protocol names and
  # in `PromptingProtocol.features`, so the order does not depend on imports.
  # All versions of a feature share its rank.
  RANK: ClassVar[int] = 0

  # Name -> version -> class.
  _REGISTRY: ClassVar[dict[str, dict[str, Type['_ProtocolFeature']]]] = {}

  def __init_subclass__(cls):
    super().__init_subclass__()
    _check_name_part('Feature name', cls.NAME)
    _check_name_part('Feature version', cls.VERSION)
    versions = dict(_ProtocolFeature._REGISTRY.get(cls.NAME, {}))
    registered = versions.get(cls.VERSION)
    if registered is not None and (
        registered.__module__, registered.__qualname__
    ) != (cls.__module__, cls.__qualname__):
      raise ValueError(
          f'Feature {cls.NAME}:{cls.VERSION} is already registered by '
          f'{registered!r}.'
      )
    versions[cls.VERSION] = cls
    defaults = sorted(v for v, c in versions.items() if c.DEFAULT)
    if len(versions) > 1 and len(defaults) != 1:
      raise ValueError(
          f'Feature {cls.NAME!r} has versions {sorted(versions)}, so exactly '
          f'one of them must set `DEFAULT = True`; found {defaults}.'
      )
    ranks = sorted({c.RANK for c in versions.values()})
    if len(ranks) > 1:
      raise ValueError(
          f'Versions of feature {cls.NAME!r} have ranks {ranks}, but all '
          'versions of a feature must share its rank.'
      )
    _ProtocolFeature._REGISTRY[cls.NAME] = versions

  @classmethod
  def get(cls, name: str, version: str) -> Type['_ProtocolFeature']:
    """Returns the registered class of feature `name:version`."""
    return cls._REGISTRY[name][version]

  @classmethod
  def supports(cls, base: str, version: str | None) -> bool:
    """Returns whether this feature supports `base:version` (None: any)."""
    if base not in cls.BASES:
      return False
    versions = cls.BASES[base]
    return version is None or versions is None or version in versions

  @classmethod
  def resolve(
      cls,
      requested: dict[str, str | None],
      base: str,
      base_version: str | None,
      protocol: str,
  ) -> tuple[tuple[str, str], ...]:
    """Returns (name, version) of the requested features.

    Args:
      requested: Feature name -> version, or None for its default version.
      base: The base protocol name.
      base_version: The base protocol version, or None if not known yet.
      protocol: The protocol name, for error messages.

    Returns:
      The features, by rank and then by name.

    Raises:
      ValueError: If a feature or version is unknown, or does not support the
        base protocol.
    """
    resolved: list[Type[_ProtocolFeature]] = []
    for name, version in requested.items():
      versions = cls._REGISTRY.get(name)
      if versions is None:
        raise ValueError(
            f'Unknown feature {name!r} in protocol {protocol!r}. Available '
            f'features: {list(cls._REGISTRY)}.'
        )
      if version is None:
        # The marked default, or the only version (see `__init_subclass__`).
        version = next(
            (v for v, c in versions.items() if c.DEFAULT), next(iter(versions))
        )
      feature_cls = versions.get(version)
      if feature_cls is None:
        raise ValueError(
            f'Version {version!r} is not supported for feature {name!r}. '
            f'Available versions: {sorted(versions)}.'
        )
      if not feature_cls.supports(base, base_version):
        raise ValueError(
            f'Feature {name}:{version} in protocol {protocol!r} does not '
            f'support {base}:{base_version or "*"}. Supported: '
            f'{feature_cls.BASES}.'
        )
      resolved.append(feature_cls)
    return tuple(
        (f.NAME, f.VERSION)
        for f in sorted(resolved, key=lambda c: (c.RANK, c.NAME))
    )

  def applies_to(self, schema: 'Schema | None') -> bool:
    """Returns False if queries of `schema` should run without this feature."""
    del schema
    return True

  def query_preamble(self, base_preamble: str) -> str | None:
    """Returns the `lf.query` preamble to use instead of `base_preamble`."""
    del base_preamble
    return None

  def parse_instructions(self) -> str | None:
    """Returns instructions to append to `lf.parse`'s or a caller's preamble."""
    return None

  def field_hint(self, field: pg.typing.Field, schema: 'Schema') -> str | None:
    """Returns a comment to render above `field` in class definitions.

    Args:
      field: A field of a class of `schema`.
      schema: The schema whose classes are rendered.
    """
    del field, schema
    return None

  def render_value(
      self,
      value: Any,
      schema: 'Schema | None',
      render: Callable[[Any], str],
  ) -> str:
    """Renders a value (not a class) for the prompt.

    Args:
      value: The value.
      schema: The schema of `value`, if known.
      render: Renders a value as the protocol without this feature does.

    Returns:
      The representation of `value`.
    """
    del schema
    return render(value)

  def parse_hooks(self, schema: 'Schema | None') -> '_ParseHooks':
    """Returns the parse hooks for one response of `schema`."""
    del schema
    return _ParseHooks()


class _ParseHooks:
  """Parse hooks of a feature for one response.

  The base protocol runs each phase for every feature, in feature order (see
  `_ProtocolFeature.RANK`), before the next phase.
  """

  def before_exec(self, text: str) -> str:
    """Phase 1: returns the response with this feature's syntax handled."""
    return text

  def check_code(self, code: str, global_vars: dict[str, Any]) -> None:
    """Phase 2: checks the code to run, and may add names to `global_vars`."""
    del code, global_vars

  def after_exec(self, value: Any, final_code: str) -> Any:
    """Phase 3: checks the parsed value.

    Args:
      value: The parsed value.
      final_code: The code that produced `value`; autofix may have rewritten
        the code passed to `check_code`.

    Returns:
      The value, possibly rewritten.
    """
    del final_code
    return value


_PROTOCOL_NAME_RE = re.compile(
    r'(?P<base>[^:+]+)(?::(?P<version>[^:+]+))?'
    r'(?P<features>(?:\+[^:+]+(?::[^:+]+)?)*)'
)
_PROTOCOL_FEATURE_RE = re.compile(r'\+([^:+]+)(?::([^:+]+))?')


@dataclasses.dataclass(frozen=True)
class ProtocolName:
  """A prompting protocol name: `base[:version](+feature[:version])*`.

  E.g. `python`, `python:2.0`, or a base with features such as
  `python:2.0+feature:1.0`. `parse` fills in feature versions and orders
  features by rank, so a protocol has a single name once its base version is
  known.
  """

  base: str
  version: str | None = None
  features: tuple[tuple[str, str], ...] = ()

  @classmethod
  def parse(
      cls, name: str, default_version: str | None = None
  ) -> 'ProtocolName':
    """Parses and normalizes a protocol name.

    Args:
      name: The protocol name.
      default_version: The base version to use if `name` has none.

    Returns:
      The normalized name.

    Raises:
      ValueError: If `name` is malformed, or has an unknown, repeated or
        unsupported feature.
    """
    match = _PROTOCOL_NAME_RE.fullmatch(name) if isinstance(name, str) else None
    if match is None:
      raise ValueError(
          f'Invalid protocol {name!r}: expected '
          '`base[:version](+feature[:version])*`.'
      )
    version = match['version'] or default_version
    requested = {}
    for feature, feature_version in _PROTOCOL_FEATURE_RE.findall(
        match['features']
    ):
      if feature in requested:
        raise ValueError(
            f'Feature {feature!r} appears twice in protocol {name!r}.'
        )
      requested[feature] = feature_version or None
    features = _ProtocolFeature.resolve(
        requested, match['base'], version, name
    )
    return cls(match['base'], version, features)

  def __str__(self) -> str:
    name = self.base if self.version is None else f'{self.base}:{self.version}'
    return name + ''.join(f'+{f}:{v}' for f, v in self.features)


class PromptingProtocol(metaclass=abc.ABCMeta):
  """Base class for prompting protocols for structured data."""

  NAME: ClassVar[str]

  # Whether `parse_value` repairs a malformed response with an LM when given
  # `autofix`; `lf.query` passes `autofix=0` to protocols without support.
  SUPPORTS_AUTOFIX: ClassVar[bool] = False

  _PROTOCOLS: ClassVar[dict[str, Type['PromptingProtocol']]] = {}

  # The base version and features of the protocol name. `from_name` sets them
  # on an instance it builds without arguments, so a subclass may have its own
  # constructor. The base version selects the `lf.query` prompt; it does not
  # change the representations here.
  version: str | None = None
  features: Sequence[_ProtocolFeature] = ()

  def __init_subclass__(cls):
    _check_name_part('Protocol name', cls.NAME)
    PromptingProtocol._PROTOCOLS[cls.NAME] = cls

  @property
  def name(self) -> str:
    """The protocol name, with its version and features if any."""
    return str(
        ProtocolName(
            self.NAME,
            self.version,
            tuple((f.NAME, f.VERSION) for f in self.features),
        )
    )

  @classmethod
  def from_name(cls, name: str) -> 'PromptingProtocol':
    """Returns the prompting protocol for a name such as `python:2.0`."""
    parsed = ProtocolName.parse(name)
    protocol_cls = cls._PROTOCOLS.get(parsed.base)
    if protocol_cls is None:
      raise ValueError(f'Unsupported protocol: {name}.')
    protocol = protocol_cls()
    protocol.version = parsed.version
    protocol.features = tuple(
        _ProtocolFeature.get(f, v)() for f, v in parsed.features
    )
    return protocol

  @classmethod
  def supports_autofix(cls, name: str) -> bool:
    """Returns True if the base protocol of `name` supports autofix."""
    protocol_cls = cls._PROTOCOLS.get(ProtocolName.parse(name).base)
    return protocol_cls is not None and protocol_cls.SUPPORTS_AUTOFIX

  def for_schema(self, schema: Schema | None) -> 'PromptingProtocol':
    """Returns this protocol without the features that skip `schema`."""
    return self._with_features(
        [f for f in self.features if f.applies_to(schema)]
    )

  def _with_features(
      self, features: Sequence[_ProtocolFeature]
  ) -> 'PromptingProtocol':
    """Returns a copy of this protocol with `features` instead of its own."""
    # A copy, not a new instance, keeps a subclass's constructor arguments.
    protocol = copy.copy(self)
    protocol.features = tuple(features)
    return protocol

  def query_preamble(self, base_preamble: str) -> str | None:
    """Returns a feature's `lf.query` preamble to use instead of the base one."""
    for feature in self.features:
      preamble = feature.query_preamble(base_preamble)
      if preamble is not None:
        return preamble
    return None

  def parse_instructions(self) -> str | None:
    """Returns the instructions features add to a preamble."""
    instructions = [
        text for f in self.features if (text := f.parse_instructions())
    ]
    return '\n\n'.join(instructions) or None

  def add_instructions(self, preamble: str | lf.Template) -> str | lf.Template:
    """Returns `preamble` followed by the `parse_instructions`, if any."""
    instructions = self.parse_instructions()
    if instructions is None:
      return preamble
    if isinstance(preamble, lf.Template):
      return lf.Template.from_value(
          preamble, template_str=f'{preamble.template_str}\n\n{instructions}'
      )
    # Dedented as `lf.Template` would, which the instructions would prevent.
    return f'{inspect.cleandoc(preamble)}\n\n{instructions}'

  @abc.abstractmethod
  def schema_repr(self, schema: Schema) -> str:
    """Returns the representation of the schema."""

  @abc.abstractmethod
  def value_repr(
      self,
      value: Any,
      schema: Schema | None = None,
      **kwargs
  ) -> str:
    """Returns the representation of a structured value."""

  @abc.abstractmethod
  def parse_value(
      self,
      text: str,
      schema: Schema | None = None,
      **kwargs
  ) -> Any:
    """Parses a LM generated text into a structured value."""


def schema_repr(
    schema: Schema,
    *,
    protocol: str = 'python',
    **kwargs
) -> str:
  """Returns the representation of the schema based on the protocol."""
  return PromptingProtocol.from_name(protocol).schema_repr(schema, **kwargs)


def value_repr(
    value: Any,
    schema: Schema | None = None,
    *,
    protocol: str = 'python',
    **kwargs) -> str:
  """Returns the representation of a structured value based on the protocol."""
  return PromptingProtocol.from_name(protocol).value_repr(
      value, schema, **kwargs
  )


def parse_value(
    text: str,
    schema: Schema | None = None,
    *,
    protocol: str = 'python',
    **kwargs
) -> Any:
  """Parses a LM generated text into a structured value."""
  return PromptingProtocol.from_name(protocol).parse_value(
      text, schema=schema, **kwargs
  )


#
# Special value markers.
#


class Missing(pg.Object, pg.typing.CustomTyping):
  """Value marker for a missing field.

  This class differs from pg.MISSING_VALUE in two aspects:
  * When a field is assigned with lf.Missing(), it's considered non-partial.
  * lf.Missing() could format the value spec as Python annotations that are
    consistent with `lf.structured.Schema.schema_repr()`.
  """

  def _on_bound(self):
    super()._on_bound()
    self._value_spec = None

  @property
  def value_spec(self) -> pg.ValueSpec | None:
    """Returns the value spec that applies to the current missing value."""
    return self._value_spec

  def custom_apply(
      self, path: pg.KeyPath, value_spec: pg.ValueSpec, *args, **kwargs
  ) -> tuple[bool, Any]:
    self._value_spec = value_spec
    return (False, self)

  def format(self, *args, **kwargs) -> str:
    if self._value_spec is None:
      return 'MISSING'
    return f'MISSING({annotation(self._value_spec)})'

  @classmethod
  def find_missing(cls, value: Any) -> dict[str, 'Missing']:
    """Lists all missing values contained in the value."""
    missing = {}

    def _visit(k, v, p):
      del p
      if isinstance(v, Missing):
        missing[k] = v
      return pg.TraverseAction.ENTER

    pg.traverse(value, _visit)
    return missing


MISSING = Missing()


def mark_missing(value: Any) -> Any:
  """Replaces pg.MISSING within the value with lf.structured.Missing objects."""
  if isinstance(value, list):
    value = pg.List(value)
  elif isinstance(value, dict):
    value = pg.Dict(value)
  if isinstance(value, pg.Symbolic):

    def _mark_missing(k, v, p):
      del k, p
      if pg.MISSING_VALUE == v:
        v = Missing()
      return v

    return value.rebind(_mark_missing, raise_on_no_change=False)
  return value


class Unknown(pg.Object, pg.typing.CustomTyping):
  """Value marker for a field that LMs could not provide."""

  def custom_apply(self, *args, **kwargs) -> tuple[bool, Any]:
    return (False, self)

  def format(self, *args, **kwargs) -> str:
    return 'UNKNOWN'


UNKNOWN = Unknown()
