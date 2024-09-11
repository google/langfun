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
"""langfun Component."""

import contextlib
import dataclasses
import threading
from typing import Annotated, Any, ContextManager, Iterator, Type
import pyglove as pg


# Default value marker that indicates to raise error.
RAISE_IF_HAS_ERROR = (pg.MISSING_VALUE,)


class Component(pg.Object):
  """Base class for langfun components."""

  # Override __repr__ format to use inferred values when available.
  __repr_format_kwargs__ = dict(
      compact=True,
      use_inferred=True,
  )

  # Override __str__ format to use inferred values when available.
  __str_format_kwargs__ = dict(
      compact=False,
      verbose=False,
      use_inferred=True,
  )

  # Allow symbolic assignment, which invalidates the object and recomputes
  # states upon update.
  allow_symbolic_assignment = True

  # Class property that indicates whether to use `sym_eq` for `__eq__`,
  # `sym_ne` for `__ne__`, and `sym_hash` for `__hash__`.
  use_symbolic_comparison = False

  def __init_subclass__(cls):
    super().__init_subclass__()

    # Find class attributes that do not have annotations but are `Component`,
    # and treat them as symbolic fields.
    additional_fields = []
    for attr_name in dir(cls):
      if (
          attr_name.startswith('_')
          or attr_name.isupper()
          or attr_name in cls.__schema__.fields
      ):
        continue
      attr_value = getattr(cls, attr_name)
      if isinstance(attr_value, pg.Inferentiable):
        value_spec = pg.typing.Any()
      elif isinstance(attr_value, Component):
        value_spec = pg.typing.Object(Component)
      else:
        value_spec = None
      if value_spec:
        field = pg.typing.create_field((attr_name, value_spec))
        field.value.set_default(attr_value)
        additional_fields.append(field)
    if additional_fields:
      cls.update_schema(additional_fields)

  def _on_bound(self):
    super()._on_bound()
    self._tls = threading.local()

  def _sym_inferred(self, key: str, **kwargs):
    """Override to allow attribute to access scoped value.

    Args:
      key: attribute name.
      **kwargs: Optional keyword arguments for value inference.

    Returns:
      The value of the symbolic attribute. If not available, returns the
        default value.

    Raises:
      AttributeError: If the attribute does not exist or contextual attribute
        is not ready.
    """
    if key not in self._sym_attributes:
      raise AttributeError(key)

    # Step 1: Try use value from `self.override`.
    # The reason is that `self.override` is short-lived and explicitly specified
    # by the user in scenarios like `LangFunc.render`, which should not be
    # affected by `lf.context`.
    v = _get_scoped_value(self._tls, _CONTEXT_OVERRIDES, key)
    if v is not None:
      return v.value

    # Step 2: Try use value from `lf.context` with `override_attrs`.
    # This gives users a chance to override the bound attributes of components
    # from the top, allowing change of bindings without modifying the code
    # that produces the components.
    override = get_contextual_override(key)
    if override and override.override_attrs:
      return override.value

    # Step 3: Try use value from the symbolic tree, starting from self to
    # the root of the tree.
    # Step 4: If the value is not present, use the value from `context()` (
    # override_attrs=False).
    # Step 5: Otherwise use the default value from `ContextualAttribute`.
    return super()._sym_inferred(key, context_override=override, **kwargs)

  def override(
      self, **kwargs) -> ContextManager[dict[str, 'ContextualOverride']]:
    """Context manager to override the attributes of this component."""
    vs = {k: ContextualOverride(v) for k, v in kwargs.items()}
    return _contextual_scope(self._tls, _CONTEXT_OVERRIDES, **vs)

  def __getattribute__(self, name: str) -> Any:
    """Override __getattribute__ to deal with class attribute override."""
    if not name.startswith('_') and hasattr(self.__class__, name):
      tls = self.__dict__.get('_tls', None)
      if tls is not None:
        v = _get_scoped_value(tls, _CONTEXT_OVERRIDES, name)
        if v is not None:
          return v.value
    return super().__getattribute__(name)


_global_tls = threading.local()

_CONTEXT_OVERRIDES = 'context_overrides'


@dataclasses.dataclass(frozen=True)
class ContextualOverride:
  """Value marker for contextual override for an attribute."""

  # Overridden value.
  value: Any

  # If True, this override will apply to both current scope and nested scope,
  # meaning current `lf.context` will take precedence over all nested
  # `lf.context` on this attribute.
  cascade: bool = False

  # If True, this override will apply to attributes that already have values.
  override_attrs: bool = False


def context(
    *,
    cascade: bool = False,
    override_attrs: bool = False,
    **variables,
) -> ContextManager[dict[str, ContextualOverride]]:
  """Context manager to provide overriden values for contextual attributes.

  Args:
    cascade: If True, this override will apply to both current scope and nested
      scope, meaning that this `lf.context` will take precedence over all
      nested `lf.context` on the overriden variables.
    override_attrs: If True, this override will apply to attributes that already
      have values. Otherwise overridden variables will only be used for
      contextual attributes whose values are not present.
    **variables: Key/values as override for contextual attributes.

  Returns:
    A dict of attribute names to their contextual overrides.
  """
  vs = {}
  for k, v in variables.items():
    if not isinstance(v, ContextualOverride):
      v = ContextualOverride(v, cascade, override_attrs)
    vs[k] = v
  return _contextual_scope(_global_tls, _CONTEXT_OVERRIDES, **vs)


def use_settings(
    *,
    cascade: bool = False,
    **settings,
) -> ContextManager[dict[str, ContextualOverride]]:
  """Shortcut method for overriding component attributes.

  Args:
    cascade: If True, this override will apply to both current scope and nested
      scope, meaning that this `lf.context` will take precedence over all
      nested `lf.context` on the overriden variables.
    **settings: Key/values as override for component attributes.

  Returns:
    A dict of attribute names to their contextual overrides.
  """
  return context(cascade=cascade, override_attrs=True, **settings)


def get_contextual_override(var_name: str) -> ContextualOverride | None:
  """Returns the overriden contextual value in current scope."""
  return _get_scoped_value(_global_tls, _CONTEXT_OVERRIDES, var_name)


def context_value(var_name: str, default: Any = RAISE_IF_HAS_ERROR) -> Any:
  """Returns the value of a variable defined in `lf.context`."""
  override = get_contextual_override(var_name)
  if override is None:
    if default == RAISE_IF_HAS_ERROR:
      raise KeyError(f'{var_name!r} does not exist in current context.')
    return default
  return override.value


def all_contextual_values() -> dict[str, Any]:
  """Returns all contextual values provided from `lf.context` in scope."""
  overrides = getattr(_global_tls, _CONTEXT_OVERRIDES, {})
  return {k: v.value for k, v in overrides.items()}


@contextlib.contextmanager
def _contextual_scope(
    tls: threading.local, tls_key, **variables
) -> Iterator[dict[str, ContextualOverride]]:
  """Context manager to set variables within a scope."""
  previous_values = getattr(tls, tls_key, {})
  current_values = dict(previous_values)
  for k, v in variables.items():
    old_v = current_values.get(k, None)
    if old_v and old_v.cascade:
      v = old_v
    current_values[k] = v
  try:
    setattr(tls, tls_key, current_values)
    yield current_values
  finally:
    setattr(tls, tls_key, previous_values)


def _get_scoped_value(
    tls: threading.local, tls_key: str, var_name: str, default: Any = None
) -> ContextualOverride:
  """Gets the value for requested variable from current scope."""
  scoped_values = getattr(tls, tls_key, {})
  return scoped_values.get(var_name, default)


class ContextualAttribute(pg.symbolic.ValueFromParentChain):
  """Attributes whose values are inferred from the context of the component.

  Please see go/langfun-component#attribute-value-retrieval for details.
  """

  NO_DEFAULT = (pg.MISSING_VALUE,)

  type: Annotated[Type[Any] | None, 'An optional type constraint.'] = None

  default: Any = NO_DEFAULT

  def value_from(
      self,
      parent,
      *,
      context_override: ContextualOverride | None = None,
      **kwargs,
  ):
    if parent not in (None, self.sym_parent) and isinstance(parent, Component):
      # Apply original search logic along the component containing chain.
      return super().value_from(parent, **kwargs)
    elif parent is None:
      # When there is no value inferred from the symbolic tree.
      # Search context override, and then attribute-level default.
      if context_override:
        return context_override.value
      if self.default == ContextualAttribute.NO_DEFAULT:
        return pg.MISSING_VALUE
      return self.default
    else:
      return pg.MISSING_VALUE


# NOTE(daiyip): Returning Any instead of `lf.ContextualAttribute` to avoid
# pytype check error as `contextual()` can be assigned to any type.
def contextual(
    type: Type[Any] | None = None,  # pylint: disable=redefined-builtin
    default: Any = ContextualAttribute.NO_DEFAULT,
) -> Any:
  """Value marker for a contextual attribute."""
  return ContextualAttribute(type=type, default=default, allow_partial=True)


# Decorator for setting the positional arguments for Component.
use_init_args = pg.use_init_args
