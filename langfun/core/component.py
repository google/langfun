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

from typing import ContextManager
import pyglove as pg


# Default value marker that indicates to raise error.
RAISE_IF_HAS_ERROR = (pg.MISSING_VALUE,)


class Component(pg.ContextualObject):
  """Base class for langfun components."""

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


# Aliases from PyGlove for ease of access.
context = pg.contextual_override
get_contextual_override = pg.utils.get_contextual_override
context_value = pg.utils.contextual_value
all_contextual_values = pg.utils.all_contextual_values
contextual = pg.contextual_attribute

# Decorator for setting the positional arguments for Component.
use_init_args = pg.use_init_args


def use_settings(
    *,
    cascade: bool = False,
    **settings,
) -> ContextManager[dict[str, pg.utils.ContextualOverride]]:
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
