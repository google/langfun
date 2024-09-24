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
"""Interface for modality (e.g. Image, Video, etc.)."""

import abc
import functools
import hashlib
from typing import Any, ContextManager
from langfun.core import component
import pyglove as pg


_TLS_MODALITY_AS_REF = '__format_modality_as_ref__'


def format_modality_as_ref(enabled: bool = True) -> ContextManager[None]:
  """A context manager that formats modality objects as references."""
  return pg.object_utils.thread_local_value_scope(
      _TLS_MODALITY_AS_REF, enabled, False
  )


class Modality(component.Component):
  """Base class for multimodal object."""

  REF_START = '<<[['
  REF_END = ']]>>'

  def _on_bound(self):
    super()._on_bound()
    # Invalidate cached hash if modality member is changed.
    self.__dict__.pop('hash', None)

  def format(self, *args, **kwargs) -> str:
    if self.referred_name is None or not pg.object_utils.thread_local_get(
        _TLS_MODALITY_AS_REF, False
    ):
      return super().format(*args, **kwargs)
    return Modality.text_marker(self.referred_name)

  def __str_kwargs__(self) -> dict[str, Any]:
    # For modality objects, we don't want to use markdown format when they
    # are rendered as parts of the prompt.
    kwargs = super().__str_kwargs__()
    kwargs.pop('markdown', None)
    return kwargs

  @abc.abstractmethod
  def to_bytes(self) -> bytes:
    """Returns content in bytes."""

  @functools.cached_property
  def hash(self) -> str:
    """Returns a 8-byte MD5 hash as the identifier for this modality object."""
    return hashlib.md5(self.to_bytes()).hexdigest()[:8]

  @classmethod
  def text_marker(cls, var_name: str) -> str:
    """Returns a marker in the text for this object."""
    return Modality.REF_START + var_name + Modality.REF_END

  @property
  def referred_name(self) -> str | None:
    """Returns the referred name of this object in its template."""
    if not self.sym_path:
      return None
    # Strip the metadata prefix under message.
    path = str(self.sym_path)
    return path[9:] if path.startswith('metadata.') else path

  @classmethod
  def from_value(cls, value: pg.Symbolic) -> dict[str, 'Modality']:
    """Returns a dict of path to modality from a symbolic value."""
    modalities = {}
    def _visit(k, v, p):
      del k, p
      if isinstance(v, Modality):
        modalities[v.referred_name] = v
        return pg.TraverseAction.CONTINUE
      return pg.TraverseAction.ENTER

    pg.traverse(value, _visit)
    return modalities


class ModalityRef(pg.Object, pg.typing.CustomTyping):
  """References of modality objects in a symbolic tree.

  `ModalityRef` was introduced to placehold modality objects in a symbolic
  tree, to prevent message from being chunked in the middle of a Python
  structure.
  """

  name: str

  def custom_apply(
      self, path: pg.KeyPath, value_spec: pg.ValueSpec, *args, **kwargs
  ) -> tuple[bool, Any]:
    return (False, self)

  @classmethod
  def placehold(cls, value: pg.Symbolic) -> pg.Symbolic:
    """Returns a copy of value by replacing modality objects with refs.

    Args:
      value: A symbolic value.

    Returns:
      A copy of value with all child `Modality` objects replaced with
        `ModalityRef` objects.
    """

    def _placehold(k, v, p):
      del p
      if isinstance(v, Modality):
        return ModalityRef(name=value.sym_path + k)
      return v
    return value.clone().rebind(_placehold, raise_on_no_change=False)


class ModalityError(RuntimeError):  # pylint: disable=g-bad-exception-name
  """Exception raised when modality is not supported."""
