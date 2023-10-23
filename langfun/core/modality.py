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
from typing import ContextManager
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

  REF_START = '{{'
  REF_END = '}}'

  def format(self, *args, **kwargs) -> str:
    if self.referred_name is None or not pg.object_utils.thread_local_get(
        _TLS_MODALITY_AS_REF, False
    ):
      return super().format(*args, **kwargs)
    return Modality.text_marker(self.referred_name)

  @abc.abstractmethod
  def to_bytes(self) -> bytes:
    """Returns content in bytes."""

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
