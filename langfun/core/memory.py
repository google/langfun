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
"""Interface for memories."""

import abc
from typing import Any

from langfun.core import template as template_lib
from langfun.core.component import Component
from langfun.core.natural_language import NaturalLanguageFormattable


class Memory(NaturalLanguageFormattable, Component):
  """Interface for memory.

  The role of memory is to form the conversation context within a prompt.
  Instead of simply dumping all the historical messages to the context, the
  abstraction of memory allows agent developers to select relevant pieces of
  information, summary such information to best direct LLMs' attentions and
  utilize their context window.
  """

  def natural_language_format(self) -> str:
    return self.recollect().render().text

  def recollect(self, **kwargs) -> template_lib.Template:
    """Recollects a message from the memory."""
    return template_lib.Template.from_value(self._recollect(**kwargs))

  def remember(
      self,
      value: Any,
      **kwargs
  ) -> None:
    self._remember(value, **kwargs)

  def reset(self, **kwargs) -> None:
    """Resets the memory."""
    self._reset(**kwargs)

  @abc.abstractmethod
  def _recollect(self, **kwargs) -> str | template_lib.Template:
    """Recollects a message from the memory."""

  @abc.abstractmethod
  def _remember(self, value: Any, **kwargs) -> None:
    """Remembers the input message and the output message."""

  @abc.abstractmethod
  def _reset(self, **kwargs) -> None:
    """Reset the memory."""
