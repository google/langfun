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
from langfun.core import message as message_lib
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
    return self.recollect().text

  def recollect(self, **kwargs) -> message_lib.MemoryRecord:
    """Recollects a message from the memory."""
    return self._recollect(**kwargs)

  def remember(
      self,
      input_message: str | message_lib.Message,
      output_message: str | message_lib.Message,
      **kwargs
  ) -> str:
    if isinstance(input_message, str):
      input_message = message_lib.UserMessage(input_message)
    if isinstance(output_message, str):
      output_message = message_lib.AIMessage(output_message)
    self._remember(input_message, output_message, **kwargs)
    return output_message.text

  def reset(self, **kwargs) -> None:
    """Resets the memory."""
    self._reset()

  @abc.abstractmethod
  def _recollect(self, **kwargs) -> message_lib.MemoryRecord:
    """Recollects a message from the memory."""

  @abc.abstractmethod
  def _remember(
      self,
      input_message: message_lib.Message,
      output_message: message_lib.Message,
      **kwargs
  ) -> None:
    """Remembers the input message and the output message."""

  @abc.abstractmethod
  def _reset(self, **kwargs) -> None:
    """Reset the memory."""
