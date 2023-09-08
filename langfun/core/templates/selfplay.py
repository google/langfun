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
"""Self-play."""

import abc
from typing import Annotated, Union
import langfun.core as lf
from langfun.core.templates.conversation import Conversation


@lf.use_init_args(['preamble'])
class SelfPlay(Conversation):
  """LM-based self play."""

  max_turns: Annotated[
      int | None,
      (
          'Max number of turns to play. If None, the self play will continue '
          'until StopIteration is raised from the `next_turn` method.'
      ),
  ] = None

  @property
  def num_turns(self) -> int:
    """Returns number of turns currently played."""
    return self._num_turns

  def __call__(self, **kwargs) -> lf.Message:
    """Starts the play."""
    while self.max_turns is None or self._num_turns < self.max_turns:
      output = self.step(**kwargs)
      if output is None:
        break
    return self._last_response

  def step(self, **kwargs) -> lf.Message | None:
    """Play the next step and return the response."""
    with lf.context(**kwargs):
      input_message = self.next_turn(self._last_response)
      if input_message is None:
        return None
      self._last_response = super().__call__(input_message=input_message)
    self._num_turns += 1
    return self._last_response

  def reset(self):
    super().reset()
    self._num_turns = 0
    self._last_response = None

  @abc.abstractmethod
  def next_turn(
      self, last_response: lf.Message | None) -> Union[None, str, lf.Message]:
    """Returns the `input_message` for the next turn.

    Args:
      last_response: Last LM response. If None, then this will be the first
        message to send.

    Returns:
      The `input_message` for the next turn. None if there is no next turn.
    """
