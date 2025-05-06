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
"""Conversation."""

from typing import Annotated
import langfun.core as lf
import langfun.core.memories as lf_memories
from langfun.core.templates.completion import Completion


@lf.use_init_args(['preamble', 'input_message'])
class Conversation(Completion):
  """LM-based conversation."""

  prompt = """
      {%- if preamble -%}
      {{ preamble }}
      {%- endif%}
      {%if conversation_context -%}
      {{ conversation_context }}
      {%- endif%}
      {{ next_input }}
      """

  preamble: Annotated[
      lf.LangFunc | None,
      '(Optional) Preamble before beginning the conversation.',
  ] = None

  role: Annotated[
      str | None,
      '(Optional) User defined role for the AI response in the conversation.',
  ] = None

  conversation_context: Annotated[
      lf.LangFunc | None,
      (
          '(Optional) Conversation context, which could be history of existing '
          'chat messages.'
      ),
  ] = lf.LangFunc('{{ memory }}')

  memory: Annotated[
      lf.Memory, 'Memory to use in formulating the conversation context.'
  ] = lf_memories.ConversationHistory()

  next_input: Annotated[
      lf.LangFunc, '(Required) next input message for this chat.'
  ] = lf.LangFunc('User: {{ input_message }}')

  input_message: Annotated[
      lf.Message | str, '(Contextual) input message for this chat.'
  ] = lf.contextual()

  def _on_bound(self):
    super()._on_bound()
    self.reset()

  def __call__(self, **kwargs) -> lf.Message:
    """Sends a message to LM and receives a response."""

    # Clear (maybe) cached LM response from the last turn.
    self.clear_lm_response()

    with lf.context(**kwargs):
      # Call LM based on the prompt generated from `input_message`.
      lm_response = super().__call__()
      if self.role is not None:
        lm_response.rebind(
            sender=self.role, skip_notification=True, raise_on_no_change=False
        )

      # Add current turn to memory.
      self.add(self.input_message, lm_response)

    return lm_response

  def add(
      self, input_message: lf.Message | str, lm_response: lf.Message | str
  ) -> None:
    """Adds a turn."""
    self.memory.remember((input_message, lm_response))

  def reset(self) -> None:
    """Resets current conversation."""
    self.memory.reset()
