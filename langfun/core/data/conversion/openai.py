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
"""OpenAI API message conversion."""

from typing import Annotated, Any, Callable

import langfun.core as lf
from langfun.core import modalities as lf_modalities


class OpenAIMessageConverter(lf.MessageConverter):
  """Converter to OpenAI API."""

  FORMAT_ID = 'openai'

  chunk_preprocessor: Annotated[
      Callable[[str | lf.Modality], Any] | None,
      (
          'Chunk preprocessor for Langfun chunk to OpenAI chunk conversion. '
          'It will be applied before each Langfun chunk is converted. '
          'If returns None, the chunk will be skipped.'
      )
  ] = None

  def to_value(self, message: lf.Message) -> dict[str, Any]:
    """Converts a Langfun message to OpenAI API."""
    parts = []
    for chunk in message.chunk():
      if self.chunk_preprocessor is not None:
        chunk = self.chunk_preprocessor(chunk)
        if chunk is None:
          continue

      if isinstance(chunk, str):
        item = dict(type='text', text=chunk)
      elif isinstance(chunk, lf_modalities.Image):
        item = dict(
            type='image_url', image_url=dict(url=chunk.embeddable_uri)
        )
      # TODO(daiyip): Support audio_input.
      else:
        raise ValueError(f'Unsupported content type: {chunk!r}.')
      parts.append(item)
    return dict(
        role=self.get_role(message),
        content=parts,
    )

  def get_role(self, message: lf.Message) -> str:
    """Returns the role of the message."""
    if isinstance(message, lf.SystemMessage):
      return 'system'
    elif isinstance(message, lf.UserMessage):
      return 'user'
    elif isinstance(message, lf.AIMessage):
      return 'assistant'
    else:
      raise ValueError(f'Unsupported message type: {message!r}.')

  def get_message_cls(self, role: str) -> type[lf.Message]:
    """Returns the message class of the message."""
    match role:
      case 'system':
        return lf.SystemMessage
      case 'user':
        return lf.UserMessage
      case 'assistant':
        return lf.AIMessage
      case _:
        raise ValueError(f'Unsupported role: {role!r}.')

  def from_value(self, value: dict[str, Any]) -> lf.Message:
    """Returns a Langfun message from OpenAI message."""
    message_cls = self.get_message_cls(
        self._safe_read(value, 'role', default='assistant')
    )
    content = self._safe_read(value, 'content')
    if isinstance(content, str):
      return message_cls(content)

    assert isinstance(content, list)
    chunks = []
    for item in content:
      t = self._safe_read(item, 'type')
      if t == 'text':
        chunk = self._safe_read(item, 'text')
      elif t == 'image_url':
        chunk = lf_modalities.Image.from_uri(
            self._safe_read(self._safe_read(item, 'image_url'), 'url')
        )
      else:
        raise ValueError(f'Unsupported content type: {item!r}.')
      chunks.append(chunk)
    return message_cls.from_chunks(chunks)


def _as_openai_format(
    self,
    chunk_preprocessor: Callable[[str | lf.Modality], Any] | None = None,
    **kwargs
) -> dict[str, Any]:
  """Returns an OpenAI format message."""
  return OpenAIMessageConverter(
      chunk_preprocessor=chunk_preprocessor, **kwargs
  ).to_value(self)


@classmethod
def _from_openai_format(
    cls,
    openai_message: dict[str, Any],
    **kwargs
) -> lf.Message:
  """Creates a Langfun message from the OpenAI format message."""
  del cls
  return OpenAIMessageConverter(**kwargs).from_value(openai_message)

# Set shortcut methods in lf.Message.
lf.Message.as_openai_format = _as_openai_format
lf.Message.from_openai_format = _from_openai_format
