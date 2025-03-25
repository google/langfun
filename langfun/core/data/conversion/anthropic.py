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
"""Anthropic API message conversion."""

import base64
from typing import Annotated, Any, Callable

import langfun.core as lf
from langfun.core import modalities as lf_modalities


class AnthropicMessageConverter(lf.MessageConverter):
  """Converter to Anthropic public API."""

  FORMAT_ID = 'anthropic'

  chunk_preprocessor: Annotated[
      Callable[[str | lf.Modality], Any] | None,
      (
          'Chunk preprocessor for Langfun chunk to Anthropic chunk conversion. '
          'It will be applied before each Langfun chunk is converted. '
          'If returns None, the chunk will be skipped.'
      )
  ] = None

  def to_value(self, message: lf.Message) -> dict[str, Any]:
    """Converts a Langfun message to Gemini API."""
    content = []
    for chunk in message.chunk():
      if self.chunk_preprocessor:
        chunk = self.chunk_preprocessor(chunk)
        if chunk is None:
          continue

      if isinstance(chunk, str):
        content.append({'type': 'text', 'text': chunk})
      elif isinstance(chunk, lf_modalities.Mime):
        if isinstance(chunk, lf_modalities.Image):
          content.append(
              dict(
                  type='image',
                  source=dict(
                      type='base64',
                      media_type=chunk.mime_type,
                      data=base64.b64encode(chunk.to_bytes()).decode(),
                  ),
              )
          )
        elif isinstance(chunk, lf_modalities.PDF):
          content.append(
              dict(
                  type='document',
                  source=dict(
                      type='base64',
                      media_type=chunk.mime_type,
                      data=base64.b64encode(chunk.to_bytes()).decode(),
                  ),
              )
          )
        else:
          raise NotImplementedError(
              f'Modality conversion not implemented: {chunk!r}'
          )
    return dict(role=self.get_role(message), content=content)

  def from_value(self, value: dict[str, Any]) -> lf.Message:
    """Returns a Langfun message from Anthropic message."""
    message_cls = self.get_message_cls(
        self._safe_read(value, 'role', default='assistant')
    )
    content = self._safe_read(value, 'content', default=[])
    assert isinstance(content, list)

    chunks = []
    thought_chunks = []
    for part in content:
      t = self._safe_read(part, 'type')
      if t == 'text':
        chunks.append(self._safe_read(part, 'text'))
      elif t == 'thinking':
        thought_chunks.append(self._safe_read(part, 'thinking'))
      elif t in ('image', 'document'):
        source = self._safe_read(part, 'source')
        chunks.append(
            lf_modalities.Mime.class_from_mime_type(
                self._safe_read(source, 'media_type')
            ).from_bytes(base64.b64decode(self._safe_read(source, 'data')))
        )
      else:
        raise ValueError(f'Unsupported content part: {part!r}.')
    message = message_cls.from_chunks(chunks)
    if thought_chunks:
      message.set('thought', message_cls.from_chunks(thought_chunks))
    return message


def _as_anthropic_format(
    self,
    chunk_preprocessor: Callable[[str | lf.Modality], Any] | None = None,
    **kwargs
) -> dict[str, Any]:
  """Returns an Anthropic format message."""
  return AnthropicMessageConverter(
      chunk_preprocessor=chunk_preprocessor, **kwargs
  ).to_value(self)


@classmethod
def _from_anthropic_format(
    cls,
    anthropic_message: dict[str, Any],
    **kwargs
) -> lf.Message:
  """Creates a Langfun message from the Anthropic format message."""
  del cls
  return AnthropicMessageConverter(**kwargs).from_value(anthropic_message)

# Set shortcut methods in lf.Message.
lf.Message.as_anthropic_format = _as_anthropic_format
lf.Message.from_anthropic_format = _from_anthropic_format
