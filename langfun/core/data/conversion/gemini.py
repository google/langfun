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
"""Gemini API message conversion."""

import base64
from typing import Annotated, Any, Callable

import langfun.core as lf
from langfun.core import modalities as lf_modalities


class GeminiMessageConverter(lf.MessageConverter):
  """Converter to Gemini public API."""

  FORMAT_ID = 'gemini'

  chunk_preprocessor: Annotated[
      Callable[[str | lf.Modality], Any] | None,
      (
          'Chunk preprocessor for Langfun chunk to Gemini chunk conversion. '
          'It will be applied before each Langfun chunk is converted. '
          'If returns None, the chunk will be skipped.'
      ),
  ] = None

  def to_value(
      self, message: lf.Message, always_send_content: bool = True
  ) -> dict[str, Any]:
    """Converts a Langfun message to Gemini API."""
    parts = []
    for chunk in message.chunk():
      if self.chunk_preprocessor:
        chunk = self.chunk_preprocessor(chunk)
        if chunk is None:
          continue

      if isinstance(chunk, str):
        parts.append(self._convert_chunk(chunk, always_send_content))
      else:
        if isinstance(chunk, lf_modalities.Mime):
          modalities = [chunk]
        # NOTE(daiyip): preprocessing may convert a single chunk into
        # a list of chunks
        elif isinstance(chunk, list):
          modalities = chunk
        else:
          raise ValueError(f'Unsupported content type: {chunk!r}.')
        parts.extend(
            self._convert_chunk(c, always_send_content) for c in modalities
        )
    return dict(role=self.get_role(message), parts=parts)

  def _convert_chunk(
      self, chunk: str | lf.Modality, always_send_content: bool = True
  ) -> Any:
    """Converts a Langfun chunk to Gemini chunk."""
    if isinstance(chunk, str):
      return {'text': chunk}
    if not isinstance(chunk, lf_modalities.Mime):
      raise ValueError(f'Unsupported content chunk: {chunk!r}.')
    # NOTE(daiyip): special handling for YouTube video.
    if chunk.uri and chunk.uri.startswith('https://www.youtube.com/watch?v='):
      return {
          'fileData': {
              'mimeType': 'video/*',
              'fileUri': chunk.uri
          }
      }
    if chunk.is_text:
      return {'text': chunk.to_text()}
    if (
        not always_send_content
        and chunk.uri
        and chunk.uri.lower().startswith(('http:', 'https:', 'ftp:'))
    ):
      return {
          'fileData': {
              'mimeType': chunk.mime_type,
              'fileUri': chunk.uri,
          }
      }
    return {
        'inlineData': {
            'data': base64.b64encode(chunk.to_bytes()).decode(),
            'mimeType': chunk.mime_type,
        }
    }

  def from_value(self, value: dict[str, Any]) -> lf.Message:
    """Returns a Langfun message from Gemini message."""
    message_cls = self.get_message_cls(
        self._safe_read(value, 'role', default='model')
    )
    parts = self._safe_read(value, 'parts', default=[])
    assert isinstance(parts, list)

    chunks = []
    thought_chunks = []
    for part in parts:
      if 'text' in part:
        text = self._safe_read(part, 'text')
        if 'thought' in part:
          thought_chunks.append(text)
        else:
          chunks.append(text)
      elif 'thought' in part:
        # We occasionally encounter 'thought' without text.
        pass
      elif 'inlineData' in part:
        data = self._safe_read(part, 'inlineData')
        chunks.append(
            lf_modalities.Mime.class_from_mime_type(
                self._safe_read(data, 'mimeType')
            ).from_bytes(base64.b64decode(self._safe_read(data, 'data')))
        )
      elif 'fileData' in part:
        data = self._safe_read(part, 'fileData')
        chunks.append(
            lf_modalities.Mime.class_from_mime_type(
                self._safe_read(data, 'mimeType')
            ).from_uri(self._safe_read(data, 'fileUri'))
        )
      else:
        raise ValueError(f'Unsupported content part: {part!r}.')
    message = message_cls.from_chunks(chunks)
    if thought_chunks:
      message.set('thought', message_cls.from_chunks(thought_chunks))
    return message

  @classmethod
  def get_role(cls, message: lf.Message) -> str:
    """Returns the role of the message."""
    if isinstance(message, lf.AIMessage):
      return 'model'
    return super().get_role(message)

  @classmethod
  def get_message_cls(cls, role: str) -> type[lf.Message]:
    """Returns the message class of the message."""
    if role == 'model':
      return lf.AIMessage
    return super().get_message_cls(role)


def _as_gemini_format(
    self,
    chunk_preprocessor: Callable[[str | lf.Modality], Any] | None = None,
    **kwargs
) -> dict[str, Any]:
  """Returns a Gemini (REST) format message."""
  return GeminiMessageConverter(
      chunk_preprocessor=chunk_preprocessor, **kwargs
  ).to_value(self)


@classmethod
def _from_gemini_format(
    cls,
    gemini_message: dict[str, Any],
    **kwargs
) -> lf.Message:
  """Creates a Langfun message from the Gemini (REST) format message."""
  del cls
  return GeminiMessageConverter(**kwargs).from_value(gemini_message)

# Set shortcut methods in lf.Message.
lf.Message.as_gemini_format = _as_gemini_format
lf.Message.from_gemini_format = _from_gemini_format
