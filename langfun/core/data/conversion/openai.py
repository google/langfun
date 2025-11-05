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


class OpenAIChatCompletionAPIMessageConverter(lf.MessageConverter):
  """Converter for OpenAI Chat Completion API.

  This converter translates `lf.Message` objects into the JSON format
  required by the OpenAI Chat Completions API
  (https://platform.openai.com/docs/api-reference/chat) and vice versa.
  It handles text and image modalities, mapping Langfun roles to OpenAI
  roles ('system', 'user', 'assistant'). An optional `chunk_preprocessor`
  can be provided to modify or filter chunks before conversion.
  """

  FORMAT_ID = 'openai_chat_completion_api'

  chunk_preprocessor: Annotated[
      Callable[[str | lf.Modality], Any] | None,
      (
          'Chunk preprocessor for Langfun chunk to OpenAI chunk conversion. '
          'It will be applied before each Langfun chunk is converted. '
          'If it returns None, the chunk will be skipped.'
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
      parts.append(self.chunk_to_json(type(message), chunk))
    return dict(
        role=self.get_role(message),
        content=parts,
    )

  def chunk_to_json(
      self,
      message_cls: type[lf.Message],
      chunk: str | lf.Modality
  ) -> dict[str, Any]:
    """Converts a Langfun chunk to OpenAI chunk."""
    del message_cls
    if isinstance(chunk, str):
      return dict(type='text', text=chunk)
    elif isinstance(chunk, lf_modalities.Image):
      return dict(
          type='image_url', image_url=dict(url=chunk.embeddable_uri)
      )
    # TODO(daiyip): Support audio_input.
    else:
      raise ValueError(f'Unsupported content type: {chunk!r}.')

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
      chunks.append(self.json_to_chunk(item))
    return message_cls.from_chunks(chunks)

  def json_to_chunk(self, json: dict[str, Any]) -> str | lf.Modality:
    """Returns a Langfun chunk from OpenAI chunk JSON."""
    t = self._safe_read(json, 'type')
    if t == 'text':
      return self._safe_read(json, 'text')
    elif t == 'image_url':
      return lf_modalities.Image.from_uri(
          self._safe_read(self._safe_read(json, 'image_url'), 'url')
      )
    else:
      raise ValueError(f'Unsupported content type: {json!r}.')


def _as_openai_chat_completion_api_format(
    self,
    chunk_preprocessor: Callable[[str | lf.Modality], Any] | None = None,
    **kwargs
) -> dict[str, Any]:
  """Returns an OpenAI format message."""
  return OpenAIChatCompletionAPIMessageConverter(
      chunk_preprocessor=chunk_preprocessor, **kwargs
  ).to_value(self)


@classmethod
def _from_openai_chat_completion_api_format(
    cls,
    openai_message: dict[str, Any],
    **kwargs
) -> lf.Message:
  """Creates a Langfun message from the OpenAI format message."""
  del cls
  return OpenAIChatCompletionAPIMessageConverter(
      **kwargs
  ).from_value(openai_message)

# Set shortcut methods in lf.Message.
lf.Message.as_openai_chat_completion_api_format = (
    _as_openai_chat_completion_api_format
)

lf.Message.from_openai_chat_completion_api_format = (
    _from_openai_chat_completion_api_format
)


#
# OpenAI Responses API message converter.
#


class OpenAIResponsesAPIMessageConverter(
    OpenAIChatCompletionAPIMessageConverter
):
  """Converter for OpenAI Responses API.

  This converter translates `lf.Message` objects into the JSON format
  required by the OpenAI Responses API
  (https://platform.openai.com/docs/api-reference/responses/create),
  which is used for human-in-the-loop rating, and vice versa.
  It extends `OpenAIChatCompletionAPIMessageConverter` but uses different
  type names for content chunks (e.g., 'input_text', 'output_image').
  """

  FORMAT_ID = 'openai_responses_api'

  def to_value(self, message: lf.Message) -> dict[str, Any]:
    """Converts a Langfun message to OpenAI API."""
    message_json = super().to_value(message)
    message_json['type'] = 'message'
    return message_json

  def chunk_to_json(
      self,
      message_cls: type[lf.Message],
      chunk: str | lf.Modality
  ) -> dict[str, Any]:
    """Converts a Langfun chunk to OpenAI chunk."""
    source = 'output' if issubclass(message_cls, lf.AIMessage) else 'input'

    if isinstance(chunk, str):
      return dict(type=f'{source}_text', text=chunk)
    elif isinstance(chunk, lf_modalities.Image):
      return dict(
          type=f'{source}_image', image_url=chunk.embeddable_uri
      )
    # TODO(daiyip): Support audio_input.
    else:
      raise ValueError(f'Unsupported content type: {chunk!r}.')

  def json_to_chunk(self, json: dict[str, Any]) -> str | lf.Modality:
    """Returns a Langfun chunk from OpenAI chunk JSON."""
    t = self._safe_read(json, 'type')
    if t in ('input_text', 'output_text'):
      return self._safe_read(json, 'text')
    elif t in ('input_image', 'output_image'):
      return lf_modalities.Image.from_uri(self._safe_read(json, 'image_url'))
    else:
      raise ValueError(f'Unsupported content type: {json!r}.')


def _as_openai_responses_api_format(
    self,
    chunk_preprocessor: Callable[[str | lf.Modality], Any] | None = None,
    **kwargs
) -> dict[str, Any]:
  """Returns an OpenAI format message."""
  return OpenAIResponsesAPIMessageConverter(
      chunk_preprocessor=chunk_preprocessor, **kwargs
  ).to_value(self)


@classmethod
def _from_openai_responses_api_format(
    cls,
    openai_message: dict[str, Any],
    **kwargs
) -> lf.Message:
  """Creates a Langfun message from the OpenAI format message."""
  del cls
  return OpenAIResponsesAPIMessageConverter(
      **kwargs
  ).from_value(openai_message)


# Set shortcut methods in lf.Message.
lf.Message.as_openai_responses_api_format = (
    _as_openai_responses_api_format
)

lf.Message.from_openai_responses_api_format = (
    _from_openai_responses_api_format
)
