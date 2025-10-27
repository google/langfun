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
"""MCP tool."""

import base64
from typing import Annotated, Any, ClassVar

from langfun.core import async_support
from langfun.core import message as lf_message
from langfun.core import modalities as lf_modalities
from langfun.core.structured import schema as lf_schema
import mcp
import pyglove as pg


class _McpToolMeta(pg.symbolic.ObjectMeta):

  def __repr__(self) -> str:
    return f'<tool-class \'{self.__name__}\'>'


class McpTool(pg.Object, metaclass=_McpToolMeta):
  """Base class for MCP tools."""

  TOOL_NAME: Annotated[
      ClassVar[str],
      'Tool name.'
  ]

  @classmethod
  def python_definition(cls, markdown: bool = True) -> str:
    """Returns the Python definition of the tool."""
    return lf_schema.Schema.from_value(cls).schema_str(
        protocol='python', markdown=markdown
    )

  @classmethod
  def result_to_message(
      cls, result: mcp.types.CallToolResult
  ) -> lf_message.ToolMessage:
    """Converts a tool call result to a message.

    This method allows users to convert an existing mcp.CallToolResult to a
    Langfun ToolMessage.

    Args:
      result: The MCP tool call result.

    Returns:
      A ToolMessage object.
    """
    chunks = []
    for item in result.content:
      if isinstance(item, mcp.types.TextContent):
        chunk = item.text
      elif isinstance(item, mcp.types.ImageContent):
        chunk = lf_modalities.Image.from_bytes(_base64_decode(item.data))
      elif isinstance(item, mcp.types.AudioContent):
        chunk = lf_modalities.Audio.from_bytes(_base64_decode(item.data))
      else:
        raise ValueError(f'Unsupported item type: {type(item)}')
      chunks.append(chunk)
    message = lf_message.ToolMessage.from_chunks(chunks)
    if result.structuredContent:
      message.metadata.update(result.structuredContent)
    return message

  def __call__(
      self,
      session,
      *,
      returns_message: bool = False) -> Any:
    """Calls a MCP tool synchronously.

    Args:
      session: A MCP session.
      returns_message: If True, always returns a ToolMessage. Otherwise,
        return structured content (result) if available, otherwise return
        ToolMessage if there is multi-modal content, otherwise return text.

    Returns:
      The call result, or the result from structured content, or content.
    """
    return async_support.invoke_sync(
        self.acall,
        session,
        returns_message=returns_message
    )

  async def acall(
      self,
      session,
      *,
      returns_message: bool = False
  ) -> Any:
    """Calls a MCP tool asynchronously.

    Args:
      session: McpSession or mcp.ClientSession.
      returns_message: If True, always returns a ToolMessage. Otherwise,
        return structured content (result) if available, otherwise return
        ToolMessage if there is multi-modal content, otherwise return text.

    Returns:
      The call result, or the result from structured content, or content.
    """
    if not isinstance(session, mcp.ClientSession):
      session = getattr(session, '_session', None)
    assert session is not None, 'MCP session is not entered.'
    tool_call_result = await session.call_tool(
        self.TOOL_NAME, self.input_parameters()
    )
    message = self.result_to_message(tool_call_result)
    if returns_message:
      return message
    if message.result:
      return message.result
    if message.referred_modalities:
      return message
    return message.text

  def input_parameters(self) -> dict[str, Any]:
    """Returns the input parameters of the tool."""
    # Optional fields are represented as fields with default values. Therefore,
    # we need to remove the default values from the JSON representation of the
    # tool.
    json = self.to_json(hide_default_values=True)

    # Remove the type name key from the JSON representation of the tool.
    def _transform(path: pg.KeyPath, x: Any) -> Any:
      del path
      if isinstance(x, dict):
        x.pop(pg.JSONConvertible.TYPE_NAME_KEY, None)
      return x
    return pg.utils.transform(json, _transform)

  @classmethod
  def make_class(cls, tool_definition: mcp.Tool) -> type['McpTool']:
    """Makes a MCP tool class from tool definition."""

    class _McpTool(cls):
      auto_schema = False

    tool_cls = _McpTool
    tool_cls.TOOL_NAME = tool_definition.name
    tool_cls.__name__ = _snake_to_camel(tool_definition.name)
    tool_cls.__doc__ = tool_definition.description
    schema = pg.Schema.from_json_schema(
        tool_definition.inputSchema, class_fn=McpToolInput.make_class
    )
    tool_cls.apply_schema(schema)
    return tool_cls


class _McpToolInputMeta(pg.symbolic.ObjectMeta):

  def __repr__(self) -> str:
    return f'<input-class \'{self.__name__}\'>'


class McpToolInput(pg.Object, metaclass=_McpToolInputMeta):
  """Base class for MCP tool inputs."""

  @classmethod
  def make_class(cls, name: str, schema: pg.Schema):
    """Converts a schema to an input class."""

    class _McpToolInput(cls):
      pass

    input_cls = _McpToolInput
    input_cls.__name__ = _snake_to_camel(name)
    input_cls.__doc__ = schema.description
    input_cls.apply_schema(schema)
    return input_cls


def _snake_to_camel(name: str) -> str:
  """Converts a snake_case name to a CamelCase name."""
  return ''.join(x.capitalize() for x in name.split('_'))


def _base64_decode(data: str) -> bytes:
  """Decodes a base64 string."""
  return base64.b64decode(data.encode('utf-8'))
