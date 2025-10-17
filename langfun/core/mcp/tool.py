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

from typing import Annotated, Any, ClassVar

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

  def __call__(
      self,
      session,
      *,
      returns_call_result: bool = False) -> Any:
    """Calls a MCP tool synchronously.

    Args:
      session: A MCP session.
      returns_call_result: If True, returns the call result. Otherwise returns
        the result from structured content, or return content.

    Returns:
      The call result, or the result from structured content, or content.
    """
    return session.call_tool(
        self,
        returns_call_result=returns_call_result
    )

  async def acall(
      self,
      session,
      *,
      returns_call_result: bool = False) -> Any:
    """Calls a MCP tool asynchronously."""
    return await session.acall_tool(self, returns_call_result=returns_call_result)

  def input_parameters(self) -> dict[str, Any]:
    """Returns the input parameters of the tool."""
    json = self.to_json()
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
