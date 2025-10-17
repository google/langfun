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
"""MCP client."""

import abc
from typing import Annotated, Type

from langfun.core.mcp import session as mcp_session
from langfun.core.mcp import tool as mcp_tool
from mcp.server import fastmcp as fastmcp_lib
import pyglove as pg


class McpClient(pg.Object):
  """Base class for MCP client.

  Usage:

  ```python

  def tool_use():
    client = lf.mcp.McpClient.from_command('<MCP_CMD>', ['<ARG1>', 'ARG2'])
    tools = client.list_tools()
    tool_cls = tools['<TOOL_NAME>']

    # Print the python definition of the tool.
    print(tool_cls.python_definition())

    with client.session() as session:
      return tool_cls(x=1, y=2)(session)

  async def tool_use_async_version():
    client = lf.mcp.McpClient.from_url('http://localhost:8000/mcp')
    tools = client.list_tools()
    tool_cls = tools['<TOOL_NAME>']

    # Print the python definition of the tool.
    print(tool_cls.python_definition())

    async with client.session() as session:
      return await tool_cls(x=1, y=2).acall(session)
  ```
  """

  def _on_bound(self):
    super()._on_bound()
    self._tools = None

  def list_tools(
      self, refresh: bool = False
  ) -> dict[str, Type[mcp_tool.McpTool]]:
    """Lists all MCP tools."""
    if self._tools is None or refresh:
      with self.session() as session:
        self._tools = session.list_tools()
    return self._tools

  @abc.abstractmethod
  def session(self) -> mcp_session.McpSession:
    """Creates a MCP session."""

  @classmethod
  def from_command(cls, command: str, args: list[str]) -> 'McpClient':
    """Creates a MCP client from a tool."""
    return _StdioMcpClient(command=command, args=args)

  @classmethod
  def from_url(
      cls,
      url: str,
      headers: dict[str, str] | None = None
  ) -> 'McpClient':
    """Creates a MCP client from a URL."""
    return _HttpMcpClient(url=url, headers=headers or {})

  @classmethod
  def from_fastmcp(cls, fastmcp: fastmcp_lib.FastMCP) -> 'McpClient':
    """Creates a MCP client from a MCP server."""
    return _InMemoryFastMcpClient(fastmcp=fastmcp)


class _StdioMcpClient(McpClient):
  """Stdio-based MCP client."""

  command: Annotated[str, 'Command to execute.']
  args: Annotated[list[str], 'Arguments to pass to the command.']

  def session(self) -> mcp_session.McpSession:
    """Creates a MCP session."""
    return mcp_session.McpSession.from_command(self.command, self.args)


class _HttpMcpClient(McpClient):
  """Server-Sent Events (SSE)/Streamable HTTP-based MCP client."""

  url: Annotated[str, 'URL to connect to.']
  headers: Annotated[dict[str, str], 'Headers to send with the request.'] = {}

  def session(self) -> mcp_session.McpSession:
    """Creates a MCP session."""
    return mcp_session.McpSession.from_url(self.url, self.headers)


class _InMemoryFastMcpClient(McpClient):
  """In-memory MCP client."""

  fastmcp: Annotated[fastmcp_lib.FastMCP, 'MCP server to connect to.']

  def session(self) -> mcp_session.McpSession:
    """Creates a MCP session."""
    return mcp_session.McpSession.from_fastmcp(self.fastmcp)
