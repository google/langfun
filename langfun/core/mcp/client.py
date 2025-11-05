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
  """Interface for Model Context Protocol (MCP) client.

  An MCP client serves as a bridge to an MCP server, enabling users to interact
  with tools hosted on the server. It provides methods for listing available
  tools and creating sessions for tool interaction.

  There are three types of MCP clients:

  * **Stdio-based client**: Ideal for interacting with tools exposed as
    command-line executables through stdin/stdout.
    Created by `lf.mcp.McpClient.from_command`.
  * **HTTP-based client**: Designed for tools accessible via HTTP,
    supporting Server-Sent Events (SSE) for streaming.
    Created by `lf.mcp.McpClient.from_url`.
  * **In-memory client**: Useful for testing or embedding MCP servers
    within the same process.
    Created by `lf.mcp.McpClient.from_fastmcp`.

  **Example Usage:**

  ```python
  import langfun as lf

  # Example 1: Stdio-based client
  client = lf.mcp.McpClient.from_command('<MCP_CMD>', ['<ARG1>', 'ARG2'])
  tools = client.list_tools()
  tool_cls = tools['<TOOL_NAME>']

  # Print the Python definition of the tool.
  print(tool_cls.python_definition())

  with client.session() as session:
    result = tool_cls(x=1, y=2)(session)
    print(result)

  # Example 2: HTTP-based client (async)
  async def main():
    client = lf.mcp.McpClient.from_url('http://localhost:8000/mcp')
    tools = client.list_tools()
    tool_cls = tools['<TOOL_NAME>']

    # Print the Python definition of the tool.
    print(tool_cls.python_definition())

    async with client.session() as session:
      result = await tool_cls(x=1, y=2).acall(session)
      print(result)
  ```
  """

  def _on_bound(self):
    super()._on_bound()
    self._tools = None

  def list_tools(
      self, refresh: bool = False
  ) -> dict[str, Type[mcp_tool.McpTool]]:
    """Lists all available tools on the MCP server.

    Args:
      refresh: If True, forces a refresh of the tool list from the server.
        Otherwise, a cached list may be returned.

    Returns:
      A dictionary mapping tool names to their corresponding `McpTool` classes.
    """
    if self._tools is None or refresh:
      with self.session() as session:
        self._tools = session.list_tools()
    return self._tools

  @abc.abstractmethod
  def session(self) -> mcp_session.McpSession:
    """Creates a new session for interacting with MCP tools.

    Returns:
      An `McpSession` object.
    """

  @classmethod
  def from_command(cls, command: str, args: list[str]) -> 'McpClient':
    """Creates an MCP client from a command-line executable.

    Args:
      command: The command to execute.
      args: A list of arguments to pass to the command.

    Returns:
      A `McpClient` instance that communicates via stdin/stdout.
    """
    return _StdioMcpClient(command=command, args=args)

  @classmethod
  def from_url(
      cls,
      url: str,
      headers: dict[str, str] | None = None
  ) -> 'McpClient':
    """Creates an MCP client from an HTTP URL.

    Args:
      url: The URL of the MCP server.
      headers: An optional dictionary of HTTP headers to include in requests.

    Returns:
      A `McpClient` instance that communicates via HTTP.
    """
    return _HttpMcpClient(url=url, headers=headers or {})

  @classmethod
  def from_fastmcp(cls, fastmcp: fastmcp_lib.FastMCP) -> 'McpClient':
    """Creates an MCP client from an in-memory FastMCP instance.

    Args:
      fastmcp: An instance of `fastmcp_lib.FastMCP`.

    Returns:
      A `McpClient` instance that communicates with the in-memory server.
    """
    return _InMemoryFastMcpClient(fastmcp=fastmcp)


class _StdioMcpClient(McpClient):
  """Stdio-based MCP client."""

  command: Annotated[str, 'Command to execute.']
  args: Annotated[list[str], 'Arguments to pass to the command.']

  def session(self) -> mcp_session.McpSession:
    """Creates an McpSession from command."""
    return mcp_session.McpSession.from_command(self.command, self.args)


class _HttpMcpClient(McpClient):
  """HTTP-based MCP client."""

  url: Annotated[str, 'URL to connect to.']
  headers: Annotated[dict[str, str], 'Headers to send with the request.'] = {}

  def session(self) -> mcp_session.McpSession:
    """Creates an McpSession from URL."""
    return mcp_session.McpSession.from_url(self.url, self.headers)


class _InMemoryFastMcpClient(McpClient):
  """In-memory MCP client."""

  fastmcp: Annotated[fastmcp_lib.FastMCP, 'MCP server to connect to.']

  def session(self) -> mcp_session.McpSession:
    """Creates an McpSession from an in-memory FastMCP instance."""
    return mcp_session.McpSession.from_fastmcp(self.fastmcp)
