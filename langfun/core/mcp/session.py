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
"""MCP session."""

import contextlib
from typing import Any, Type
import anyio
from langfun.core import async_support
from langfun.core.mcp import tool as mcp_tool
import mcp
from mcp.client import sse
from mcp.client import streamable_http
from mcp.server import fastmcp as fastmcp_lib
from mcp.shared import memory


class McpSession:
  """Represents a session for interacting with an MCP server.

  `McpSession` provides the context for making calls to tools hosted on an
  MCP server. It wraps the standard `mcp.ClientSession` to offer both
  synchronous and asynchronous usage patterns.

  Sessions are created using `lf.mcp.McpClient.session()` and should be used
  as context managers (either sync or async) to ensure proper initialization
  and teardown of the connection to the server.

  **Example Sync Usage:**

  ```python
  import langfun as lf

  client = lf.mcp.McpClient.from_command(...)
  with client.session() as session:
    tools = session.list_tools()
    result = tools['my_tool'](x=1)(session)
  ```

  **Example Async Usage:**

  ```python
  import langfun as lf

  client = lf.mcp.McpClient.from_url(...)
  async with client.session() as session:
    tools = await session.alist_tools()
    result = await tools['my_tool'](x=1).acall(session)
  ```
  """

  def __init__(self, stream) -> None:
    self._stream = stream
    self._session = None
    self._session_exit_stack = None
    self._in_session = False

    # For supporting sync context manager.
    self._sync_context_manager_exit_stack = None

  def __enter__(self) -> 'McpSession':
    exit_stack = contextlib.ExitStack()
    exit_stack.enter_context(async_support.sync_context_manager(self))
    self._sync_context_manager_exit_stack = exit_stack
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    assert self._sync_context_manager_exit_stack is not None
    self._sync_context_manager_exit_stack.close()

  async def __aenter__(self) -> 'McpSession':
    assert self._session_exit_stack is None, 'Session cannot be re-entered.'

    self._session_exit_stack = contextlib.AsyncExitStack()
    stream_output = await self._session_exit_stack.enter_async_context(
        self._stream
    )
    assert isinstance(stream_output, tuple) and len(stream_output) in (2, 3)
    read, write = stream_output[:2]
    self._session = mcp.ClientSession(read, write)
    await self._session_exit_stack.enter_async_context(self._session)
    await self._session.initialize()
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    del exc_type, exc_val, exc_tb
    if self._session is None:
      return
    assert self._session_exit_stack is not None
    await self._session_exit_stack.aclose()
    self._session = None

  def list_tools(self) -> dict[str, Type[mcp_tool.McpTool]]:
    """Lists all available tools on the MCP server synchronously.

    Returns:
      A dictionary mapping tool names to their corresponding `McpTool` classes.
    """
    return async_support.invoke_sync(self.alist_tools)

  async def alist_tools(self) -> dict[str, Type[mcp_tool.McpTool]]:
    """Lists all available tools on the MCP server asynchronously.

    Returns:
      A dictionary mapping tool names to their corresponding `McpTool` classes.
    """
    assert self._session is not None, 'MCP session is not entered.'
    return {
        t.name: mcp_tool.McpTool.make_class(t)
        for t in (await self._session.list_tools()).tools
    }

  def call_tool(
      self,
      tool: mcp_tool.McpTool,
      *,
      returns_message: bool = False
  ) -> Any:
    """Calls an MCP tool synchronously.

    Args:
      tool: The `McpTool` instance to call.
      returns_message: If True, the tool call will return an `mcp.Message`
        object; otherwise, it returns the tool's direct result.

    Returns:
      The result of the tool call.
    """
    return tool(self, returns_message=returns_message)

  async def acall_tool(
      self,
      tool: mcp_tool.McpTool,
      *,
      returns_message: bool = False
  ) -> Any:
    """Calls an MCP tool asynchronously.

    Args:
      tool: The `McpTool` instance to call.
      returns_message: If True, the tool call will return an `mcp.Message`
        object; otherwise, it returns the tool's direct result.

    Returns:
      The result of the tool call.
    """
    return await tool.acall(self, returns_message=returns_message)

  @classmethod
  def from_command(
      cls,
      command: str,
      args: list[str] | None = None
  ) -> 'McpSession':
    """Creates an MCP session from a command-line executable.

    Args:
      command: The command to execute.
      args: An optional list of arguments to pass to the command.

    Returns:
      An `McpSession` instance.
    """
    return cls(
        mcp.stdio_client(
            mcp.StdioServerParameters(command=command, args=args or [])
        )
    )

  @classmethod
  def from_url(
      cls,
      url: str,
      headers: dict[str, str] | None = None
  ) -> 'McpSession':
    """Creates an MCP session from an HTTP URL.

    The transport protocol (e.g., 'mcp' or 'sse') is inferred from the
    last part of the URL path.

    Args:
      url: The URL of the MCP server.
      headers: An optional dictionary of HTTP headers to include in requests.

    Returns:
      An `McpSession` instance.
    """
    transport = url.removesuffix('/').split('/')[-1].lower()
    if transport == 'mcp':
      return cls(streamable_http.streamablehttp_client(url, headers or {}))
    elif transport == 'sse':
      return cls(sse.sse_client(url, headers or {}))
    else:
      raise ValueError(f'Unsupported transport: {transport}')

  @classmethod
  def from_fastmcp(
      cls,
      fastmcp: fastmcp_lib.FastMCP
  ):
    """Creates an MCP session from an in-memory FastMCP instance.

    Args:
      fastmcp: An instance of `fastmcp_lib.FastMCP`.

    Returns:
      An `McpSession` instance.
    """
    return cls(_client_streams_from_fastmcp(fastmcp))


@contextlib.asynccontextmanager
async def _client_streams_from_fastmcp(fastmcp: fastmcp_lib.FastMCP):
  """Creates client streams from an in-memory FastMCP instance."""
  server = fastmcp._mcp_server  # pylint: disable=protected-access
  async with memory.create_client_server_memory_streams(
      ) as (client_streams, server_streams):
    client_read, client_write = client_streams
    server_read, server_write = server_streams

    # Create a cancel scope for the server task
    async with anyio.create_task_group() as tg:
      tg.start_soon(
          lambda: server.run(
              server_read,
              server_write,
              server.create_initialization_options(),
              raise_exceptions=True,
          )
      )
      yield client_read, client_write
