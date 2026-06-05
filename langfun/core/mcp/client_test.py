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
"""Tests for MCP client."""

import unittest
from langfun.core import async_support
from langfun.core import mcp as lf_mcp
from langfun.core import message as lf_message
from mcp.server import fastmcp as fastmcp_lib

mcp = fastmcp_lib.FastMCP(host='0.0.0.0', port=1234)


@mcp.tool()
async def add(a: int, b: int) -> int:
  """Adds two integers and returns their sum.

  Args:
    a: The first integer.
    b: The second integer.

  Returns:
    The sum of the two integers.
  """
  return a + b


class McpTest(unittest.TestCase):

  def test_sync_usages(self):
    client = lf_mcp.McpClient.from_fastmcp(mcp)
    tools = client.list_tools()
    self.assertEqual(len(tools), 1)
    with client.session() as session:
      self.assertEqual(
          # Test `session.call_tool` method as `tool.__call__` is already tested
          # in `tool_test.py`.
          session.call_tool(tools['add'](a=1, b=2)), 3
      )

  def test_async_usages(self):
    async def _test():
      client = lf_mcp.McpClient.from_fastmcp(mcp)
      tools = client.list_tools()
      self.assertEqual(len(tools), 1)
      tool_cls = tools['add']
      self.assertEqual(tool_cls.__name__, 'Add')
      self.assertEqual(tool_cls.TOOL_NAME, 'add')
      async with client.session() as session:
        self.assertEqual(
            # Test `session.acall_tool` method as `tool.acall` is already
            # tested in `tool_test.py`.
            await session.acall_tool(tool_cls(a=1, b=2), returns_message=True),
            lf_message.ToolMessage(text='3', result=3)
        )
    async_support.invoke_sync(_test)


if __name__ == '__main__':
  unittest.main()
