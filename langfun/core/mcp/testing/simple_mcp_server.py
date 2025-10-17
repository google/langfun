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
"""Simple MCP server for testing."""

from absl import app as absl_app
from mcp.server import fastmcp as fastmcp_lib

mcp = fastmcp_lib.FastMCP(host='0.0.0.0', port=8000)


@mcp.tool()
async def add(a: int, b: int) -> int:
  """Adds two integers and returns their sum."""
  return a + b


def main(_):
  mcp.run(transport='streamable-http', mount_path='/mcp')


if __name__ == '__main__':
  absl_app.run(main)
