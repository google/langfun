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
"""A simple MCP client for testing."""

from absl import app
from absl import flags
from langfun.core import mcp


_URL = flags.DEFINE_string(
    'url',
    'http://localhost:8000/mcp',
    'URL of the MCP server.',
)


def main(_):
  print(mcp.McpClient.from_url(url=_URL.value).list_tools())


if __name__ == '__main__':
  app.run(main)
