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
"""Tests for MCP session."""

import unittest
from unittest import mock

from langfun.core.mcp import session as mcp_session
import mcp
from mcp.client import sse
from mcp.client import streamable_http


class McpSessionTest(unittest.TestCase):

  @mock.patch.object(mcp, 'stdio_client', autospec=True)
  def test_from_command(self, mock_stdio_client):
    mcp_session.McpSession.from_command('my-command', ['--foo'])
    mock_stdio_client.assert_called_once_with(
        mcp.StdioServerParameters(command='my-command', args=['--foo'])
    )

  @mock.patch.object(streamable_http, 'streamablehttp_client', autospec=True)
  def test_from_url_mcp(self, mock_streamablehttp_client):
    mcp_session.McpSession.from_url(
        'http://localhost/mcp', headers={'k': 'v'}
    )
    mock_streamablehttp_client.assert_called_once_with(
        'http://localhost/mcp', {'k': 'v'}
    )

  @mock.patch.object(sse, 'sse_client', autospec=True)
  def test_from_url_sse(self, mock_sse_client):
    mcp_session.McpSession.from_url('http://localhost/sse', headers={'k': 'v'})
    mock_sse_client.assert_called_once_with('http://localhost/sse', {'k': 'v'})

  def test_from_url_unsupported(self):
    with self.assertRaisesRegex(ValueError, 'Unsupported transport: foo'):
      mcp_session.McpSession.from_url('http://localhost/foo')


if __name__ == '__main__':
  unittest.main()
