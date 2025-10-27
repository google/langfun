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
"""Tests for MCP tool."""

import base64
import inspect
import unittest

from langfun.core import async_support
from langfun.core import message as lf_message
from langfun.core import modalities as lf_modalities
from langfun.core.mcp import client as mcp_client
from langfun.core.mcp import tool as mcp_tool
import mcp
from mcp.server import fastmcp as fastmcp_lib
import pyglove as pg


# MCP server setup for testing.
_mcp_server = fastmcp_lib.FastMCP(host='0.0.0.0', port=1235)


@_mcp_server.tool()
async def add(a: int, b: int) -> int:
  """Adds two integers."""
  return a + b


class McpToolTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.client = mcp_client.McpClient.from_fastmcp(_mcp_server)
    self.tools = self.client.list_tools()

  def test_snake_to_camel(self):
    self.assertEqual(mcp_tool._snake_to_camel('foo_bar'), 'FooBar')
    self.assertEqual(mcp_tool._snake_to_camel('foo'), 'Foo')

  def test_base64_decode(self):
    self.assertEqual(
        mcp_tool._base64_decode(base64.b64encode(b'foo').decode('utf-8')),
        b'foo'
    )

  def test_make_input_class(self):
    schema = pg.Schema(
        description='Foo input.',
        fields=[
            pg.typing.Field('x', pg.typing.Int(), 'Integer x.'),
            pg.typing.Field('y', pg.typing.Str(), 'String y.'),
        ],
    )
    input_cls = mcp_tool.McpToolInput.make_class('foo_input', schema)
    self.assertTrue(issubclass(input_cls, mcp_tool.McpToolInput))
    self.assertEqual(input_cls.__name__, 'FooInput')
    self.assertEqual(input_cls.__doc__, 'Foo input.')
    s = input_cls.__schema__
    self.assertEqual(list(s.fields.keys()), ['x', 'y'])
    self.assertEqual(repr(input_cls), "<input-class 'FooInput'>")
    self.assertEqual(
        repr(input_cls(x=1, y='abc')),
        "FooInput(x=1, y='abc')",
    )

  def test_make_tool_class(self):
    tool_def = mcp.Tool(
        name='my_tool',
        inputSchema={
            'type': 'object',
            'properties': {
                'a': {'type': 'integer', 'description': 'Integer a.'},
                'b': {'type': 'string', 'description': 'String b.'},
            },
            'required': ['a'],
        },
        description='My tool.',
    )
    tool_cls = mcp_tool.McpTool.make_class(tool_def)
    self.assertTrue(issubclass(tool_cls, mcp_tool.McpTool))
    self.assertEqual(tool_cls.__name__, 'MyTool')
    self.assertEqual(tool_cls.TOOL_NAME, 'my_tool')
    self.assertEqual(tool_cls.__doc__, 'My tool.')
    s = tool_cls.__schema__
    self.assertEqual(list(s.fields.keys()), ['a', 'b'])
    self.assertEqual(repr(tool_cls), "<tool-class 'MyTool'>")
    self.assertEqual(s.fields['a'].description, 'Integer a.')
    self.assertEqual(s.fields['b'].description, 'String b.')

    self.assertEqual(
        tool_cls.python_definition(markdown=True),
        inspect.cleandoc(
            """
            MyTool

            ```python
            class MyTool:
              \"\"\"My tool.\"\"\"
              # Integer a.
              a: int
              # String b.
              b: str | None
            ```
            """
        ),
    )
    self.assertEqual(
        tool_cls.python_definition(markdown=False),
        inspect.cleandoc(
            """
            MyTool

            class MyTool:
              \"\"\"My tool.\"\"\"
              # Integer a.
              a: int
              # String b.
              b: str | None
            """
        ),
    )

  def test_input_parameters(self):
    tool_cls = self.tools['add']
    self.assertEqual(tool_cls(a=1, b=2).input_parameters(), {'a': 1, 'b': 2})

  def test_result_to_message(self):
    img_data = base64.b64encode(b'image-data').decode('utf-8')
    audio_data = base64.b64encode(b'audio-data').decode('utf-8')

    tool_def = self.tools['add']
    result = mcp.types.CallToolResult(
        content=[
            mcp.types.TextContent(type='text', text='hello'),
            mcp.types.ImageContent(
                type='image', data=img_data, mimeType='image/png'
            ),
            mcp.types.AudioContent(
                type='audio', data=audio_data, mimeType='audio/wav'
            ),
        ],
        structuredContent={'x': 1},
    )
    message = tool_def.result_to_message(result)
    self.assertIsInstance(message, lf_message.ToolMessage)
    self.assertIn('hello', message.text)
    self.assertIn('<<[[image', message.text)
    self.assertIn('<<[[audio', message.text)
    self.assertEqual(message.metadata, {'x': 1})
    modalities = message.modalities()
    self.assertEqual(len(modalities), 2)
    self.assertIsInstance(modalities[0], lf_modalities.Image)
    self.assertEqual(modalities[0].to_bytes(), b'image-data')
    self.assertIsInstance(modalities[1], lf_modalities.Audio)
    self.assertEqual(modalities[1].to_bytes(), b'audio-data')

  def test_sync_call(self):
    add_tool_cls = self.tools['add']
    with self.client.session() as session:
      # Test returning structured content.
      self.assertEqual(add_tool_cls(a=1, b=2)(session), 3)

      # Test returning message.
      self.assertEqual(
          add_tool_cls(a=1, b=2)(session, returns_message=True),
          lf_message.ToolMessage(text='3', result=3),
      )

  def test_async_call(self):
    async def _test():
      add_tool_cls = self.tools['add']
      async with self.client.session() as session:
        # Test returning structured content.
        self.assertEqual(await add_tool_cls(a=1, b=2).acall(session), 3)

        # Test returning message.
        self.assertEqual(
            await add_tool_cls(a=1, b=2).acall(session, returns_message=True),
            lf_message.ToolMessage(text='3', result=3),
        )

    async_support.invoke_sync(_test)


if __name__ == '__main__':
  unittest.main()
