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
import base64
import unittest
import langfun.core as lf
from langfun.core import modalities as lf_modalities
from langfun.core.data.conversion import openai  # pylint: disable=unused-import


image_content = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x18\x00\x00\x00\x18\x04'
    b'\x03\x00\x00\x00\x12Y \xcb\x00\x00\x00\x18PLTE\x00\x00'
    b'\x00fff_chaag_cg_ch^ci_ciC\xedb\x94\x00\x00\x00\x08tRNS'
    b'\x00\n\x9f*\xd4\xff_\xf4\xe4\x8b\xf3a\x00\x00\x00>IDATx'
    b'\x01c \x05\x08)"\xd8\xcc\xae!\x06pNz\x88k\x19\\Q\xa8"\x10'
    b'\xc1\x14\x95\x01%\xc1\n\xa143Ta\xa8"D-\x84\x03QM\x98\xc3'
    b'\x1a\x1a\x1a@5\x0e\x04\xa0q\x88\x05\x00\x07\xf8\x18\xf9'
    b'\xdao\xd0|\x00\x00\x00\x00IEND\xaeB`\x82'
)


class OpenAIConversionTest(unittest.TestCase):

  def test_as_format_with_role(self):
    self.assertEqual(
        lf.UserMessage('hi').as_format('openai'),
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'hi'}],
        },
    )
    self.assertEqual(
        lf.AIMessage('hi').as_format('openai'),
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'hi'}],
        },
    )
    self.assertEqual(
        lf.SystemMessage('hi').as_format('openai'),
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': 'hi'}],
        },
    )

  def test_as_format_with_image(self):
    self.assertEqual(
        lf.Template(
            'What is this {{image}}?',
            image=lf_modalities.Image.from_bytes(image_content)
        ).render().as_format('openai'),
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'What is this'
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': (
                            'data:image/png;base64,'
                            + base64.b64encode(image_content).decode('utf-8')
                        )
                    }
                },
                {
                    'type': 'text',
                    'text': '?'
                }
            ],
        },
    )

  def test_as_format_with_chunk_preprocessor(self):
    self.assertEqual(
        lf.Template(
            'What is this {{image}}?',
            image=lf_modalities.Image.from_bytes(image_content)
        ).render().as_openai_format(
            chunk_preprocessor=lambda x: x if isinstance(x, str) else None
        ),
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'What is this'
                },
                {
                    'type': 'text',
                    'text': '?'
                }
            ],
        },
    )

  def test_from_value_with_simple_text(self):
    self.assertEqual(
        lf.Message.from_value(
            {
                'content': 'this is a text',
            },
            format='openai',
        ),
        lf.AIMessage('this is a text'),
    )

  def test_from_value_with_role(self):
    self.assertEqual(
        lf.Message.from_value(
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'hi'}],
            },
            format='openai',
        ),
        lf.UserMessage('hi'),
    )
    self.assertEqual(
        lf.Message.from_value(
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'hi'}],
            },
            format='openai',
        ),
        lf.AIMessage('hi'),
    )
    self.assertEqual(
        lf.Message.from_value(
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': 'hi'}],
            },
            format='openai',
        ),
        lf.SystemMessage('hi'),
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported role: .*'):
      lf.Message.from_value(
          {
              'role': 'function',
              'content': [{'type': 'text', 'text': 'hi'}],
          },
          format='openai',
      )

  def test_from_value_with_image(self):
    m = lf.Message.from_openai_format(
        lf.Template(
            'What is this {{image}}?',
            image=lf_modalities.Image.from_bytes(image_content)
        ).render().as_format('openai'),
    )
    self.assertEqual(m.text, 'What is this <<[[obj0]]>> ?')
    self.assertIsInstance(m.obj0, lf_modalities.Image)
    self.assertEqual(m.obj0.mime_type, 'image/png')
    self.assertEqual(m.obj0.to_bytes(), image_content)


if __name__ == '__main__':
  unittest.main()
