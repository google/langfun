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
from langfun.core.data.conversion import anthropic  # pylint: disable=unused-import


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

pdf_content = (
    b'%PDF-1.1\n%\xc2\xa5\xc2\xb1\xc3\xab\n\n1 0 obj\n'
    b'<< /Type /Catalog\n     /Pages 2 0 R\n  >>\nendobj\n\n2 0 obj\n '
    b'<< /Type /Pages\n     /Kids [3 0 R]\n     '
    b'/Count 1\n     /MediaBox [0 0 300 144]\n  '
    b'>>\nendobj\n\n3 0 obj\n  '
    b'<<  /Type /Page\n      /Parent 2 0 R\n      /Resources\n       '
    b'<< /Font\n'
    b'<< /F1\n'
    b'<< /Type /Font\n'
    b'/Subtype /Type1\n'
    b'/BaseFont /Times-Roman\n'
    b'>>\n>>\n>>\n      '
    b'/Contents 4 0 R\n  >>\nendobj\n\n4 0 obj\n  '
    b'<< /Length 55 >>\nstream\n  BT\n    /F1 18 Tf\n    0 0 Td\n    '
    b'(Hello World) Tj\n  ET\nendstream\nendobj\n\nxref\n0 5\n0000000000 '
    b'65535 f \n0000000018 00000 n \n0000000077 00000 n \n0000000178 00000 n '
    b'\n0000000457 00000 n \ntrailer\n  <<  /Root 1 0 R\n      /Size 5\n  '
    b'>>\nstartxref\n565\n%%EOF\n'
)


class AnthropicConversionTest(unittest.TestCase):

  def test_as_format_with_role(self):
    self.assertEqual(
        lf.UserMessage('hi').as_format('anthropic'),
        {
            'role': 'user',
            'content': [{'type': 'text', 'text': 'hi'}],
        },
    )
    self.assertEqual(
        lf.AIMessage('hi').as_format('anthropic'),
        {
            'role': 'assistant',
            'content': [{'type': 'text', 'text': 'hi'}],
        },
    )
    self.assertEqual(
        lf.SystemMessage('hi').as_format('anthropic'),
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': 'hi'}],
        },
    )

  def test_as_format_with_image(self):
    self.assertEqual(
        lf.Template(
            'What are the common words from {{image}} and {{pdf}}?',
            image=lf_modalities.Image.from_bytes(image_content),
            pdf=lf_modalities.PDF.from_bytes(pdf_content),
        ).render().as_anthropic_format(),
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'What are the common words from'
                },
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': base64.b64encode(
                            image_content
                        ).decode('utf-8'),
                    }
                },
                {
                    'type': 'text',
                    'text': 'and'
                },
                {
                    'type': 'document',
                    'source': {
                        'type': 'base64',
                        'media_type': 'application/pdf',
                        'data': base64.b64encode(
                            pdf_content
                        ).decode('utf-8'),
                    }
                },
                {
                    'type': 'text',
                    'text': '?'
                },
            ],
        },
    )

  def test_as_format_with_chunk_preprocessor(self):
    self.assertEqual(
        lf.Template(
            'What is this {{image}}?',
            image=lf_modalities.Image.from_bytes(image_content)
        ).render().as_format(
            'anthropic',
            chunk_preprocessor=lambda x: x if isinstance(x, str) else None
        ),
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text', 'text': 'What is this'
                },
                {
                    'type': 'text', 'text': '?'
                }
            ],
        },
    )

  def test_from_value_with_simple_text(self):
    self.assertEqual(
        lf.Message.from_value(
            {
                'content': [{'type': 'text', 'text': 'this is a text'}],
            },
            format='anthropic',
        ),
        lf.AIMessage('this is a text'),
    )

  def test_from_value_with_role(self):
    self.assertEqual(
        lf.Message.from_value(
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'this is a text'}],
            },
            format='anthropic',
        ),
        lf.UserMessage('this is a text'),
    )
    self.assertEqual(
        lf.Message.from_value(
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'this is a text'}],
            },
            format='anthropic',
        ),
        lf.AIMessage('this is a text'),
    )
    self.assertEqual(
        lf.Message.from_value(
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': 'this is a text'}],
            },
            format='anthropic',
        ),
        lf.SystemMessage('this is a text'),
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported role: .*'):
      lf.Message.from_value(
          {
              'role': 'function',
              'content': [{'type': 'text', 'text': 'this is a text'}],
          },
          format='anthropic',
      )

  def test_from_value_with_thoughts(self):
    message = lf.Message.from_anthropic_format(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'thinking',
                    'thinking': 'this is a red round object',
                },
                {
                    'type': 'text',
                    'text': 'this is a apple',
                },
            ],
        },
    )
    self.assertEqual(message.text, 'this is a apple')
    self.assertEqual(message.thought, 'this is a red round object')

  def test_from_value_with_modalities(self):
    m = lf.Message.from_value(
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'What are the common words from'
                },
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/png',
                        'data': base64.b64encode(image_content).decode('utf-8'),
                    }
                },
                {
                    'type': 'text',
                    'text': 'and'
                },
                {
                    'type': 'document',
                    'source': {
                        'type': 'base64',
                        'media_type': 'application/pdf',
                        'data': base64.b64encode(pdf_content).decode('utf-8'),
                    }
                },
                {
                    'type': 'text',
                    'text': '?'
                },
            ],
        },
        format='anthropic',
    )
    self.assertEqual(
        m.text,
        'What are the common words from <<[[obj0]]>> and <<[[obj1]]>> ?'
    )
    self.assertIsInstance(m.obj0, lf_modalities.Image)
    self.assertEqual(m.obj0.mime_type, 'image/png')
    self.assertEqual(m.obj0.to_bytes(), image_content)

    self.assertIsInstance(m.obj1, lf_modalities.PDF)
    self.assertEqual(m.obj1.to_bytes(), pdf_content)


if __name__ == '__main__':
  unittest.main()
