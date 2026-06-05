# Copyright 2023 The Langfun Authors
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
"""MIME tests."""
import inspect
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core.modalities import mime
import pyglove as pg


def mock_request(*args, **kwargs):
  del args, kwargs
  return pg.Dict(content=b'foo')


def mock_readfile(*args, **kwargs):
  del args, kwargs
  return b'bar'


class CustomMimeTest(unittest.TestCase):

  def test_is_text(self):
    self.assertTrue(mime.Custom('text/plain', b'foo').is_text)
    self.assertTrue(mime.Custom('text/xml', b'foo').is_text)
    self.assertTrue(mime.Custom('application/json', b'foo').is_text)
    self.assertTrue(mime.Custom('application/x-python-code', b'foo').is_text)
    self.assertFalse(mime.Custom('application/pdf', b'foo').is_text)
    self.assertFalse(mime.Custom('application/octet-stream', b'foo').is_text)

  def test_from_byes(self):
    content = mime.Mime.from_bytes(b'hello')
    self.assertIs(content.__class__, mime.Mime)

    content = mime.Custom('text/plain', b'foo')
    self.assertEqual(content.to_bytes(), b'foo')
    self.assertEqual(content.mime_type, 'text/plain')
    self.assertTrue(content.is_text)
    self.assertFalse(content.is_binary)
    self.assertEqual(content.to_text(), 'foo')
    self.assertTrue(content.is_compatible('text/plain'))
    self.assertFalse(content.is_compatible('text/xml'))
    self.assertIs(content.make_compatible('text/plain'), content)

    with self.assertRaisesRegex(
        lf.ModalityError, '.* cannot be converted to supported types'
    ):
      content.make_compatible('application/pdf')

    with self.assertRaisesRegex(
        ValueError, 'Either uri or content must be provided.'
    ):
      mime.Custom('text/plain')

  def test_uri(self):
    content = mime.Custom.from_uri('http://mock/web/a.txt', mime='text/plain')
    with mock.patch('requests.get') as mock_requests_stub:
      mock_requests_stub.side_effect = mock_request
      self.assertEqual(content.uri, 'http://mock/web/a.txt')
      self.assertEqual(content.content_uri, 'data:text/plain;base64,Zm9v')
      self.assertEqual(content.embeddable_uri, 'http://mock/web/a.txt')

    content = mime.Custom.from_uri('a.txt', mime='text/plain')
    with mock.patch('pyglove.io.readfile') as mock_readfile_stub:
      mock_readfile_stub.side_effect = mock_readfile
      self.assertEqual(content.uri, 'a.txt')
      self.assertEqual(content.content_uri, 'data:text/plain;base64,YmFy')
      self.assertEqual(content.embeddable_uri, 'data:text/plain;base64,YmFy')

  def test_from_uri(self):
    content = mime.Custom.from_uri('http://mock/web/a.txt', mime='text/plain')
    with mock.patch('requests.get') as mock_requests_stub:
      mock_requests_stub.side_effect = mock_request
      self.assertEqual(content.to_bytes(), b'foo')
      self.assertEqual(content.mime_type, 'text/plain')

    content = mime.Custom.from_uri('a.txt', mime='text/plain')
    with mock.patch('pyglove.io.readfile') as mock_readfile_stub:
      mock_readfile_stub.side_effect = mock_readfile
      self.assertEqual(content.to_bytes(), b'bar')
      self.assertEqual(content.mime_type, 'text/plain')

    content = mime.Mime.from_uri('data:text/plain;base64,Zm9v')
    self.assertIsNone(content.uri)
    self.assertEqual(content.mime_type, 'text/plain')
    self.assertEqual(content.content, b'foo')
    self.assertEqual(content.content_uri, 'data:text/plain;base64,Zm9v')
    self.assertEqual(content.embeddable_uri, 'data:text/plain;base64,Zm9v')

    with self.assertRaisesRegex(ValueError, 'Invalid data URI'):
      mime.Mime.from_uri('data:text/plain')

    with self.assertRaisesRegex(ValueError, 'Invalid data URI'):
      mime.Mime.from_uri('data:text/plain;abcd')

    with self.assertRaisesRegex(ValueError, 'Unsupported encoding'):
      mime.Mime.from_uri('data:text/plain;base16,abcd')

    # Test YouTube URI
    yt_uri = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    with mock.patch(
        'langfun.core.modalities.mime.Mime.download'
    ) as mock_download:
      content = mime.Mime.from_uri(yt_uri)
      self.assertIsInstance(content, mime.Custom)
      self.assertEqual(content.mime_type, 'text/html')
      self.assertEqual(content.uri, yt_uri)
      mock_download.assert_not_called()

  def assert_html_content(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_html(self):
    self.assert_html_content(
        mime.Custom('text/plain', b'foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
        ),
        """
        <details open class="pyglove custom"><summary><div class="summary-title">Custom(...)</div></summary><embed type="text/plain" src="data:text/plain;base64,Zm9v"/></details>
        """
    )
    self.assert_html_content(
        mime.Custom('text/plain', b'foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            extra_flags=dict(
                raw_mime_content=True,
            )
        ),
        """
        <embed type="text/plain" src="data:text/plain;base64,Zm9v"/>
        """
    )
    self.assert_html_content(
        mime.Custom('text/plain', b'foo').to_html(
            enable_summary_tooltip=False,
            enable_key_tooltip=False,
            extra_flags=dict(
                display_modality_when_hover=True,
            )
        ),
        """
        <details open class="pyglove custom"><summary><div class="summary-title">Custom(...)</div><span class="tooltip"><embed type="text/plain" src="data:text/plain;base64,Zm9v"/></span></summary><embed type="text/plain" src="data:text/plain;base64,Zm9v"/></details>
        """
    )


class ToTextEncodingTest(unittest.TestCase):
  """Tests for to_text() encoding handling."""

  def test_utf8_decoding(self):
    """Test that valid UTF-8 content is decoded correctly."""
    content = mime.Custom('text/plain', b'Hello, World!')
    self.assertEqual(content.to_text(), 'Hello, World!')

    # UTF-8 with multi-byte characters.
    utf8_content = 'こんにちは'.encode('utf-8')
    content = mime.Custom('text/plain', utf8_content)
    self.assertEqual(content.to_text(), 'こんにちは')

  def test_utf16_le_bom_decoding(self):
    """Test that UTF-16 Little Endian with BOM is decoded correctly."""
    # UTF-16 LE BOM: 0xff 0xfe
    utf16_le_content = 'Hello'.encode('utf-16-le')
    content_with_bom = b'\xff\xfe' + utf16_le_content
    content = mime.Custom('text/plain', content_with_bom)
    self.assertEqual(content.to_text(), 'Hello')

  def test_utf16_be_bom_decoding(self):
    """Test that UTF-16 Big Endian with BOM is decoded correctly."""
    # UTF-16 BE BOM: 0xfe 0xff
    utf16_be_content = 'Hello'.encode('utf-16-be')
    content_with_bom = b'\xfe\xff' + utf16_be_content
    content = mime.Custom('text/plain', content_with_bom)
    self.assertEqual(content.to_text(), 'Hello')

  def test_invalid_bytes_fallback_with_replacement(self):
    """Test that invalid bytes are replaced with replacement character."""
    # 0xff alone is invalid in UTF-8 and doesn't have UTF-16 BOM pattern.
    invalid_content = b'\xff\xfdHello'
    content = mime.Custom('text/plain', invalid_content)
    result = content.to_text()
    # Invalid bytes should be replaced with U+FFFD (replacement character).
    self.assertIn('\ufffd', result)
    self.assertIn('Hello', result)

  def test_binary_mime_type_raises_error(self):
    """Test that binary MIME types raise ModalityError."""
    content = mime.Custom('application/octet-stream', b'\x00\x01\x02')
    with self.assertRaisesRegex(
        lf.ModalityError, 'cannot be converted to text'
    ):
      content.to_text()


class TextCompatibilityTest(unittest.TestCase):

  def test_exact_mime_match(self):
    content = mime.Custom('text/plain', b'hello')
    self.assertTrue(content.is_compatible('text/plain'))
    result = content.make_compatible('text/plain')
    self.assertIs(result, content)

  def test_text_content_with_misdetected_mime_type(self):
    ts_code = b'import type { Foo } from "bar";\nexport const x = 1;\n'
    content = mime.Custom('video/mp2t', ts_code)
    self.assertTrue(content.is_compatible(['text/plain', 'image/png']))
    result = content.make_compatible(['text/plain', 'image/png'])
    self.assertIsInstance(result, mime.Custom)
    self.assertEqual(result.mime_type, 'text/plain')
    self.assertEqual(result.to_bytes(), ts_code)

  def test_text_mime_not_in_supported_list(self):
    content = mime.Custom('text/x-typescript', b'const x: number = 1;\n')
    self.assertTrue(content.is_compatible(['text/plain']))
    result = content.make_compatible(['text/plain'])
    self.assertEqual(result.mime_type, 'text/plain')

  def test_application_text_compatible_with_text_plain(self):
    content = mime.Custom('application/x-yaml', b'key: value\n')
    self.assertTrue(content.is_compatible(['text/plain']))
    result = content.make_compatible(['text/plain'])
    self.assertEqual(result.mime_type, 'text/plain')

  def test_binary_content_not_compatible_with_text_plain(self):
    binary_data = bytes(range(256))
    content = mime.Custom('application/octet-stream', binary_data)
    self.assertFalse(content.is_compatible(['text/plain']))
    with self.assertRaises(lf.ModalityError):
      content.make_compatible(['text/plain'])

  def test_no_fallback_without_text_plain_in_targets(self):
    ts_code = b'const x = 1;\n'
    content = mime.Custom('video/mp2t', ts_code)
    self.assertFalse(content.is_compatible(['image/png', 'audio/wav']))
    with self.assertRaises(lf.ModalityError):
      content.make_compatible(['image/png', 'audio/wav'])

  def test_make_compatible_preserves_content(self):
    original = b'# Markdown\n\nHello world\n'
    content = mime.Custom('text/markdown', original)
    result = content.make_compatible(['text/plain'])
    self.assertEqual(result.to_bytes(), original)
    self.assertEqual(result.to_text(), '# Markdown\n\nHello world\n')

  def test_unicode_content_compatible_with_text_plain(self):
    unicode_bytes = 'こんにちは世界 🎉'.encode('utf-8')
    content = mime.Custom('video/mp2t', unicode_bytes)
    self.assertTrue(content.is_compatible(['text/plain']))
    result = content.make_compatible(['text/plain'])
    self.assertEqual(result.to_text(), 'こんにちは世界 🎉')


if __name__ == '__main__':
  unittest.main()
