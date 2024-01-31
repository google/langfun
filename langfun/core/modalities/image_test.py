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
"""Image tests."""
import unittest
from unittest import mock

from langfun.core.modalities import image as image_lib
import pyglove as pg


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


def mock_request(*args, **kwargs):
  del args, kwargs
  return pg.Dict(content=image_content)


class ImageContentTest(unittest.TestCase):

  def test_image_content(self):
    image = image_lib.Image.from_bytes(image_content)
    self.assertEqual(image.image_format, 'png')
    self.assertIn('data:image/png;base64,', image._repr_html_())
    self.assertEqual(image.to_bytes(), image_content)

  def test_bad_image(self):
    image = image_lib.Image.from_bytes(b'bad')
    with self.assertRaisesRegex(ValueError, 'Unsupported image format'):
      _ = image.image_format


class ImageFileTest(unittest.TestCase):

  def test_image_file(self):
    image = image_lib.Image.from_uri('http://mock/web/a.png')
    with mock.patch('requests.get') as mock_requests_get:
      mock_requests_get.side_effect = mock_request
      self.assertEqual(image.image_format, 'png')
      self.assertEqual(image._repr_html_(), '<img src="http://mock/web/a.png">')
      self.assertEqual(image.to_bytes(), image_content)


if __name__ == '__main__':
  unittest.main()
