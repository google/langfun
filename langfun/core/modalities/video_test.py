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
"""Video tests."""
import unittest
from unittest import mock

from langfun.core.modalities import video as video_lib
import pyglove as pg


mp4_bytes = (
    b'\x00\x00\x00\x20ftypmp42'
    b'\x00\x00\x00\x08moov'
    b'\x00\x00\x00\x0ctrak'
    b'\x00\x00\x00\x08mdia'
)


def mock_request(*args, **kwargs):
  del args, kwargs
  return pg.Dict(content=mp4_bytes)


class VideoContentTest(unittest.TestCase):

  def test_video_content(self):
    video = video_lib.Video.from_bytes(mp4_bytes)
    self.assertEqual(video.mime_type, 'video/mp4')
    self.assertEqual(video.video_format, 'mp4')
    self.assertIn('data:video/mp4;base64,', video._raw_html())
    self.assertEqual(video.to_bytes(), mp4_bytes)

  def test_bad_video(self):
    video = video_lib.Video.from_bytes(b'bad')
    with self.assertRaisesRegex(ValueError, 'Expected MIME type'):
      _ = video.video_format


class VideoFileTest(unittest.TestCase):

  def test_video_file(self):
    video = video_lib.Video.from_uri('http://mock/web/a.mp4')
    with mock.patch('requests.get') as mock_requests_get:
      mock_requests_get.side_effect = mock_request
      self.assertEqual(video.video_format, 'mp4')
      self.assertEqual(video.mime_type, 'video/mp4')
      self.assertEqual(
          video._raw_html(),
          '<video controls> <source src="http://mock/web/a.mp4"> </video>',
      )
      self.assertEqual(video.to_bytes(), mp4_bytes)


if __name__ == '__main__':
  unittest.main()
