# Copyright 2024 The Langfun Authors
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
"""Audio tests."""
import unittest
from unittest import mock

from langfun.core.modalities import audio as audio_lib
import pyglove as pg


content_bytes = (
    b'RIFF$\x00\x00\x00WAVEfmt'
    b' \x10\x00\x00\x00\x01\x00\x01\x00\x00\x04\x00\x00\x00\x04\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'
)


def mock_request(*args, **kwargs):
  del args, kwargs
  return pg.Dict(content=content_bytes)


class AudioTest(unittest.TestCase):

  def test_audio_content(self):
    audio = audio_lib.Audio.from_bytes(content_bytes)
    self.assertEqual(audio.mime_type, 'audio/x-wav')
    self.assertEqual(audio.audio_format, 'x-wav')
    self.assertEqual(audio.to_bytes(), content_bytes)

  def test_bad_audio(self):
    audio = audio_lib.Audio.from_bytes(b'bad')
    with self.assertRaisesRegex(ValueError, 'Expected MIME type'):
      _ = audio.audio_format


class AudioFileTest(unittest.TestCase):

  def test_audio_file(self):
    audio = audio_lib.Audio.from_uri('http://mock/web/a.wav')
    with mock.patch('requests.get') as mock_requests_get:
      mock_requests_get.side_effect = mock_request
      self.assertEqual(audio.audio_format, 'x-wav')
      self.assertEqual(audio.mime_type, 'audio/x-wav')
      self.assertEqual(
          audio._repr_html_(),
          '<audio controls> <source src="http://mock/web/a.wav"> </audio>',
      )
      self.assertEqual(audio.to_bytes(), content_bytes)


if __name__ == '__main__':
  unittest.main()
