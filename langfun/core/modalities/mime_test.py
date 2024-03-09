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
import unittest
from unittest import mock

from langfun.core.modalities import mime
import pyglove as pg


def mock_request(*args, **kwargs):
  del args, kwargs
  return pg.Dict(content='foo')


def mock_readfile(*args, **kwargs):
  del args, kwargs
  return 'bar'


class CustomMimeTest(unittest.TestCase):

  def test_content(self):
    content = mime.Custom('text/plain', 'foo')
    self.assertEqual(content.to_bytes(), 'foo')
    self.assertEqual(content.mime_type, 'text/plain')

    with self.assertRaisesRegex(
        ValueError, 'Either uri or content must be provided.'
    ):
      mime.Custom('text/plain')

  def test_from_uri(self):
    content = mime.Custom.from_uri('http://mock/web/a.txt', type='text/plain')
    with mock.patch('requests.get') as mock_requests_stub:
      mock_requests_stub.side_effect = mock_request
      self.assertEqual(content.to_bytes(), 'foo')
      self.assertEqual(content.mime_type, 'text/plain')

    content = mime.Custom.from_uri('a.txt', type='text/plain')
    with mock.patch('pyglove.io.readfile') as mock_readfile_stub:
      mock_readfile_stub.side_effect = mock_readfile
      self.assertEqual(content.to_bytes(), 'bar')
      self.assertEqual(content.mime_type, 'text/plain')


if __name__ == '__main__':
  unittest.main()
