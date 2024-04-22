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
"""Tests for Anthropic models."""

import base64
import os
from typing import Any
import unittest
from unittest import mock
from langfun.core import modalities as lf_modalities
from langfun.core.llms import anthropic
import pyglove as pg
import requests


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs

  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'content': [{
          'type': 'text',
          'text': (
              f'hello with temperature={json.get("temperature")}, '
              f'top_k={json.get("top_k")}, '
              f'top_p={json.get("top_p")}, '
              f'max_tokens={json.get("max_tokens")}, '
              f'stop={json.get("stop_sequences")}.'
          ),
      }],
      'usage': {
          'input_tokens': 2,
          'output_tokens': 1,
      },
  }).encode()
  return response


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


def mock_mm_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  v = json['messages'][0]['content'][0]
  image = lf_modalities.Image.from_bytes(base64.b64decode(v['source']['data']))

  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'content': [{
          'type': 'text',
          'text': f'{v["type"]}: {image.mime_type}',
      }],
      'usage': {
          'input_tokens': 2,
          'output_tokens': 1,
      },
  }).encode()
  return response


def mock_requests_post_error(status_code, error_type, error_message):
  def _mock_requests(url: str, json: dict[str, Any], **kwargs):
    del url, json, kwargs
    response = requests.Response()
    response.status_code = status_code
    response._content = pg.to_json_str(
        {
            'error': {
                'type': error_type,
                'message': error_message,
            }
        }
    ).encode()
    return response

  return _mock_requests


class AnthropicTest(unittest.TestCase):

  def test_basics(self):
    self.assertEqual(
        anthropic.Claude3Haiku().model_id, 'claude-3-haiku-20240307'
    )
    self.assertGreater(anthropic.Claude3Haiku().max_concurrency, 0)

  def test_api_key(self):
    lm = anthropic.Claude3Haiku()
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      lm('hi')

    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post

      lm = anthropic.Claude3Haiku(api_key='fake key')
      self.assertRegex(lm('hi').text, 'hello.*')

      os.environ['ANTHROPIC_API_KEY'] = 'abc'
      lm = anthropic.Claude3Haiku()
      self.assertRegex(lm('hi').text, 'hello.*')
      del os.environ['ANTHROPIC_API_KEY']

  def test_call(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      response = lm('hello', temperature=0.0, top_k=0.1, top_p=0.2, stop=['\n'])
      self.assertEqual(
          response.text,
          (
              'hello with temperature=0.0, top_k=0.1, top_p=0.2, '
              "max_tokens=4096, stop=['\\n']."
          ),
      )
      self.assertIsNotNone(response.usage)
      self.assertIsNotNone(response.usage.prompt_tokens, 2)
      self.assertIsNotNone(response.usage.completion_tokens, 1)
      self.assertIsNotNone(response.usage.total_tokens, 3)

  def test_mm_call(self):
    with mock.patch('requests.Session.post') as mock_mm_request:
      mock_mm_request.side_effect = mock_mm_requests_post
      lm = anthropic.Claude3Haiku(api_key='fake_key')
      response = lm(lf_modalities.Image.from_bytes(image_content), lm=lm)
      self.assertEqual(response.text, 'image: image/png')

  def test_call_errors(self):
    for status_code, error_type, error_message in [
        (429, 'rate_limit', 'Rate limit exceeded.'),
        (529, 'service_unavailable', 'Service unavailable.'),
        (500, 'bad_request', 'Bad request.'),
    ]:
      with mock.patch('requests.Session.post') as mock_mm_request:
        mock_mm_request.side_effect = mock_requests_post_error(
            status_code, error_type, error_message
        )
        lm = anthropic.Claude3Haiku(api_key='fake_key')
        with self.assertRaisesRegex(
            Exception, f'.*{status_code}: .*{error_message}'
        ):
          lm('hello', lm=lm, max_attempts=1)


if __name__ == '__main__':
  unittest.main()
