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
"""Tests for Groq models."""

import os
from typing import Any
import unittest
from unittest import mock
from langfun.core import modalities as lf_modalities
from langfun.core.llms import groq
import pyglove as pg
import requests


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs

  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'choices': [{
          'message': {
              'content': [{
                  'type': 'text',
                  'text': (
                      f'hello with temperature={json.get("temperature")}, '
                      f'top_p={json.get("top_p")}, '
                      f'max_tokens={json.get("max_tokens")}, '
                      f'stop={json.get("stop")}.'
                  ),
              }],
          }
      }],
      'usage': {
          'prompt_tokens': 2,
          'completion_tokens': 1,
          'total_tokens': 3,
      },
  }).encode()
  return response


def mock_mm_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  v = json['messages'][0]['content'][0]
  image = lf_modalities.Image.from_uri(v['image_url'])

  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'choices': [
          {
              'message': {
                  'content': [{
                      'type': 'text',
                      'text': image.uri,
                  }],
              }
          }
      ],
      'usage': {
          'prompt_tokens': 2,
          'completion_tokens': 1,
          'total_tokens': 3,
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


class AuthropicTest(unittest.TestCase):

  def test_basics(self):
    self.assertEqual(groq.GroqMistral_8x7B().model_id, 'mixtral-8x7b-32768')
    self.assertEqual(groq.GroqMistral_8x7B().max_concurrency, 16)

  def test_api_key(self):
    lm = groq.GroqMistral_8x7B()
    with self.assertRaisesRegex(ValueError, 'Please specify `api_key`'):
      lm('hi')

    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post

      lm = groq.GroqMistral_8x7B(api_key='fake key')
      self.assertRegex(lm('hi').text, 'hello.*')

      os.environ['GROQ_API_KEY'] = 'abc'
      lm = groq.GroqMistral_8x7B()
      self.assertRegex(lm('hi').text, 'hello.*')
      del os.environ['GROQ_API_KEY']

  def test_call(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = groq.GroqLlama3_70B(api_key='fake_key')
      response = lm(
          'hello',
          temperature=0.0,
          max_tokens=1024,
          top_k=0.1,
          top_p=0.2,
          stop=['\n'],
      )
      self.assertEqual(
          response.text,
          (
              'hello with temperature=0.0, top_p=0.2, '
              "max_tokens=1024, stop=['\\n']."
          ),
      )
      self.assertIsNotNone(response.usage)
      self.assertIsNotNone(response.usage.prompt_tokens, 2)
      self.assertIsNotNone(response.usage.completion_tokens, 1)
      self.assertIsNotNone(response.usage.total_tokens, 3)

  def test_mm_call(self):
    with mock.patch('requests.Session.post') as mock_mm_request:
      mock_mm_request.side_effect = mock_mm_requests_post
      lm = groq.GroqLlama3_70B(multimodal=True, api_key='fake_key')
      response = lm(lf_modalities.Image.from_uri('https://fake/image.jpg'))
      self.assertEqual(response.text, 'https://fake/image.jpg')

  def test_call_errors(self):
    for status_code, error_type, error_message in [
        (429, 'rate_limit', 'Rate limit exceeded.'),
        (503, 'service_unavailable', 'Service unavailable.'),
        (500, 'bad_request', 'Bad request.'),
    ]:
      with mock.patch('requests.Session.post') as mock_mm_request:
        mock_mm_request.side_effect = mock_requests_post_error(
            status_code, error_type, error_message
        )
        lm = groq.GroqLlama3_70B(api_key='fake_key')
        with self.assertRaisesRegex(
            Exception, f'{status_code}:.*{error_type}'
        ):
          lm('hello', lm=lm, max_attempts=1)


if __name__ == '__main__':
  unittest.main()
