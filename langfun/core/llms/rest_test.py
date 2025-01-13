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
"""Tests for REST models."""

from typing import Any
import unittest
from unittest import mock
import langfun.core as lf
from langfun.core.llms import rest
import pyglove as pg
import requests


def mock_requests_post(url: str, json: dict[str, Any], **kwargs):
  del url, kwargs
  response = requests.Response()
  response.status_code = 200
  response._content = pg.to_json_str({
      'content': [(
          f'hello with temperature={json.get("temperature")}, '
          f'top_k={json.get("top_k")}, '
          f'top_p={json.get("top_p")}, '
          f'max_tokens={json.get("max_tokens")}, '
          f'stop={json.get("stop_sequences")}.'
      )],
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


class RestTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._lm = rest.REST(
        api_endpoint='https://fake-api.com',
        request=lambda x, o: dict(
            model='test-model',
            prompt=x.text,
            temperature=0.0,
            top_k=0.1,
            top_p=0.2,
            stop_sequences=['\n'],
            max_tokens=4096,
        ),
        result=lambda x: lf.LMSamplingResult(
            [lf.LMSample(c) for c in x['content']]),
        headers=dict(api_key='fake_key'),
    )

  def test_call(self):
    with mock.patch('requests.Session.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      self.assertEqual(self._lm.model_id, 'unknown')
      response = self._lm(
          'hello', temperature=0.0, top_k=0.1, top_p=0.2, stop=['\n'])
      self.assertEqual(
          response.text,
          (
              'hello with temperature=0.0, top_k=0.1, top_p=0.2, '
              "max_tokens=4096, stop=['\\n']."
          ),
      )
      self.assertIsInstance(response.usage, lf.UsageNotAvailable)

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
        with self.assertRaisesRegex(
            Exception, f'.*{status_code}: .*{error_message}'
        ):
          self._lm('hello', max_attempts=1)


if __name__ == '__main__':
  unittest.main()
