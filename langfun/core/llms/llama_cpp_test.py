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
"""Tests for llama cpp models."""

import unittest
from unittest import mock
import typing
import langfun.core as lf
from langfun.core.llms import llama_cpp


def mock_requests_post(url: str, json: typing.Dict[str, typing.Any], **kwargs):
  del kwargs
  class TEMP:
    def json(self):
      return {"content": json["prompt"] + "\n" + url}
  return TEMP()


class LlamaCppRemoteTest(unittest.TestCase):
  """Tests for the LlamaCppRemote model."""

  def test_call_completion(self):
    with mock.patch('requests.post') as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = llama_cpp.LlamaCppRemote(url="http://127.0.0.1:8080")
      response = lm('hello', sampling_options=lf.LMSamplingOptions(n=1))
      self.assertEqual(
          response.text,
          'hello\nhttp://127.0.0.1:8080/completion',
      )

if __name__ == '__main__':
  unittest.main()
