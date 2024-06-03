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

import typing
import unittest
from unittest import mock

from langfun.core.llms import llama_cpp


def mock_requests_post(url: str, json: typing.Dict[str, typing.Any], **kwargs):
  del kwargs

  class TEMP:
    @property
    def status_code(self):
      return 200

    def json(self):
      return {"content": json["prompt"] + "\n" + url}

  return TEMP()


class LlamaCppRemoteTest(unittest.TestCase):
  """Tests for the LlamaCppRemote model."""

  def test_call_completion(self):
    with mock.patch("requests.Session.post") as mock_request:
      mock_request.side_effect = mock_requests_post
      lm = llama_cpp.LlamaCppRemote("http://127.0.0.1:8080")
      [result] = lm.sample(["hello"], n=2)
      self.assertEqual(
          len(result.samples),
          2
      )
      self.assertEqual(
          str(result.samples[0].response),
          "hello\nhttp://127.0.0.1:8080/completion",
      )

  def test_model_id(self):
    lm = llama_cpp.LlamaCppRemote("http://127.0.0.1:8080")
    self.assertEqual(lm.model_id, "LLaMAC++()")
    lm = llama_cpp.LlamaCppRemote("xxx", model="x")
    self.assertEqual(lm.model_id, "LLaMAC++(x)")


if __name__ == "__main__":
  unittest.main()
