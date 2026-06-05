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
import unittest
from langfun.core.llms import llama_cpp


class LlamaCppRemoteTest(unittest.TestCase):
  """Tests for the LlamaCppRemote model."""

  def test_basics(self):
    lm = llama_cpp.LlamaCppRemote("http://127.0.0.1:8080")
    self.assertEqual(lm.api_endpoint, "http://127.0.0.1:8080/completion")
    self.assertEqual(lm.model_id, "LLaMAC++()")
    lm = llama_cpp.LlamaCppRemote("xxx", model="x")
    self.assertEqual(lm.model_id, "LLaMAC++(x)")


if __name__ == "__main__":
  unittest.main()
