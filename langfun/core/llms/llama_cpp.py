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
"""Language models from llama.cpp."""

from typing import Annotated

import langfun.core as lf
import requests


@lf.use_init_args(["url"])
class LlamaCppRemote(lf.LanguageModel):
  """The remote LLaMA C++ model.

  The Remote LLaMA C++ models can be launched via
  https://github.com/ggerganov/llama.cpp/tree/master/examples/server
  """

  url: Annotated[
      str,
      "The name of the model to use.",
  ] = ""

  name: Annotated[
      str,
      "The abbreviation for the LLaMA CPP-based model name.",
  ] = ""

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f"LLaMAC++({self.name})"

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    def _complete_fn(cur_prompts):
      results = []
      for prompt in cur_prompts:
        result = lf.LMSamplingResult()
        for _ in range(self.sampling_options.n or 1):
          data = {
              "prompt": prompt.text,
              "n_predict": self.sampling_options.max_tokens,
              "top_k": self.sampling_options.top_k or 50,
              "top_p": self.sampling_options.top_p or 0.95,
          }
          if self.sampling_options.temperature is not None:
            data["temperature"] = self.sampling_options.temperature

          response = requests.post(
              f"{self.url}/completion",
              json=data,
              headers={"Content-Type": "application/json"},
              timeout=self.timeout,
          )
          decoded_response = response.json()
          response = decoded_response["content"]
          result.samples.append(lf.LMSample(response, score=0.0))
        results.append(result)
      return results

    return self._parallel_execute_with_currency_control(
        _complete_fn, [prompts]
    )[0]
