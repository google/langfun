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

from typing import Any

import langfun.core as lf
from langfun.core.llms import rest
import pyglove as pg


class LlamaCppRemote(rest.REST):
  """The remote LLaMA C++ model.

  The Remote LLaMA C++ models can be launched via
  https://github.com/ggerganov/llama.cpp/tree/master/examples/server
  """

  @pg.explicit_method_override
  def __init__(self, url: str, model: str | None = None, **kwargs):
    super().__init__(api_endpoint=f'{url}/completion', model=model, **kwargs)

  @property
  def model_id(self) -> str:
    """Returns a string to identify the model."""
    return f'LLaMAC++({self.model or ""})'

  def request(
      self, prompt: lf.Message, sampling_options: lf.LMSamplingOptions
  ) -> dict[str, Any]:
    """Returns the JSON input for a message."""
    request = dict()
    request.update(self._request_args(sampling_options))
    # NOTE(daiyip): multi-modal is current not supported.
    request['prompt'] = prompt.text
    return request

  def _request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    """Returns a dict as request arguments."""
    args = dict(
        n_predict=options.max_tokens or 1024,
        top_k=options.top_k or 50,
        top_p=options.top_p or 0.95,
    )
    if options.temperature is not None:
      args['temperature'] = options.temperature
    return args

  def result(self, json: dict[str, Any]) -> lf.LMSamplingResult:
    return lf.LMSamplingResult(
        [lf.LMSample(item['content'], score=0.0) for item in json['items']]
    )

  def _sample_single(self, prompt: lf.Message) -> lf.LMSamplingResult:
    request = self.request(prompt, self.sampling_options)

    def _sample_one_example(request):
      response = self._session.post(
          self.api_endpoint,
          json=request,
          timeout=self.timeout,
      )
      if response.status_code == 200:
        return response.json()
      else:
        error_cls = self._error_cls_from_status(response.status_code)
        raise error_cls(f'{response.status_code}: {response.content}')

    items = self._parallel_execute_with_currency_control(
        _sample_one_example,
        [request] * (self.sampling_options.n or 1),
    )
    return self.result(dict(items=items))
