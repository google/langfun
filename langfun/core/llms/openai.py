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
"""Language models from OpenAI."""

import collections
import os
from typing import Annotated, Any
import langfun.core as lf
import openai


@lf.use_init_args(['model'])
class OpenAI(lf.LanguageModel):
  """OpenAI model."""

  model: Annotated[
      str,
      'The name of the model to use.',
  ] = 'gpt-3.5-turbo'

  api_key: Annotated[
      str | None,
      (
          'API key. If None, the key will be read from environment variable '
          "'OPENAI_API_KEY'."
      ),
  ] = None

  organization: Annotated[
      str | None,
      (
          'Organization. If None, the key will be read from environment '
          "variable 'OPENAI_ORGANIZATION'. Based on the value, usages from "
          "these API requests will count against the organization's quota. "
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()
    api_key = self.api_key or os.environ.get('OPENAI_API_KEY', None)
    if not api_key:
      raise ValueError(
          'Please specify `api_key` or set environment variable '
          '`OPENAI_API_KEY` with your OpenAI API key.'
      )
    openai.api_key = api_key
    org = self.organization or os.environ.get('OPENAI_ORGANIZATION', None)
    if org:
      openai.organization = org

  @classmethod
  def dir(cls):
    return openai.Model.list()

  def _get_request_args(self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    # Reference:
    # https://platform.openai.com/docs/api-reference/completions/create
    # NOTE(daiyip): options.top_k is not applicable.
    args = dict(
        engine=self.model,
        n=options.n,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        stream=False,
        timeout=self.timeout,
    )
    if options.top_p is not None:
      args['top_p'] = options.top_k
    return args

  def _get_results(self, response) -> list[lf.LMSamplingResult]:
    samples_by_index = collections.defaultdict(list)
    for choice in response.choices:
      samples_by_index[choice.index].append(
          lf.LMSample(text=choice.text.strip(), score=choice.logprobs or 0.0))
    return [
        lf.LMSamplingResult(samples_by_index[index])
        for index in sorted(samples_by_index.keys())
    ]

  def _sample(self, prompts: list[str]) -> list[lf.LMSamplingResult]:
    return self._with_max_attempts(
        self._sample_batch,
        (openai.error.ServiceUnavailableError, openai.error.RateLimitError),
        lambda i, e: 60,
    )(prompts)

  def _sample_batch(self, prompts: list[str]) -> list[lf.LMSamplingResult]:
    response = openai.Completion.create(
        prompt=prompts, **self._get_request_args(self.sampling_options)
    )
    return self._get_results(response)


class Gpt4(OpenAI):
  """GPT-4."""
  model = 'gpt-4'


class Gpt4_32K(Gpt4):       # pylint:disable=invalid-name
  """GPT-4 with 32K context window size."""
  model = 'gpt-4-32k'


class Gpt35(OpenAI):
  """GPT-3.5. 4K max tokens, trained up on data up to Sep, 2021."""
  model = 'text-davinci-003'


class Gpt35Turbo(Gpt35):
  """Most capable GPT-3.5 model, 10x cheaper than GPT35 (text-davinci-003)."""
  model = 'gpt-3.5-turbo'


class Gpt35Turbo16K(Gpt35Turbo):
  """Latest GPT-3.5 model with 16K context window size."""
  model = 'gpt-3.5-turbo-16k'


class Gpt3(OpenAI):
  """Most capable GPT-3 model (Davinci) 2K context window size.

  All GPT3 models have 2K max tokens and trained on data up to Oct 2019.
  """
  model = 'davinci'


class Gpt3Curie(Gpt3):
  """Very capable, but faster and lower cost than Davici."""
  model = 'curie'


class Gpt3Babbage(Gpt3):
  """Capable of straightforward tasks, very fast and low cost."""
  model = 'babbage'


class Gpt3Ada(Gpt3):
  """Capable of very simple tasks, the fastest/lowest cost among GPT3 models."""
  model = 'ada'
