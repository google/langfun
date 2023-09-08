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
import pyglove as pg


class Usage(pg.Object):
  """Usage information per completion."""

  prompt_tokens: int
  completion_tokens: int
  total_tokens: int


class LMSamplingResult(lf.LMSamplingResult):
  """LMSamplingResult with usage information."""

  usage: Usage | None = None


@lf.use_init_args(['model'])
class OpenAI(lf.LanguageModel):
  """OpenAI model."""

  model: pg.typing.Annotated[
      pg.typing.Enum[
          'gpt-4',
          'gpt-4-32k',
          'gpt-3.5-turbo',
          'gpt-3.5-turbo-16k',
          'text-davinci-003',
          'davinci',
          'curie',
          'babbage',
          'ada',
      ],
      'The name of the model to use.',
  ] = 'gpt-3.5-turbo'

  _CHAT_MODELS = frozenset([
      'gpt-4',
      'gpt-4-32k',
      'gpt-3.5-turbo',
      'gpt-3.5-turbo-16k',
  ])

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

  @property
  def is_chat_model(self):
    """Returns True if the model is a chat model."""
    return self.model in OpenAI._CHAT_MODELS

  def _get_request_args(
      self, options: lf.LMSamplingOptions) -> dict[str, Any]:
    # Reference:
    # https://platform.openai.com/docs/api-reference/completions/create
    # NOTE(daiyip): options.top_k is not applicable.
    args = dict(
        n=options.n,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        stream=False,
        timeout=self.timeout,
    )
    # Completion and ChatCompletion uses different parameter name for model.
    args['model' if self.is_chat_model else 'engine'] = self.model

    if options.top_p is not None:
      args['top_p'] = options.top_k
    return args

  def _sample(self, prompts: list[lf.Message]) -> list[LMSamplingResult]:
    if self.is_chat_model:
      return self._chat_complete_batch(prompts)
    return self._complete_batch(prompts)

  def _complete_batch(
      self, prompts: list[lf.Message]) -> list[LMSamplingResult]:

    def _open_ai_completion(prompts):
      response = openai.Completion.create(
          prompt=[p.text for p in prompts],
          **self._get_request_args(self.sampling_options),
      )

      # Parse response.
      samples_by_index = collections.defaultdict(list)
      for choice in response.choices:
        samples_by_index[choice.index].append(
            lf.LMSample(choice.text.strip(), score=choice.logprobs or 0.0)
        )

      usage = Usage(
          prompt_tokens=response.usage.prompt_tokens,
          completion_tokens=response.usage.completion_tokens,
          total_tokens=response.usage.total_tokens,
      )
      return [
          LMSamplingResult(
              samples_by_index[index], usage=usage if index == 0 else None
          )
          for index in sorted(samples_by_index.keys())
      ]

    return lf.with_retry(
        _open_ai_completion,
        retry_on_errors=(
            openai.error.ServiceUnavailableError,
            openai.error.RateLimitError,
        ),
        max_attempts=self.max_attempts,
        retry_interval=(1, 60),
        exponential_backoff=True,
    )(prompts)

  def _chat_complete_batch(
      self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:

    def _open_ai_chat_completion(prompt):
      response = openai.ChatCompletion.create(
          # TODO(daiyip): support conversation history.
          messages=[{'role': 'user', 'content': prompt.text}],
          **self._get_request_args(self.sampling_options),
      )
      return LMSamplingResult(
          [
              lf.LMSample(choice.message.content, score=0.0)
              for choice in response.choices
          ],
          usage=Usage(
              prompt_tokens=response.usage.prompt_tokens,
              completion_tokens=response.usage.completion_tokens,
              total_tokens=response.usage.total_tokens,
          ),
      )

    return lf.concurrent_execute(
        _open_ai_chat_completion,
        prompts,
        max_workers=8,
        retry_on_errors=(
            openai.error.ServiceUnavailableError,
            openai.error.RateLimitError,
        ),
        max_attempts=self.max_attempts,
        retry_interval=(1, 60),
        exponential_backoff=True,
    )


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
