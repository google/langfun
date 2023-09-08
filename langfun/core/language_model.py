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
"""Interface for language model."""

import abc
import time
from typing import Annotated
from langfun.core import component
from langfun.core import console
from langfun.core import message as message_lib
import pyglove as pg


class LMSample(pg.Object):
  """Response candidate."""

  response: pg.typing.Annotated[
      pg.typing.Object(
          message_lib.Message,
          # Allowing automatic conversion from text to AIMessage.
          transform=message_lib.AIMessage.from_value
      ),
      'The natural language response of LM.'
  ]

  score: Annotated[
      float, 'The score of sampled response. The larger is better'
  ] = 0.0


class LMSamplingResult(pg.Object):
  """Language model response."""

  samples: Annotated[
      list[LMSample],
      (
          'Multiple samples of the same prompt, sorted by their score. '
          'The first candidate has the highest score.'
      ),
  ] = []

  error: Annotated[
      Exception | None,
      (
          'Error information if sampling request failed. If Not None, '
          '`samples` will be an empty list.'
      ),
  ] = None


class LMSamplingOptions(component.Component):
  """Language model sampling options."""

  temperature: Annotated[float, 'Model temperature between [0, 1.0].'] = 0.0
  max_tokens: Annotated[int, 'Per example max tokens to generate.'] = 1024
  n: Annotated[int | None, 'Max number of samples to return.'] = 1
  top_k: Annotated[int | None, 'Top k tokens to sample the next token.'] = 40
  top_p: Annotated[
      float | None,
      (
          'Only sample the next token from top N tokens whose accumulated '
          'probability // mass <= p. Not applicable to OpenAI models and '
          'BigBard.'
      ),
  ] = None
  random_seed: Annotated[
      int | None, 'A fixed random seed used during model inference.'
  ] = None


class LanguageModel(component.Component):
  """Interface of a language model.

  Language models are at the center of LLM-based agents. ``LanguageModel``
  is the interface to interact with different language modles.

  In langfun, users can use different language models with the same agents,
  allowing fast prototype, as well as side-by-side comparisons.
  """

  sampling_options: LMSamplingOptions = LMSamplingOptions()

  timeout: Annotated[
      float | None, 'Timeout in seconds. If None, there is no timeout.'
  ] = 30.0

  max_attempts: Annotated[
      int,
      (
          'A number of max attempts to request the LM if fails.'
          'The retry wait time is determined per LM serivice.'
      ),
  ] = 5

  debug: Annotated[
      bool, 'If True, the prompt and the response will be output to stdout.'
  ] = False

  @pg.explicit_method_override
  def __init__(self, *args, **kwargs) -> None:
    """Overrides __init__ to pass through **kwargs to sampling options."""

    sampling_options = kwargs.pop('sampling_options', LMSamplingOptions())
    sampling_options_delta = {}

    for k, v in kwargs.items():
      if LMSamplingOptions.__schema__.get_field(k) is not None:
        sampling_options_delta[k] = v

    if sampling_options_delta:
      sampling_options.rebind(sampling_options_delta)

    for k in sampling_options_delta:
      del kwargs[k]

    super().__init__(*args, sampling_options=sampling_options, **kwargs)

  def _on_bound(self):
    super()._on_bound()
    self._call_counter = 0

  def sample(self,
             prompts: list[str | message_lib.Message],
             **kwargs) -> list[LMSamplingResult]:
    """Samples one or multiple prompts."""
    with component.context(override_attrs=True, **kwargs):
      return self._sample([
          message_lib.UserMessage.from_value(p)
          for p in prompts
      ])

  @abc.abstractmethod
  def _sample(
      self,
      prompt: list[message_lib.Message],
  ) -> list[LMSamplingResult]:
    """Subclass should override."""

  def __call__(self, prompt: message_lib.Message, **kwargs) -> str:
    """Returns the first candidate."""
    with component.context(override_attrs=True, **kwargs):
      sampling_options = self.sampling_options
      if sampling_options.n != 1:
        sampling_options = sampling_options.clone(override=dict(n=1))

      call_counter = self._call_counter
      self._call_counter += 1

      request_start = time.time()
      result = self.sample([prompt], sampling_options=sampling_options)[0]
      response = result.samples[0].response
      elapse = time.time() - request_start

      if self.debug:
        console.write(
            self.format(compact=True),
            title=f'[{call_counter}] LM INFO:',
            color='magenta',
        )
        console.write(
            prompt,
            title=f'\n[{call_counter}] PROMPT SENT TO LM:',
            color='green',
        )
        console.write(
            str(response) + '\n',
            title=(
                f'\n[{call_counter}] LM RESPONSE '
                f'(in {elapse:.2f} seconds):'
            ),
            color='blue',
        )

      if result.error:
        raise result.error
      return response
