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
"""Fake LMs for testing."""

import abc
import functools
from typing import Annotated
import langfun.core as lf


class Fake(lf.LanguageModel):
  """The base class for all fake language models."""

  def _score(self, prompt: lf.Message| list[lf.Message],
             completions: list[lf.Message]):
    return [lf.LMScoringResult(score=-i * 1.0) for i in range(len(completions))]

  def _tokenize(self, prompt: lf.Message) -> list[tuple[str | bytes, int]]:
    return [(w, i) for i, w in enumerate(prompt.text.split(' '))]

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    results = []
    for prompt in prompts:
      response = self._response_from(prompt)
      results.append(
          lf.LMSamplingResult(
              [lf.LMSample(response, 1.0)],
              usage=lf.LMSamplingUsage(
                  prompt_tokens=len(prompt.text),
                  completion_tokens=len(response.text),
                  total_tokens=len(prompt.text) + len(response.text),
              )
          )
      )
    return results

  @functools.cached_property
  def model_info(self) -> lf.ModelInfo:
    """Returns the specification of the model."""
    return lf.ModelInfo(model_id=self.__class__.__name__)

  @abc.abstractmethod
  def _response_from(self, prompt: lf.Message) -> lf.Message:
    """Returns the response for the given prompt."""


class Echo(Fake):
  """A simple echo language model for testing."""

  def _response_from(self, prompt: lf.Message) -> lf.Message:
    return lf.AIMessage(prompt.text)


@lf.use_init_args(['response'])
class StaticResponse(Fake):
  """Language model that always gives the same canned response."""

  response: Annotated[
      str | lf.Message,
      'A canned response that will be returned regardless of the prompt.'
  ]

  def _response_from(self, prompt: lf.Message) -> lf.Message:
    return lf.AIMessage.from_value(self.response)


@lf.use_init_args(['mapping'])
class StaticMapping(Fake):
  """A static mapping from prompt to response."""

  mapping: Annotated[
      dict[str, str | lf.Message],
      'A mapping from prompt to response.'
  ]

  def _response_from(self, prompt: lf.Message) -> lf.Message:
    return lf.AIMessage.from_value(self.mapping[prompt])


@lf.use_init_args(['sequence'])
class StaticSequence(Fake):
  """A static sequence of responses to use."""

  sequence: Annotated[
      list[str | lf.Message],
      'A sequence of strings as the response.'
  ]

  def _on_bound(self):
    super()._on_bound()
    self._pos = 0

  def _response_from(self, prompt: lf.Message) -> lf.Message:
    r = lf.AIMessage.from_value(self.sequence[self._pos])
    self._pos += 1
    return r
