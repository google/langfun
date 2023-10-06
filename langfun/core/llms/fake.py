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

from typing import Annotated
import langfun.core as lf


class Echo(lf.LanguageModel):
  """A simple echo language model for testing."""

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    return [
        lf.LMSamplingResult([lf.LMSample(prompt.text, 1.0)])
        for prompt in prompts
    ]


@lf.use_init_args(['response'])
class StaticResponse(lf.LanguageModel):
  """Language model that always gives the same canned response."""

  response: Annotated[
      str,
      'A canned response that will be returned regardless of the prompt.'
  ]

  def _sample(self, prompts: list[lf.Message]) -> list[lf.LMSamplingResult]:
    return [
        lf.LMSamplingResult([lf.LMSample(self.response, 1.0)])
        for _ in prompts
    ]


@lf.use_init_args(['mapping'])
class StaticMapping(lf.LanguageModel):
  """A static mapping from prompt to response."""

  mapping: Annotated[
      dict[str, str],
      'A mapping from prompt to response.'
  ]

  def _sample(self, prompts: list[str]) -> list[lf.LMSamplingResult]:
    return [
        lf.LMSamplingResult([lf.LMSample(self.mapping[prompt], 1.0)])
        for prompt in prompts
    ]


@lf.use_init_args(['sequence'])
class StaticSequence(lf.LanguageModel):
  """A static sequence of responses to use."""

  sequence: Annotated[
      list[str],
      'A sequence of strings as the response.'
  ]

  def _on_bound(self):
    super()._on_bound()
    self._pos = 0

  def _sample(self, prompts: list[str]) -> list[lf.LMSamplingResult]:
    results = []
    for _ in prompts:
      results.append(lf.LMSamplingResult(
          [lf.LMSample(self.sequence[self._pos], 1.0)]))
      self._pos += 1
    return results
