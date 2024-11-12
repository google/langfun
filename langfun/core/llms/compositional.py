# Copyright 2024 The Langfun Authors
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
"""Compositions of different LLM models."""
import random
from typing import Annotated

import langfun.core as lf
import pyglove as pg


@pg.use_init_args(['candidates', 'seed'])
class RandomChoice(lf.LanguageModel):
  """Random choice of a list of LLM models."""

  candidates: Annotated[
      list[lf.LanguageModel],
      (
          'A list of LLMs as candidates to choose from.'
      )
  ]

  seed: Annotated[
      int,
      (
          'The random seed to use for the random choice.'
      )
  ] = 0

  def _on_bound(self):
    super()._on_bound()
    self._rand = random.Random(self.seed)
    # Applying sampling options to all candidates.
    parent_non_default = self.sampling_options.sym_nondefault()
    if parent_non_default:
      for c in self.candidates:
        c.sampling_options.rebind(
            parent_non_default, notify_parents=False, raise_on_no_change=False
        )

  @property
  def model_id(self) -> str:
    model_ids = ', '.join(
        sorted(c.model_id for c in self.candidates)
    )
    return f'RandomChoice({model_ids})'

  @property
  def resource_id(self) -> str:
    resource_ids = ', '.join(
        sorted(c.resource_id for c in self.candidates)
    )
    return f'RandomChoice({resource_ids})'

  def _select_lm(self) -> lf.LanguageModel:
    """Selects a random LLM from the candidates."""
    return self._rand.choice(self.candidates)

  def sample(
      self,
      prompts: list[str | lf.Message],
      *,
      cache_seed: int = 0,
      **kwargs,
  ) -> list[lf.LMSamplingResult]:
    return self._select_lm().sample(
        prompts, cache_seed=cache_seed, **kwargs
    )

  def __call__(
      self, prompt: lf.Message, *, cache_seed: int = 0, **kwargs
  ) -> lf.Message:
    return self._select_lm()(prompt, cache_seed=cache_seed, **kwargs)

  def score(
      self,
      prompt: str | lf.Message | list[lf.Message],
      completions: list[str | lf.Message],
      **kwargs,
  ) -> list[lf.LMScoringResult]:
    return self._select_lm().score(prompt, completions, **kwargs)

  def tokenize(
      self,
      prompt: str | lf.Message,
      **kwargs,
  ) -> list[tuple[str | bytes, int]]:
    return self._select_lm().tokenize(prompt, **kwargs)

  def _sample(self, *arg, **kwargs):
    assert False, 'Should never trigger.'
