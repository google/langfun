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
"""Tests for langfun.core.templates.selfplay."""


import unittest
import langfun.core as lf
from langfun.core.templates import selfplay


class NumberGuesser(lf.language_model.LanguageModel):
  guesses: list[int]

  def _on_bound(self):
    super()._on_bound()
    self._index = 0

  def _sample(self, prompts):
    results = []
    for _ in range(len(prompts)):
      guess = self.guesses[self._index]
      self._index = (self._index + 1) % len(self.guesses)
      results.append(lf.LMSamplingResult([lf.LMSample(str(guess))]))
    return results


class NumberGuess(selfplay.SelfPlay):
  target_num: int

  def next_turn(self, last_response):
    if last_response is None:
      return 'Let us get started. Tell me your first guess.'
    guess = int(last_response.text)
    if guess > self.target_num:
      return 'Too large'
    elif guess < self.target_num:
      return 'Too small'
    else:
      return None


class SelfPlayTest(unittest.TestCase):

  def test_play(self):
    g = NumberGuess(target_num=10)

    with lf.context(lm=NumberGuesser(guesses=[50, 20, 5, 10])):
      self.assertEqual(
          g(), lf.AIMessage('10', score=0.0, logprobs=None, usage=None)
      )

    self.assertEqual(g.num_turns, 4)

  def test_play_with_num_turns(self):
    g = NumberGuess(target_num=10, max_turns=10)

    with lf.context(lm=NumberGuesser(guesses=[50, 20, 5, 2, 5, 4])):
      self.assertEqual(
          g(), lf.AIMessage('2', score=0.0, logprobs=None, usage=None)
      )

    self.assertEqual(g.num_turns, 10)


if __name__ == '__main__':
  unittest.main()
