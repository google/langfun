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
"""Tests for compositional models."""
import unittest

import langfun.core as lf
from langfun.core.llms import compositional
from langfun.core.llms import fake


class RandomChoiceTest(unittest.TestCase):

  def test_basic(self):
    lm = compositional.RandomChoice([
        fake.StaticResponse('hi'),
        fake.StaticSequence(['hello', 'world'])
    ])
    self.assertEqual(
        lm.model_id,
        'RandomChoice(StaticResponse, StaticSequence)'
    )
    self.assertEqual(
        lm.resource_id,
        'RandomChoice(StaticResponse, StaticSequence)'
    )
    self.assertEqual(
        [lm('a'), lm('b'), lm('c')],
        ['hello', 'world', 'hi']
    )
    lm = lm.clone()
    self.assertEqual(
        [
            x.samples[0].response for x in [
                lm.sample(['a'])[0],
                lm.sample(['b'])[0],
                lm.sample(['c'])[0],
            ]
        ],
        ['hello', 'world', 'hi']
    )
    self.assertEqual(
        lm.score('hello', ['world']),
        [lf.LMScoringResult(0.0)]
    )
    self.assertEqual(
        lm.tokenize('hello'),
        [('hello', 0)]
    )

  def test_sampling_options(self):
    lm = compositional.RandomChoice([
        fake.StaticResponse('hi'),
        fake.StaticSequence(['hello', 'world'])
    ], temperature=0.5)
    self.assertEqual(
        lm.candidates[0].sampling_options.temperature,
        0.5
    )


if __name__ == '__main__':
  unittest.main()
