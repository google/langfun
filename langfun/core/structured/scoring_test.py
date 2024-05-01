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

import unittest
import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.structured import scoring


class ScoringTest(unittest.TestCase):

  def test_bad_call(self):
    with self.assertRaisesRegex(ValueError, '`completions` must not be empty'):
      scoring.score('hi', [])

    with self.assertRaisesRegex(
        ValueError, '`schema` cannot be inferred from completions'
    ):
      scoring.score('hi', [1, 'b'])

    with self.assertRaisesRegex(ValueError, '`lm` must be specified'):
      scoring.score('hi', [1, 2])

  def test_score(self):
    self.assertEqual(scoring.score('hi', [1, 2], lm=fake.Echo()), [0.0, -1.0])

  def test_score_returning_scoring_results(self):
    self.assertEqual(scoring.score(
        'hi', [1, 2], lm=fake.Echo(), return_scoring_results=True),
                     [lf.LMScoringResult(score=0.0, gradients=None),
                      lf.LMScoringResult(score=-1.0, gradients=None)])

  def test_scope_with_lm_from_the_context(self):
    with lf.context(lm=fake.Echo()):
      self.assertEqual(scoring.score('hi', [1, 2]), [0.0, -1.0])


if __name__ == '__main__':
  unittest.main()
