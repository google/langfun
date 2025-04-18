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
"""Tests for scoring evaluation."""

import os
import tempfile
import unittest

import langfun.core as lf
from langfun.core.eval import scoring
from langfun.core.llms import fake
import pyglove as pg


@pg.functor()
def float_list():
  return list[float]


@pg.functor()
def constrained_by_upperbound(upper_bound: int):
  return [
      (
          'Give an example of three numbers (x, y, z) '
          f'such that x + y + z <={upper_bound}',
      ),
      (
          'Give an example of two numbers (x, y) '
          f'such that x + y <={upper_bound}',
      ),
  ]


class ConstraintFollowing(scoring.Scoring):
  inputs = constrained_by_upperbound(1)
  prompt = '{{example}}'
  method = 'query'
  schema_fn = float_list()
  use_cache = True
  max_workers = 1

  def score(self, example, output):
    return 1.0 if sum(output) <= self.inputs.upper_bound else 0.0


def eval_set(lm: lf.LanguageModel):
  """Creates an evaluation object for testing."""
  tmp_dir = tempfile.gettempdir()
  return ConstraintFollowing(root_dir=tmp_dir, lm=lm)


class ScoringTest(unittest.TestCase):
  """Scoring test."""

  def setUp(self):
    super().setUp()
    pg.symbolic.set_save_handler(pg.symbolic.default_save_handler)
    pg.symbolic.set_load_handler(pg.symbolic.default_load_handler)

  def test_run(self):
    lm = fake.StaticSequence([
        '[0.5, 0.2, 0.3]',
        '[0.6, 0.7]',
    ])

    s = eval_set(lm=lm)
    self.assertEqual(s.avg_score, 0.0)
    s.run()
    result_copy = s.result.copy()
    del result_copy['experiment_setup']['id']
    self.assertEqual(
        result_copy,
        dict(
            experiment_setup=dict(
                dir=s.dir,
                model='StaticSequence',
                prompt_template='{{example}}',
                method='query',
                schema_fn='float_list()',
            ),
            cache_stats=dict(
                use_cache=True, num_queries=2, num_hits=0, num_updates=2
            ),
            metrics=dict(
                total=2,
                failures=0,
                failure_rate=0.0,
                oop_failures=0,
                oop_failure_rate=0.0,
                non_oop_failures=0,
                non_oop_failure_rate=0.0,
                failure_breakdown={},
                num_scored=2,
                score_rate=1.0,
                avg_score=0.5,
            ),
            usage=s.result.usage,
        ),
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, scoring.Scoring.EXPERIMENT_JSON
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, scoring.Scoring.CACHE_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, scoring.Scoring.RESULT_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, scoring.Scoring.OOP_FAILURES_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, scoring.Scoring.NON_OOP_FAILURES_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, scoring.Scoring.SCORED_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(os.path.join(s.root_dir, scoring.Scoring.SUMMARY_HTML))
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, scoring.Scoring.INDEX_HTML)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, scoring.Scoring.OOP_FAILURES_HTML
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, scoring.Scoring.NON_OOP_FAILURES_HTML
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, scoring.Scoring.SCORED_HTML
            )
        )
    )


if __name__ == '__main__':
  unittest.main()
