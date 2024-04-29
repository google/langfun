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
"""Tests for groundtruth matching."""

import os
import tempfile
from typing import Any
import unittest

import langfun.core as lf
from langfun.core.eval import base
from langfun.core.eval import matching
from langfun.core.llms import fake
import pyglove as pg


# We put class definitions outside the functors just to make it easier
# to refer to them in test.


class Solution(pg.Object):
  final_answer: int


class SolutionForCompletion(pg.Object):
  question: str
  final_answer: int


@pg.functor
def answer_schema():
  return Solution


@pg.functor
def complete_schema():
  return SolutionForCompletion


class MyTask(matching.Matching):

  def groundtruth(self, example: Any) -> Any:
    return example.groundtruth

  def answer(self, output: Any, example: Any) -> Any:
    return output.final_answer


def eval_set(
    eval_id: str,
    method: str,
    schema_fn,
    lm: lf.LanguageModel,
    use_cache: bool = True,
):
  """Creates an evaluation object for testing."""
  return MyTask(
      root_dir=os.path.join(tempfile.gettempdir(), eval_id),
      inputs=base.as_inputs([
          pg.Dict(question='Compute 1 + 1', groundtruth=2),
          pg.Dict(question='Compute 1 + 2', groundtruth=3),
          pg.Dict(question='Compute 1 + 3', groundtruth=4),
          pg.Dict(question='Compute 1 + 1', groundtruth=2),
      ]),
      method=method,
      prompt='{{example.question}}',
      schema_fn=schema_fn,
      lm=lm,
      use_cache=use_cache,
      max_workers=1,
  )


class MatchingTest(unittest.TestCase):
  """Matching test."""

  def setUp(self):
    super().setUp()
    pg.symbolic.set_save_handler(pg.symbolic.default_save_handler)
    pg.symbolic.set_load_handler(pg.symbolic.default_load_handler)

  def test_run(self):
    lm = fake.StaticSequence([
        'Solution(final_answer=2)',
        '3',
        'Solution(final_answer=3)',
    ])

    s = eval_set('match_run_test', 'query', schema_fn=answer_schema(), lm=lm)
    s.run()
    self.assertEqual(
        s.result,
        dict(
            experiment_setup=dict(
                id='MyTask@739a174b',
                dir=s.dir,
                model='StaticSequence',
                prompt_template='{{example.question}}',
                method='query',
                schema_fn='answer_schema()',
            ),
            cache_stats=dict(
                use_cache=True,
                num_queries=4,
                num_hits=1,
                num_updates=3,
            ),
            metrics=dict(
                total=4,
                failures=1,
                failure_rate=0.25,
                oop_failures=1,
                oop_failure_rate=0.25,
                non_oop_failures=0,
                non_oop_failure_rate=0.0,
                failure_breakdown={
                    'MappingError.SchemaError.TypeError': 1
                },
                num_matches=2,
                match_rate=0.5,
                num_mismatches=1,
                mismatch_rate=0.25,
            ),
            usage=s.result.usage,
        ),
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.EXPERIMENT_JSON
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, matching.Matching.RESULT_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, matching.Matching.CACHE_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, matching.Matching.MATCHES_JSON)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.MISMATCHES_JSON
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.OOP_FAILURES_JSON
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.NON_OOP_FAILURES_JSON
            )
        )
    )
    self.assertTrue(
        os.path.exists(os.path.join(s.root_dir, matching.Matching.SUMMARY_HTML))
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, matching.Matching.INDEX_HTML)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.OOP_FAILURES_HTML
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.NON_OOP_FAILURES_HTML
            )
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(s.dir, matching.Matching.MATCHES_HTML)
        )
    )
    self.assertTrue(
        os.path.exists(
            os.path.join(
                s.dir, matching.Matching.MISMATCHES_HTML
            )
        )
    )


if __name__ == '__main__':
  unittest.main()
