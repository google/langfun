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
"""Tests for action evaluation."""

import os
import tempfile
import unittest

from langfun.core import eval as lf_eval
from langfun.core import llms as lf_llms
from langfun.core.agentic import action as action_lib
from langfun.core.agentic import action_eval
import pyglove as pg


_TIMING_KEYS = ('start_time', 'end_time', 'wall_clock_s')


def _strip_timing(result):
  """Removes non-deterministic per-leaf timing fields for stable eq checks.

  `Evaluation.finalize()` now emits `start_time`/`end_time`/`wall_clock_s` into
  every leaf result. Those wall-clock values are non-deterministic, so tests
  that assert the full result dict must drop them before comparing. Their
  presence and sanity are covered separately by `test_run_timing`.

  Args:
    result: The leaf result dict (or None) to strip timing fields from.

  Returns:
    A shallow copy of `result` without the timing fields, or None if `result`
    is None.
  """
  if result is None:
    return None
  stripped = dict(result)
  for key in _TIMING_KEYS:
    stripped.pop(key, None)
  return stripped


class Foo(action_lib.Action):
  x: int

  def call(self, session, **kwargs):
    del session, kwargs
    return self.x


@pg.functor()
def foo_inputs():
  return [
      pg.Dict(action=Foo(1), groundtruth=1),
      pg.Dict(action=Foo(2), groundtruth=1),
  ]


class ActionEvalTest(unittest.TestCase):

  def test_basics(self):

    class FooEval(action_eval.ActionEval):
      inputs = foo_inputs()
      metrics = [lf_eval.v2.metrics.Match()]
      action_args = dict(
          lm=lf_llms.Echo()
      )

    s = FooEval()
    root_dir = os.path.join(tempfile.gettempdir(), 'foo_eval')
    s.run(root_dir, plugins=[])
    self.assertEqual(s.metrics[0].matches, 0.5)
    self.assertEqual(s.metrics[0].mismatches, 0.5)


class ActionEvalV1Test(unittest.TestCase):

  def test_basics(self):

    class FooEval(action_eval.ActionEvalV1):
      lm = lf_llms.Echo()
      inputs = foo_inputs()

    s = FooEval()
    result = s.run(summary=False)
    self.assertEqual(
        _strip_timing(result),
        dict(
            experiment_setup=dict(
                id=s.id,
                dir=None,
                model='Echo',
                prompt_template='<unused>',
                method='query',
                schema_fn='_dummy_schema()'
            ),
            cache_stats=dict(
                use_cache=True,
                num_queries=0,
                num_hits=0,
                num_updates=0,
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
                num_matches=0,
                match_rate=0.0,
                num_mismatches=2,
                mismatch_rate=1.0
            ),
            usage=None
        )
    )


if __name__ == '__main__':
  unittest.main()
