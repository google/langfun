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
import os
import tempfile
import unittest

from langfun.core.eval.v2 import evaluation as evaluation_lib
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib

from langfun.core.eval.v2 import test_helper

import pyglove as pg

Example = example_lib.Example
Evaluation = evaluation_lib.Evaluation
RunId = experiment_lib.RunId
Run = experiment_lib.Run


class EvaluationTest(unittest.TestCase):

  def test_hyper_evaluation(self):
    exp = test_helper.TestEvaluation(
        lm=test_helper.TestLLM(offset=pg.oneof(range(3)))
    )
    self.assertFalse(exp.is_leaf)
    self.assertTrue(
        pg.eq(
            exp.children,
            [
                test_helper.TestEvaluation(lm=test_helper.TestLLM(offset=0)),
                test_helper.TestEvaluation(lm=test_helper.TestLLM(offset=1)),
                test_helper.TestEvaluation(lm=test_helper.TestLLM(offset=2)),
            ]
        )
    )
    self.assertEqual(exp.children[0].num_examples, 10)
    self.assertEqual(
        [c.is_leaf for c in exp.children],
        [True] * len(exp.children)
    )
    self.assertEqual(
        [r.resource_ids() for r in exp.leaf_nodes],
        [set(['test_llm:0']), set(['test_llm:1']), set(['test_llm:2'])]
    )

  def test_input(self):
    exp = test_helper.TestEvaluation()
    self.assertEqual(exp.num_examples, 10)
    exp = test_helper.TestEvaluation(inputs=test_helper.test_inputs(None))
    self.assertEqual(exp.num_examples, 20)
    @pg.functor
    def my_inputs():
      yield pg.Dict(x=1, y=2)
      yield pg.Dict(x=3, y=4)
    exp = test_helper.TestEvaluation(inputs=my_inputs())
    self.assertEqual(exp.num_examples, 2)

  def test_evaluate(self):
    exp = test_helper.TestEvaluation()
    example = exp.evaluate(Example(id=3))
    self.assertIs(exp.state.get(3), example)
    self.assertTrue(example.newly_processed)
    self.assertEqual(example.input, pg.Dict(x=2, y=4, groundtruth=6))
    self.assertEqual(example.output, 6)
    self.assertIsNone(example.error)
    self.assertEqual(example.metadata, {})
    self.assertEqual(example.metric_metadata, dict(match=True))
    self.assertIsNotNone(example.usage_summary)
    self.assertGreater(example.usage_summary.total.total_tokens, 0)
    self.assertEqual(example.usage_summary.total.num_requests, 1)
    self.assertIsNotNone(example.execution_status)
    self.assertIsNotNone(example.start_time)
    self.assertIsNotNone(example.end_time)

    exp = test_helper.TestEvaluation(lm=test_helper.TestLLM(offset=1))
    example = exp.evaluate(3)
    self.assertTrue(example.newly_processed)
    self.assertEqual(example.input, pg.Dict(x=2, y=4, groundtruth=6))
    self.assertEqual(example.output, 7)
    self.assertIsNone(example.error)
    self.assertEqual(example.metadata, {})
    self.assertEqual(example.metric_metadata, dict(mismatch=True))

    with self.assertRaisesRegex(ValueError, 'x should not be 5'):
      _ = exp.evaluate(6, raise_if_has_error=True)
    example = exp.evaluate(6)
    self.assertTrue(example.newly_processed)
    self.assertEqual(example.input, pg.Dict(x=5, y=25, groundtruth=30))
    self.assertEqual(pg.MISSING_VALUE, example.output)
    self.assertEqual(example.error.tag, 'ValueError')
    self.assertEqual(example.metadata, {})
    self.assertEqual(example.metric_metadata, dict(error='ValueError'))

  def test_evaluate_with_state(self):
    eval_dir = os.path.join(tempfile.gettempdir(), 'test_eval')
    pg.io.mkdirs(eval_dir, exist_ok=True)
    state_file = os.path.join(eval_dir, 'state.jsonl')
    with pg.io.open_sequence(state_file, 'w') as f:
      exp = test_helper.TestEvaluation()
      example = exp.evaluate(3)
      self.assertTrue(example.newly_processed)
      self.assertEqual(example.input, pg.Dict(x=2, y=4, groundtruth=6))
      self.assertEqual(example.output, 6)
      self.assertEqual(len(exp._state.evaluated_examples), 1)
      f.add(pg.to_json_str(example))

    exp.reset()
    self.assertEqual(len(exp._state.evaluated_examples), 0)
    exp.load_state(state_file)
    self.assertEqual(len(exp._state.evaluated_examples), 1)
    example = exp.evaluate(3)
    self.assertFalse(example.newly_processed)
    self.assertEqual(example.input, pg.Dict(x=2, y=4, groundtruth=6))
    self.assertEqual(example.output, 6)
    self.assertGreater(example.usage_summary.total.total_tokens, 0)
    self.assertGreater(example.usage_summary.cached.total.total_tokens, 0)
    self.assertEqual(example.usage_summary.cached.total.num_requests, 1)
    self.assertEqual(example.usage_summary.uncached.total.total_tokens, 0)
    self.assertEqual(example.usage_summary.uncached.total.num_requests, 0)

  def test_html_view(self):
    exp = test_helper.TestEvaluation()
    self.assertIn(
        exp.id,
        exp.to_html(extra_flags=dict(card_view=True, current_run=None)).content
    )
    self.assertIn(
        exp.id,
        exp.to_html(
            extra_flags=dict(
                card_view=False,
                current_run=Run(
                    root_dir='/tmp/test_run',
                    id=RunId.from_id('20241031_1'),
                    experiment=pg.Ref(exp),
                )
            )
        ).content
    )


if __name__ == '__main__':
  unittest.main()
