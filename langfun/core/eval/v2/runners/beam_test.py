# Copyright 2025 The Langfun Authors
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
"""Tests for beam runner."""

import os
import tempfile
from typing import Any
import unittest

from langfun.core.eval.v2 import checkpointing  # pylint: disable=unused-import
from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import reporting  # pylint: disable=unused-import
from langfun.core.eval.v2.runners import beam  # pylint: disable=unused-import
import pyglove as pg


@unittest.skip(
    'These tests are flaky due to writing ckpt files with standard IO.'
    'We will move to `beam.io` and re-enable these tests later.'
)
class BeamRunnerTest(unittest.TestCase):

  def assert_same_list(self, actual: list[Any], expected: list[Any]):
    self.assertEqual(len(actual), len(expected))
    for i, (x, y) in enumerate(zip(actual, expected)):
      if x is not y:
        print(i, pg.diff(x, y))
      self.assertIs(x, y)

  def setUp(self):
    super().setUp()
    self.test_dir = os.path.join(tempfile.mkdtemp(), 'test_dir')

  def test_basic(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(self.test_dir, 'test_beam_runner')
    run = exp.run(
        root_dir,
        runner='beam',
        plugins=[
            plugin,
            reporting.ExampleHtmlGenerator(),
            checkpointing.PerExampleCheckpointer(
                checkpoint_filename='checkpoint.jsonl'
            ),
        ],
        concurrent_startup_delay=(1, 2),
        use_cache='no',
        ckpt_format='jsonl',
    )

    self.assertIsNotNone(plugin.start_time)
    self.assertIsNotNone(plugin.complete_time)
    self.assertGreater(plugin.complete_time, plugin.start_time)

    self.assertEqual(len(plugin.started_experiments), len(exp.nodes))
    self.assertEqual(len(plugin.completed_experiments), len(exp.nodes))
    self.assertEqual(len(plugin.started_example_ids), 6 * 10)
    self.assertEqual(len(plugin.completed_example_ids), 6 * 10)
    self.assert_same_list(plugin.skipped_experiments, [])

    for node in exp.leaf_nodes:
      for i in range(node.num_examples):
        self.assertTrue(
            pg.io.path_exists(
                run.output_path_for(node, f'{i + 1}.html')
            )
        )

    for node in exp.nodes:
      if node.is_leaf:
        self.assertTrue(node.progress.is_started)
        self.assertTrue(node.progress.is_completed)
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_completed, 10)
        self.assertEqual(node.progress.num_failed, 1)
      else:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_failed, 0)
        self.assertEqual(node.progress.num_processed, node.progress.num_total)

    # Test warm start.
    root_dir2 = os.path.join(self.test_dir, 'test_warm_start')
    exp = eval_test_helper.test_experiment()
    plugin = eval_test_helper.TestPlugin()
    run2 = exp.run(
        root_dir2,
        warm_start_from=run.output_root,
        runner='beam',
        plugins=[plugin],
        use_cache='no',
        ckpt_format='jsonl',
    )
    for node in run2.experiment.nodes:
      if node.is_leaf:
        self.assertTrue(node.progress.is_started)
        self.assertTrue(node.progress.is_completed)
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_completed, 10)
        self.assertEqual(node.progress.num_failed, 1)
      else:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_failed, 0)
        self.assertEqual(node.progress.num_processed, node.progress.num_total)

  def test_shuffle_inputs(self):
    root_dir = os.path.join(self.test_dir, 'test_shuffle_inputs')
    exp = eval_test_helper.test_experiment()
    plugin = eval_test_helper.TestPlugin()
    run = exp.run(
        root_dir,
        runner='beam',
        plugins=[plugin],
        shuffle_inputs=True,
        use_cache='no',
        ckpt_format='jsonl',
    )
    self.assertTrue(run.shuffle_inputs)

  def test_beam_runner_does_not_support_cache(self):
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(self.test_dir, 'test_beam_runner_cache')
    with self.assertRaisesRegex(ValueError, 'Cache is not supported'):
      exp.run(
          root_dir,
          runner='beam',
          use_cache='global',
      )

  def test_no_beam(self):
    orig_beam = beam.beam
    beam.beam = None
    with self.assertRaisesRegex(ValueError, 'Beam is not installed'):
      exp = eval_test_helper.TestEvaluation()
      root_dir = os.path.join(self.test_dir, 'test_no_beam')
      exp.run(root_dir, runner='beam')
    beam.beam = orig_beam


if __name__ == '__main__':
  unittest.main()
