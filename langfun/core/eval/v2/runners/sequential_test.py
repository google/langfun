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
"""Tests for sequential runner."""
import os
import tempfile
from typing import Any
import unittest

from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2.runners import sequential  # pylint: disable=unused-import

import pyglove as pg


class SequentialRunnerTest(unittest.TestCase):

  def assert_same_list(self, actual: list[Any], expected: list[Any]):
    self.assertEqual(len(actual), len(expected))
    for i, (x, y) in enumerate(zip(actual, expected)):
      if x is not y:
        print(i, pg.diff(x, y))
      self.assertIs(x, y)

  def test_basic(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_sequential_runner')
    _ = exp.run(root_dir, runner='sequential', plugins=[plugin])

    self.assertIsNotNone(plugin.start_time)
    self.assertIsNotNone(plugin.complete_time)
    self.assertGreater(plugin.complete_time, plugin.start_time)

    self.assert_same_list(
        plugin.started_experiments,
        exp.nonleaf_nodes + exp.leaf_nodes
    )
    self.assert_same_list(
        plugin.completed_experiments,
        exp.leaf_nodes + list(reversed(exp.nonleaf_nodes))
    )
    self.assert_same_list(
        plugin.started_example_ids, list(range(1, 11)) * 6
    )
    self.assert_same_list(
        plugin.completed_example_ids, list(range(1, 11)) * 6
    )
    self.assert_same_list(plugin.skipped_experiments, [])

    for node in exp.nodes:
      self.assertTrue(node.progress.is_started)
      self.assertTrue(node.progress.is_completed)
      if node.is_leaf:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_completed, 10)
        self.assertEqual(node.progress.num_failed, 1)
      else:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_failed, 0)
        self.assertEqual(node.progress.num_processed, node.progress.num_total)

  def test_raise_if_has_error(self):
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_raise_if_has_error')
    exp = eval_test_helper.TestEvaluation()
    with self.assertRaisesRegex(ValueError, 'x should not be 5'):
      exp.run(
          root_dir, runner='sequential', plugins=[], raise_if_has_error=True
      )

  def test_example_ids(self):
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_example_ids')
    exp = eval_test_helper.test_experiment()
    plugin = eval_test_helper.TestPlugin()
    _ = exp.run(
        root_dir, runner='sequential', plugins=[plugin], example_ids=[5, 7, 9]
    )
    self.assertEqual(plugin.started_example_ids, [5, 7, 9] * 6)
    self.assertEqual(plugin.completed_example_ids, [5, 7, 9] * 6)

  def test_shuffle_inputs(self):
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_shuffle_inputs')
    exp = eval_test_helper.test_experiment()
    plugin = eval_test_helper.TestPlugin()
    run = exp.run(
        root_dir, runner='sequential', plugins=[plugin], shuffle_inputs=True
    )
    self.assertTrue(run.shuffle_inputs)

  def test_filter(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_filter')

    _ = exp.run(
        root_dir, runner='sequential', plugins=[plugin],
        filter=lambda e: e.lm.offset != 0
    )
    self.assert_same_list(
        plugin.started_experiments,
        exp.nonleaf_nodes + exp.leaf_nodes[2:]
    )
    self.assert_same_list(
        plugin.skipped_experiments, exp.leaf_nodes[:2]
    )
    self.assert_same_list(
        plugin.completed_experiments,
        exp.leaf_nodes[2:] + [exp.children[1], exp]
    )

  def test_use_cache(self):
    @pg.functor()
    def test_inputs(num_examples: int = 10):
      return [
          pg.Dict(
              x=i // 2, y=(i // 2) ** 2,
              groundtruth=(i // 2 + (i // 2) ** 2)
          ) for i in range(num_examples)
      ]

    exp = eval_test_helper.TestEvaluation(
        inputs=test_inputs(num_examples=pg.oneof([2, 4]))
    )
    # Global cache.
    root_dir = os.path.join(tempfile.mkdtemp(), 'global_cache')
    run = exp.run(
        root_dir, 'new', runner='sequential', use_cache='global', plugins=[]
    )
    self.assertTrue(pg.io.path_exists(run.output_path_for(exp, 'cache.json')))
    self.assertEqual(exp.usage_summary.cached.total.num_requests, 4)
    self.assertEqual(exp.usage_summary.uncached.total.num_requests, 2)

    # Per-dataset cache.
    root_dir = os.path.join(tempfile.mkdtemp(), 'per_dataset')
    run = exp.run(
        root_dir, 'new', runner='sequential',
        use_cache='per_dataset', plugins=[]
    )
    for leaf in exp.leaf_nodes:
      self.assertTrue(
          pg.io.path_exists(run.output_path_for(leaf, 'cache.json'))
      )
    self.assertEqual(exp.usage_summary.cached.total.num_requests, 3)
    self.assertEqual(exp.usage_summary.uncached.total.num_requests, 3)

    # No cache.
    root_dir = os.path.join(tempfile.mkdtemp(), 'no')
    run = exp.run(root_dir, runner='sequential', use_cache='no', plugins=[])
    self.assertFalse(pg.io.path_exists(run.output_path_for(exp, 'cache.json')))
    for leaf in exp.leaf_nodes:
      self.assertFalse(
          pg.io.path_exists(run.output_path_for(leaf, 'cache.json'))
      )
    self.assertEqual(exp.usage_summary.cached.total.num_requests, 0)
    self.assertEqual(exp.usage_summary.uncached.total.num_requests, 6)


if __name__ == '__main__':
  unittest.main()
