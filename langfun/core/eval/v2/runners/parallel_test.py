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
"""Tests for parallel runner."""
import os
import tempfile
import threading
from typing import Any
import unittest

from langfun.core.eval.v2 import checkpointing
from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import reporting
from langfun.core.eval.v2.runners import parallel  # pylint: disable=unused-import

import pyglove as pg


class ParallelRunnerTest(unittest.TestCase):

  def assert_same_list(self, actual: list[Any], expected: list[Any]):
    self.assertEqual(len(actual), len(expected))
    for i, (x, y) in enumerate(zip(actual, expected)):
      if x is not y:
        print(i, pg.diff(x, y))
      self.assertIs(x, y)

  def test_parallel_runner(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_parallel_runner')
    _ = exp.run(root_dir, runner='parallel', plugins=[plugin])

    self.assertIsNotNone(plugin.start_time)
    self.assertIsNotNone(plugin.complete_time)
    self.assertGreater(plugin.complete_time, plugin.start_time)

    self.assertEqual(
        len(plugin.started_experiments), len(exp.nodes)
    )
    self.assertEqual(
        len(plugin.completed_experiments), len(exp.nodes)
    )
    self.assertEqual(
        len(plugin.started_example_ids), 6 * 10
    )
    self.assertEqual(
        len(plugin.completed_example_ids), 6 * 10
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
      exp.run(root_dir, runner='parallel', plugins=[], raise_if_has_error=True)

  def test_concurrent_startup_delay(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(
        tempfile.mkdtemp(), 'test_concurrent_startup_delay'
    )
    _ = exp.run(
        root_dir,
        runner='parallel',
        plugins=[plugin],
        concurrent_startup_delay=(0, 5),
    )


class MultiProcessParallelRunnerTest(unittest.TestCase):

  def assert_same_list(self, actual: list[Any], expected: list[Any]):
    self.assertEqual(len(actual), len(expected))
    for i, (x, y) in enumerate(zip(actual, expected)):
      if x is not y:
        print(i, pg.diff(x, y))
      self.assertIs(x, y)

  def test_basic(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_parallel_runner')
    runs = [None, None]

    def run_slice(slice_id: int):
      runs[slice_id] = exp.run(
          root_dir,
          runner='sliced-parallel',
          plugins=[
              pg.Ref(plugin),
              # Ignored as PerExampleCheckpointer is used.
              checkpointing.BulkCheckpointer(),
              reporting.ExampleHtmlGenerator(),
          ],
          use_cache='no',
          slice_id=slice_id,
          num_slices=2,
          ckpt_format='jsonl',
      )

    # We simulate two slices running in parallel.
    threads = [threading.Thread(target=run_slice, args=(i,)) for i in range(2)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    self.assertIsNotNone(plugin.start_time)
    self.assertIsNotNone(plugin.complete_time)
    self.assertGreater(plugin.complete_time, plugin.start_time)

    self.assertEqual(
        len(plugin.started_experiments), len(exp.nodes)
    )
    self.assertEqual(
        len(plugin.completed_experiments), len(exp.nodes)
    )
    self.assertEqual(
        len(plugin.started_example_ids), 6 * 10
    )
    self.assertEqual(
        len(plugin.completed_example_ids), 6 * 10
    )
    self.assert_same_list(plugin.skipped_experiments, [])

    for node in exp.nodes:
      self.assertTrue(node.progress.is_started)
      self.assertTrue(node.progress.is_completed)
      if node.is_leaf:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_completed, 10)
        self.assertEqual(node.progress.num_failed, 1)
        for example_id in runs[0].examples_to_evaluate(node):
          self.assertTrue(
              pg.io.path_exists(
                  runs[0].output_path_for(node, f'{example_id}.html')
              )
          )
      else:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_failed, 0)
        self.assertEqual(node.progress.num_processed, node.progress.num_total)

  def test_parallel_mp_runner_does_not_support_cache(self):
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_parallel_mp_runner_cache')
    with self.assertRaisesRegex(ValueError, 'Cache is not supported'):
      exp.run(
          root_dir,
          runner='sliced-parallel',
          use_cache='global',
          slice_id=0,
          num_slices=1,
      )


if __name__ == '__main__':
  unittest.main()
