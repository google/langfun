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
import threading
import time
from typing import Any
import unittest

from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2 import runners as runners_lib  # pylint: disable=unused-import

import pyglove as pg


Runner = experiment_lib.Runner
Example = example_lib.Example
Experiment = experiment_lib.Experiment
Suite = experiment_lib.Suite
Plugin = experiment_lib.Plugin


class TestPlugin(Plugin):
  started_experiments: list[Experiment] = []
  completed_experiments: list[Experiment] = []
  skipped_experiments: list[Experiment] = []
  started_example_ids: list[int] = []
  completed_example_ids: list[int] = []
  skipped_example_ids: list[int] = []
  start_time: float | None = None
  complete_time: float | None = None

  def _on_bound(self):
    super()._on_bound()
    self._lock = threading.Lock()

  def on_run_start(self, runner: Runner, root: Experiment):
    del root
    with pg.notify_on_change(False), pg.allow_writable_accessors(True):
      self.start_time = time.time()

  def on_run_complete(self, runner: Runner, root: Experiment):
    del root
    with pg.notify_on_change(False), pg.allow_writable_accessors(True):
      self.complete_time = time.time()

  def on_experiment_start(self, runner: Runner, experiment: Experiment):
    del runner
    with pg.notify_on_change(False), self._lock:
      self.started_experiments.append(pg.Ref(experiment))

  def on_experiment_skipped(self, runner: Runner, experiment: Experiment):
    del runner
    with pg.notify_on_change(False), self._lock:
      self.skipped_experiments.append(pg.Ref(experiment))

  def on_experiment_complete(self, runner: Runner, experiment: Experiment):
    del runner
    with pg.notify_on_change(False), self._lock:
      self.completed_experiments.append(pg.Ref(experiment))

  def on_example_start(
      self, runner: Runner, experiment: Experiment, example: Example):
    del runner, experiment
    with pg.notify_on_change(False), self._lock:
      self.started_example_ids.append(example.id)

  def on_example_skipped(
      self, runner: Runner, experiment: Experiment, example: Example):
    del runner, experiment
    with pg.notify_on_change(False), self._lock:
      self.skipped_example_ids.append(example.id)

  def on_example_complete(
      self, runner: Runner, experiment: Experiment, example: Example):
    del runner, experiment
    with pg.notify_on_change(False), self._lock:
      self.completed_example_ids.append(example.id)


class RunnerTest(unittest.TestCase):

  def assert_same_list(self, actual: list[Any], expected: list[Any]):
    self.assertEqual(len(actual), len(expected))
    for i, (x, y) in enumerate(zip(actual, expected)):
      if x is not y:
        print(i, pg.diff(x, y))
      self.assertIs(x, y)

  def test_basic(self):
    plugin = TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.gettempdir(), 'test_sequential_runner')
    run = exp.run(root_dir, runner='sequential', plugins=[plugin])

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
    self.assert_same_list(plugin.skipped_example_ids, [])
    self.assertTrue(
        pg.io.path_exists(os.path.join(run.output_root, 'run.json'))
    )

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
    root_dir = os.path.join(tempfile.gettempdir(), 'test_raise_if_has_error')
    exp = eval_test_helper.TestEvaluation()
    with self.assertRaisesRegex(ValueError, 'x should not be 5'):
      exp.run(
          root_dir, runner='sequential', plugins=[], raise_if_has_error=True
      )

    with self.assertRaisesRegex(ValueError, 'x should not be 5'):
      exp.run(root_dir, runner='parallel', plugins=[], raise_if_has_error=True)

  def test_example_ids(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'test_example_ids')
    exp = eval_test_helper.test_experiment()
    plugin = TestPlugin()
    _ = exp.run(
        root_dir, runner='sequential', plugins=[plugin], example_ids=[5, 7, 9]
    )
    self.assertEqual(plugin.started_example_ids, [5, 7, 9] * 6)
    self.assertEqual(plugin.completed_example_ids, [5, 7, 9] * 6)

  def test_filter(self):
    plugin = TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.gettempdir(), 'test_filter')

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
    root_dir = os.path.join(tempfile.gettempdir(), 'global_cache')
    run = exp.run(
        root_dir, 'new', runner='sequential', use_cache='global', plugins=[]
    )
    self.assertTrue(pg.io.path_exists(run.output_path_for(exp, 'cache.json')))
    self.assertEqual(exp.usage_summary.cached.total.num_requests, 4)
    self.assertEqual(exp.usage_summary.uncached.total.num_requests, 2)

    # Per-dataset cache.
    root_dir = os.path.join(tempfile.gettempdir(), 'per_dataset')
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
    root_dir = os.path.join(tempfile.gettempdir(), 'no')
    run = exp.run(root_dir, runner='sequential', use_cache='no', plugins=[])
    self.assertFalse(pg.io.path_exists(run.output_path_for(exp, 'cache.json')))
    for leaf in exp.leaf_nodes:
      self.assertFalse(
          pg.io.path_exists(run.output_path_for(leaf, 'cache.json'))
      )
    self.assertEqual(exp.usage_summary.cached.total.num_requests, 0)
    self.assertEqual(exp.usage_summary.uncached.total.num_requests, 6)


class ParallelRunnerTest(RunnerTest):

  def test_parallel_runner(self):
    plugin = TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.gettempdir(), 'test_parallel_runner')
    run = exp.run(root_dir, runner='parallel', plugins=[plugin])

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
    self.assert_same_list(plugin.skipped_example_ids, [])
    self.assertTrue(
        pg.io.path_exists(os.path.join(run.output_root, 'run.json'))
    )

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

  def test_concurrent_startup_delay(self):
    plugin = TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(
        tempfile.gettempdir(), 'test_concurrent_startup_delay'
    )
    _ = exp.run(
        root_dir,
        runner='parallel',
        plugins=[plugin],
        concurrent_startup_delay=(0, 5),
    )


class DebugRunnerTest(RunnerTest):

  def test_debug_runner(self):
    plugin = TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.gettempdir(), 'test_debug_runner')
    run = exp.run(root_dir, runner='debug', plugins=[plugin])

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
        len(plugin.started_example_ids), 6 * 1
    )
    self.assertEqual(
        len(plugin.completed_example_ids), 6 * 1
    )
    self.assert_same_list(plugin.skipped_experiments, [])
    self.assert_same_list(plugin.skipped_example_ids, [])
    self.assertFalse(
        pg.io.path_exists(os.path.join(run.output_root, 'run.json'))
    )

    for node in exp.nodes:
      self.assertTrue(node.progress.is_started)
      self.assertTrue(node.progress.is_completed)
      if node.is_leaf:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_completed, 1)
        self.assertEqual(node.progress.num_failed, 0)
      else:
        self.assertEqual(node.progress.num_skipped, 0)
        self.assertEqual(node.progress.num_failed, 0)
        self.assertEqual(node.progress.num_processed, node.progress.num_total)


if __name__ == '__main__':
  unittest.main()
