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
import time
import unittest

import langfun.core as lf
from langfun.core.eval.v2 import checkpointing
from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2.runners import ckpt_monitor
from langfun.core.eval.v2.runners import sequential  # pylint: disable=unused-import
import pyglove as pg


class CheckpointMonitorTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()

  def test_aggregate(self):
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(self.test_dir, 'test_aggregate')
    ckpt_start_time = time.time()
    run = exp.run(
        root_dir,
        runner='sequential',
        progress_tracker=None,
        plugins=[
            checkpointing.PerExampleCheckpointer(
                checkpoint_filename='checkpoint.jsonl'
            )
        ],
        use_cache='no',
    )
    # Try to corrupt one of the checkpoint files.
    pg.io.writefile(
        run.output_path_for(exp.leaf_nodes[0], 'checkpoint_1.jsonl'),
        'bad ckpt'
    )
    plugin = eval_test_helper.TestPlugin()
    monitor = ckpt_monitor.CheckpointMonitor(
        run,
        plugins=[plugin],
        checkpoint_pattern='checkpoint_*.jsonl',
        monitor_inprogress_files=True,
        ckpt_start_time=ckpt_start_time,
    )
    monitor.run()

    # Assert that the in-progress files are created and not removed.
    for entry in monitor._aggregation_entries:
      self.assertEqual(len(entry.example_ids_inprogress), 10)

    # 6 leaf nodes + 1 suite + 1 hyper.
    self.assertEqual(len(plugin.started_experiments), 6 + 2)
    self.assertEqual(len(plugin.completed_experiments), 6 + 2)
    self.assertEqual(len(plugin.started_example_ids), 10 * 6)
    self.assertEqual(len(plugin.completed_example_ids), 10 * 6)
    for e in exp.leaf_nodes:
      self.assertEqual(e.progress.num_completed, 10)

  def test_ignore_old_ckpt_files_with_non_oop_errors(self):
    exp = eval_test_helper.test_evaluation()
    root_dir = os.path.join(self.test_dir, 'test_ignore_old_ckpt_files')
    run = exp.run(
        root_dir,
        runner='sequential',
        progress_tracker=None,
        plugins=[
            checkpointing.PerExampleCheckpointer(
                checkpoint_filename='checkpoint.jsonl'
            )
        ],
        use_cache='no',
    )
    monitor = ckpt_monitor.CheckpointMonitor(
        run,
        plugins=[],
        checkpoint_pattern='checkpoint_*.jsonl',
        monitor_inprogress_files=True
    )
    monitor.start()
    time.sleep(2)
    # Example 6 is a non-oop error, we simulate a re-evaluation.
    ex = example_lib.Example(
        id=6, output=1, metric_metadata={'match': {'is_correct': True}},
        start_time=time.time() - 2, end_time=time.time(),
        usage_summary=lf.UsageSummary(),
        execution_status={
            'evaluate': pg.utils.TimeIt.Status(name='evaluate', elapse=1)
        }
    )
    with pg.io.open_sequence(
        run.output_path_for(exp, 'checkpoint_6.jsonl'),
        mode='w'
    ) as f:
      f.add(pg.to_json_str(ex))
    print(time.time(), pg.io.listdir(run.output_dir(exp)))
    monitor.join()
    self.assertEqual(exp.progress.num_processed, 10)
    self.assertEqual(exp.progress.num_completed, 10)
    self.assertEqual(exp.progress.num_failed, 0)

  def test_aggregate_with_filter(self):
    ckpt_start_time = time.time()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(self.test_dir, 'test_aggregate_with_filter')

    node_to_skip = exp.leaf_nodes[2]
    # Run experiment to generate checkpoint files for all examples.
    run = exp.run(
        root_dir,
        runner='sequential',
        filter=lambda e: e.id != node_to_skip.id,
        progress_tracker=None,
        plugins=[
            checkpointing.PerExampleCheckpointer(
                checkpoint_filename='checkpoint.jsonl'
            )
        ],
        use_cache='no',
    )
    plugin = eval_test_helper.TestPlugin()
    monitor = ckpt_monitor.CheckpointMonitor(
        run,
        plugins=[plugin],
        checkpoint_pattern='checkpoint_*.jsonl',
        ckpt_start_time=ckpt_start_time,
    )
    monitor.run()

    # Assert that on_experiment_skipped was called for the filtered node.
    self.assertEqual(len(plugin.skipped_experiments), 1)
    self.assertEqual(plugin.skipped_experiments[0].id, node_to_skip.id)

    # Assert that the skipped node was not started.
    started_ids = [e.id for e in plugin.started_experiments]
    self.assertNotIn(node_to_skip.id, started_ids)

  def test_plugin_raise(self):

    class TestPlugin(eval_test_helper.TestPlugin):
      simulate_raise_on_example_complete: bool = False
      simulate_raise_on_experiment_complete: bool = False

      def on_example_complete(
          self,
          runner: experiment_lib.Runner,
          experiment: experiment_lib.Experiment,
          example: example_lib.Example
      ):
        if self.simulate_raise_on_example_complete:
          raise ValueError('example complete error')

      def on_experiment_complete(
          self,
          runner: experiment_lib.Runner,
          experiment: experiment_lib.Experiment
      ):
        if self.simulate_raise_on_experiment_complete:
          raise ValueError('experiment complete error')

    ckpt_start_time = time.time()
    exp = eval_test_helper.test_evaluation()
    root_dir = os.path.join(self.test_dir, 'test_plugin_raise')

    # Run experiment to generate checkpoint files for all examples.
    run = exp.run(
        root_dir,
        runner='sequential',
        progress_tracker=None,
        plugins=[
            checkpointing.PerExampleCheckpointer(
                checkpoint_filename='checkpoint.jsonl'
            )
        ],
        use_cache='no',
    )

    with self.assertRaisesRegex(ValueError, 'example complete error'):
      ckpt_monitor.CheckpointMonitor(
          run,
          plugins=[TestPlugin(simulate_raise_on_example_complete=True)],
          checkpoint_pattern='checkpoint_*.jsonl',
          ckpt_start_time=ckpt_start_time,
      ).run()

    with self.assertRaisesRegex(ValueError, 'experiment complete error'):
      ckpt_monitor.CheckpointMonitor(
          run,
          plugins=[TestPlugin(simulate_raise_on_experiment_complete=True)],
          checkpoint_pattern='checkpoint_*.jsonl',
          ckpt_start_time=ckpt_start_time,
      ).run()


if __name__ == '__main__':
  unittest.main()
