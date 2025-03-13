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

from langfun.core.eval.v2 import checkpointing
from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2 import runners as runners_lib  # pylint: disable=unused-import
import pyglove as pg

Example = example_lib.Example


class SequenceWriterTest(unittest.TestCase):

  def test_basic(self):
    file = os.path.join(tempfile.gettempdir(), 'test.jsonl')
    writer = checkpointing.SequenceWriter(file)
    example = Example(id=1, input=pg.Dict(x=1), output=2)
    writer.add(example)
    del writer
    self.assertTrue(pg.io.path_exists(file))

  def test_error_handling(self):
    file = os.path.join(tempfile.gettempdir(), 'test_error_handling.jsonl')
    writer = checkpointing.SequenceWriter(file)
    writer.add(Example(id=1, input=pg.Dict(x=1), output=2))

    def f():
      raise ValueError('Intentional error')

    try:
      writer.add(f())
    except ValueError:
      del writer

    self.assertTrue(pg.io.path_exists(file))
    with pg.io.open_sequence(file, 'r') as f:
      self.assertEqual(len(list(iter(f))), 1)


class ExampleCollector(experiment_lib.Plugin):
  """Collects all examples."""

  def _on_bound(self):
    super()._on_bound()
    self._examples = {}

  @property
  def examples(self) -> dict[int, example_lib.Example]:
    return self._examples

  def on_example_complete(
      self, runner: runners_lib.Runner,
      experiment: experiment_lib.Experiment,
      example: example_lib.Example,
  ):
    assert experiment.is_leaf, None
    self._examples[example.id] = example


class CheckpointerTest(unittest.TestCase):

  def assert_found_in_log(self, experiment, message):
    found_error_log = False
    for log_entry in experiment._log_entries:
      if log_entry.message.startswith(message):
        found_error_log = True
        break
    self.assertTrue(found_error_log)


class PerExampleCheckpointerTest(CheckpointerTest):

  def test_checkpointing(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'per_example_checkpointer')
    experiment = eval_test_helper.test_experiment()
    checkpoint_filename = 'checkpoint.jsonl'
    checkpointer = checkpointing.PerExampleCheckpointer(checkpoint_filename)
    collector = ExampleCollector()
    run = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer, collector]
    )
    num_processed = {}
    for leaf in experiment.leaf_nodes:
      for i in range(leaf.num_examples):
        self.assertIn(i + 1, collector.examples)
        example = collector.examples[i + 1]
        ckpt = run.output_path_for(leaf, f'checkpoint_{example.id}.jsonl')
        if example.has_error:
          self.assertFalse(pg.io.path_exists(ckpt))
        else:
          self.assertTrue(pg.io.path_exists(ckpt))
          with pg.io.open_sequence(ckpt) as f:
            self.assertEqual(len(list(iter(f))), 1)
      if leaf.id not in num_processed:
        self.assertEqual(leaf.progress.num_skipped, 0)
        num_processed[leaf.id] = leaf.progress.num_processed

    # Run again, should skip existing.
    _ = experiment.run(
        root_dir, 'latest', runner='sequential', plugins=[checkpointer]
    )
    for leaf in experiment.leaf_nodes:
      self.assertEqual(leaf.progress.num_skipped, num_processed[leaf.id])

    # Test warm start without reprocess.
    root_dir = os.path.join(tempfile.gettempdir(), 'per_example_checkpointer2')
    experiment = eval_test_helper.test_experiment()
    _ = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer],
        warm_start_from=run.output_root
    )
    for leaf in experiment.leaf_nodes:
      self.assertEqual(leaf.progress.num_skipped, num_processed[leaf.id])

    # Test warm start with reprocess.
    root_dir = os.path.join(tempfile.gettempdir(), 'per_example_checkpointer3')
    experiment = eval_test_helper.test_experiment()
    _ = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer],
        warm_start_from=run.output_root,
        reprocess=True
    )
    for leaf in experiment.leaf_nodes:
      self.assertEqual(leaf.progress.num_skipped, 0)

    root_dir = os.path.join(tempfile.gettempdir(), 'per_example_checkpointer4')
    experiment = eval_test_helper.test_experiment()
    _ = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer],
        warm_start_from=run.output_root,
        reprocess=[1, 2, 3]
    )
    for leaf in experiment.leaf_nodes:
      self.assertEqual(leaf.progress.num_skipped, num_processed[leaf.id] - 3)

  def test_loading_corrupted_checkpoint(self):
    root_dir = os.path.join(
        tempfile.gettempdir(),
        'per_example_checkpointer_with_corrupted_checkpoint'
    )
    experiment = eval_test_helper.TestEvaluation()
    checkpoint_filename = 'checkpoint.jsonl'
    checkpointer = checkpointing.PerExampleCheckpointer(checkpoint_filename)
    collector = ExampleCollector()

    run = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer, collector]
    )
    num_processed = {}
    for i in range(experiment.num_examples):
      self.assertIn(i + 1, collector.examples)
      example = collector.examples[i + 1]
      ckpt = run.output_path_for(experiment, f'checkpoint_{example.id}.jsonl')
      if not example.has_error:
        self.assertTrue(pg.io.path_exists(ckpt))
        with pg.io.open_sequence(ckpt) as f:
          self.assertEqual(len(list(iter(f))), 1)

        # Simulate corrupting the first checkpoint.
        if i == 0:
          pg.io.writefile(ckpt, 'bad file')
        num_processed[example.id] = i + 1

    root_dir = os.path.join(
        tempfile.gettempdir(),
        'per_example_checkpointer_with_corrupted_checkpoint_warm_start'
    )
    experiment = eval_test_helper.TestEvaluation()
    _ = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer],
        warm_start_from=run.output_root,
    )
    for leaf in experiment.leaf_nodes:
      self.assertEqual(leaf.progress.num_skipped, len(num_processed) - 1)
    self.assert_found_in_log(experiment, 'Failed to load checkpoint')

  def test_checkpointing_error(self):
    root_dir = os.path.join(
        tempfile.gettempdir(),
        'per_example_checkpointer_with_checkpointing_error'
    )
    experiment = (eval_test_helper
                  .test_experiment_with_example_checkpointing_error())
    checkpointer = checkpointing.PerExampleCheckpointer('checkpoint.jsonl')
    _ = experiment.run(
        root_dir, 'new', runner='parallel', plugins=[checkpointer]
    )
    self.assert_found_in_log(experiment, 'Failed to checkpoint')


class BulkCheckpointerTest(CheckpointerTest):

  def test_checkpointing(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'test_bulk_checkpointer')
    experiment = eval_test_helper.test_experiment()
    checkpoint_filename = 'checkpoint.jsonl'
    checkpointer = checkpointing.BulkCheckpointer(checkpoint_filename)
    run = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer]
    )
    self.assertEqual(len(checkpointer._sequence_writer), 0)
    num_processed = {}
    for leaf in experiment.leaf_nodes:
      ckpt = run.output_path_for(leaf, checkpoint_filename)
      self.assertTrue(pg.io.path_exists(ckpt))
      with pg.io.open_sequence(ckpt) as f:
        self.assertEqual(
            len(list(iter(f))),
            leaf.progress.num_completed - leaf.progress.num_failed
        )
      if leaf.id not in num_processed:
        self.assertEqual(leaf.progress.num_skipped, 0)
        num_processed[leaf.id] = leaf.progress.num_processed

    # Run again, should skip existing.
    _ = experiment.run(
        root_dir, 'latest', runner='sequential', plugins=[checkpointer]
    )
    self.assertEqual(len(checkpointer._sequence_writer), 0)
    for leaf in experiment.leaf_nodes:
      self.assertEqual(leaf.progress.num_skipped, num_processed[leaf.id])

  def test_checkpointing_error(self):
    root_dir = os.path.join(
        tempfile.gettempdir(),
        'bulk_checkpointer_with_checkpointing_error'
    )
    experiment = (eval_test_helper
                  .test_experiment_with_example_checkpointing_error())
    checkpointer = checkpointing.BulkCheckpointer('checkpoint.jsonl')
    _ = experiment.run(
        root_dir, 'new', runner='parallel', plugins=[checkpointer]
    )
    self.assert_found_in_log(experiment, 'Failed to checkpoint')


if __name__ == '__main__':
  unittest.main()
