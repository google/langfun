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


class PerExampleCheckpointerTest(unittest.TestCase):

  def test_checkpointing(self):
    pg.defaults.loggers.use_stdout()
    root_dir = os.path.join(tempfile.gettempdir(), 'per_example_checkpointer')
    experiment = eval_test_helper.test_experiment()
    checkpoint_filename = 'checkpoint.jsonl'
    checkpointer = checkpointing.PerExampleCheckpointer(checkpoint_filename)
    run = experiment.run(
        root_dir, 'new', runner='sequential', plugins=[checkpointer]
    )
    num_processed = {}
    for leaf in experiment.leaf_nodes:
      for i in range(leaf.num_examples):
        example = leaf.state.get(i + 1)
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


class BulkCheckpointerTest(unittest.TestCase):

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


if __name__ == '__main__':
  unittest.main()
