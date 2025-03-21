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
from langfun.core.eval.v2 import reporting
from langfun.core.eval.v2 import runners as runners_lib  # pylint: disable=unused-import
import pyglove as pg


class ReportingTest(unittest.TestCase):

  def test_reporting(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'test_reporting')
    experiment = eval_test_helper.test_experiment()
    checkpointer = checkpointing.BulkCheckpointer('checkpoint.jsonl')
    reporter = reporting.HtmlReporter()
    run = experiment.run(root_dir, 'new', plugins=[checkpointer, reporter])
    self.assertTrue(
        pg.io.path_exists(os.path.join(run.output_root, 'summary.html'))
    )
    for leaf in experiment.leaf_nodes:
      self.assertTrue(
          pg.io.path_exists(run.output_path_for(leaf, 'index.html'))
      )
      for i in range(leaf.num_examples):
        self.assertTrue(
            pg.io.path_exists(run.output_path_for(leaf, f'{i + 1}.html'))
        )
      found_generation_log = False
      for log_entry in leaf._log_entries:
        if 'generated in' in log_entry.message:
          found_generation_log = True
          break
      self.assertTrue(found_generation_log)

    # Test warm start.
    root_dir = os.path.join(tempfile.gettempdir(), 'test_reporting2')
    experiment = eval_test_helper.test_experiment()
    run = experiment.run(
        root_dir, 'new', plugins=[checkpointer, reporter],
        warm_start_from=run.output_root
    )
    self.assertTrue(
        pg.io.path_exists(os.path.join(run.output_root, 'summary.html'))
    )
    for leaf in experiment.leaf_nodes:
      self.assertTrue(
          pg.io.path_exists(run.output_path_for(leaf, 'index.html'))
      )
      for i in range(leaf.num_examples):
        self.assertTrue(
            pg.io.path_exists(run.output_path_for(leaf, f'{i + 1}.html'))
        )
      found_copy_log = False
      for log_entry in leaf._log_entries:
        if 'copied in' in log_entry.message:
          found_copy_log = True
          break
      self.assertTrue(found_copy_log)

  def test_index_html_generation_error(self):
    root_dir = os.path.join(
        tempfile.gettempdir(),
        'test_reporting_with_index_html_generation_error'
    )
    experiment = (eval_test_helper
                  .test_experiment_with_index_html_generation_error())
    reporter = reporting.HtmlReporter()
    run = experiment.run(root_dir, 'new', plugins=[reporter])
    self.assertFalse(
        pg.io.path_exists(os.path.join(run.output_root, 'summary.html'))
    )
    for leaf in experiment.leaf_nodes:
      self.assertFalse(
          pg.io.path_exists(run.output_path_for(leaf, 'index.html'))
      )
    found_error_log = False
    for log_entry in experiment._log_entries:
      if log_entry.message.startswith('Failed to generate'):
        found_error_log = True
        break
    self.assertTrue(found_error_log)

  def test_example_html_generation_error(self):
    root_dir = os.path.join(
        tempfile.gettempdir(),
        'test_reporting_with_example_html_generation_error'
    )
    experiment = (eval_test_helper
                  .test_experiment_with_example_html_generation_error())
    checkpointer = checkpointing.BulkCheckpointer('checkpoint.jsonl')
    reporter = reporting.HtmlReporter()
    run = experiment.run(root_dir, 'new', plugins=[checkpointer, reporter])
    self.assertTrue(
        pg.io.path_exists(os.path.join(run.output_root, 'summary.html'))
    )
    for leaf in experiment.leaf_nodes:
      self.assertTrue(
          pg.io.path_exists(run.output_path_for(leaf, 'index.html'))
      )
      for i in range(leaf.num_examples):
        self.assertFalse(
            pg.io.path_exists(run.output_path_for(leaf, f'{i + 1}.html'))
        )
    found_error_log = False
    for log_entry in experiment._log_entries:
      if log_entry.message.startswith('Failed to generate'):
        found_error_log = True
        break
    self.assertTrue(found_error_log)

    # Test warm start.
    root_dir = os.path.join(
        tempfile.gettempdir(),
        'test_reporting_with_example_html_generation_error2'
    )
    experiment = (eval_test_helper
                  .test_experiment_with_example_html_generation_error())
    run = experiment.run(
        root_dir, 'new', plugins=[checkpointer, reporter],
        warm_start_from=run.output_root
    )
    self.assertTrue(
        pg.io.path_exists(os.path.join(run.output_root, 'summary.html'))
    )
    for leaf in experiment.leaf_nodes:
      self.assertTrue(
          pg.io.path_exists(run.output_path_for(leaf, 'index.html'))
      )
      for i in range(leaf.num_examples):
        self.assertFalse(
            pg.io.path_exists(run.output_path_for(leaf, f'{i + 1}.html'))
        )
    found_error_log = False
    for log_entry in experiment._log_entries:
      if log_entry.message.startswith('Skip copying'):
        found_error_log = True
        break
    self.assertTrue(found_error_log)


if __name__ == '__main__':
  unittest.main()
