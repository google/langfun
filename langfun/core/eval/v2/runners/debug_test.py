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
"""Tests for debug runner."""
import os
import tempfile
from typing import Any
import unittest

from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2.runners import debug  # pylint: disable=unused-import

import pyglove as pg


class DebugRunnerTest(unittest.TestCase):

  def assert_same_list(self, actual: list[Any], expected: list[Any]):
    self.assertEqual(len(actual), len(expected))
    for i, (x, y) in enumerate(zip(actual, expected)):
      if x is not y:
        print(i, pg.diff(x, y))
      self.assertIs(x, y)

  def test_debug_runner(self):
    plugin = eval_test_helper.TestPlugin()
    exp = eval_test_helper.test_experiment()
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_debug_runner')
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
