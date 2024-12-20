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

from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import reporting
from langfun.core.eval.v2 import runners as runners_lib  # pylint: disable=unused-import
import pyglove as pg


class ReportingTest(unittest.TestCase):

  def test_reporting(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'test_reporting')
    experiment = eval_test_helper.test_experiment()
    reporter = reporting.HtmlReporter()
    run = experiment.run(root_dir, 'new', plugins=[reporter])
    pg.io.path_exists(run.output_path_for(experiment, 'summary.html'))
    for leaf in experiment.leaf_nodes:
      self.assertTrue(
          pg.io.path_exists(run.output_path_for(leaf, 'index.html'))
      )
      for i in range(leaf.num_examples):
        pg.io.path_exists(run.output_path_for(leaf, f'{i + 1}.html'))


if __name__ == '__main__':
  unittest.main()
