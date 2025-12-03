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
"""Config saver test."""

import os
import tempfile
import unittest
from langfun.core.eval.v2 import config_saver
from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2.runners import parallel  # pylint: disable=unused-import


class RunConfigSaverTest(unittest.TestCase):

  def test_save_run_config(self):
    root_dir = os.path.join(tempfile.mkdtemp(), 'test_run_config_saver')
    experiment = eval_test_helper.test_evaluation()
    run = experiment.run(
        root_dir, 'new', plugins=[config_saver.RunConfigSaver()]
    )
    self.assertTrue(os.path.exists(os.path.join(run.output_root, 'run.json')))


if __name__ == '__main__':
  unittest.main()
