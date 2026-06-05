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
"""Config saver plugins."""

import os
from langfun.core.eval.v2 import experiment as experiment_lib


class RunConfigSaver(experiment_lib.Plugin):
  """Saves the current run."""

  def on_run_start(
      self,
      runner: experiment_lib.Runner,
      root: experiment_lib.Experiment
  ) -> None:
    del root  # Unused.
    self._save_run_config(runner)

  def _save_run_config(self, runner: experiment_lib.Runner) -> None:
    def _save():
      runner.current_run.save(
          os.path.join(runner.current_run.output_root, 'run.json'),
          hide_default_values=True,
      )
    runner.background_run(_save)
