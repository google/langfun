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
"""Debug runner."""

from langfun.core.eval.v2.runners import sequential


class DebugRunner(sequential.SequentialRunner):
  """A runner for debugging evaluations.

  The debug runner is a sequential runner that only runs the first example
  of each evaluation, with `raise_if_has_error` enabled. This is useful for
  quickly identifying issues in evaluation logic during development.
  Checkpointers are disabled for this runner.
  """

  NAME = 'debug'

  # Do not use the checkpointer for debug runner.
  plugins = []

  def _on_bound(self):
    super()._on_bound()
    if self.current_run.example_ids is None:
      self.current_run.rebind(example_ids=[1], skip_notification=True)
    self.current_run.rebind(raise_if_has_error=True, skip_notification=True)

  def _save_run_manifest(self) -> None:
    """Do nothing to avoid overriden existing runs."""
