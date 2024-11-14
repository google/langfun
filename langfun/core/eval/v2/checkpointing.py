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
"""Checkpointing evaluation runs."""
import threading

from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
import pyglove as pg

Example = example_lib.Example
Experiment = experiment_lib.Experiment
Runner = experiment_lib.Runner


class Checkpointer(experiment_lib.Plugin):
  """Plugin for checkpointing evaluation runs."""

  checkpoint_filename: str = 'checkpoint.bagz'

  def _on_bound(self):
    super()._on_bound()
    self._lock = threading.Lock()
    self._state_writer = None

  def on_run_start(
      self,
      runner: Runner,
      root: Experiment,
  ) -> None:
    self._state_writer = {}

  def on_run_abort(
      self,
      runner: Runner,
      root: Experiment,
      error: BaseException
  ) -> None:
    with self._lock:
      if self._state_writer is not None:
        self._state_writer.clear()

  def on_run_complete(
      self,
      runner: Runner,
      root: Experiment,
  ) -> None:
    with self._lock:
      assert self._state_writer is not None and not self._state_writer

  def on_experiment_start(
      self,
      runner: Runner,
      experiment: Experiment,
  ) -> None:
    """Creates the checkpoint file."""
    if not experiment.is_leaf:
      return
    # For refresh runs, we don't want to load the previous state.
    if not runner.current_run.refresh:
      experiment.load_state(
          runner.current_run.input_path_for(
              experiment, self.checkpoint_filename
          ),
          raise_if_not_exist=False
      )
    state_writer = StateWriter(
        runner.current_run.output_path_for(
            experiment, self.checkpoint_filename
        )
    )
    with self._lock:
      if self._state_writer is not None:
        self._state_writer[experiment.id] = state_writer

  def on_experiment_complete(
      self,
      runner: Runner,
      experiment: Experiment,
  ) -> None:
    """Closes the checkpoint file."""
    if not experiment.is_leaf:
      return
    assert experiment.id in self._state_writer
    with self._lock:
      if self._state_writer is not None:
        del self._state_writer[experiment.id]

  def on_example_complete(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves the example to the checkpoint file."""
    assert experiment.id in self._state_writer
    if not example.has_error:
      runner.background_run(self._state_writer[experiment.id].add, example)


class StateWriter:
  """Thread safe state writer."""

  def __init__(self, path: str):
    self._lock = threading.Lock()
    self._sequence_writer = pg.io.open_sequence(path, 'w')

  def add(self, example: Example):
    example_blob = pg.to_json_str(
        example,
        hide_default_values=True,
        save_ref_value=True,
        exclude_input=True
    )
    with self._lock:
      if self._sequence_writer is None:
        return
      self._sequence_writer.add(example_blob)

  def __del__(self):
    # Make sure there is no write in progress.
    with self._lock:
      assert self._sequence_writer is not None
      self._sequence_writer.close()
      self._sequence_writer = None
