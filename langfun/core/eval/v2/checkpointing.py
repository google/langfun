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

import langfun.core as lf
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
import pyglove as pg

Example = example_lib.Example
Experiment = experiment_lib.Experiment
Runner = experiment_lib.Runner


class Checkpointer(experiment_lib.Plugin):
  """Base class for checkpointing evaluation examples."""


class PerExampleCheckpointer(Checkpointer):
  """Checkpointer that saves each example to a separate file."""

  checkpoint_filename: str = 'checkpoint.bagz'

  def _on_bound(self):
    super()._on_bound()
    prefix, ext = self._file_prefix_and_ext(self.checkpoint_filename)
    self._checkpoint_file_prefix = prefix
    self._checkpoint_file_ext = ext

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
      def _load_state(ckpt_file):
        experiment.load_state(ckpt_file)

      experiment_dir = runner.current_run.input_dir(experiment)
      if pg.io.path_exists(experiment_dir):
        ckpt_files = [
            runner.current_run.input_path_for(experiment, filename)
            for filename in pg.io.listdir(experiment_dir)
            if filename.startswith(self._checkpoint_file_prefix)
            and filename.endswith(self._checkpoint_file_ext)
        ]
      else:
        ckpt_files = []

      for ckpt_file, _, error in lf.concurrent_map(
          _load_state, ckpt_files, max_workers=64,
      ):
        if error is not None:
          pg.logging.warning(
              'Failed to load checkpoint file %s: %s. Skipping the file.',
              ckpt_file, error
          )

  def on_example_complete(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves the example to the checkpoint file."""
    if not example.has_error:
      def save_state(example: Example):
        writer = SequenceWriter(
            runner.current_run.output_path_for(
                experiment,
                (
                    f'{self._checkpoint_file_prefix}_{example.id}'
                    f'{self._checkpoint_file_ext}'
                )
            )
        )
        writer.add(example)
        writer.close()
      runner.background_run(save_state, example)

  def _file_prefix_and_ext(self, filename: str) -> tuple[str, str]:
    ext_index = filename.rfind('.')
    if ext_index == -1:
      return filename, ''
    else:
      return filename[:ext_index], filename[ext_index:]


class BulkCheckpointer(Checkpointer):
  """Checkpointer that saves all examples to a single file."""

  checkpoint_filename: str = 'checkpoint.bagz'

  def _on_bound(self):
    super()._on_bound()
    self._lock = threading.Lock()
    self._sequence_writer = None

  def on_run_start(
      self,
      runner: Runner,
      root: Experiment,
  ) -> None:
    self._sequence_writer = {}

  def on_run_abort(
      self,
      runner: Runner,
      root: Experiment,
      error: BaseException
  ) -> None:
    with self._lock:
      if self._sequence_writer is not None:
        for writer in self._sequence_writer.values():
          writer.close()
        self._sequence_writer.clear()

  def on_run_complete(
      self,
      runner: Runner,
      root: Experiment,
  ) -> None:
    with self._lock:
      assert self._sequence_writer is not None and not self._sequence_writer

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
    sequence_writer = SequenceWriter(
        runner.current_run.output_path_for(
            experiment, self.checkpoint_filename
        )
    )
    with self._lock:
      if self._sequence_writer is not None:
        self._sequence_writer[experiment.id] = sequence_writer

  def on_experiment_complete(
      self,
      runner: Runner,
      experiment: Experiment,
  ) -> None:
    """Closes the checkpoint file."""
    if not experiment.is_leaf:
      return
    assert experiment.id in self._sequence_writer
    with self._lock:
      if self._sequence_writer is not None:
        # Make sure the writer is closed without delay so the file will be
        # available immediately.
        self._sequence_writer[experiment.id].close()
        del self._sequence_writer[experiment.id]

  def on_example_complete(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves the example to the checkpoint file."""
    assert experiment.id in self._sequence_writer
    if not example.has_error:
      runner.background_run(self._sequence_writer[experiment.id].add, example)


class SequenceWriter:
  """Thread safe sequence writer."""

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

  def close(self):
    # Make sure there is no write in progress.
    with self._lock:
      if self._sequence_writer is None:
        return
      self._sequence_writer.close()
      self._sequence_writer = None

  def __del__(self):
    self.close()
