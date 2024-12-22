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
import abc
import threading
import traceback

import langfun.core as lf
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
import pyglove as pg

Example = example_lib.Example
Experiment = experiment_lib.Experiment
Runner = experiment_lib.Runner


class Checkpointer(experiment_lib.Plugin):
  """Base class for checkpointing evaluation examples."""

  def on_experiment_start(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    if not experiment.is_leaf:
      return

    # For refresh runs, we don't want to load the previous state.
    if not runner.current_run.refresh:
      if runner.current_run.input_root != runner.current_run.output_root:
        experiment.info(
            f'Warm starting from directory: {runner.current_run.input_root}.'
        )
      self._load_experiment(runner, experiment)

    if experiment.state.evaluated_examples:
      loaded_example_ids = list(
          sorted(experiment.state.evaluated_examples.keys())
      )
      example_ids_to_evaluate = (
          set(runner.current_run.example_ids) if runner.current_run.example_ids
          else set(range(1, experiment.num_examples + 1))
      )
      example_ids_to_evaluate -= set(loaded_example_ids)

      experiment.info(
          f'{len(experiment.state.evaluated_examples)} examples have been '
          'loaded from checkpoint files. Their outputs will be used '
          f'for recomputing metrics. Example IDs: {loaded_example_ids}'
      )
      experiment.info(
          f'{len(example_ids_to_evaluate)} examples will be processed from '
          f'scratch. Example IDs: {list(sorted(example_ids_to_evaluate))}'
      )
    else:
      experiment.info(
          'No examples are loaded from checkpoint files. '
          f'Experiment {experiment.id} starts from scratch.'
      )

  def on_example_complete(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves the example to the checkpoint file."""
    if example.has_error:
      experiment.warning(
          f'Example {example.id} has error. Skipping checkpointing.'
      )
    else:
      self._save_example(runner, experiment, example)

  @abc.abstractmethod
  def _load_experiment(self, runner: Runner, experiment: Experiment) -> None:
    """Loads the experiment state from checkpoint files."""

  @abc.abstractmethod
  def _save_example(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves an evaluated example."""


class PerExampleCheckpointer(Checkpointer):
  """Checkpointer that saves each example to a separate file."""

  checkpoint_filename: str = 'checkpoint.bagz'

  def _on_bound(self):
    super()._on_bound()
    prefix, ext = self._file_prefix_and_ext(self.checkpoint_filename)
    self._checkpoint_file_prefix = prefix
    self._checkpoint_file_ext = ext

  def _load_experiment(
      self,
      runner: Runner,
      experiment: Experiment,
  ) -> None:
    """Creates the checkpoint file."""
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

    experiment.info(f'Found {len(ckpt_files)} checkpoint files to load.')

    # Load the checkpoint files in parallel.
    context = dict(counter=0, counter_lock=threading.Lock())
    def _load_state(ckpt_file):
      error = None
      with pg.timeit() as t:
        try:
          experiment.load_state(ckpt_file)
        except BaseException as e:  # pylint: disable=broad-except
          error = e
        finally:
          with context['counter_lock']:
            context['counter'] += 1

          progress_str = f'{context["counter"]}/{len(ckpt_files)}'
          if error is None:
            experiment.info(
                f'Loaded checkpoint file {ckpt_file} in {t.elapse:.2f} '
                f'seconds. ({progress_str})'
            )
          else:
            experiment.warning(
                f'Failed to load checkpoint file {ckpt_file}: {error}. '
                f'Skipping the file. ({progress_str})'
            )

    _ = list(
        lf.concurrent_map(
            _load_state, ckpt_files, max_workers=16, silence_on_errors=None
        )
    )

  def _save_example(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves the example to the checkpoint file."""
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
      try:
        writer.add(example)
        writer.close()
        experiment.info(
            f'Example {example.id} saved to {writer.path}.',
        )
      except BaseException as e:  # pylint: disable=broad-except
        experiment.error(
            f'Failed to save example {example.id} to {writer.path}. '
            f'Error: {e}, Stacktrace: \n{traceback.format_exc()}.',
        )
        raise e
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
    super().on_experiment_start(runner, experiment)

    # Prepare the sequence writer for the experiment.
    if experiment.is_leaf:
      sequence_writer = SequenceWriter(
          runner.current_run.output_path_for(
              experiment, self.checkpoint_filename
          )
      )
      with self._lock:
        if self._sequence_writer is not None:
          self._sequence_writer[experiment.id] = sequence_writer

  def _load_experiment(
      self,
      runner: Runner,
      experiment: Experiment,
  ) -> None:
    """Creates the checkpoint file."""
    experiment.load_state(
        runner.current_run.input_path_for(
            experiment, self.checkpoint_filename
        ),
        raise_if_not_exist=False
    )

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
        writer = self._sequence_writer.pop(experiment.id)
        writer.close()
        experiment.info(
            f'{len(experiment.state.evaluated_examples)} examples are '
            f'checkpointed to {writer.path}.'
        )

  def _save_example(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example,
  ) -> None:
    """Saves the example to the checkpoint file."""
    assert experiment.id in self._sequence_writer
    def _save_example(example: Example):
      writer = self._sequence_writer[experiment.id]
      try:
        writer.add(example)
        experiment.info(
            f'Example {example.id} added to {writer.path}.',
        )
      except BaseException as e:  # pylint: disable=broad-except
        experiment.error(
            f'Failed to save example {example.id} to {writer.path}. '
            f'Error: {e}, Stacktrace: \n{traceback.format_exc()}.',
        )
        raise e
    runner.background_run(_save_example, example)


class SequenceWriter:
  """Thread safe sequence writer."""

  def __init__(self, path: str):
    self._lock = threading.Lock()
    self._path = path
    self._sequence_writer = pg.io.open_sequence(path, 'w')

  @property
  def path(self) -> str:
    return self._path

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
