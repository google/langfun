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
import re
import threading
import traceback
from typing import Annotated

import langfun.core as lf
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
import pyglove as pg

Example = example_lib.Example
Experiment = experiment_lib.Experiment
Runner = experiment_lib.Runner


class Checkpointer(experiment_lib.Plugin):
  """Base class for checkpointing evaluation examples."""

  checkpoint_filename: Annotated[
      str,
      'Checkpoint file pattern.'
  ] = 'checkpoint.bagz'

  def on_experiment_start(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    if not experiment.is_leaf:
      return

    current_run = runner.current_run
    if current_run.reprocess is not True:  # pylint: disable=g-bool-id-comparison
      if current_run.input_root != current_run.output_root:
        experiment.info(
            f'Warm starting from directory: {current_run.input_root}.'
        )
      self._load_experiment(runner, experiment)

    example_ids_to_evaluate = current_run.examples_to_evaluate(experiment)
    if experiment.state.ckpt_examples:
      loaded_example_ids = list(
          sorted(experiment.state.ckpt_examples.keys())
      )
      example_ids_to_evaluate -= set(loaded_example_ids)
      example_ids_to_evaluate = list(sorted(example_ids_to_evaluate))
      experiment.info(
          f'{len(experiment.state.ckpt_examples)} examples '
          'loaded from checkpoint files. Their outputs will be used '
          f'for recomputing metrics. Example IDs: {loaded_example_ids}.'
      )
      experiment.info(
          f'{len(example_ids_to_evaluate)} examples will be processed from '
          f'scratch. Example IDs: {example_ids_to_evaluate}.'
      )
    else:
      experiment.info(
          'No examples are loaded from checkpoint files. '
          f'{len(example_ids_to_evaluate)} examples will be processed from '
          f'scratch. Example IDs: {example_ids_to_evaluate}.'
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
    elif example.newly_processed:
      self._save_example(runner, experiment, example)

  def _load_experiment(
      self,
      runner: Runner,
      experiment: Experiment,
  ) -> None:
    """Creates the checkpoint file."""
    ckpt_files = self._list_checkpoint_filenames(runner, experiment)
    experiment.info(f'Found {len(ckpt_files)} checkpoint files to load.')

    # Load the checkpoint files in parallel.
    current_run = runner.current_run
    examples_to_load = current_run.examples_to_load(experiment)
    examples_to_load_metadata = current_run.examples_to_load_metadata(
        experiment
    )
    context = dict(counter=0, counter_lock=threading.Lock())
    copy_ckpt = current_run.input_root != current_run.output_root

    def _load_state(ckpt_file):
      error = None
      with pg.timeit() as t:
        try:
          experiment.load_state(
              current_run.input_path_for(experiment, ckpt_file),
              filter=lambda x: x.id in examples_to_load,
              load_example_metadata=lambda x: x.id in examples_to_load_metadata,
          )
        except BaseException as e:  # pylint: disable=broad-except
          error = e
        finally:
          with context['counter_lock']:
            context['counter'] += 1

          progress_str = f'{context["counter"]}/{len(ckpt_files)}'
          if error is None:
            experiment.info(
                f'Checkpoint file {ckpt_file!r} loaded in {t.elapse:.2f} '
                f'seconds. ({progress_str})'
            )
          else:
            experiment.warning(
                f'Failed to load checkpoint file {ckpt_file!r}: {error}. '
                f'Skipping the file. ({progress_str})'
            )

        if not copy_ckpt:
          return

        # Copy the checkpoint records to the output directory.
        try:
          with pg.io.open_sequence(
              current_run.output_path_for(experiment, ckpt_file), 'w'
          ) as o, pg.io.open_sequence(
              current_run.input_path_for(experiment, ckpt_file), 'r'
          ) as i:
            for x in i:
              o.add(x)
        except BaseException as e:  # pylint: disable=broad-except
          experiment.warning(
              f'Failed to copy checkpoint {ckpt_file!r}: {e}.'
          )

    _ = list(
        lf.concurrent_map(
            _load_state, ckpt_files, max_workers=16, silence_on_errors=None
        )
    )

  @abc.abstractmethod
  def _list_checkpoint_filenames(
      self, runner: Runner, experiment: Experiment
  ) -> list[str]:
    """Lists the checkpoint filenames to restore."""

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

  def _on_bound(self):
    super()._on_bound()
    prefix, ext = self._file_prefix_and_ext(self.checkpoint_filename)
    self._checkpoint_file_prefix = prefix
    self._checkpoint_file_ext = ext

  def _list_checkpoint_filenames(
      self, runner: Runner, experiment: Experiment
  ) -> list[str]:
    experiment_dir = runner.current_run.input_dir(experiment)
    filenames = []
    examples_to_load = runner.current_run.examples_to_load(experiment)
    if pg.io.path_exists(experiment_dir):
      regex = re.compile(
          f'{self._checkpoint_file_prefix}_(\\d+){self._checkpoint_file_ext}'
          .replace('.', '\\.')
      )
      for filename in pg.io.listdir(experiment_dir):
        match = regex.match(filename)
        if match and int(match.group(1)) in examples_to_load:
          filenames.append(filename)
    return filenames

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
            f'Example {example.id} checkpointed to {writer.path}.',
        )
      except BaseException as e:  # pylint: disable=broad-except
        experiment.error(
            f'Failed to checkpoint example {example.id} to {writer.path}. '
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

  def _list_checkpoint_filenames(
      self, runner: Runner, experiment: Experiment
  ) -> list[str]:
    if pg.io.path_exists(
        runner.current_run.input_path_for(experiment, self.checkpoint_filename)
    ):
      return [self.checkpoint_filename]
    return []

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
            f'{len(experiment.state.evaluation_status)} examples are '
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
            f'Example {example.id} checkpointed to {writer.path}.',
        )
      except BaseException as e:  # pylint: disable=broad-except
        experiment.error(
            f'Failed to checkpoint example {example.id} to {writer.path}. '
            f'Error: {e}, Stacktrace: \n{traceback.format_exc()}.',
        )
        raise e
    runner.background_run(_save_example, example)


class SequenceWriter:
  """Thread safe sequence writer."""

  def __init__(self, path: str):
    self._lock = threading.Lock()
    self._path = path
    self._sequence_writer = pg.io.open_sequence(path, 'a')

  @property
  def path(self) -> str:
    return self._path

  def add(self, example: Example):
    example_blob = pg.to_json_str(
        example,
        hide_default_values=True,
        save_ref_value=True,
        exclude_input=False,
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
