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
"""Base experiment runner."""

import abc
import concurrent.futures
import random
import threading
import traceback
from typing import Any, Annotated, Callable, Iterator, Literal

from langfun import core as lf
from langfun.core.eval.v2 import checkpointing
from langfun.core.eval.v2 import config_saver
from langfun.core.eval.v2 import evaluation as evaluation_lib
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2 import progress_tracking
from langfun.core.eval.v2 import reporting
from langfun.core.llms.cache import in_memory
import pyglove as pg

Runner = experiment_lib.Runner
Example = example_lib.Example
Evaluation = evaluation_lib.Evaluation
Experiment = experiment_lib.Experiment
Plugin = experiment_lib.Plugin


class RunnerBase(Runner):
  """Base class for runners with plugin support and IO pooling.

  `RunnerBase` provides the basic runner functionalities such as plugin
  integration for checkpointing, reporting and progress tracking.
  It also manages a thread pool for background IO operations.
  Subclasses should implement `_run` and `_evaluate_items` for different
  execution strategies.
  """

  progress_tracker: Annotated[
      Literal['tqdm', 'html', 'auto', None],
      (
          'If `tqdm`, force using tqdm for progress update. '
          'If `html`, force using html for progress update. '
          'If `auto`, determine it automatically based on the running '
          'environment (console vs. notebook)'
          'If `none`, disable progress update.'
      )
  ] = 'auto'

  plugins = [
      checkpointing.BulkCheckpointer(),
      reporting.HtmlReporter(),
      config_saver.RunConfigSaver(),
  ]

  max_background_threads: Annotated[
      int,
      'Max number of background threads for IO operations.'
  ] = 128

  def _on_bound(self):
    super()._on_bound()

    # Install the tqdm plugin if needed.
    if self.progress_tracker is not None:
      with pg.notify_on_change(False):
        self.plugins.append(
            progress_tracking.progress_tracker(self.progress_tracker)
        )

    if self.max_background_threads > 0:
      self._io_pool_lock = threading.Lock()
      self._io_pool = concurrent.futures.ThreadPoolExecutor(
          max_workers=self.max_background_threads
      )
    else:
      self._io_pool_lock = None
      self._io_pool = None

    # TODO(daiyip): render background errors.
    self._background_last_error = None

  def background_run(self, func: Callable[..., Any], *args, **kwargs) -> None:
    """Runs the function with the IO pool."""
    def _background_run(*args, **kwargs):
      try:
        func(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-except
        self._background_last_error = e

    if self.max_background_threads > 0:
      with self._io_pool_lock:
        if self._io_pool is not None:
          self._io_pool.submit(_background_run, *args, **kwargs)
    else:
      _background_run(*args, **kwargs)

  def _all_plugins(self, experiment: Experiment) -> Iterator[Plugin]:
    """Returns all plugins for the experiment."""
    for plugin in self.plugins:
      yield plugin
    for plugin in experiment.plugins:
      yield plugin

  def on_run_start(self) -> None:
    """Called when a runner is started."""
    for plugin in self._all_plugins(self.current_run.experiment):
      plugin.on_run_start(self, self.current_run.experiment)

  def on_run_complete(self) -> None:
    """Called when a runner is complete."""
    for plugin in self._all_plugins(self.current_run.experiment):
      plugin.on_run_complete(self, self.current_run.experiment)

  def on_run_abort(self, error: Exception) -> None:
    """Called when a runner is aborted."""
    for plugin in self._all_plugins(self.current_run.experiment):
      plugin.on_run_abort(self, self.current_run.experiment, error)

  def on_experiment_start(self, experiment: Experiment) -> None:
    """Called when an evaluation is started."""
    # Start the progress of the evaluation.
    num_examples_to_evaluate = 0
    if experiment.is_leaf:
      assert isinstance(experiment, Evaluation)
      num_examples_to_evaluate = len(
          self.current_run.examples_to_evaluate(experiment)
      )
      experiment.progress.start(total=num_examples_to_evaluate)
      pg.io.mkdirs(self.current_run.output_dir(experiment))
    else:
      experiment.progress.start(total=len(experiment.leaf_nodes))

    # Notify the plugins of the experiment start.
    for plugin in self._all_plugins(experiment):
      plugin.on_experiment_start(self, experiment)

    if experiment.is_leaf:
      pg.io.mkdirs(self.current_run.output_dir(experiment))
      experiment.info(
          f'Starting evaluation {experiment.id!r} with '
          f'{num_examples_to_evaluate} examples to evaluate.'
      )

  def on_experiment_skipped(self, experiment: Experiment) -> None:
    """Called when an evaluation is skipped."""
    # Skip event will only be triggered for leaf evaluations.
    assert experiment.is_leaf
    experiment.progress.start(total=1)
    experiment.progress.increment_skipped(1)

    if experiment.is_leaf:
      experiment.info(
          f'Evaluation {experiment.id!r} is skipped.'
      )

    # Notify the plugins of the experiment skip.
    for plugin in self._all_plugins(experiment):
      plugin.on_experiment_skipped(self, experiment)

    # Only leaf evaluations will trigger the complete notification of the
    # ancestors.
    self._update_ancestor_progresses(experiment)

  def on_experiment_complete(self, experiment: Experiment) -> None:
    """Called when an evaluation is complete."""
    progress = experiment.progress
    progress.stop()

    # Notify the plugins of the experiment complete.
    for plugin in self._all_plugins(experiment):
      plugin.on_experiment_complete(self, experiment)

    # Only leaf evaluations will trigger the complete notification of the
    # ancestors.
    if experiment.is_leaf:
      self._update_ancestor_progresses(experiment)
      self._log_experiment_completion(experiment)

  def _log_experiment_completion(self, experiment: Experiment):
    example_ids = sorted(self.current_run.examples_to_evaluate(experiment))
    num_from_checkpoint, num_processed = 0, 0
    for example_id in example_ids:
      status = experiment.state.get_status(example_id)
      if status.newly_processed:
        num_processed += 1
      else:
        num_from_checkpoint += 1
    experiment.info(
        f'{experiment.id} completed with {num_from_checkpoint + num_processed} '
        f'examples evaluated ({num_from_checkpoint} from checkpoint, '
        f'{num_processed} newly processed).'
    )

  def on_experiment_abort(
      self, experiment: Experiment, error: BaseException) -> None:
    """Called when an evaluation is complete."""
    assert experiment.is_leaf
    experiment.fatal(f'{error}\n\n{traceback.format_exc()}')

    # Notify the plugins of the experiment abort.
    for plugin in self._all_plugins(experiment):
      plugin.on_experiment_abort(self, experiment, error)

  def _update_ancestor_progresses(self, experiment: Experiment):
    """Updates the progresses of the parent nodes of the experiment."""
    parent = experiment.parent
    progress = experiment.progress
    while parent is not None:
      parent_progress = parent.progress
      if progress.is_failed:
        parent_progress.increment_failed()
      elif progress.is_skipped:
        parent_progress.increment_skipped()
      else:
        # A evaluation could be considered as done if it has processed all the
        # examples specified by `example_ids`.
        assert progress.is_completed, progress
        parent_progress.increment_processed()

      if parent_progress.is_completed:
        self.on_experiment_complete(parent)
      elif parent_progress.is_skipped:
        self.on_experiment_skipped(parent)
      parent = parent.parent

  def on_example_start(
      self,
      experiment: Experiment,
      example: Example
  ) -> None:
    """Called when an evaluation example is started."""
    assert isinstance(experiment, Evaluation), experiment
    experiment.state.update(example, in_progress=True)
    for plugin in self._all_plugins(experiment):
      plugin.on_example_start(self, experiment, example)
    experiment.info(f'Starting to evaluate example {example.id}.')

  def on_example_complete(
      self,
      experiment: Experiment,
      example: Example
  ) -> None:
    """Called when an evaluation example is complete."""
    assert isinstance(experiment, Evaluation), experiment
    experiment.state.update(example, in_progress=False)
    if example.newly_processed:
      if example.error is None:
        experiment.progress.increment_processed()
        experiment.info(
            f'Example {example.id} is successfully evaluated (with process) in '
            f'{example.elapse:.2f} seconds.'
        )
      else:
        experiment.progress.increment_failed()
        experiment.error(
            (
                f'Failed to evaluate example {example.id} in '
                f'{example.elapse:.2f} seconds.'
            ),
            error=example.error
        )
    else:
      experiment.progress.increment_skipped()
      experiment.info(
          f'Example {example.id} is successfully evaluated (without reprocess) '
          f'in {example.elapse:.2f} seconds.'
      )

    experiment.usage_summary.merge(example.usage_summary)
    experiment.progress.update_execution_summary(example.execution_status)

    parent = experiment.parent
    while parent is not None:
      parent.usage_summary.merge(example.usage_summary)
      parent = parent.parent

    for plugin in self._all_plugins(experiment):
      plugin.on_example_complete(self, experiment, example)

  def run(self) -> None:
    """Runs the experiment."""
    # Resets the experiment before getting start.
    for node in self.current_run.experiment.nodes:
      node.reset()

    # Start the run.
    self.on_run_start()
    cache = None

    try:
      # Start the non-leaf nodes.
      for node in self.current_run.experiment.nonleaf_nodes:
        self.on_experiment_start(node)

      # Skip evaluations if needed.
      if self.current_run.filter is not None:
        targets = []
        for evaluation in self.current_run.experiment.leaf_nodes:
          if self.current_run.filter(evaluation):
            targets.append(evaluation)
          else:
            self.on_experiment_skipped(evaluation)
      else:
        targets = self.current_run.experiment.leaf_nodes

      # Prepare the global cache if needed.
      global_settings = {}
      if self.current_run.use_cache == 'global':
        cache = self._load_or_create_cache(self.current_run.experiment)
        global_settings['cache'] = cache

      # Evaluate the leaf evaluations if not skipped.
      with lf.use_settings(**global_settings):
        self._run(targets)

      self.on_run_complete()
    except BaseException as e:  # pylint: disable=broad-except
      self.on_run_abort(e)
      raise e
    finally:
      if cache is not None:
        self.background_run(cache.save)

      # Wait for the background tasks to finish.
      if self.max_background_threads > 0:
        with self._io_pool_lock:
          self._io_pool, io_pool = None, self._io_pool
        io_pool.shutdown(wait=True)

  @abc.abstractmethod
  def _run(self, evaluations: list[Evaluation]) -> None:
    """Runs multiple evaluations."""

  def run_evaluation(self, evaluation: Evaluation) -> None:
    """Runs the evaluation."""
    try:
      evaluation.setup()
      self.on_experiment_start(evaluation)

      per_evaluation_settings = {}
      cache = None
      if self.current_run.use_cache == 'per_dataset':
        cache = self._load_or_create_cache(evaluation)
        per_evaluation_settings['cache'] = cache

      with lf.use_settings(**per_evaluation_settings):
        items = (
            Example(
                id=example_id,
                input=evaluation.example_input_by_id(example_id)
            ) for example_id in sorted(
                self.current_run.examples_to_evaluate(evaluation)
            )
        )
        if self.current_run.shuffle_inputs:
          items = list(items)
          random.shuffle(items)
        self._evaluate_items(evaluation, items)

      if cache:
        self.background_run(cache.save)
      self.on_experiment_complete(evaluation)
    except BaseException as e:  # pylint: disable=broad-except
      self.on_experiment_abort(evaluation, e)
      raise e
    finally:
      evaluation.teardown()

  @abc.abstractmethod
  def _evaluate_items(
      self, evaluation: Evaluation, items: Iterator[Example]
  ) -> None:
    """Evaluates the items of an evaluation."""

  def evaluate_item(
      self,
      evaluation: Evaluation,
      item: Example
  ) -> Example:
    """Runs the evaluation example."""
    self.on_example_start(evaluation, item)
    item = evaluation.evaluate(
        item,
        raise_if_has_error=self.current_run.raise_if_has_error,
        reevaluate_upon_previous_errors=self.current_run.reevaluate_upon_previous_errors,
        force_recompute_metrics=self.current_run.force_recompute_metrics,
    )
    self.on_example_complete(evaluation, item)
    return item

  def _load_or_create_cache(self, experiment: Experiment) -> lf.LMCache | None:
    """Loads or creates the cache."""
    return in_memory.InMemory(
        self.current_run.output_path_for(experiment, 'cache.json')
    )
