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
"""Evaluation experiment runners."""
import abc
import collections
import concurrent.futures
import random
import threading
import time
import traceback
from typing import Any, Annotated, Callable, Iterator

from langfun import core as lf
from langfun.core.eval.v2 import checkpointing
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


_RUN_MANIFEST = 'run.json'


class RunnerBase(Runner):
  """A simple runner that runs evaluations and their examples sequentially."""

  tqdm: Annotated[
      bool,
      (
          'If True, force using tqdm for progress update. Otherwise, determine '
          'it automatically based on the running environment (console vs. '
          'notebook)'
      )
  ] = False

  plugins = [
      checkpointing.BulkCheckpointer(),
      reporting.HtmlReporter(),
  ]

  def _on_bound(self):
    super()._on_bound()

    # Install the tqdm plugin if needed.
    with pg.notify_on_change(False):
      self.plugins.append(progress_tracking.progress_tracker(self.tqdm))

    self._io_pool_lock = threading.Lock()
    self._io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    # TODO(daiyip): render background errors.
    self._background_last_error = None

  def background_run(self, func: Callable[..., Any], *args, **kwargs) -> None:
    """Runs the function with the IO pool."""
    def _background_run(*args, **kwargs):
      try:
        func(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-except
        self._background_last_error = e

    with self._io_pool_lock:
      if self._io_pool is not None:
        self._io_pool.submit(_background_run, *args, **kwargs)

  def _all_plugins(self, experiment: Experiment) -> Iterator[Plugin]:
    """Returns all plugins for the experiment."""
    for plugin in self.plugins:
      yield plugin
    for plugin in experiment.plugins:
      yield plugin

  #
  # IO operations for saving running files.
  #

  def _save_run_manifest(self) -> None:
    def _save():
      pg.symbolic.deref(self.current_run.clone(), recursive=True).save(
          self.current_run.output_path_for(
              self.current_run.experiment, _RUN_MANIFEST
          ),
          hide_default_values=True
      )
    self.background_run(_save)

  def on_run_start(self) -> None:
    """Called when a runner is started."""
    self._save_run_manifest()

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
    if experiment.is_leaf:
      assert isinstance(experiment, Evaluation)
      num_examples_to_evaluate = (
          len(self.current_run.example_ids)
          if self.current_run.example_ids else experiment.num_examples
      )
      experiment.progress.start(total=num_examples_to_evaluate)
      experiment.info(
          'Starting evaluation %s with %d examples to evaluate.'
          % (experiment.id, num_examples_to_evaluate)
      )
    else:
      experiment.progress.start(total=len(experiment.leaf_nodes))

    # Notify the plugins of the experiment start.
    for plugin in self._all_plugins(experiment):
      plugin.on_experiment_start(self, experiment)

  def on_experiment_skipped(self, experiment: Experiment) -> None:
    """Called when an evaluation is skipped."""
    # Skip event will only be triggered for leaf evaluations.
    assert experiment.is_leaf
    experiment.progress.start(total=1)
    experiment.progress.increment_skipped(1)

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
    example_ids = (
        self.current_run.example_ids if self.current_run.example_ids else
        list(range(1, experiment.num_examples + 1))
    )
    num_from_checkpoint, num_processed = 0, 0
    for example_id in example_ids:
      example = experiment.state.get(example_id)
      if example.newly_processed:
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
        assert progress.is_completed
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
    for plugin in self._all_plugins(experiment):
      plugin.on_example_start(self, experiment, example)

  def on_example_complete(
      self,
      experiment: Experiment,
      example: Example
  ) -> None:
    """Called when an evaluation example is complete."""
    if example.newly_processed:
      if example.error is None:
        experiment.progress.increment_processed()
      else:
        experiment.progress.increment_failed()
    else:
      experiment.progress.increment_skipped()

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
    except Exception as e:  # pylint: disable=broad-except
      self.on_run_abort(e)
      raise e
    finally:
      if cache is not None:
        self.background_run(cache.save)

      # Wait for the background tasks to finish.
      with self._io_pool_lock:
        self._io_pool, io_pool = None, self._io_pool
      io_pool.shutdown(wait=True)

  @abc.abstractmethod
  def _run(self, evaluations: list[Evaluation]) -> None:
    """Runs multiple evaluations."""

  def run_evaluation(self, evaluation: Evaluation) -> None:
    """Runs the evaluation."""
    try:
      self.on_experiment_start(evaluation)

      per_evaluation_settings = {}
      cache = None
      if self.current_run.use_cache == 'per_dataset':
        cache = self._load_or_create_cache(evaluation)
        per_evaluation_settings['cache'] = cache

      with lf.use_settings(**per_evaluation_settings):
        if self.current_run.example_ids is None:
          items = (
              Example(id=i + 1, input=ex) for i, ex in enumerate(
                  evaluation.example_inputs)
          )
        else:
          items = (
              Example(
                  id=example_id,
                  input=evaluation.example_input_by_id(example_id)
              ) for example_id in self.current_run.example_ids
          )
        self._evaluate_items(evaluation, items)

      if cache:
        self.background_run(cache.save)
      self.on_experiment_complete(evaluation)
    except BaseException as e:  # pylint: disable=broad-except
      self.on_experiment_abort(evaluation, e)
      raise e

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
        item, raise_if_has_error=self.current_run.raise_if_has_error
    )
    self.on_example_complete(evaluation, item)
    return item

  def _load_or_create_cache(self, experiment: Experiment) -> lf.LMCache | None:
    """Loads or creates the cache."""
    return in_memory.InMemory(
        self.current_run.output_path_for(experiment, 'cache.json')
    )


class SequentialRunner(RunnerBase):
  """Sequential runner.

  Sequential runner runs all evaluations and their examples in sequence,
  as well as the background tasks, it allows the developer to catch all
  exceptions thrown from the background tasks, making it easier to debug.
  """

  NAME = 'sequential'

  def background_run(
      self, func: Callable[..., Any], *args: Any, **kwargs: Any
  ) -> None:
    """Runs the function with the IO pool."""
    func(*args, **kwargs)

  def _run(self, evaluations: list[Evaluation]) -> None:
    """Runs the experiment in sequence."""
    for e in evaluations:
      self.run_evaluation(e)

  def _evaluate_items(
      self, evaluation: Evaluation, items: Iterator[Example]
  ) -> None:
    """Runs the evaluation items in sequence."""
    for item in items:
      self.evaluate_item(evaluation, item)


class DebugRunner(SequentialRunner):
  """Debug runner."""

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


class ParallelRunner(RunnerBase):
  """Parallel runner."""

  NAME = 'parallel'

  timeout: Annotated[
      int | None,
      'Timeout for each evaluation example.'
  ] = None

  concurrent_startup_delay: Annotated[
      tuple[int, int] | None,
      (
          'A range of seconds to delay the initial evaluation of each thread '
          'in the thread pool, helping to prevent a burst in LLM QPS at '
          'startup. If set to None, no delay will be applied.'
      )
  ] = None

  def _run(self, evaluations: list[Evaluation]) -> None:
    """Runs the evaluations in parallel."""
    def _run_group(evaluation_group: list[Evaluation]):
      for e in evaluation_group:
        self.run_evaluation(e)

    # Run evaluations in parallel groupped by resource key.
    groups: dict[str, list[Evaluation]] = collections.defaultdict(list)
    for e in evaluations:
      resource_ids = e.resource_ids()
      if not resource_ids:
        group_id = e.id
      else:
        # TODO(daiyip): support group that requires multiple resources.
        group_id = resource_ids.pop()
      groups[group_id].append(e)

    for _, _, _ in lf.concurrent_map(
        _run_group,
        groups.values(),
        max_workers=max(64, len(groups)),
        timeout=self.timeout,
        silence_on_errors=None,
    ):
      pass

  def _evaluate_items(
      self, evaluation: Evaluation, items: Iterator[Example]
  ) -> None:
    """Override run items to run in parallel."""
    if self.concurrent_startup_delay is not None:
      thread_delayed = {}
      def _evaluate_item(item: Example):
        thread_id = threading.current_thread().ident
        if thread_id not in thread_delayed:
          thread_delayed[thread_id] = True
          time.sleep(random.randint(*self.concurrent_startup_delay))
        return self.evaluate_item(evaluation, item)
    else:
      def _evaluate_item(item: Example):
        return self.evaluate_item(evaluation, item)

    for _, _, _ in lf.concurrent_map(
        _evaluate_item,
        items,
        max_workers=evaluation.max_workers,
        timeout=self.timeout,
        silence_on_errors=None,
    ):
      pass
