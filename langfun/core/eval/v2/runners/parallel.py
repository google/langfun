# Copyright 2025 The Langfun Authors
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
"""Parallel runner."""

import collections
import math
import random
import threading
import time

from typing import Annotated, Iterator
import langfun.core as lf
from langfun.core.eval.v2 import checkpointing
from langfun.core.eval.v2 import experiment as experiment_lib
from langfun.core.eval.v2.runners import base
from langfun.core.eval.v2.runners import ckpt_monitor
import pyglove as pg


class ParallelRunner(base.RunnerBase):
  """A runner that executes evaluations and examples in parallel.

  The parallel runner groups evaluations by their required resources
  (e.g., specific LLMs) and runs evaluations that do not share resources in
  parallel. Within each evaluation, examples are also processed in parallel
  using threads, up to `Evaluation.max_workers`.
  """

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

  def _run(self, evaluations: list[base.Evaluation]) -> None:
    """Runs the evaluations in parallel."""
    def _run_group(evaluation_group: list[base.Evaluation]):
      for e in evaluation_group:
        self.run_evaluation(e)

    # Run evaluations in parallel groupped by resource key.
    groups: dict[str, list[base.Evaluation]] = collections.defaultdict(list)
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
      self, evaluation: base.Evaluation, items: Iterator[base.Example]
  ) -> None:
    """Override run items to run in parallel."""
    if self.concurrent_startup_delay is not None:
      thread_delayed = {}
      def _evaluate_item(item: base.Example):
        thread_id = threading.current_thread().ident
        if thread_id not in thread_delayed:
          thread_delayed[thread_id] = True
          time.sleep(random.randint(*self.concurrent_startup_delay))
        return self.evaluate_item(evaluation, item)
    else:
      def _evaluate_item(item: base.Example):
        return self.evaluate_item(evaluation, item)

    for _, _, _ in lf.concurrent_map(
        _evaluate_item,
        items,
        max_workers=self._max_workers(evaluation),
        timeout=self.timeout,
        silence_on_errors=None,
    ):
      pass

  def _max_workers(self, evaluation: base.Evaluation) -> int | None:
    return evaluation.max_workers


class _SingleSliceRunner(ParallelRunner):
  """A single slice runner."""

  NAME = '__single_slice_runner__'

  # Do not track progress in single slice runner.
  progress_tracker = None

  num_slices: Annotated[
      int,
      'The number of slices to run the evaluations in.'
  ] = 1

  def _max_workers(self, evaluation: base.Evaluation) -> int | None:
    max_workers = super()._max_workers(evaluation)
    if max_workers is None:
      return None
    return max(1, math.ceil(max_workers / self.num_slices))


class MultiSliceParallelRunner(experiment_lib.Runner):
  """A sliced parallel runner.

  An evaluation is split into `num_slices` slices. Each MultiSliceParallelRunner
  instance is responsible for evaluating a single slice. The instance with
  `slice_id` 0 will also aggregate checkpoints from all slices.

  Sliced parallel runner allows running multiple instances across different
  machines/hosts parallelly. This can be utilize for scaling the evaluation
  jobs to run on multiple machines, or running evaluations in a fault-tolerant
  way by splitting each evaluation into multiple slices.
  """

  NAME = 'sliced-parallel'

  slice_id: Annotated[
      int,
      (
          'The slice ID of the runner. If 0, it will also run as the '
          'aggregator for collecting results from other slices. '
      )
  ] = 0

  num_slices: Annotated[
      int,
      'The number of slices to run the evaluations in parallel.'
  ] = 1

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

  ckpt_format: Annotated[
      str,
      'The file extension of the checkpoint files.'
  ] = 'bagz'

  max_aggregation_threads: Annotated[
      int,
      'The maximum number of threads to aggregate checkpoints.'
  ] = 32

  def _on_bound(self):
    super()._on_bound()
    if self.current_run.use_cache != 'no':
      raise ValueError(
          'Cache is not supported in MultiProcessParallelRunner. '
          f'Encountered: {self.current_run.use_cache}'
      )
    monitor_plugins = []
    worker_plugins = [
        checkpointing.PerExampleCheckpointer(
            checkpoint_filename=f'checkpoint.{self.ckpt_format}'
        ),
    ]
    for plugin in self.plugins:
      if isinstance(plugin, checkpointing.Checkpointer):
        pg.logging.warning(
            'Built-in checkpointing is enabled on MultiProcessParallelRunner. '
            f'Ignoring checkpointer: {plugin!r}.'
        )
      elif plugin.is_per_example():
        worker_plugins.append(pg.Ref(plugin))
      else:
        monitor_plugins.append(pg.Ref(plugin))

    if self.slice_id == 0:
      self._ckpt_monitor = ckpt_monitor.CheckpointMonitor(
          pg.Ref(self.current_run),
          plugins=monitor_plugins,
          monitor_inprogress_files=True,
          checkpoint_pattern=f'checkpoint_*.{self.ckpt_format}',
          max_aggregation_threads=self.max_aggregation_threads,
      )
    else:
      self._ckpt_monitor = None

    self._slice_runner = _SingleSliceRunner(
        current_run=self.current_run.clone(
            override=dict(
                # Clone the experiment to avoid updating the original one.
                experiment=self.current_run.experiment.clone(),
                example_ids=self._examples_to_evaluate,
            )
        ),
        plugins=worker_plugins,
        timeout=self.timeout,
        concurrent_startup_delay=self.concurrent_startup_delay,
    )

  def _examples_to_evaluate(
      self,
      experiment: experiment_lib.Experiment
  ) -> list[int]:
    all_ids = self.current_run.examples_to_evaluate(experiment)
    return [x for x in all_ids if x % self.num_slices == self.slice_id]

  def run(self) -> None:
    if self._ckpt_monitor is not None:
      self._ckpt_monitor.start()
    self._slice_runner.run()
    if self._ckpt_monitor is not None:
      self._ckpt_monitor.join()
