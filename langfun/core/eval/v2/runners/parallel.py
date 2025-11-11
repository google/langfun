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
import random
import threading
import time

from typing import Annotated, Iterator
import langfun.core as lf
from langfun.core.eval.v2.runners import base


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
        max_workers=evaluation.max_workers,
        timeout=self.timeout,
        silence_on_errors=None,
    ):
      pass
