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
"""Tracking evaluation run progress."""

import langfun.core as lf
from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import experiment as experiment_lib
import pyglove as pg

Runner = experiment_lib.Runner
Experiment = experiment_lib.Experiment
Example = example_lib.Example


def progress_tracker(tqdm: bool = False) -> experiment_lib.Plugin:
  """Creates a progress tracker as a plugin.

  Args:
    tqdm: If True, force using tqdm for progress update.

  Returns:
    The progress tracker plugin.
  """
  if tqdm or not lf.console.under_notebook():
    return _TqdmProgressTracker()
  else:
    return _HtmlProgressTracker()


class _HtmlProgressTracker(experiment_lib.Plugin):
  """HTML progress tracker plugin."""

  def on_run_start(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    # Display the experiment if running under notebook and not using tqdm.
    assert lf.console.under_notebook()
    with pg.view_options(
        collapse_level=None,
        extra_flags=dict(
            current_run=runner.current_run,
            interactive=True,
        )
    ):
      lf.console.display(runner.current_run.experiment)


ProgressBarId = int


class _TqdmProgressTracker(experiment_lib.Plugin):
  """Tqdm process updater plugin."""

  def _on_bound(self):
    super()._on_bound()
    self._overall_progress: ProgressBarId | None = None
    self._leaf_progresses: dict[str, ProgressBarId] = {}

  def experiment_progress(
      self, experiment: Experiment) -> lf.concurrent.ProgressBar:
    """Returns the progress of the experiment."""
    assert experiment.is_leaf
    return self._leaf_progresses[experiment.id]

  def on_run_start(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    """Called when a runner is started."""
    self._overall_progress = lf.concurrent.ProgressBar.install(
        label='All', total=len(root.leaf_nodes), color='blue'
    )
    self._leaf_progresses = {
        leaf.id: lf.concurrent.ProgressBar.install(
            label=f'[#{i + 1} - {leaf.id}]',
            total=(len(runner.current_run.example_ids)
                   if runner.current_run.example_ids else leaf.num_examples),
            color='cyan',
            status=None
        )
        for i, leaf in enumerate(root.leaf_nodes)
    }
    summary_link = Experiment.link(
        runner.current_run.output_path_for(root, 'summary.html')
    )
    lf.console.write(f'Summary: {summary_link}.', color='green')

  def on_run_complete(
      self,
      runner: Runner,
      root: Experiment
  ) -> None:
    """Called when a runner is complete."""
    lf.concurrent.ProgressBar.update(
        self._overall_progress,
        color='green',
        status='ALL COMPLETED.',
    )
    lf.concurrent.ProgressBar.uninstall(self._overall_progress)
    self._overall_progress = None
    for progress in self._leaf_progresses.values():
      lf.concurrent.ProgressBar.uninstall(progress)
    self._leaf_progresses = {}

  def on_experiment_start(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    """Called when an evaluation is started."""

  def on_experiment_skipped(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    """Called when an evaluation is skipped."""
    if experiment.is_leaf:
      lf.concurrent.ProgressBar.update(
          self.experiment_progress(experiment),
          delta=experiment.progress.num_total,
          status='Skipped.',
      )
      lf.concurrent.ProgressBar.update(
          self._overall_progress,
          status=f'Skipped {experiment.id}.',
      )

  def on_experiment_complete(
      self,
      runner: Runner,
      experiment: Experiment
  ) -> None:
    """Called when an evaluation is complete."""
    if experiment.is_leaf:
      lf.concurrent.ProgressBar.update(
          self.experiment_progress(experiment),
          color='green',
      )
      lf.concurrent.ProgressBar.update(
          self._overall_progress,
          delta=1,
          status=f'{experiment.id} COMPLETED.',
      )

  def on_example_start(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example
  ) -> None:
    """Called when an evaluation example is started."""

  def on_example_skipped(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example
  ) -> None:
    """Called when an evaluation example is skipped."""
    del runner, example
    lf.concurrent.ProgressBar.update(
        self.experiment_progress(experiment),
        delta=1,
    )

  def on_example_complete(
      self,
      runner: Runner,
      experiment: Experiment,
      example: Example
  ) -> None:
    """Called when an evaluation example is complete."""
    lf.concurrent.ProgressBar.update(
        self.experiment_progress(experiment),
        delta=1,
        status=self.status(experiment),
    )

  def status(self, experiment: Experiment) -> str:
    """Returns the progress text of the evaluation."""
    items = []
    for metric in experiment.metrics:
      for metric_value in metric.values():
        items.append(
            f'{metric_value.sym_path.key}={metric_value.format(verbose=True)}'
        )
    error_tags = {}
    for entry in experiment.progress.execution_summary.breakdown.values():
      error_tags.update(entry.error_tags)

    if error_tags:
      items.extend(
          [f'{k}={v}' for k, v in error_tags.items() if v > 0]
      )
    return ', '.join(items)
