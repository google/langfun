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
"""Progress reporting for evaluation."""

import datetime
import threading
import time
from typing import Annotated, Any
import pyglove as pg


class Progress(pg.Object, pg.views.HtmlTreeView.Extension):
  """Evaluation progress."""

  num_total: Annotated[
      int | None,
      (
          'Total number of items to be processed. '
          'If None, the progress is not started.'
      )
  ] = None
  num_processed: Annotated[
      int,
      (
          'Number of items that have been processed without errors.'
      )
  ] = 0
  num_failed: Annotated[
      int,
      (
          'Number of items that have failed.'
      )
  ] = 0
  num_skipped: Annotated[
      int,
      (
          'Number of items that have been skipped.'
      )
  ] = 0
  start_time: Annotated[
      float | None,
      (
          'The start time of the progress. '
          'If None, the progress is not started.'
      )
  ] = None
  stop_time: Annotated[
      float | None,
      (
          'The stop time of the progress. '
          'If None, the progress is not stopped.'
      )
  ] = None
  execution_summary: Annotated[
      pg.object_utils.TimeIt.StatusSummary,
      'The execution summary of the progress.'
  ] = pg.object_utils.TimeIt.StatusSummary()

  def _on_bound(self):
    super()._on_bound()
    self._progress_bar = None
    self._time_label = None
    self._lock = threading.Lock()

  def reset(self) -> None:
    """Resets the progress."""
    self._sync_members(
        num_total=None,
        num_processed=0,
        num_failed=0,
        num_skipped=0,
        start_time=None,
        stop_time=None,
        execution_summary=pg.object_utils.TimeIt.StatusSummary(),
    )

  @property
  def num_completed(self) -> int:
    """Returns the number of completed examples."""
    return self.num_processed + self.num_failed + self.num_skipped

  def __float__(self) -> float:
    """Returns the complete rate in range [0, 1]."""
    if self.num_total is None:
      return float('nan')
    return self.num_completed / self.num_total

  @property
  def is_started(self) -> bool:
    """Returns whether the evaluation is started."""
    return self.start_time is not None

  @property
  def is_stopped(self) -> bool:
    """Returns whether the evaluation is stopped."""
    return self.stop_time is not None

  @property
  def is_completed(self) -> bool:
    """Returns whether the evaluation is completed."""
    return (
        self.num_total is not None
        and self.num_completed == self.num_total
    )

  @property
  def is_skipped(self) -> bool:
    """Returns whether the evaluation is skipped."""
    return (
        self.num_total is not None
        and self.num_skipped == self.num_total
    )

  @property
  def is_failed(self) -> bool:
    """Returns whether the evaluation is failed."""
    return (
        self.num_failed > 0
        and self.num_failed + self.num_skipped == self.num_total
    )

  @property
  def elapse(self) -> float | None:
    """Returns the elapse time in seconds."""
    if self.start_time is None:
      return None
    if self.stop_time is None:
      return time.time() - self.start_time
    return self.stop_time - self.start_time

  @property
  def start_time_str(self) -> str | None:
    """Returns the start time string of the evaluation."""
    if self.start_time is None:
      return None
    return time.strftime('%Y/%m/%d %H:%M:%S', time.localtime(self.start_time))

  @property
  def stop_time_str(self) -> str | None:
    """Returns the complete time string of the evaluation."""
    if self.stop_time is None:
      return None
    return time.strftime(
        '%Y/%m/%d %H:%M:%S', time.localtime(self.stop_time)
    )

  def start(self, total: int) -> None:
    """Marks the evaluation as started."""
    assert self.start_time is None, self
    self._sync_members(start_time=time.time(), num_total=total)
    if self._progress_bar is not None:
      self._progress_bar.update(total=total)
    self._update_time_label()

  def stop(self) -> None:
    """Marks the evaluation as stopped."""
    assert self.stop_time is None, self
    self._sync_members(stop_time=time.time())
    self._update_time_label()

  def _sync_members(self, **kwargs: Any):
    """Synchronizes the members of the progress."""
    self.rebind(
        **kwargs,
        skip_notification=True,
        raise_on_no_change=False,
    )

  def increment_processed(self, delta: int = 1) -> None:
    """Updates the number of processed examples."""
    assert self.is_started and not self.is_stopped, self
    with self._lock:
      self._sync_members(num_processed=self.num_processed + delta)
    if self._progress_bar is not None:
      self._progress_bar['Processed'].increment(delta)
    self._update_time_label()

  def increment_failed(self, delta: int = 1) -> None:
    """Updates the number of failed examples."""
    assert self.is_started and not self.is_stopped, self
    with self._lock:
      self._sync_members(num_failed=self.num_failed + delta)
    if self._progress_bar is not None:
      self._progress_bar['Failed'].increment(delta)
    self._update_time_label()

  def increment_skipped(self, delta: int = 1) -> None:
    """Updates the number of skipped examples."""
    assert self.is_started and not self.is_stopped, self
    with self._lock:
      self._sync_members(num_skipped=self.num_skipped + delta)
    if self._progress_bar is not None:
      self._progress_bar['Skipped'].increment(delta)
    self._update_time_label()

  def update_execution_summary(
      self,
      execution_status: dict[str, pg.object_utils.TimeIt.Status]
  ) -> None:
    """Updates the execution summary of the progress."""
    with self._lock:
      self.execution_summary.aggregate(execution_status)

  def _sym_nondefault(self) -> dict[str, Any]:
    """Overrides nondefault values so volatile values are not included."""
    return dict()

  #
  # HTML view.
  #

  def _duration_text(self) -> str:
    if self.start_time is None:
      return '00:00:00'
    return str(datetime.timedelta(seconds=self.elapse)).split('.')[0]

  def _time_tooltip(self) -> pg.Html.WritableTypes:
    time_info = pg.Dict(
        duration=self._duration_text(),
        last_update=(
            time.strftime(    # pylint: disable=g-long-ternary
                '%Y/%m/%d %H:%M:%S',
                time.localtime(time.time())
            ) if not self.is_stopped else self.stop_time_str
        ),
        start_time=self.start_time_str,
        stop_time=self.stop_time_str,
    )
    if self.execution_summary:
      time_info['execution'] = pg.Dict(
          {
              k: pg.Dict(
                  num_started=v.num_started,
                  num_ended=v.num_ended,
                  num_failed=v.num_failed,
                  avg_duration=round(v.avg_duration, 2),
              ) for k, v in self.execution_summary.breakdown.items()
          }
      )
    return pg.format(time_info, verbose=False)

  def _html_tree_view(
      self,
      *,
      view: pg.views.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs
  ) -> pg.Html:
    """Renders the content of the progress bar."""
    def _progress_bar():
      return pg.views.html.controls.ProgressBar(
          [
              pg.views.html.controls.SubProgress(
                  name='Skipped', value=self.num_skipped,
              ),
              pg.views.html.controls.SubProgress(
                  name='Processed', value=self.num_processed,
              ),
              pg.views.html.controls.SubProgress(
                  name='Failed', value=self.num_failed,
              ),
          ],
          total=self.num_total,
          interactive=interactive,
      )

    def _time_label():
      css_class = 'not-started'
      if self.is_started and not self.is_stopped:
        css_class = 'started'
      elif self.is_stopped:
        css_class = 'stopped'
      return pg.views.html.controls.Label(
          self._duration_text(),
          tooltip=self._time_tooltip(),
          css_classes=[
              'progress-time', css_class
          ],
          interactive=interactive,
      )

    extra_flags = extra_flags or {}
    interactive = extra_flags.pop('interactive', True)
    if interactive:
      if self._progress_bar is None:
        self._progress_bar = _progress_bar()
      if self._time_label is None:
        self._time_label = _time_label()
      progress_bar = self._progress_bar
      time_label = self._time_label
    else:
      progress_bar = _progress_bar()
      time_label = _time_label()
    return pg.Html.element(
        'div', [progress_bar, time_label], css_classes=['eval-progress'],
    )

  def _update_time_label(self):
    """Updates the time label of the progress."""
    if self._time_label is None:
      return
    self._time_label.update(
        text=self._duration_text(),
        tooltip=self._time_tooltip(),
        styles=dict(
            color=(
                'dodgerblue' if self.is_started
                and not self.is_stopped else '#ccc'
            ),
        ),
    )

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .eval-progress {
          display: inline-block;
        }
        .sub-progress.skipped {
          background-color:yellow;
        }
        .sub-progress.processed {
          background-color:#00B000;
        }
        .sub-progress.failed {
          background-color:red;
        }
        .progress-time {
          font-weight: normal;
          margin-left: 10px;
          border-radius: 5px;
          color: #CCC;
          padding: 5px;
        }
        """
    ]
