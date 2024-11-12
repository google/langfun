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
"""Common metrics for Langfun evaluation."""


import abc
import collections
import threading
from typing import Annotated, Any

from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import metric_values
import pyglove as pg


Rate = metric_values.Rate
Average = metric_values.Average


class Metric(pg.Object, pg.views.HtmlTreeView.Extension):
  """Interface for an evaluation metric."""

  name: Annotated[
      str,
      (
          'Name of the metric, which will be used as the key in the dict '
          'returned by `Experiment.metric_values()`'
      )
  ]

  def _on_bound(self):
    super()._on_bound()
    self._label_group = None
    self._lock = threading.Lock()

  def audit(self, example: example_lib.Example) -> dict[str, Any]:
    """Audits a processed example and returns metric metadata for it."""
    # NOTE(daiyip): the metric values are being updated concurrently, so we
    # uses a lock to avoid race condition. We might consider relaxing the lock
    # later if metric auditing becomes a bottleneck.
    with self._lock:
      for v in self.values():
        v.increment_total()

      metadata = self._audit(example)

      self._update_view()
      return metadata

  @abc.abstractmethod
  def _audit(self, example: example_lib.Example) -> dict[str, Any]:
    """Subclasses should override this method to implement the metric logic."""

  @abc.abstractmethod
  def values(self) -> list[metric_values.MetricValue]:
    """Returns all the values computed by this metric."""

  def reset(self) -> None:
    """Resets the metric values."""
    for v in self.values():
      v.reset()

  def _update_view(self):
    """Refreshes the metric values."""
    if self._label_group is None:
      return

    for label, value in zip(self._label_group.labels, self.values()):
      label.update(
          text=self._metric_value_text(value),
          tooltip=self._metric_value_tooltip(value),
      )

  def _metric_value_text(self, metric_value: metric_values.MetricValue) -> str:
    """Returns the label text for the metric value."""
    return str(metric_value)

  def _metric_value_tooltip(
      self, metric_value: metric_values.MetricValue) -> str:
    """Returns the label text for the metric value."""
    with pg.str_format(verbose=True):
      return f'{metric_value.sym_path.key}: {metric_value}'

  def _metric_label_text(self) -> str:
    return ''.join(
        c for c in self.__class__.__name__
        if c.isalnum() and not c.islower()
    )

  def _metric_label_tooltip(self) -> str:
    return self.__class__.__type_name__

  def _html_tree_view(
      self,
      *,
      view: pg.views.HtmlTreeView,
      extra_flags: dict[str, Any] | None = None,
      **kwargs,
  ) -> pg.Html:
    """Renders the content of the metric value."""
    extra_flags = extra_flags or {}
    interactive = extra_flags.get('interactive', True)
    label_group = self._label_group
    if label_group is None:
      label_group = pg.views.html.controls.LabelGroup(
          [
              pg.views.html.controls.Label(
                  self._metric_value_text(mv),
                  tooltip=self._metric_value_tooltip(mv),
                  css_classes=[mv.sym_path.key, 'metric-value'],
                  interactive=interactive,
              ) for mv in self.values()
          ],
          name=pg.views.html.controls.Label(
              self._metric_label_text(),
              tooltip=self._metric_label_tooltip(),
              css_classes=[
                  'metric-name',
                  pg.object_utils.camel_to_snake(self.__class__.__name__, '-')
              ],
              interactive=False,
          ),
          css_classes=['metric-container'],
      )
      if interactive:
        self._label_group = label_group
    return label_group.to_html()

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .metric-container {
          display: inline-flex;
          overflow: hidden;
          border-radius: 5px;
          border: 0px;
          margin: 5px;
          padding: 0px;
        }
        .metric-container .label-container {
          vertical-align: middle;
        }
        .metric-value.oop_errors {
          color: magenta;
          background-color: #f9e6eb;
        }
        .metric-value.non_oop_errors {
          color: red;
          background-color: #fdcccc;
        }
        """
    ]

#
# Common metrics.
#


class MetricBase(Metric):
  """Base class for common metrics."""

  oop_errors: Rate | None = Rate()
  non_oop_errors: Rate | None = Rate()

  def _on_bound(self) -> None:
    super()._on_bound()
    self._error_breakdown = collections.defaultdict(list)

  def reset(self) -> None:
    """Resets the metric."""
    super().reset()
    self._error_breakdown = collections.defaultdict(list)

  def _audit(self, example: example_lib.Example) -> dict[str, Any]:
    """Audits the evaluation example after processing."""
    if example.error is None:
      return self._audit_processed(example)
    else:
      return self._audit_error(example)

  def _audit_error(self, example: example_lib.Example) -> dict[str, Any]:
    """Audits the evaluation example after processing."""
    assert example.error is not None
    tag = example.error.tag
    if tag.startswith('MappingError'):
      self.oop_errors.add(example.id, 1)
    else:
      self.non_oop_errors.add(example.id, 1)
    self._error_breakdown[tag].append(example.id)
    return dict(error=tag)

  @abc.abstractmethod
  def _audit_processed(self, example: example_lib.Example) -> dict[str, Any]:
    """Audits the evaluation example after processing."""

  def _oop_errors_breakdown(self) -> str | None:
    """Returns the OOP error breakdown as a string."""
    return '\n'.join(
        [
            f'- {k}: {len(v)}' for k, v in self._error_breakdown.items()
            if k.startswith('MappingError')
        ]
    ) or None

  def _non_oop_errors_breakdown(self) -> str | None:
    """Returns the non-OOP error breakdown as a string."""
    return '\n'.join(
        [
            f'- {k}: {len(v)}' for k, v in self._error_breakdown.items()
            if not k.startswith('MappingError')
        ]
    ) or None

  def _sym_nondefault(self) -> dict[str, Any]:
    """Overrides nondefault valuesso volatile values are not included."""
    return dict()


class Match(MetricBase):
  """Metric for matching outputs against groundtruth."""

  name = 'match'
  matches: Rate = Rate()
  mismatches: Rate = Rate()

  def match(
      self, example_input: Any, output: Any
  ) -> bool | tuple[bool, dict[str, Any]]:
    """Returns whether the output matches the groundtruth from the example.

    Args:
      example_input: The example input which contains the groundtruth.
      output: The output to match against.

    Returns:
      True if the output matches the groundtruth, False otherwise.
      Or a tuple of (match, metadata).
    """
    groundtruth = getattr(example_input, 'groundtruth', pg.MISSING_VALUE)
    if pg.MISSING_VALUE == groundtruth:
      raise ValueError(
          f'`groundtruth` is not present in the example ({example_input}). '
          'Please subclassing `Match` and override the `match` method to '
          'support custom example format.'
      )
    return pg.eq(output, groundtruth)

  def _audit_processed(self, example: example_lib.Example) -> dict[str, Any]:
    """Audits the evaluation example after processing."""
    metadata = {}
    is_match = self.match(example.input, example.output)
    if isinstance(is_match, tuple):
      is_match, metadata = is_match
    if is_match:
      self.matches.add(example.id, 1)
      metadata['match'] = True
    else:
      self.mismatches.add(example.id, 1)
      metadata['mismatch'] = True
    return metadata

  def values(self) -> list[metric_values.MetricValue]:
    """Returns all the values computed by this metric."""
    return [
        self.matches,
        self.mismatches,
        self.oop_errors,
        self.non_oop_errors
    ]

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .metric-name.match {
          padding: 5px;
          color: white;
          background-color: purple;
        }
        .metric-value.matches {
          color: green;
          background-color: #dcefbe;
        }
        .metric-value.mismatches {
          color: orange;
          background-color: #ffefc4;
        }
        """
    ]


class Score(MetricBase):
  """Base class for scoring."""

  name = 'score'
  average_score: Average = Average()

  @abc.abstractmethod
  def score(
      self,
      example_input: Any,
      output: Any) -> float | tuple[float, dict[str, Any]]:
    """Returns the score based on the example and output.

    Args:
      example_input: The example input based on which the output is generated.
      output: The output to score.

    Returns:
      A float score. Or a tuple of (score, metadata).
    """

  def _audit_processed(self, example: example_lib.Example) -> dict[str, Any]:
    """Audits the evaluation example after processing."""
    metadata = {}
    score = self.score(example.input, example.output)
    if isinstance(score, tuple):
      score, metadata = score
    self.average_score.add(example.id, score)
    metadata['score'] = score
    return metadata

  def values(self) -> list[metric_values.MetricValue]:
    """Returns all the values computed by this metric."""
    return [
        self.average_score,
        self.oop_errors,
        self.non_oop_errors
    ]

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        .metric-name.score {
          padding: 5px;
          color: white;
          background-color: blue;
        }
        .metric-value.average_score {
          color: blue;
          background-color: #b0c7f6;
        }
        """
    ]
