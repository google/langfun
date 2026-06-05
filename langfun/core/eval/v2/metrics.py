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
  """Interface for an evaluation metric.

  A metric is used to evaluate the quality of the outputs produced by an
  evaluation. It works by auditing each processed example via its `audit`
  method, which in turn calls the user-overridable `_audit` method to perform
  metric-specific logic and update metric values. Metrics can compute multiple
  values (e.g., precision, recall, F1 score) which are exposed via the
  `values` method.
  """

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

  def update(
      self,
      example: example_lib.Example,
      force_recompute: bool = False
  ) -> dict[str, Any]:
    """Updates metric values with a processed example.

    Args:
      example: The processed example.
      force_recompute: Whether to force recompute the metric metadata even if
        they are already present.

    Returns:
      A dict of metric metadata.
    """
    if (force_recompute
        or example.metric_metadata is None
        or self.name not in example.metric_metadata):
      metadata = self.compute_metric_metadata(example)
    else:
      metadata = example.metric_metadata[self.name]
    self.update_metric_values(example.id, metadata)
    self._update_view()
    return metadata

  @abc.abstractmethod
  def compute_metric_metadata(
      self, example: example_lib.Example
  ) -> dict[str, Any]:
    """Subclasses should override this method to implement the metric logic."""

  @abc.abstractmethod
  def update_metric_values(
      self, example_id: int, metric_metadata: dict[str, Any]
  ) -> None:
    """Update metric values based on metric metadata."""

  @abc.abstractmethod
  def values(self) -> list[metric_values.MetricValue]:
    """Returns all the values computed by this metric."""

  def reset(self) -> None:
    """Resets the metric values."""
    for v in self.values():
      v.reset()

  def merge_from(self, other: 'Metric') -> 'Metric':
    """Merges the values from another metric."""
    for v1, v2 in zip(self.values(), other.values()):
      v1.merge_from(v2)
    return self

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
  """Base class for common metrics.

  `MetricBase` provides common functionalities for metrics, such as automatic
  error counting based on whether an example has an error during evaluation.
  It distinguishes between Object-Oriented Programming (OOP) errors
  (e.g. `MappingError` during structured output generation) and other errors.
  Subclasses should implement `_audit_processed` for metric computation on
  successfully processed examples.
  """

  oop_errors: Rate | None = Rate()
  non_oop_errors: Rate | None = Rate()

  def _on_bound(self) -> None:
    super()._on_bound()
    self._error_breakdown = collections.defaultdict(list)

  def reset(self) -> None:
    """Resets the metric."""
    super().reset()
    self._error_breakdown = collections.defaultdict(list)

  def compute_metric_metadata(
      self, example: example_lib.Example
  ) -> dict[str, Any]:
    """Computes the metric metadata for the example."""
    if example.error is None:
      return self._compute_metric_metadata(example)
    return self._compute_metric_metadata_with_processing_error(example)

  def update_metric_values(
      self,
      example_id: int,
      metric_metadata: dict[str, Any]
  ) -> None:
    """Collects the metric metadata."""
    # NOTE(daiyip): the metric values are being updated concurrently, so we
    # uses a lock to avoid race condition. We might consider relaxing the lock
    # later if metric auditing becomes a bottleneck.
    with self._lock:
      for v in self.values():
        v.increment_total()

    if 'error' in metric_metadata:
      self._update_metric_values_with_processing_error(
          example_id, metric_metadata
      )
    else:
      self._update_metric_values(example_id, metric_metadata)

  @abc.abstractmethod
  def _compute_metric_metadata(
      self,
      example: example_lib.Example
  ) -> dict[str, Any]:
    """Computes the metric metadata for the example."""

  def _compute_metric_metadata_with_processing_error(
      self,
      example: example_lib.Example
  ) -> dict[str, Any]:
    """Audits the evaluation example after processing."""
    assert example.error is not None
    return dict(error=example.error.tag)

  @abc.abstractmethod
  def _update_metric_values(self, metadata: dict[str, Any]) -> None:
    """Update metric values based metric metadata."""

  def _update_metric_values_with_processing_error(
      self,
      example_id: int,
      metric_metadata: dict[str, Any]
  ) -> None:
    """Updates metric values with processing error."""
    error_tag = metric_metadata.get('error')
    assert error_tag is not None, (example_id, metric_metadata)
    self._error_breakdown[error_tag].append(example_id)
    if error_tag.startswith('MappingError'):
      self.oop_errors.add(example_id, 1)
    else:
      self.non_oop_errors.add(example_id, 1)
    self._error_breakdown[error_tag].append(example_id)

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
  """Metric for matching outputs against ground truth.

  This metric computes match and mismatch rates by comparing the output of
  an example with its ground truth. By default, it looks for a `groundtruth`
  attribute in `example.input` for comparison. Users can customize this behavior
  by subclassing `Match` and overriding the `match` method.
  """

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

  def _compute_metric_metadata(
      self, example: example_lib.Example
  ) -> dict[str, Any]:
    """Computes the metric metadata for the example."""
    metadata = {}
    is_correct = self.match(example.input, example.output)
    if isinstance(is_correct, tuple):
      is_correct, metadata = is_correct

    metadata['is_correct'] = is_correct
    return metadata

  def _update_metric_values(
      self, example_id: int, metadata: dict[str, Any]
  ) -> None:
    """Update metric values based metric metadata."""
    is_correct = metadata.get('is_correct')
    assert is_correct is not None, (example_id, metadata)
    if is_correct:
      self.matches.add(example_id, 1)
    else:
      assert not is_correct
      self.mismatches.add(example_id, 1)

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
  """Base class for scoring metrics.

  `Score` is a base class for metrics that assign a numerical score to each
  example's output (e.g., evaluating quality on a scale of 1-5).
  It automatically computes the average score across all examples.
  Subclasses must implement the `score` method to define how an example
  should be scored.
  """

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

  def _compute_metric_metadata(
      self, example: example_lib.Example
  ) -> dict[str, Any]:
    """Computes the metric metadata for the example."""
    metadata = {}
    score = self.score(example.input, example.output)
    if isinstance(score, tuple):
      score, metadata = score
    metadata['score'] = score
    return metadata

  def _update_metric_values(
      self, example_id: int, metadata: dict[str, Any]
  ) -> None:
    """Update metric values based metric metadata."""
    score = metadata.get('score')
    assert score is not None, (example_id, metadata)
    self.average_score.add(example_id, score)

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
