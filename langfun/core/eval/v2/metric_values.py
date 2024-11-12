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
"""Common value types for evaluation metrics and metadata."""


import abc
from typing import Annotated, Any, Union
import pyglove as pg


class MetricValue(pg.Object):
  """Base class for metric values."""

  class DataPoint(pg.Object):
    """A data point for a metric value."""
    example_id: int
    value: float
    weight: float = 1.0

  # NOTE(daiyip): For evaluations, usually the number of examples is within 10K,
  # therefore it's beneficial to store all accumulated values with their example
  # IDs so we are able to track the individual examples that contributed to this
  # metric value. If this premise changes, we might consider using a more
  # efficient data structure.
  data_points: Annotated[
      list[DataPoint],
      'Accumulated computed values with example IDs and weights.'
  ] = []

  total: Annotated[
      int,
      'The total number of examples being evaluated. Including errors.'
  ] = 0

  def _on_bound(self):
    super()._on_bound()
    self._weighted_sum = sum(dp.value * dp.weight for dp in self.data_points)

  def reset(self) -> None:
    """Resets the value to its initial state."""
    self._sync_members(data_points=[], total=0)
    self._weighted_sum = 0.0

  def _sync_members(self, **kwargs) -> None:
    """Synchronizes the members of this object."""
    self.rebind(**kwargs, skip_notification=True, raise_on_no_change=False)

  def __float__(self) -> float:
    """Returns the float representation of this object."""
    if self.total == 0:
      return float('nan')
    return self.reduce()

  @abc.abstractmethod
  def reduce(self) -> float:
    """Reduces the accumulated values into a single value."""

  def increment_total(self, delta: int = 1) -> 'MetricValue':
    """Increments the total number of examples being evaluated."""
    self._sync_members(total=self.total + delta)
    return self

  def add(
      self,
      example_id: int,
      value: float,
      weight: float = 1.0,
      increment_total: bool = False,
  ) -> 'MetricValue':
    """Adds a value to the accumulated values."""
    self._weighted_sum += value * weight
    with pg.notify_on_change(False), pg.allow_writable_accessors(True):
      self.data_points.append(
          MetricValue.DataPoint(example_id, value, weight)
      )
      if increment_total:
        self.increment_total()
    return self

  def __gt__(self, other: Union['MetricValue', float]) -> bool:
    if isinstance(other, self.__class__):
      return float(self) > float(other)
    return float(self) > other

  def __lt__(self, other: Union['MetricValue', float]) -> bool:
    if isinstance(other, self.__class__):
      return float(self) < float(other)
    return float(self) < other

  def __eq__(self, other: Union['MetricValue', float]) -> bool:
    if isinstance(other, self.__class__):
      return super().__eq__(other)
    return float(self) == other

  def __nonzero__(self) -> bool:
    return float(self) != 0

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      *args,
      **kwargs
  ) -> str:
    if compact:
      return super().format(compact, *args, **kwargs)
    if self.total == 0:
      return 'n/a'
    if verbose:
      return (
          f'{self.scalar_repr()} ({len(self.data_points)}/{self.total})'
      )
    return self.scalar_repr()

  @abc.abstractmethod
  def scalar_repr(self) -> str:
    """Returns the format string for the value."""

  def _sym_nondefault(self) -> dict[str, Any]:
    """Overrides nondefault valuesso volatile values are not included."""
    return dict()


class Rate(MetricValue):
  """Representing a rate in range [0, 1]."""

  def reduce(self) -> float:
    return self._weighted_sum / self.total

  def scalar_repr(self):
    if self.total == 0:
      return 'n/a'
    return f'{self.reduce():.1%}'


class Average(MetricValue):
  """Average of a aggregated values."""

  def reduce(self) -> float:
    if not self.data_points:
      return float('nan')
    return self._weighted_sum / len(self.data_points)

  def scalar_repr(self):
    return f'{self.reduce():.3f}'
