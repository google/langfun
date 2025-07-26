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
"""Structures for location objects in an image."""

import math
import random
from typing import Optional, Union
import pyglove as pg


class Coordinate(pg.Object):
  """A coordinate in a 2D image."""
  x: int
  y: int

  def as_tuple(self) -> tuple[int, int]:
    """Returns the coordinate as a tuple."""
    return self.x, self.y

  def __add__(
      self, other: Union['Coordinate', tuple[int, int]]
  ) -> 'Coordinate':
    """Returns the coordinate plus the other coordinate."""
    other = Coordinate.from_value(other)
    return Coordinate(self.x + other.x, self.y + other.y)

  def __radd__(
      self, other: Union['Coordinate', tuple[int, int]]
  ) -> 'Coordinate':
    """Returns the coordinate plus the other coordinate."""
    return self + other

  def __sub__(
      self, other: Union['Coordinate', tuple[int, int]]
  ) -> 'Coordinate':
    """Returns the coordinate minus the other coordinate."""
    other = Coordinate.from_value(other)
    return Coordinate(self.x - other.x, self.y - other.y)

  def __rsub__(
      self, other: Union['Coordinate', tuple[int, int]]
  ) -> 'Coordinate':
    """Returns the coordinate minus the other coordinate."""
    other = Coordinate.from_value(other)
    return other - self

  def __mul__(self, ratio: float) -> 'Coordinate':
    """Returns the coordinate multiplied by a ratio."""
    return Coordinate(self.x * ratio, self.y * ratio)

  def __rmul__(self, ratio: float) -> 'Coordinate':
    """Returns the coordinate multiplied by a ratio."""
    return self * ratio

  @classmethod
  def random(cls,
             bound: 'BBox',
             rand: random.Random | None = None) -> 'Coordinate':
    """Generates a random coordinate.

    Args:
      bound: The bounding box within which the coordinate will be generated.
      rand: The random number generator to use. If None, the default random
        number generator will be used.

    Returns:
      A random coordinate.
    """
    rand = rand or random
    x = rand.randint(bound.left, bound.right)
    y = rand.randint(bound.top, bound.bottom)
    return cls(x, y)

  def distance_to(self, point: 'Coordinate') -> float:
    """Returns the distance to the point."""
    return math.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

  @classmethod
  def from_value(
      cls, value: Union[tuple[int, int], 'Coordinate']) -> 'Coordinate':
    """Creates a coordinate from a tuple or coordinate."""
    if isinstance(value, tuple):
      if len(value) == 2:
        return cls(*value)
      else:
        raise ValueError(f'Invalid tuple size: {len(value)}')
    assert isinstance(value, Coordinate), value
    return value


class BBox(pg.Object):
  """A bounding box in a 2D image."""
  x: int
  y: int
  right: int
  bottom: int

  def _on_bound(self):
    super()._on_bound()
    assert self.left < self.right, self
    assert self.top < self.bottom, self

  @property
  def left(self) -> int:
    """Returns the left coordinate of the bounding box."""
    return self.x

  @property
  def top(self) -> int:
    """Returns the top coordinate of the bounding box."""
    return self.y

  @property
  def width(self) -> int:
    """Returns the width of the bounding box."""
    return self.right - self.left

  @property
  def height(self) -> int:
    """Returns the height of the bounding box."""
    return self.bottom - self.top

  @property
  def center(self) -> Coordinate:
    """Returns the center of the bounding box."""
    return Coordinate(
        (self.left + self.right) // 2, (self.top + self.bottom) // 2)

  @property
  def area(self) -> int:
    """Returns the area of the bounding box."""
    return self.width * self.height

  @property
  def top_left(self) -> Coordinate:
    """Returns the top left corner of the bounding box."""
    return Coordinate(self.left, self.top)

  @property
  def bottom_right(self) -> Coordinate:
    """Returns the bottom right corner of the bounding box."""
    return Coordinate(self.right, self.bottom)

  def clip(self, image_size: tuple[int, int]) -> Optional['BBox']:
    """Clips the bounding box to the given image size.

    Args:
      image_size: The image size (width, height).

    Returns:
      The clipped bounding box.
    """
    width, height = image_size
    x = min(max(0, self.x), width)
    y = min(max(0, self.y), height)
    right = min(max(0, self.right), width)
    bottom = min(max(0, self.bottom), height)
    try:
      return BBox(x, y, right, bottom)
    except AssertionError:
      return None

  def __contains__(
      self,
      v: Union[
          tuple[int, int],
          tuple[int, int, int, int],
          Coordinate,
          'BBox'
      ]) -> bool:
    """Contains operator."""
    if isinstance(v, tuple):
      if len(v) == 2:
        v = Coordinate(*v)
      elif len(v) == 4:
        v = BBox(*v)
      else:
        raise ValueError(f'Invalid tuple size: {len(v)}')
    if isinstance(v, Coordinate):
      return self.left <= v.x <= self.right and self.top <= v.y <= self.bottom
    elif isinstance(v, BBox):
      return (
          v.left >= self.left
          and v.right <= self.right
          and v.top >= self.top
          and v.bottom <= self.bottom
      )
    else:
      raise ValueError(f'Invalid type: {type(v)}')

  def intersects(self, other: 'BBox') -> bool:
    """Returns the intersection of two bounding boxes."""
    return not (self.right < other.left or
                self.left > other.right or
                self.bottom < other.top or
                self.top > other.bottom)

  def as_tuple(self) -> tuple[int, int, int, int]:
    """Returns the bounding box as a tuple."""
    return self.x, self.y, self.right, self.bottom

  @classmethod
  def random(
      cls,
      bound: 'BBox',
      min_width: int = 30,
      max_width: int = 200,
      min_height: int = 30,
      max_height: int = 200,
      rand: random.Random | None = None
  ) -> 'BBox':
    """Generates a random bounding box.

    Args:
      bound: The bounding box within which the random bounding box will be
        generated.
      min_width: The minimum width of the random bounding box.
      max_width: The maximum width of the random bounding box.
      min_height: The minimum height of the random bounding box.
      max_height: The maximum height of the random bounding box.
      rand: The random number generator to use. If None, the default random
        number generator will be used.

    Returns:
      A random bounding box.
    """
    assert max_width >= min_width, (min_width, max_width)
    assert max_height >= min_height, (min_height, max_height)

    if min_width > bound.width or min_height > bound.height:
      raise ValueError('Minimum width or height is larger than the bound.')

    rand = rand or random

    max_width = min(max_width, bound.width)
    max_height = min(max_height, bound.height)

    width = rand.randint(min_width, max_width)
    height = rand.randint(min_height, max_height)

    max_left = bound.right - width
    max_top = bound.bottom - height

    left = rand.randint(bound.left, max_left)
    top = rand.randint(bound.top, max_top)

    right = left + width
    bottom = top + height

    return cls(left, top, right, bottom)

  def matches(
      self,
      other: 'BBox',
      area_diff_threshold: float = 5.0,
      max_center_distance: float = 100
  ) -> bool:
    """Returns whether the bounding boxes match."""
    if self.area == 0 or other.area == 0:
      return False

    return (
        abs(self.area - other.area) / self.area < area_diff_threshold
        and self.center.distance_to(other.center) < max_center_distance
    )

  def expand(
      self,
      width_scale: float = 1.0,
      height_scale: float = 1.0,
  ) -> 'BBox':
    """Expands the bounding box from the center."""
    new_width = int(self.width * width_scale)
    new_height = int(self.height * height_scale)
    new_x = self.x - (new_width - self.width) // 2
    new_y = self.y - (new_height - self.height) // 2
    return BBox(new_x, new_y, new_x + new_width, new_y + new_height)
