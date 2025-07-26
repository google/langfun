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

import random
import unittest
from langfun.assistant.capabilities.gui import location


class CoordinateTest(unittest.TestCase):

  def test_basics(self):
    pt = location.Coordinate(1, 2)
    self.assertEqual(pt.x, 1)
    self.assertEqual(pt.y, 2)
    self.assertEqual(pt.as_tuple(), (1, 2))

  def test_random(self):
    bound = location.BBox(0, 0, 10, 10)
    rand = random.Random(0)
    for _ in range(10):
      pt = location.Coordinate.random(bound, rand=rand)
      self.assertIsInstance(pt, location.Coordinate)
      self.assertIn(pt, bound)

  def test_from_value(self):
    pt = location.Coordinate.from_value((1, 2))
    self.assertEqual(pt.x, 1)
    self.assertEqual(pt.y, 2)
    self.assertIs(location.Coordinate.from_value(pt), pt)

  def test_arithmetic(self):
    pt1 = location.Coordinate(1, 2)
    pt2 = location.Coordinate(3, 4)
    self.assertEqual(pt1 + pt2, location.Coordinate(4, 6))
    self.assertEqual(pt2 + pt1, location.Coordinate(4, 6))
    self.assertEqual(pt1 - pt2, location.Coordinate(-2, -2))
    self.assertEqual(pt2 - pt1, location.Coordinate(2, 2))

    pt1 = location.Coordinate(1, 2)
    pt2 = (3, 4)
    self.assertEqual(pt1 + pt2, location.Coordinate(4, 6))
    self.assertEqual(pt2 + pt1, location.Coordinate(4, 6))
    self.assertEqual(pt1 - pt2, location.Coordinate(-2, -2))
    self.assertEqual(pt2 - pt1, location.Coordinate(2, 2))

    self.assertEqual(pt1 * 2, location.Coordinate(2, 4))
    self.assertEqual(2 * pt1, location.Coordinate(2, 4))

  def test_distance_to(self):
    pt1 = location.Coordinate(1, 2)
    pt2 = location.Coordinate(3, 4)
    # compare float values with a tolerance of 1e-6
    self.assertAlmostEqual(pt1.distance_to(pt2), 2.8284271247461903, delta=1e-6)


class BBoxTest(unittest.TestCase):

  def test_invalid_bbox_creation(self):
    with self.assertRaisesRegex(AssertionError, '.*'):
      location.BBox(100, 50, 100, 450)  # Zero width.
    with self.assertRaisesRegex(AssertionError, '.*'):
      location.BBox(100, 50, 300, 50)  # Zero height.
    with self.assertRaisesRegex(AssertionError, '.*'):
      location.BBox(100, 50, 50, 50)  # Zero width and height

  def test_basics(self):
    bbox = location.BBox(100, 50, 300, 450)
    self.assertEqual(bbox.x, 100)
    self.assertEqual(bbox.y, 50)
    self.assertEqual(bbox.right, 300)
    self.assertEqual(bbox.bottom, 450)
    self.assertEqual(bbox.left, 100)
    self.assertEqual(bbox.top, 50)
    self.assertEqual(bbox.width, 200)
    self.assertEqual(bbox.height, 400)

    self.assertEqual(bbox.center, location.Coordinate(200, 250))
    self.assertEqual(bbox.top_left, location.Coordinate(100, 50))
    self.assertEqual(bbox.bottom_right, location.Coordinate(300, 450))
    self.assertEqual(bbox.area, 80000)
    self.assertEqual(bbox.as_tuple(), (100, 50, 300, 450))

  def test_contains(self):
    bbox = location.BBox(100, 50, 300, 450)
    self.assertIn((120, 200), bbox)
    self.assertIn(location.Coordinate(120, 200), bbox)
    self.assertNotIn(location.Coordinate(80, 200), bbox)
    self.assertNotIn(location.Coordinate(120, 20), bbox)
    self.assertNotIn(location.Coordinate(320, 200), bbox)
    self.assertNotIn(location.Coordinate(120, 470), bbox)

    self.assertIn((100, 50, 300, 450), bbox)
    self.assertIn(location.BBox(100, 50, 300, 450), bbox)
    self.assertIn(location.BBox(120, 70, 260, 410), bbox)
    self.assertNotIn(location.BBox(60, 70, 260, 410), bbox)
    self.assertNotIn(location.BBox(120, 45, 260, 410), bbox)
    self.assertNotIn(location.BBox(120, 60, 310, 410), bbox)
    self.assertNotIn(location.BBox(120, 60, 260, 470), bbox)
    self.assertNotIn(location.BBox(0, 0, 400, 500), bbox)

    with self.assertRaisesRegex(ValueError, 'Invalid tuple size'):
      _ = (1, 2, 3) in bbox

    with self.assertRaisesRegex(ValueError, 'Invalid type'):
      _ = 'abc' in bbox  # pytype: disable=unsupported-operands

  def test_intersects(self):
    bbox = location.BBox(100, 50, 300, 450)
    self.assertTrue(bbox.intersects(location.BBox(100, 50, 300, 450)))
    self.assertTrue(bbox.intersects(location.BBox(120, 70, 260, 410)))
    self.assertTrue(bbox.intersects(location.BBox(60, 70, 260, 470)))
    self.assertFalse(bbox.intersects(location.BBox(0, 70, 90, 410)))
    self.assertFalse(bbox.intersects(location.BBox(60, 10, 400, 30)))

  def test_clip(self):
    # BBox within image bounds
    bbox = location.BBox(100, 50, 300, 450)
    clipped_bbox = bbox.clip((400, 500))
    self.assertEqual(clipped_bbox, bbox)

    # BBox exceeds right and bottom bounds
    bbox = location.BBox(100, 50, 500, 600)
    clipped_bbox = bbox.clip((400, 500))
    self.assertEqual(clipped_bbox, location.BBox(100, 50, 400, 500))

    # BBox exceeds left and top bounds
    bbox = location.BBox(-10, -20, 300, 450)
    clipped_bbox = bbox.clip((400, 500))
    self.assertEqual(clipped_bbox, location.BBox(0, 0, 300, 450))

    # BBox larger than the image
    bbox = location.BBox(-10, -20, 800, 700)
    clipped_bbox = bbox.clip((400, 500))
    self.assertEqual(clipped_bbox, location.BBox(0, 0, 400, 500))

    # BBox starts outside the image boundaries (bottom/right)
    bbox = location.BBox(500, 600, 700, 800)
    clipped_bbox = bbox.clip((400, 500))
    self.assertIsNone(clipped_bbox)

  def test_random(self):
    bound = location.BBox(0, 0, 800, 600)
    rand = random.Random(0)
    for _ in range(10):
      bbox = location.BBox.random(bound, rand=rand)
      self.assertIsInstance(bbox, location.BBox)
      self.assertIn(bbox, bound)

    with self.assertRaisesRegex(
        ValueError, 'Minimum width or height is larger than the bound'
    ):
      _ = location.BBox.random(location.BBox(0, 0, 100, 100), min_width=110)

  def test_matches(self):
    bbox1 = location.BBox(100, 50, 300, 450)
    self.assertFalse(bbox1.matches(location.BBox(10, 10, 11, 11)))

    bbox2 = location.BBox(100, 50, 300, 450)
    self.assertTrue(bbox1.matches(bbox2))
    self.assertTrue(bbox2.matches(bbox1))

    bbox3 = location.BBox(110, 40, 290, 450)
    self.assertTrue(bbox1.matches(bbox3))
    self.assertFalse(bbox3.matches(bbox1, area_diff_threshold=0.05))
    self.assertFalse(bbox3.matches(bbox1, max_center_distance=1))

  def test_expand(self):
    # Test case 1: Expanding both width and height.
    bbox = location.BBox(100, 50, 300, 450)
    expanded_bbox = bbox.expand(width_scale=1.2, height_scale=1.1)
    self.assertEqual(expanded_bbox.x, 80)
    self.assertEqual(expanded_bbox.y, 30)
    self.assertEqual(expanded_bbox.right, 320)
    self.assertEqual(expanded_bbox.bottom, 470)
    self.assertEqual(expanded_bbox.width, 240)
    self.assertEqual(expanded_bbox.height, 440)
    self.assertEqual(expanded_bbox.center, location.Coordinate(200, 250))

    # Test case 2: Expanding only width.
    bbox = location.BBox(100, 50, 300, 450)
    expanded_bbox = bbox.expand(width_scale=1.2)
    self.assertEqual(expanded_bbox.x, 80)
    self.assertEqual(expanded_bbox.y, 50)
    self.assertEqual(expanded_bbox.right, 320)
    self.assertEqual(expanded_bbox.bottom, 450)
    self.assertEqual(expanded_bbox.width, 240)
    self.assertEqual(expanded_bbox.height, 400)
    self.assertEqual(expanded_bbox.center, location.Coordinate(200, 250))

    # Test case 3: Expanding only height.
    bbox = location.BBox(100, 50, 300, 450)
    expanded_bbox = bbox.expand(height_scale=1.1)
    self.assertEqual(expanded_bbox.x, 100)
    self.assertEqual(expanded_bbox.y, 30)
    self.assertEqual(expanded_bbox.right, 300)
    self.assertEqual(expanded_bbox.bottom, 470)
    self.assertEqual(expanded_bbox.width, 200)
    self.assertEqual(expanded_bbox.height, 440)
    self.assertEqual(expanded_bbox.center, location.Coordinate(200, 250))

    # Test case 4: Shrinking.
    bbox = location.BBox(100, 50, 300, 450)
    expanded_bbox = bbox.expand(width_scale=0.8, height_scale=0.9)
    self.assertEqual(expanded_bbox.x, 120)
    self.assertEqual(expanded_bbox.y, 70)
    self.assertEqual(expanded_bbox.right, 280)
    self.assertEqual(expanded_bbox.bottom, 430)
    self.assertEqual(expanded_bbox.width, 160)
    self.assertEqual(expanded_bbox.height, 360)
    self.assertEqual(expanded_bbox.center, location.Coordinate(200, 250))

    # Test case 5: No change.
    bbox = location.BBox(100, 50, 300, 450)
    expanded_bbox = bbox.expand(width_scale=1.0, height_scale=1.0)
    self.assertEqual(expanded_bbox, bbox)

if __name__ == '__main__':
  unittest.main()
