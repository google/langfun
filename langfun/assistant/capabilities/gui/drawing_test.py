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

import unittest
import langfun as lf
from langfun.assistant.capabilities.gui import drawing
from langfun.assistant.capabilities.gui import location
from PIL import Image as pil_image


class DrawingTest(unittest.TestCase):

  def test_blank_image(self):
    image = drawing.blank_image((3, 3), (0, 0, 0))
    self.assertIsInstance(image, lf.Image)
    self.assertEqual(image.size, (3, 3))
    self.assertEqual(image.image_format, 'png')
    self.assertIn(b'\x89PNG\r\n\x1a\n', image.to_bytes())
    image2 = drawing.blank_image((3, 3), (0, 0, 0), pil=True)
    self.assertIsInstance(image2, pil_image.Image)
    image2 = lf.Image.from_pil_image(image2)
    self.assertIsInstance(image2, lf.Image)
    self.assertEqual(image2.to_bytes(), image.to_bytes())

  def test_draw_bboxes(self):
    image = drawing.blank_image((5, 5), (0, 0, 0))
    image2 = drawing.draw_bboxes(image, [])
    self.assertIs(image, image2)

    image = drawing.draw_bboxes(
        image,
        [
            location.BBox(0, 0, 3, 3),
            location.BBox(1, 1, 4, 4),
        ],
        line_color='red',
        line_width=1,
    )
    self.assertEqual(image.size, (5, 5))
    self.assertIsInstance(image, lf.Image)

  def test_draw_points(self):
    image = drawing.blank_image((5, 5), (0, 0, 0))
    image2 = drawing.draw_points(image, [])
    self.assertIs(image, image2)

    image = drawing.draw_points(
        image,
        [
            location.Coordinate(1, 1),
            location.Coordinate(2, 2),
        ],
        color='red',
        radius=1,
    )
    self.assertEqual(image.size, (5, 5))
    self.assertIsInstance(image, lf.Image)

  def test_draw_calibration_lines(self):
    image = drawing.blank_image((5, 5), (0, 0, 0))
    image = drawing.draw_calibration_lines(
        image,
        location.Coordinate(2, 2),
        vline_color='red',
        vline_width=1,
        hline_color='green',
        hline_width=1,
    )
    self.assertEqual(image.size, (5, 5))
    self.assertIsInstance(image, lf.Image)

  def test_draw_cursor(self):
    image = drawing.blank_image((50, 50), (255, 0, 0))
    image = drawing.draw_cursor(image, location.Coordinate(10, 10))
    self.assertEqual(image.size, (50, 50))
    self.assertIsInstance(image, lf.Image)

  def test_draw_ref_points(self):
    image = drawing.blank_image((50, 50), (255, 0, 0))
    image = drawing.draw_ref_points(image, 10, 10, 3, 3)
    self.assertEqual(image.size, (50, 50))
    self.assertIsInstance(image, lf.Image)

  def test_draw_text(self):
    image = drawing.blank_image((50, 50), (255, 0, 0))
    image = drawing.draw_text(image, 'Hello World', location.Coordinate(10, 10))
    self.assertEqual(image.size, (50, 50))
    self.assertIsInstance(image, lf.Image)

if __name__ == '__main__':
  unittest.main()
