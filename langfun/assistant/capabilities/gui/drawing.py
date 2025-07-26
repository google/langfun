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
"""Image drawing for facilitating UI understanding."""

import functools
from typing import Callable
import langfun as lf
from langfun.assistant.capabilities.gui import location
from PIL import Image as pil_image
from PIL import ImageDraw as pil_draw


def blank_image(
    size: tuple[int, int],
    background: tuple[int, int, int] = (0, 0, 0),
    pil: bool = False
    ) -> lf.Image | pil_image.Image:
  """Creates a blank image of given size and background color.

  Args:
    size: The size of the image.
    background: The background color of the image in RGB format.
    pil: If True, the return value will be a `PIL.Image` object, otherwise it
      will be a `lf.Image` object.

  Returns:
    A blank image with requested size and background color.
  """
  image = pil_image.new('RGB', size, background)
  return image if pil else lf.Image.from_pil_image(image)


def draw(
    image: lf.Image | pil_image.Image,
    draw_fn: Callable[[pil_draw.ImageDraw], None],
    ) -> lf.Image | pil_image.Image:
  """Draws on an image with a draw_fn.

  Args:
    image: An `lf.Image` or a `PIL.Image` object.
    draw_fn: A function that takes a `PIL.ImageDraw` object and draws on it.

  Returns:
    The image after drawing. Its type will be the same as the input image.
  """
  is_pil_input = isinstance(image, pil_image.Image)
  if not is_pil_input:
    image = image.to_pil_image()
  draw_fn(pil_draw.Draw(image))
  return image if is_pil_input else lf.Image.from_pil_image(image)


def draw_bboxes(
    image: lf.Image | pil_image.Image,
    bboxes: list[location.BBox],
    line_color: str = 'red',
    line_width: int = 3,
    text: str | None = None,
) -> lf.Image | pil_image.Image:
  """Draws bounding boxes on an image.

  Args:
    image: An `lf.Image` or a `PIL.Image` object.
    bboxes: A list of `location.BBox` objects to draw on the image.
    line_color: The color of the bounding box lines.
    line_width: The width of the bounding box lines.
    text: The text to draw on each bounding box.

  Returns:
    The image after drawing. Its type will be the same as the input image.
  """
  if not bboxes:
    return image

  def _draw_fn(drawing: pil_draw.ImageDraw) -> None:
    for bbox in bboxes:
      drawing.rectangle(
          (bbox.x, bbox.y, bbox.right, bbox.bottom),
          outline=line_color, width=line_width
      )
      if text:
        drawing.text((bbox.x + 5, bbox.y + 5), text, fill=line_color)
  return draw(image, _draw_fn)


def draw_points(
    image: lf.Image | pil_image.Image,
    points: list[location.Coordinate],
    color: str = 'red',
    radius: int = 3,
) -> lf.Image | pil_image.Image:
  """Draws points on an image.

  Args:
    image: An `lf.Image` or a `PIL.Image` object.
    points: A list of `location.Coordinate` objects to draw on the image.
    color: The color of the points.
    radius: The radius of the points.

  Returns:
    The image after drawing. Its type will be the same as the input image.
  """
  if not points:
    return image

  def _draw_fn(drawing: pil_draw.ImageDraw) -> None:
    for point in points:
      drawing.ellipse(
          (
              point.x - radius,
              point.y - radius,
              point.x + radius,
              point.y + radius,
          ),
          fill=color,
      )

  return draw(image, _draw_fn)


def draw_calibration_lines(
    image: lf.Image | pil_image.Image,
    coordinate: location.Coordinate,
    vline_color: str = 'green',
    vline_width: int = 3,
    hline_color: str = 'blue',
    hline_width: int = 3
) -> lf.Image | pil_image.Image:
  """Draws calibration lines for a coordinate in an image.

  Args:
    image: An `lf.Image` or a `PIL.Image` object.
    coordinate: The coordinate to draw calibration lines for.
    vline_color: The color of the vertical line.
    vline_width: The width of the vertical line.
    hline_color: The color of the horizontal line.
    hline_width: The width of the horizontal line.

  Returns:
    The image after drawing. Its type will be the same as the input image.
  """
  def _draw_fn(drawing: pil_draw.ImageDraw) -> None:
    drawing.line(
        (0, coordinate.y, image.size[0], coordinate.y),
        fill=hline_color, width=hline_width
    )
    drawing.line(
        (coordinate.x, 0, coordinate.x, image.size[1]),
        fill=vline_color, width=vline_width
    )
  return draw(image, _draw_fn)


def draw_cursor(
    image: lf.Image | pil_image.Image,
    coordinate: location.Coordinate,
) -> lf.Image | pil_image.Image:
  """Draws a cursor on an image.

  Args:
    image: An `lf.Image` or a `PIL.Image` object.
    coordinate: The coordinate to draw the cursor at.

  Returns:
    The image after drawing. Its type will be the same as the input image.
  """
  is_pil_input = isinstance(image, pil_image.Image)
  if not is_pil_input:
    image = image.to_pil_image()

  image.paste(
      _cursor_image(), (coordinate.x - 3, coordinate.y - 2), _cursor_image()
  )
  return image if is_pil_input else lf.Image.from_pil_image(image)


def draw_ref_points(
    image: lf.Image | pil_image.Image,
    width: int,
    height: int,
    rows: int,
    cols: int,
) -> lf.Image | pil_image.Image:
  """Draw a grid of reference points on an image."""
  delta_x = width // (cols + 1)
  delta_y = height // (rows + 1)
  points = []

  for i in range(rows):
    for j in range(cols):
      x = int((i + 1) * delta_x)
      y = int((j + 1) * delta_y)
      points.append((x, y))

  def _draw_fn(drawing: pil_draw.ImageDraw) -> None:
    for point in points:
      x, y = point
      drawing.ellipse(
          (x - 2, y - 2, x + 2, y + 2), fill='green', outline='green'
      )
      drawing.text((x + 5, y - 6), f'(x={x}, y={y})', fill='green')

  return draw(image, _draw_fn)


def draw_text(
    image: lf.Image | pil_image.Image,
    text: str,
    coordinate: location.Coordinate,
    font_color: str = 'red',
) -> lf.Image | pil_image.Image:
  """Draws text on an image."""
  is_pil_input = isinstance(image, pil_image.Image)
  if not is_pil_input:
    image = image.to_pil_image()

  drawing = pil_draw.ImageDraw(image)
  drawing.text(
      (coordinate.x, coordinate.y),
      text,
      fill=font_color,
      spacing=4,
  )
  return image if is_pil_input else lf.Image.from_pil_image(image)


@functools.cache
def _cursor_image() -> pil_image.Image:
  return lf.Image.from_bytes(
      b"""\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x19\x00\x00\x00\x1e\x08\x06\x00\x00\x00\xd9\xec\xb5\xdb\x00\x00\x04gIDATx\x9c\xc5\x95_H[g\x18\xc6\x9f\x9c$\'1\xccl\x8ch\xe7\xe2\xb4%\xd2\xea\xa8s\xdd&\xd8Q\x9b\xba\x8d\x8eZA\x10\x0cQ\xca\x84\xad^l\x17e-+\xb8\x1b\x99]\xbb\xe26Ea\xa0\xb2\x8bA\x07\xcat\xa3(Sj\xa7\xad4C+v\x17\xa27\xb6\xb3\x81\xd9\xe9b\xa41mNr\xce\xc9I\x9e]\xa8\x99v\xadZk\xb7\x07^\xbes\xf1\x9d\xe7\xf7\xbd\x7f\xbes\x80\xffR\xe1p\xd8\xa5(J\xf9\xd3\xf0\x16V\xd6\xd9\xd9\xd9lQ\x14\x7f\x88\xc5b_\xf5\xf4\xf4X\x9e\x06\xcc422R\x17\x89D8;;K\x92W\xee\xdc\xb9\xb3{\xbb\xccW2\x01\xc9X(\x14\x82\xd3\xe9DOOO\x91\xddn\xbf"IR\xe9v\x81\x00@\xf4x<5\xc1`\x90V\xab\x95:\x9d\x8e\xb5\xb5\xb5T\x14E%\xf9Y[[\x9bq[ \x83\x83\x83\x9f\x84B!\xee\xd8\xb1\x83\x00\x08\x80\xc5\xc5\xc5\x9c\x9b\x9b#\xc9\xee\x9b7o\xa6?1\xa4\xbf\xbf\xffD8\x1c\xa6\xddnO@\x00p\xd7\xae]\xbcv\xed\x1aIN\x05\x02\x81C[1_\xe9\tUU\x8d\n\x82\x00\x83\xc1\xb0f\x83\xd7\xeb\xc5\xe1\xc3\x87\xd1\xdc\xdc\xbc\xdbj\xb5\xfe\xac\xaa\xea\x89\xadfb\xbcx\xf1\xe2\x07\x8a\xa2\xd0\xe1p\xac\xc9du\x1c;v\x8c\x81@\x80$\xbf\x1b\x1e\x1e~\xfe\xb1!\xed\xed\xedU\xd1h\x94\xd9\xd9\xd9\x8f\x84\x00`^^\x1e\'&&Hr\xcc\xe7\xf3\xbd\xba\x19\xf3\xc4\x08\xab\xaa\xaa\x02\x80\xd1\xb8\xfe \x8d\x8f\x8f\xa3\xb0\xb0\x10\x1d\x1d\x1do\xa4\xa6\xa6\xfe\x12\x0e\x87+7\x0b\xa1\xaa\xaaQ\x00\x14Eq\xc3\x93-..\xa2\xb2\xb2\x12\xa7N\x9d\xb2\x01\xf8\x9e\xe4\x97\r\r\rI\x1b\xbdghnn.\x8d\xc5b\xb1\x82\x82\x82u\xcb\xf5`\x14\x15\x15qff\x86$/y\xbd\xde\x9d\xebe\x02I\x92\xa2\x00h2\x996\xccd\xb5\xae^\xbd\x8a}\xfb\xf6\xa1\xb7\xb7\xf7\xdd\xf4\xf4\xf4\x81\xb9\xb9\xb9\xb7\xfe\x95\xc1\xf2\xcaH$\x12\x15\x04!&\x8a\xa2\xfeQ\x86\x82  ++\x0b\xa2(\xc2b\xb1 ))\tf\xb3\x19\x06\x83\x01\x9d\x9d\x9d\xd8\xbbw\xaf#33\xb37\x14\n}\xd4\xd7\xd7w\xc1\xe5r\xc5VCt\xb2,k\x00\xe2\x1b\xf5\xa4\xa6\xa6\x06UUU\x08\x85B\x7f\x01P\x01h$\xa3\xb2,3\x18\x0cbjj*n6\x9b\xdfKII\x99\x00\xf0\x1b\x00&n^$\x12\xd1\x00\xc4VC,\x16\x0b\xce\x9d;\x87\xee\xeen\x0c\r\r!\x1e\x8f\xe3\xfc\xf9\xf3p\xbb\xdd\x18\x19\x19\xb9TVV\xf6\xa3\xa6iPU5\x8e\xa5\xd2\x1b\x00\xc4\x00\xf8\x00\xfc\t@\x0f@[\xf1\xd3\x1f?~\xbc\x80d\xd0\xedv\x13\x00\xf7\xec\xd9\xc3\xeb\xd7\xaf\x93dtxx\x98F\xa31\xd1\xec3g\xce\x90\xa4\xdf\xe5r\x1d\x04\xf0\xc2r\xa4\x02H\x01`\x03\xf0\x1c\x003\x00\xdd\xea*\xe8].W>\xc9\xbbn\xb7\x9b\xc5\xc5\xc5\xf4\xfb\xfd\x9c\x9e\x9e\xfe\xf5\xf4\xe9\xd3\xf5$#\xd5\xd5\xd5\t\x88\xd5j\xe5\xcc\xcc\x0c%I\xfa\xf6A\xa3\xf5\xa4/))yM\x96e\xff\x8d\x1b7(I\x92:00pA\xaf\xd7\x1f\x05\xf0\xf6\xe8\xe8\xe8O~\xbf\x9f6\x9b-\x01\xaa\xa8\xa8 \xc9\x88\xc7\xe39\xb4i\xc8\x81\x03\x07^\x91e\xf9\xae\xa2(\xf7\xeb\xeb\xeb\xbf\x00P\x00 \x03\x80}\xff\xfe\xfdG5M\xf3555% \x82 \xd0\xe3\xf1\x90\xe4\x95\xb4\xb4\xb4M\xfd\xae\x85#G\x8e\xe4\xde\xbe}{\xb4\xba\xba\xfaS\x00\xafc\xa9\xb6z\x00&\x00;\xbb\xba\xba\xbe\xd64\x8d\xb9\xb9\xb9\tP~~>\xe3\xf18\xa7\xa7\xa7\xdf\xdfT&\x0e\x87\xe3\xa5\x9c\x9c\x9c\x83\x00r\xb0\xd4\xb8\x95\x8b\xaa\x03\xf0lrr\xf2\x9b\x81@`\xb2\xbf\xbf\x9f:\x9d.\x01jmm%\xc9\xdf\xcf\x9e=\x9b\xb6\x11D\x87\xa5iH\xc6C\xa6\x02K\xa3i\xaf\xab\xab\xfb\x98\xa4VZZ\x9a\x80dddP\x92$\xde\xbbw\xef\x1b\xa7\xd3i\xc0\x13\xea\x19\x00\xb9\xb7n\xdd\xba\xec\xf5zy\xf2\xe4Ivvvrrr\x92\x81@@\x9a\x9f\x9f\x9fhiiy\xf9!\x07|,\xe9\x01\xa4\x96\x97\x97W,,,\xfc\xe1\xf3\xf9\xc6\xc7\xc6\xc6:\x1a\x1b\x1b?w:\x9d\x1f\x9aL\xa6w\x00\xbc\xb8\xbco\x8d\x1e\x97j\x06\x90a\xb3\xd9v/,,h\x00\x16\x01\x84\x01H\x00\xee/\x87\x8c\xa5Rn\x19\xb2\xd2;\xcb\xf2\xb3\xb2\x1c\x1a\xfe\xf9\xfc\xff?\xfa\x1b\x9d\xbfI\x0e\x1da[h\x00\x00\x00\x00IEND\xaeB`\x82"""
  ).to_pil_image()
