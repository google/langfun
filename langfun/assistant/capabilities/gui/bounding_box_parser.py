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
"""Customized bounding box related to Gemini Object Detection."""

import json
import re
from typing import Any, Dict, List, Tuple, Union

from langfun.assistant.capabilities.gui import location
import pyglove as pg


class GeminiBBox(pg.Object):
  """Customized bounding box.

  Note: GeminiPro returns a unique JSON response structure for object detection
  tasks, which differs from the standard location.BBox format. To accommodate 
  this, we've created this custom class to adapt to GeminiPro's specific 
  response structure.
  """

  ymin: int
  xmin: int
  ymax: int
  xmax: int

  def scale(self,
            target_size: tuple[int, int],
            source_size: tuple[int, int] = (1000, 1000)) -> 'GeminiBBox':

    def _scale_x(v):
      return (v * target_size[0]) // source_size[0]

    def _scale_y(v):
      return (v * target_size[1]) // source_size[1]

    return GeminiBBox(
        ymin=_scale_y(self.ymin),
        xmin=_scale_x(self.xmin),
        ymax=_scale_y(self.ymax),
        xmax=_scale_x(self.xmax),
    )

  def resize(
      self, image_size: Tuple[int, int], resize_factor: int = 3
  ) -> 'GeminiBBox':
    """Resize a bounding box from its center.

    Args:
        image_size: A tuple (width, height) representing the size of the image.
        resize_factor: The factor by which to resize the bounding box. Defaults
          to 3.

    Returns:
        A bounding box with the expanded dimensions.
    """
    center_x = (self.xmin + self.xmax) // 2
    center_y = (self.ymin + self.ymax) // 2
    width = self.xmax - self.xmin
    height = self.ymax - self.ymin

    new_width = width * resize_factor
    new_height = height * resize_factor

    new_xmin = max(0, int(center_x - new_width // 2))
    new_ymin = max(0, int(center_y - new_height // 2))
    new_xmax = min(image_size[0], int(center_x + new_width // 2))
    new_ymax = min(image_size[1], int(center_y + new_height // 2))

    return GeminiBBox(
        xmin=new_xmin, ymin=new_ymin, xmax=new_xmax, ymax=new_ymax
    )

  def to_gui_bbox(self) -> location.BBox | None:
    try:
      return location.BBox(
          x=self.xmin, y=self.ymin, right=self.xmax, bottom=self.ymax
      )
    except AssertionError:
      # If the bounding box is not valid, return None.
      return None


def extract_json_candidate_from_text(raw_text: str) -> str:
  """Extracts a JSON candidate string from raw text."""
  # Try to find content within ```json ... ```
  match = re.search(r'```json\s*([\s\S]+?)\s*```', raw_text, re.IGNORECASE)
  if match:
    return match.group(1).strip()

  # Try to find content within ``` ... ```
  match = re.search(r'```\s*([\s\S]+?)\s*```', raw_text)
  if match:
    return match.group(1).strip()

  # If no code blocks, return the stripped raw text.
  return raw_text.strip()


def parse_and_convert_json(
    text: str, screen_size: Tuple[int, int] = (1000, 1000)
) -> Dict[str, location.BBox | None]:
  """Parse and convert json to bounding box.

  Args:
    text: The text to parse.
    screen_size: The screen size to scale the bounding box.

  Returns:
    A dictionary of bounding boxes.

  Example:
  >>> json_text = '```{"search button": [10, 20, 100, 200]}```'
  >>> bboxes = parse_and_convert_json(json_text, screen_size=(800, 600))
  >>> print(bboxes)
  {'search button': BBox(x=16, y=6, right=160, bottom=60)}
  """

  def parse_json(t: str) -> Union[Dict[str, str], List[str], None]:
    """Parse text to json."""
    try:
      return json.loads(t)
    except json.JSONDecodeError:
      return None

  def can_cast_to_int(obj: Any) -> bool:
    try:
      int(obj)
      return True
    except (ValueError, TypeError):
      return False

  def convert_to_bbox(
      data: Union[Dict[str, Any], List[Any], None], screen_size: Tuple[int, int]
  ) -> Dict[str, location.BBox | None]:
    """Convert data to bounding box."""
    result = {}
    if not data:
      return result

    if isinstance(data, list):
      if len(data) == 4 and all(can_cast_to_int(item) for item in data):
        return {
            'element': (
                GeminiBBox(
                    xmin=int(data[1]),
                    ymin=int(data[0]),
                    xmax=int(data[3]),
                    ymax=int(data[2]),
                )
                .scale(screen_size)
                .to_gui_bbox()
            )
        }
      for item in data:
        if isinstance(item, dict):
          result.update(convert_to_bbox(item, screen_size))
      return result

    for key, value in data.items():
      if not value:
        continue

      if isinstance(value, list):
        if len(value) == 4 and all(can_cast_to_int(item) for item in value):
          result[key] = (
              GeminiBBox(
                  xmin=int(value[1]),
                  ymin=int(value[0]),
                  xmax=int(value[3]),
                  ymax=int(value[2]),
              )
              .scale(screen_size)
              .to_gui_bbox()
          )
      elif isinstance(value, dict):
        result.update(convert_to_bbox(value, screen_size))
    return result

  parsed_data = parse_json(extract_json_candidate_from_text(text))
  if parsed_data is None:
    return {}

  return convert_to_bbox(parsed_data, screen_size)
