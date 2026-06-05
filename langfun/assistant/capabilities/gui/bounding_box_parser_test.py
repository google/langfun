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
from typing import Dict, Tuple
import unittest

from langfun.assistant.capabilities.gui import bounding_box_parser
from langfun.assistant.capabilities.gui import location


class BoundingBoxTest(unittest.TestCase):

  def assert_bbox_equal(
      self,
      expected: Dict[str, Tuple[int, int, int, int]],
      actual: Dict[str, location.BBox],
  ):
    self.assertEqual(len(expected), len(actual))
    for key, value in expected.items():
      self.assertIn(key, actual)
      self.assertEqual(value[0], actual[key].x)
      self.assertEqual(value[1], actual[key].y)
      self.assertEqual(value[2], actual[key].right)
      self.assertEqual(value[3], actual[key].bottom)

  def test_bbox_basic_functionality(self):
    bbox = bounding_box_parser.GeminiBBox(xmin=10, ymin=20, xmax=50, ymax=80)

    self.assertEqual(bbox.xmin, 10)
    self.assertEqual(bbox.ymin, 20)
    self.assertEqual(bbox.xmax, 50)
    self.assertEqual(bbox.ymax, 80)

    # Test the `scale` method
    target_size = (800, 600)
    source_size = (1000, 1000)
    scaled_bbox = bbox.scale(target_size, source_size)

    self.assertEqual(scaled_bbox.xmin, 8)
    self.assertEqual(scaled_bbox.ymin, 12)
    self.assertEqual(scaled_bbox.xmax, 40)
    self.assertEqual(scaled_bbox.ymax, 48)

    # Test the `resize` method
    resized_bbox = bbox.resize((1200, 800), 2)

    self.assertEqual(resized_bbox.xmin, 0)
    self.assertEqual(resized_bbox.ymin, 0)
    self.assertEqual(resized_bbox.xmax, 70)
    self.assertEqual(resized_bbox.ymax, 110)

    # Test the `to_gui_bbox` method
    gui_bbox = bbox.to_gui_bbox()
    self.assertEqual(gui_bbox.x, 10)
    self.assertEqual(gui_bbox.y, 20)
    self.assertEqual(gui_bbox.right, 50)
    self.assertEqual(gui_bbox.bottom, 80)

  def test_simple_json(self):
    json_text = '{"search button": [10, 20, 100, 200]}'
    expected = {'search button': (16, 6, 160, 60)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(800, 600)
    )
    self.assert_bbox_equal(expected, result)

  def test_multiple_objects(self):
    json_text = (
        '{"button1": [10, 20, 100, 200], "button2": [30, 40, 130, 240]}'
    )
    expected = {'button1': (16, 6, 160, 60), 'button2': (32, 18, 192, 78)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(800, 600)
    )
    self.assert_bbox_equal(expected, result)

  def test_nested_json(self):
    json_text = (
        '{"buttons": {"search": [10, 20, 100, 200], "cancel": [30, 40, 130,'
        ' 240]}}'
    )
    expected = {'search': (16, 6, 160, 60), 'cancel': (32, 18, 192, 78)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(800, 600)
    )
    self.assert_bbox_equal(expected, result)

  def test_json_in_code_block(self):
    json_text = '```\n{"search button": [10, 20, 100, 200]}\n```'
    expected = {'search button': (16, 6, 160, 60)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(800, 600)
    )
    self.assert_bbox_equal(expected, result)

  def test_extract_json_candidate_from_text(self):
    test_cases = [
        (
            'Some text before ```json\n{"key": "value"}\n``` and after',
            '{"key": "value"}',
        ),
        (
            'Some text before ```\n{"key": "value"}\n``` and after',
            '{"key": "value"}',
        ),
        ('{"key": "value"}', '{"key": "value"}'),
        (
            '```json\n{\n  "name": "Test",\n  "version": 1\n}\n```',
            '{\n  "name": "Test",\n  "version": 1\n}',
        ),
        (
            '```\n{\n  "name": "Test",\n  "version": 1\n}\n```',
            '{\n  "name": "Test",\n  "version": 1\n}',
        ),
        (
            '   ```json\n   {"spaced_json": true}   \n```   ',
            '{"spaced_json": true}',
        ),
        (
            'No code block here, just plain text.',
            'No code block here, just plain text.',
        ),
        (
            '```JSON\n{"case_test": "uppercase_json_tag"}\n```',
            '{"case_test": "uppercase_json_tag"}',
        ),
        ('', ''),
        ('   ', ''),
        (
            (
                'First block: ```json\n{"first": true}\n``` Second block:'
                ' ```json\n{"second": false}\n```'
            ),
            '{"first": true}',
        ),
        (
            (
                'First block: ```\n{"first_code": true}\n``` Second block:'
                ' ```\n{"second_code": false}\n```'
            ),
            '{"first_code": true}',
        ),
        (
            '```json  \n  {"leading_trailing_space_in_block": "test"}  \n  ```',
            '{"leading_trailing_space_in_block": "test"}',
        ),
        (
            (
                '```\n  {"leading_trailing_space_in_block_no_json_tag": "test"}'
                '  \n  ```'
            ),
            '{"leading_trailing_space_in_block_no_json_tag": "test"}',
        ),
    ]

    for raw_text, expected_json_str in test_cases:
      with self.subTest(raw_text=raw_text):
        self.assertEqual(
            bounding_box_parser.extract_json_candidate_from_text(raw_text),
            expected_json_str,
        )

  def test_dict_in_list(self):
    json_text = '```json\n[\n  {"box_2d": [61, 22, 160, 95]}\n]\n```'
    expected = {'box_2d': (22, 61, 95, 160)}
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assert_bbox_equal(expected, result)

  def test_dict_with_label(self):
    json_text = """```json
[
  {"box_2d": [634, 416, 820, 482], "label": "the inner vertical side of the rightmost lower protrusion of the green polygon"},
  {"box_2d": [820, 328, 872, 352], "label": "the purple number '1' located to its left"}
]
```"""
    expected = {'box_2d': (328, 820, 352, 872)}
    result = bounding_box_parser.parse_and_convert_json(json_text)
    print('result: ', result)
    self.assert_bbox_equal(expected, result)

  def test_invalid_json(self):
    json_text = 'This is not a valid JSON'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

  def test_empty_input(self):
    json_text = ''
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

  def test_invalid_list_length(self):
    # Test with a list of length 3
    json_text = '{"button": [10, 20, 100]}'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

    # Test with a list of length 5
    json_text = '{"button": [10, 20, 100, 200, 300]}'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

    # Test with an empty list
    json_text = '{"button": []}'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

  def test_list_input(self):
    json_text = '[10, 20, 100, 200]'
    expected = {'element': (20, 10, 200, 100)}
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assert_bbox_equal(expected, result)

  def test_default_screen_size(self):
    json_text = '{"button": [10, 20, 100, 200]}'
    expected = {'button': (20, 10, 200, 100)}
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assert_bbox_equal(expected, result)

  def test_float_numbers(self):
    json_text = '{"button": [10.5, 20.2, 100.7, 200.9]}'
    expected = {'button': (20, 10, 200, 100)}  # Expected integer coordinates
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assert_bbox_equal(expected, result)

  def test_type_error_handling(self):
    json_text = '{"button": ["text", 20, 100, 200]}'
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assertEqual({}, result)
    json_text = '{"button": [None, 20, 100, 200]}'
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assertEqual({}, result)

  def test_malformed_json(self):
    # Missing quotes around keys
    json_text = '{search button: [10, 20, 100, 200]}'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

    # Unbalanced brackets
    json_text = '{"search button": [10, 20, 100, 200}'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

    # Incorrect comma usage
    json_text = '{"search button", [10, 20, 100, 200]}'
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assertEqual({}, result)

  def test_mixed_data_types(self):
    # String values in coordinates
    json_text = '{"button": ["10", "20", "100", "200"]}'
    expected = {'button': (20, 10, 200, 100)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assert_bbox_equal(expected, result)

  def test_none_values(self):
    json_text = '{"button": [null, 20, 100, 200]}'
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assertEqual({}, result)

    json_text = '{"button": [10, null, 100, 200]}'
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assertEqual({}, result)

  def test_large_coordinates(self):
    json_text = '{"button": [10000, 20000, 100000, 200000]}'
    expected = {'button': (20000, 10000, 200000, 100000)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(1000, 1000)
    )
    self.assert_bbox_equal(expected, result)

  def test_different_string_formats(self):
    # Newlines
    json_text = '{\n"search button": [10, 20, 100, 200]\n}'
    expected = {'search button': (16, 6, 160, 60)}
    result = bounding_box_parser.parse_and_convert_json(
        json_text, screen_size=(800, 600)
    )
    self.assert_bbox_equal(expected, result)

  def test_deeply_nested_json(self):
    json_text = (
        '{"layer1": {"layer2": {"layer3": {"button": [10, 20, 100, 200]}}}}'
    )
    expected = {'button': (20, 10, 200, 100)}
    result = bounding_box_parser.parse_and_convert_json(json_text)
    self.assert_bbox_equal(expected, result)

if __name__ == '__main__':
  unittest.main()
