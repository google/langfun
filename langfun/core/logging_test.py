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
"""Tests for langfun.core.logging."""

import datetime
import inspect
import unittest

from langfun.core import logging


class LoggingTest(unittest.TestCase):

  def test_use_log_level(self):
    self.assertEqual(logging.get_log_level(), 'info')
    with logging.use_log_level('debug'):
      self.assertEqual(logging.get_log_level(), 'debug')
      with logging.use_log_level(None):
        self.assertIsNone(logging.get_log_level(), None)
      self.assertEqual(logging.get_log_level(), 'debug')
    self.assertEqual(logging.get_log_level(), 'info')

  def test_log(self):
    entry = logging.log('info', 'hi', indent=1, x=1, y=2)
    self.assertEqual(entry.level, 'info')
    self.assertEqual(entry.message, 'hi')
    self.assertEqual(entry.indent, 1)
    self.assertEqual(entry.metadata, {'x': 1, 'y': 2})

    self.assertEqual(logging.debug('hi').level, 'debug')
    self.assertEqual(logging.info('hi').level, 'info')
    self.assertEqual(logging.warning('hi').level, 'warning')
    self.assertEqual(logging.error('hi').level, 'error')
    self.assertEqual(logging.fatal('hi').level, 'fatal')

  def test_repr_html(self):
    def assert_color(entry, color):
      self.assertIn(f'background-color: {color}', entry._repr_html_())

    assert_color(logging.debug('hi', indent=0), '#EEEEEE')
    assert_color(logging.info('hi', indent=1), '#A3E4D7')
    assert_color(logging.warning('hi', x=1, y=2), '#F8C471')
    assert_color(logging.error('hi', indent=2, x=1, y=2), '#F5C6CB')
    assert_color(logging.fatal('hi', indent=2, x=1, y=2), '#F19CBB')

  def assert_html_content(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_html(self):
    time = datetime.datetime(2024, 10, 10, 12, 30, 45)
    self.assert_html_content(
        logging.LogEntry(
            level='info', message='5 + 2 > 3',
            time=time, metadata={}
        ).to_html(enable_summary_tooltip=False),
        """
        <details open class="pyglove log-entry log-info"><summary><div class="summary-title log-info"><span class="log-time">12:30:45</span><span class="log-summary">5 + 2 &gt; 3</span></div></summary><div class="complex_value"></div></details>
        """
    )
    self.assert_html_content(
        logging.LogEntry(
            level='error', message='This is a longer message: 5 + 2 > 3',
            time=time, metadata=dict(x=dict(z=1), y=2)
        ).to_html(
            extra_flags=dict(
                collapse_log_metadata_level=1,
            ),
            max_summary_len_for_str=10,
            enable_summary_tooltip=False,
        ),
        """
        <details open class="pyglove log-entry log-error"><summary><div class="summary-title log-error"><span class="log-time">12:30:45</span><span class="log-summary">This is a ...</span></div></summary><div class="complex_value"><span class="log-text">This is a longer message: 5 + 2 &gt; 3</span><div class="log-metadata"><details open class="pyglove dict"><summary><div class="summary-name">metadata<span class="tooltip">metadata</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details class="pyglove dict"><summary><div class="summary-name">x<span class="tooltip">metadata.x</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">z<span class="tooltip">metadata.x.z</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details></div></details><details open class="pyglove int"><summary><div class="summary-name">y<span class="tooltip">metadata.y</span></div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div></details></div></div></details>
        """
    )
    self.assert_html_content(
        logging.LogEntry(
            level='error', message='This is a longer message: 5 + 2 > 3',
            time=time, metadata=dict(x=dict(z=1), y=2)
        ).to_html(
            extra_flags=dict(
                max_summary_len_for_str=10,
            ),
            enable_summary_tooltip=False,
        ),
        """
        <details open class="pyglove log-entry log-error"><summary><div class="summary-title log-error"><span class="log-time">12:30:45</span><span class="log-summary">This is a longer message: 5 + 2 &gt; 3</span></div></summary><div class="complex_value"><div class="log-metadata"><details open class="pyglove dict"><summary><div class="summary-name">metadata<span class="tooltip">metadata</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove dict"><summary><div class="summary-name">x<span class="tooltip">metadata.x</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">z<span class="tooltip">metadata.x.z</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details></div></details><details open class="pyglove int"><summary><div class="summary-name">y<span class="tooltip">metadata.y</span></div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details></div></details></div></div></details>
        """
    )

if __name__ == '__main__':
  unittest.main()
