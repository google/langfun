# Copyright 2023 The Langfun Authors
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
"""Tests for langfun.core.console."""

import contextlib
import io
import unittest

from langfun.core import console
import pyglove as pg


class ConsoleTest(unittest.TestCase):

  def test_write(self):
    s = io.StringIO()
    with contextlib.redirect_stdout(s):
      console.write('foo', title='hello')
    s = s.getvalue()
    self.assertIn('hello', s)
    self.assertIn('foo', s)

  def test_under_notebook(self):
    self.assertFalse(console.under_notebook())
    console._notebook = True
    self.assertTrue(console.under_notebook())
    console._notebook = None

  def test_notebook_interaction(self):
    console._notebook = pg.Dict(
        display=lambda x: x, Javascript=lambda x: x, clear_output=lambda: None)
    self.assertEqual(console.display('hi', clear=True), 'hi')
    self.assertEqual(
        console.run_script('console.log("hi")'),
        'console.log("hi")'
    )
    console.clear()
    console._notebook = None
    self.assertIsNone(console.display('hi'))
    self.assertIsNone(console.run_script('console.log("hi")'))


if __name__ == '__main__':
  unittest.main()
