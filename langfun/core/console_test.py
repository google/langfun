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


if __name__ == '__main__':
  unittest.main()
