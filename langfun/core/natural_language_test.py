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
"""Tests for natural language utilities."""
import unittest

from langfun.core import component
from langfun.core import natural_language as nl
import pyglove as pg


class NaturalLanguageFormattableTest(unittest.TestCase):
  """Tests for lf.NaturalLanguageFormattable."""

  def test_natural_language_formattable(self):
    class A(nl.NaturalLanguageFormattable, component.Component):
      x: int
      y: str

      def natural_language_format(self) -> str:
        return f'A simple object with {self.x} and {self.y!r}.'

    a = A(1, 'abc')
    self.assertEqual(repr(a), 'A(x=1, y=\'abc\')')
    with pg.object_utils.repr_format(natural_language=True):
      self.assertEqual(repr(a), "A simple object with 1 and 'abc'.")
    self.assertEqual(str(a), "A simple object with 1 and 'abc'.")


if __name__ == '__main__':
  unittest.main()
