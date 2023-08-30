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
"""Tests for text formatting."""

import inspect
import unittest
from langfun.core import text_formatting


class TextFormattingTest(unittest.TestCase):

  def test_colored_template(self):
    original_text = inspect.cleandoc("""
        Hi {{ foo }}
        {# print x if x is present #}
        {% if x %}
        {{ x }}
        {% endif %}
        """)

    colored_text = text_formatting.colored_template(
        text_formatting.colored(original_text, color='blue')
    )
    self.assertEqual(
        colored_text,
        '\x1b[34mHi \x1b[44m\x1b[37m{{ foo }}\x1b[0m\x1b[34m\n'
        '\x1b[32m{# print x if x is present #}\x1b[0m\x1b[34m\n'
        '\x1b[31m{% if x %}\x1b[0m\x1b[34m\n'
        '\x1b[44m\x1b[37m{{ x }}\x1b[0m\x1b[34m\n'
        '\x1b[31m{% endif %}\x1b[0m\x1b[34m\x1b[0m'
    )
    self.assertEqual(text_formatting.decolored(colored_text), original_text)


if __name__ == '__main__':
  unittest.main()
