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
"""Base memory test."""

import unittest
import langfun.core as lf
from langfun.core.memories.conversation_history import ConversationHistory
import pyglove as pg


class ConversationHistoryTest(unittest.TestCase):

  def test_history_with_no_round_limit(self):
    m = ConversationHistory()
    m.remember(('hi', 'hello'))
    m.remember(
        ('how are you', 'Fine, thank you. Anything I can help with?'),
    )
    m.remember(
        ('Not for now, bye.', 'Okay, bye!'),
    )
    self.assertEqual(len(m.turns), 3)
    self.assertEqual(len(m.messages), 6)
    self.assertTrue(
        pg.eq(
            m.recollect(),
            lf.Template(
                """
                User: hi
                AI: hello
                User: how are you
                AI: Fine, thank you. Anything I can help with?
                User: Not for now, bye.
                AI: Okay, bye!
                """
            )
        )
    )

  def test_history_with_round_limit(self):
    m = ConversationHistory(max_turns=1)
    m.remember(('hi', 'hello'))
    m.remember(
        ('how are you', 'Fine, thank you. Anything I can help with?'),
    )
    m.remember(
        ('Not for now, bye.', 'Okay, bye!'),
    )
    self.assertEqual(len(m.turns), 1)
    self.assertEqual(len(m.messages), 2)
    self.assertTrue(
        pg.eq(
            m.recollect(),
            lf.Template(
                """
                User: Not for now, bye.
                AI: Okay, bye!
                """
            )
        )
    )


if __name__ == '__main__':
  unittest.main()
