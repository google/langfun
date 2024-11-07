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
"""Tests for langfun.core.templates.Conversation."""

import inspect
import unittest
import langfun.core as lf
from langfun.core.templates.conversation import Conversation


class QuestionCounter(lf.LanguageModel):

  def _on_bound(self):
    super()._on_bound()
    self._num_call = 0

  def _sample(self, prompts):
    del prompts
    self._num_call += 1
    return [lf.LMSamplingResult([lf.LMSample(f'Response {self._num_call}.')])]


class ConversationTest(unittest.TestCase):

  def test_render_input_only(self):
    c = Conversation(next_input='How are you?')
    self.assertEqual(
        c.render(),
        'How are you?')

  def test_preamble_only(self):
    c = Conversation(
        'You are a helpful and joyful chat bot. Let us chat.',
        'How are you?')
    self.assertEqual(
        c.render(),
        inspect.cleandoc("""
          You are a helpful and joyful chat bot. Let us chat.

          User: How are you?
        """))

  def test_with_conversation_context(self):
    c = Conversation(
        preamble='You are a helpful and joyful chat bot. Let us chat.',
        conversation_context=lf.LangFunc(
            """
            {% for message in conversation_history %}
            {{message.sender}}: {{message.text}}
            {%- endfor %}
            """,
            conversation_history=[
                lf.UserMessage('Hello'),
                lf.AIMessage('Hi, how can I help you?'),
            ]
        ),
        next_input=lf.LangFunc(
            """{{message.sender}}: {{message.text}}""",
            message=lf.UserMessage('What is your name?'),
        ),
    )

    self.assertEqual(
        c.render(),
        inspect.cleandoc("""
          You are a helpful and joyful chat bot. Let us chat.
          User: Hello
          AI: Hi, how can I help you?
          User: What is your name?
        """))

  def test_call(self):
    c = Conversation(
        lm=QuestionCounter(),
        role='Agent',
        preamble="You are a helpful and joyful AI bot. Now let's chat.",
    )
    # First round.
    self.assertEqual(
        c.prompt.render(input_message='Hello'),
        inspect.cleandoc("""
        You are a helpful and joyful AI bot. Now let's chat.

        User: Hello
        """),
    )
    self.assertEqual(c(input_message='Hello'), 'Response 1.')

    # Second round.
    self.assertEqual(
        c.prompt.render(input_message='How are you?'),
        inspect.cleandoc("""
        You are a helpful and joyful AI bot. Now let's chat.
        User: Hello
        Agent: Response 1.
        User: How are you?
        """),
    )
    self.assertEqual(c(input_message='How are you?'), 'Response 2.')

    # Third round.
    self.assertEqual(
        c.prompt.render(input_message='Okay, bye.'),
        inspect.cleandoc("""
        You are a helpful and joyful AI bot. Now let's chat.
        User: Hello
        Agent: Response 1.
        User: How are you?
        Agent: Response 2.
        User: Okay, bye.
        """),
    )
    self.assertEqual(c(input_message='Okay, bye.'), 'Response 3.')

    # A new round of conversation.
    c.reset()
    self.assertEqual(
        c.prompt.render(input_message='Hi'),
        inspect.cleandoc("""
        You are a helpful and joyful AI bot. Now let's chat.

        User: Hi
        """),
    )


if __name__ == '__main__':
  unittest.main()
