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
"""LangFunc test."""
import unittest

from langfun.core import component
from langfun.core import language_model
from langfun.core import message
from langfun.core import subscription
from langfun.core import template as template_lib
from langfun.core.langfunc import LangFunc
from langfun.core.langfunc import LangFuncCallEvent
from langfun.core.llms import fake
from langfun.core.llms.cache import in_memory
import pyglove as pg


class ExcitedEchoer(language_model.LanguageModel):
  """LM for testing."""

  def _sample(
      self,
      prompts: list[message.Message]) -> list[language_model.LMSamplingResult]:
    return [
        language_model.LMSamplingResult([
            language_model.LMSample(prompt.text + '!!!')
            ]) for prompt in prompts
    ]


class BasicTest(unittest.TestCase):

  def test_cached_lm_input_and_output(self):
    l = LangFunc('Hello')
    self.assertEqual(l.render(), 'Hello')
    self.assertIsNone(l.lm_input)
    with component.context(lm=ExcitedEchoer()):
      self.assertEqual(l(), 'Hello!!!')
      self.assertEqual(l.lm_input, 'Hello')
      self.assertEqual(l.lm_output, 'Hello!!!')

  def test_from_value(self):
    l1 = LangFunc.from_value('Hello')
    self.assertEqual(l1.template_str, 'Hello')

    l2 = LangFunc.from_value(l1)
    self.assertIs(l2, l1)

    l3 = LangFunc.from_value(l1, x=1)
    self.assertIsNot(l3, l1)
    self.assertTrue(pg.eq(l3, LangFunc('Hello', x=1)))

    c = template_lib.Template(
        '{{x}} + {{l}}',
        x=1,
        l=template_lib.Template('{{x}} + {{y}}', y=2))
    l3 = LangFunc.from_value(c.l)
    self.assertEqual(l3.render(), '1 + 2')

    l4 = LangFunc.from_value(1)
    self.assertEqual(l4.render(), '1')


class LangFuncCallTest(unittest.TestCase):

  def test_call(self):
    l = LangFunc('Hello', lm=ExcitedEchoer())
    self.assertEqual(l, 'Hello')
    self.assertNotEqual(l, 'Hello!!!')
    self.assertEqual(l.natural_language_format(), 'Hello')

    i = l.render()
    self.assertEqual(i, 'Hello')
    self.assertEqual(i, message.UserMessage('Hello'))
    self.assertEqual(i.tags, ['rendered'])

    r = l()
    self.assertEqual(
        r,
        message.AIMessage(
            'Hello!!!', score=0.0, logprobs=None, is_cached=False,
            usage=language_model.UsageNotAvailable()
        )
    )
    self.assertEqual(r.tags, ['lm-response', 'lm-output'])
    self.assertEqual(
        r.source,
        message.UserMessage('Hello', metadata=dict(cache_seed=0))
    )
    self.assertEqual(r.source.tags, ['rendered', 'lm-input'])

    self.assertEqual(str(l), 'Hello')
    self.assertEqual(
        repr(l),
        "LangFunc(template_str='Hello', clean=True,"
        ' lm=ExcitedEchoer(sampling_options=LMSamplingOptions(temperature=None,'
        ' max_tokens=None, n=1, top_k=40, top_p=None, stop=None,'
        ' random_seed=None, logprobs=False, top_logprobs=None), cache=None,'
        ' max_concurrency=None, timeout=120.0, max_attempts=5,'
        ' retry_interval=(5, 60), exponential_backoff=True,'
        ' max_retry_interval=300, debug=False))',
    )

    l = LangFunc('Hello')
    with component.context(lm=ExcitedEchoer()):
      self.assertEqual(l, 'Hello')
      self.assertEqual(l.natural_language_format(), 'Hello')
      self.assertEqual(l.render(), 'Hello')
      r = l(cache_seed=1)
      self.assertEqual(
          r,
          message.AIMessage(
              'Hello!!!', score=0.0, logprobs=None, is_cached=False,
              usage=language_model.UsageNotAvailable()
          )
      )
      self.assertEqual(r.tags, ['lm-response', 'lm-output'])
      self.assertEqual(r.source.metadata.cache_seed, 1)

    self.assertEqual(str(l), 'Hello')

    with self.assertRaisesRegex(
        AttributeError, '`lm` is not found under its context'
    ):
      l()

  def test_call_with_init_time_vars(self):
    l = LangFunc('Hello, {{foo}}', foo='a')
    self.assertEqual(l.foo, 'a')
    self.assertEqual(l.natural_language_format(), 'Hello, a')

    with self.assertRaises(AttributeError):
      _ = l.bar
    self.assertEqual(l.render(), 'Hello, a')

    # Use instance variable even the parent has the same variable.
    r = pg.Dict(s=LangFunc('Hello, {{x}}', x='foo'), x='world')
    self.assertEqual(r.s.x, 'foo')
    self.assertEqual(r.x, 'world')
    self.assertEqual(r.s.render(), 'Hello, foo')

  def test_call_with_call_time_vars(self):
    l = LangFunc('Hello, {{foo}} {{bar}}', bar=LangFunc('and {{foo}}'))
    with self.assertRaises(AttributeError):
      _ = l.foo
    self.assertEqual(l.render(foo='a'), 'Hello, a and a')

    l = LangFunc('Hello, {{foo}} {{bar}}',
                 foo='a',
                 bar=LangFunc('and {{foo}}', foo='b'))
    self.assertEqual(l.render(), 'Hello, a and b')
    self.assertEqual(l.render(foo='c'), 'Hello, c and b')

  def test_call_with_contextual_vars(self):
    class Chat(LangFunc):
      """Chat.

      {{conversation}}
      """

      # Generated texts.
      prompt: str = LangFunc('compute {{x}} + {{y}}')
      response: str = LangFunc('{{x + y}}')
      conversation: str = LangFunc('Q: {{prompt}}\n\nA: {{response}}')

      # Instance variables.
      x: int = 1

      # Class variables.
      y = 2

    c = Chat()
    self.assertEqual(c.prompt.x, 1)
    self.assertEqual(c.prompt.y, 2)
    self.assertEqual(c.response.x, 1)
    self.assertEqual(c.response.y, 2)
    self.assertIs(c.conversation.prompt, c.prompt)
    self.assertIs(c.conversation.response, c.response)

    self.assertEqual(c.render(), 'Q: compute 1 + 2\n\nA: 3')
    self.assertEqual(c.render(x=2), 'Q: compute 2 + 2\n\nA: 4')

    c = Chat(x=2)
    self.assertEqual(c.render(), 'Q: compute 2 + 2\n\nA: 4')
    self.assertEqual(c.render(y=3), 'Q: compute 2 + 3\n\nA: 5')

  def test_call_with_overriden_lm_input(self):
    t = LangFunc('Hello')
    with component.context(lm=ExcitedEchoer()):
      self.assertEqual(t(lm_input=message.UserMessage('Hi')), 'Hi!!!')

  def test_call_with_cache(self):
    l = LangFunc('{{x}}')
    with component.context(
        lm=fake.StaticSequence(['a', 'b', 'c', 'd'], cache=in_memory.InMemory())
    ):
      self.assertEqual(l(x=1), 'a')
      self.assertEqual(l(x=2), 'b')
      self.assertEqual(l(x=1), 'a')
      self.assertEqual(l(x=1, cache_seed=None), 'c')
      self.assertEqual(l(x=1, cache_seed=None), 'd')
      self.assertEqual(l(x=2), 'b')

  def test_call_with_skip_lm(self):
    l = LangFunc('hi')
    with component.context(lm=ExcitedEchoer()):
      self.assertEqual(l(skip_lm=True), 'hi')


class CallEventTest(unittest.TestCase):

  def test_call_event(self):
    l = LangFunc(
        'Subject: {{subject}}',
        subject=LangFunc(
            'The science of {{ask_me}}',
            ask_me=LangFunc(
                '{{inner_voice()}}', inner_voice=LangFunc('living {{name}}')
            ),
        ),
    )

    class CallEventHandler(subscription.EventHandler[LangFuncCallEvent]):

      def __init__(self):
        super().__init__()
        self.call_events = []
        self.call_stacks = []

      def on_event(self, event: LangFuncCallEvent):
        self.call_events.append((event.lm_input, event.lm_output.text))
        self.call_stacks.append(event.lm_callstack)

    x = CallEventHandler()
    subscription.subscribe(x)
    with component.context(lm=ExcitedEchoer()):
      l(name='spirit')

    self.assertEqual(
        x.call_events,
        [
            ('living spirit', 'living spirit!!!'),
            (
                'Subject: The science of living spirit!!!',
                'Subject: The science of living spirit!!!!!!',
            ),
        ],
    )
    self.assertEqual(x.call_stacks, [[l], []])
    subscription.unsubscribe(x)

    y = CallEventHandler()
    z = CallEventHandler()
    subscription.subscribe(y, l.subject)
    subscription.subscribe(z, l.subject.ask_me.inner_voice)

    with component.context(lm=ExcitedEchoer()):
      l(name='dead')

    # l.subject is not called explicitly.
    self.assertEqual(len(y.call_events), 0)

    # l.subject.ask_me.inner_voice is called explicitly.
    self.assertEqual(
        z.call_events,
        [
            ('living dead', 'living dead!!!'),
        ],
    )


if __name__ == '__main__':
  unittest.main()
