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
from langfun.core import message_transform
from langfun.core import subscription
from langfun.core import template as template_lib
from langfun.core.langfunc import call
from langfun.core.langfunc import LangFunc
from langfun.core.langfunc import LangFuncCallEvent
from langfun.core.llms import fake
# Enables as_structured() operation of LangFunc.
from langfun.core.structured import parsing  # pylint: disable=unused-import
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


class ExclaimCounter(message_transform.MessageTransform):
  """Transform for output transform testing."""

  input_path = ''
  output_path = ''

  def _transform_path(
      self, m: message.Message, path: str, v) -> message.Message:
    del v
    num_exclaims = 0
    result = ''
    for c in m.text:
      if c == '!':
        num_exclaims += 1
      else:
        result += c
    return message.AIMessage(result, metadata=dict(num_exclaims=num_exclaims))


class BasicTest(unittest.TestCase):

  def test_cached_lm_input_and_output(self):
    l = LangFunc('Hello')
    self.assertEqual(l.render(), 'Hello')
    self.assertIsNone(l.lm_input)
    with component.context(lm=ExcitedEchoer()):
      self.assertEqual(l(), 'Hello!!!')
      self.assertEqual(l.lm_input, 'Hello')
      self.assertEqual(l.lm_output, 'Hello!!!')


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
    self.assertEqual(r, message.AIMessage('Hello!!!'))
    self.assertEqual(r.tags, ['lm-response', 'lm-output'])
    self.assertEqual(r.source, message.UserMessage('Hello'))
    self.assertEqual(r.source.tags, ['rendered', 'lm-input'])

    self.assertEqual(str(l), 'Hello')
    self.assertEqual(
        repr(l),
        "LangFunc(template_str='Hello', clean=True, returns=None, "
        'lm=ExcitedEchoer(sampling_options=LMSamplingOptions(temperature=0.0, '
        'max_tokens=1024, n=1, top_k=40, top_p=None, random_seed=None), '
        'cache=None, timeout=120.0, max_attempts=5, debug=False), '
        'input_transform=None, output_transform=None)',
    )

    l = LangFunc('Hello')
    with component.context(lm=ExcitedEchoer()):
      self.assertEqual(l, 'Hello')
      self.assertEqual(l.natural_language_format(), 'Hello')
      self.assertEqual(l.render(), 'Hello')
      r = l()
      self.assertEqual(r, message.AIMessage('Hello!!!'))
      self.assertEqual(r.tags, ['lm-response', 'lm-output'])

    self.assertEqual(str(l), 'Hello')

    with self.assertRaisesRegex(
        AttributeError, '`lm` is not found under its context'
    ):
      l()

  def test_call_with_input_transform(self):
    # Test input transform on text.
    l = LangFunc(
        'Hello',
        input_transform=message_transform.Lambda(lambda x: str(x) + '?'),
    )
    i = l.render()
    self.assertEqual(i, message.UserMessage('Hello?'))
    self.assertEqual(i.tags, ['transformed'])
    self.assertEqual(i.source, message.UserMessage('Hello', result='Hello?'))
    self.assertEqual(i.source.tags, ['rendered', 'transformed'])

    with component.context(lm=ExcitedEchoer()):
      r = l()
      self.assertEqual(r, message.AIMessage('Hello?!!!'))
      self.assertEqual(r.tags, ['lm-response', 'lm-output'])
      self.assertEqual(r.source, message.UserMessage('Hello?'))
      self.assertEqual(r.source.tags, ['transformed', 'lm-input'])

    # Test `skip_input_transform``.
    with component.context(lm=ExcitedEchoer()):
      i = l.render(skip_input_transform=True)
      self.assertEqual(i, message.UserMessage('Hello'))
      self.assertEqual(i.tags, ['rendered'])
      self.assertEqual(l(skip_input_transform=True), 'Hello!!!')

    # Test transform that operates on the entire input message.
    class NewMessageTransform(message_transform.MessageTransform):
      suffix: str
      input_path = ''
      output_path = ''

      def _transform_path(self, m, path, v) -> message.Message:
        del v
        m = m.clone()
        m.text += self.suffix
        return m

    l = LangFunc('Hello', input_transform=NewMessageTransform(suffix='???'))
    with component.context(lm=ExcitedEchoer()):
      r = l()
      self.assertEqual(r, message.AIMessage('Hello???!!!'))
      self.assertEqual(r.tags, ['lm-response', 'lm-output'])

  def test_call_with_output_transform(self):
    l = LangFunc('Hello', output_transform=ExclaimCounter())
    with component.context(lm=ExcitedEchoer()):
      r = l()
      self.assertEqual(r, message.AIMessage('Hello', num_exclaims=3))
      self.assertEqual(r.tags, ['transformed', 'lm-output'])
      self.assertEqual(r.source, message.AIMessage('Hello!!!'))
      self.assertEqual(r.source.tags, ['lm-response'])

      # Test `skip_output_transform``.
      r = l(skip_output_transform=True)
      self.assertEqual(r, message.AIMessage('Hello!!!'))
      self.assertEqual(r.tags, ['lm-response', 'lm-output'])

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

  def test_call_with_structured_output(self):
    l = LangFunc('Compute 1 + 2', returns=int)
    with component.context(lm=fake.StaticSequence([
        'three', '3'
    ])):
      r = l()
      self.assertEqual(r.result, 3)

    l = LangFunc('Compute 1 + 2', returns=int, output_transform=lambda x: '3')
    with component.context(
        lm=fake.StaticSequence([
            'three', '3'
        ])
    ):
      r = l()
      self.assertEqual(r.result, 3)


class TransformTest(unittest.TestCase):

  def test_transform(self):
    t = message_transform.Identity() >> LangFunc('hi {{message.text}}')
    i = message.AIMessage('foo')
    with component.context(lm=ExcitedEchoer()):
      r = t.transform(i)
    self.assertEqual(r, message.AIMessage('hi foo!!!'))
    self.assertEqual(r.tags, ['lm-response', 'lm-output', 'transformed'])
    self.assertEqual(
        r.lm_input, message.UserMessage('hi foo', message=pg.Ref(i))
    )
    self.assertEqual(r.lm_input.tags, ['rendered', 'lm-input'])
    self.assertIs(r.root, i)

    t = LangFunc('hi {{message.text}}', input_path='result')
    with component.context(lm=ExcitedEchoer()):
      r = t.transform(message.AIMessage('abc', result='bar'))
      self.assertEqual(r, message.AIMessage('hi bar!!!'))
      self.assertEqual(r.tags, ['lm-response', 'lm-output', 'transformed'])
      self.assertEqual(
          r.source,
          message.UserMessage('hi bar', message=pg.Ref(r.source.source)),
      )
      self.assertEqual(r.source.tags, ['rendered', 'lm-input'])
      self.assertEqual(r.source.source, message.AIMessage('bar'))
      self.assertEqual(r.source.source.tags, ['transformed'])

      with self.assertRaisesRegex(TypeError, 'Metadata .* should be a string'):
        t.transform(message.AIMessage('abc', result=1))

  def test_transform_composition(self):
    self.assertTrue(
        pg.eq(
            LangFunc('hi').match(r'\d+').to_int(),
            LangFunc(
                'hi', output_transform=message_transform.Match(r'\d+').to_int()
            ),
        )
    )

    self.assertTrue(
        pg.eq(
            LangFunc('hi').to_int(),
            LangFunc('hi', output_transform=message_transform.ParseInt()),
        )
    )


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
    with component.context(lm=ExcitedEchoer()):
      print('RRR', l.render(name='spirit'))

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


class CallTest(unittest.TestCase):

  def test_call_with_const_str(self):
    with component.context(lm=fake.StaticMapping({
        'Compute 1 + 2': 'three',
    })):
      self.assertEqual(call('Compute 1 + 2'), 'three')

  def test_call_with_template_str(self):
    with component.context(lm=fake.StaticMapping({
        'Compute 1 + 2': 'three',
    })):
      self.assertEqual(call('Compute {{x}} + {{y}}', x=1, y=2), 'three')

  def test_call_with_explicit_template(self):
    with component.context(lm=fake.StaticMapping({
        'Compute 1 + 2': 'three',
    })):
      self.assertEqual(
          call(template_lib.Template('Compute {{x}} + {{y}}', x=1, y=2)),
          'three')

    with component.context(lm=fake.StaticMapping({
        'Compute 1 + 2': 'three',
    })):
      self.assertEqual(
          call(template_lib.Template('Compute {{x}} + {{y}}'), x=1, y=2),
          'three')

  def test_call_with_lfun(self):
    with component.context(lm=fake.StaticMapping({
        'Compute 1 + 2': 'three',
    })):
      self.assertEqual(
          call(LangFunc('Compute {{x}} + {{y}}', x=1, y=2)),
          'three')

  def test_call_with_returns(self):
    with component.context(lm=fake.StaticSequence(['three', '3'])):
      self.assertEqual(call('Compute 1 + 2', returns=int), 3)

    with component.context(lm=fake.StaticSequence(['three', '3'])):
      self.assertEqual(
          call(LangFunc('Compute {{x}} + {{y}}', x=1, y=2), returns=int), 3)

  def test_bad_call(self):
    with self.assertRaisesRegex(TypeError, '`prompt` should be .*'):
      call(1)


if __name__ == '__main__':
  unittest.main()
