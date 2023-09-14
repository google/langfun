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
import inspect
import unittest

from langfun.core import component
from langfun.core import language_model
from langfun.core import message
from langfun.core import message_transform
from langfun.core import subscription
# We import lf.transforms for automatic conversion from value transforms
# to message transforms.
from langfun.core.langfunc import _Template
from langfun.core.langfunc import LangFunc
from langfun.core.langfunc import LangFuncCallEvent
from langfun.core.langfunc import LangFuncEvent
from langfun.core.langfunc import LangFuncRenderEvent
from langfun.core.llms import fake
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


class LangFuncTest(unittest.TestCase):
  """Tests for LangFunc."""

  def test_hash(self):
    l = LangFunc('Hello')
    self.assertEqual(hash(l), hash(l))
    self.assertNotEqual(hash(l), hash(l.clone()))

  def test_render(self):
    l = LangFunc('Hello {{x}} and {{y}}',
                 x=1, y=LangFunc('{{z}}'), z=3)
    v = l.render(x=2)
    self.assertEqual(v.text, 'Hello 2 and 3')
    self.assertEqual(v.x, 2)
    self.assertTrue(pg.eq(l.y, v.y))
    self.assertEqual(v.y.lm_input.z, 3)

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
        "LangFunc(template_str='Hello', returns=None, lm=ExcitedEchoer("
        'sampling_options=LMSamplingOptions(temperature=0.0, max_tokens=1024, '
        'n=1, top_k=40, top_p=None, random_seed=None), timeout=30.0, '
        'max_attempts=5, debug=False), input_transform=None, '
        'output_transform=None, clean=True)',
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

  def test_no_clean(self):
    l = LangFunc('\n Hello\n ')
    self.assertEqual(l.render(), 'Hello')

    l = LangFunc('\n Hello\n ', clean=False)
    self.assertEqual(l.render(), '\n Hello\n ')

  def test_render_cache(self):
    class DynamicContent(LangFunc):
      """Random number.

      {{_counter}}
      """

      def _on_bound(self):
        super()._on_bound()
        self._counter = 0

      def render(self, **kwargs):
        self._counter += 1
        return super().render(**kwargs)

    l = LangFunc('{{x}} + {{x}} =', x=DynamicContent())
    self.assertEqual(l.render(), '1 + 1 =')

    l = LangFunc('{{x}} + {{x.render()}} =', x=DynamicContent())
    self.assertEqual(l.render(), '1 + 2 =')

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

  def test_constant(self):
    l = LangFunc('Hello')
    self.assertEqual(l.render(), 'Hello')
    self.assertEqual(l.natural_language_format(), 'Hello')

  def test_call_with_init_time_vars(self):
    l = LangFunc('Hello, {{foo}}', foo='a')
    self.assertEqual(l.foo, 'a')
    self.assertEqual(l.get('foo'), 'a')
    self.assertEqual(l.natural_language_format(), 'Hello, a')

    with self.assertRaises(AttributeError):
      _ = l.bar
    self.assertEqual(l.get('bar'), pg.MISSING_VALUE)
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

    class FakeParseStructured(message_transform.MessageTransform):

      def _transform_path(self, unused_message, input_path, value):
        return int(str(value))

    prev_as_structured = message_transform.MessageTransform.as_structured
    message_transform.MessageTransform.as_structured = (
        lambda self, *args: self >> FakeParseStructured())

    l = LangFunc('Compute 1 + 2', returns=int)
    with component.context(lm=fake.StaticMapping({
        'Compute 1 + 2': '3',
    })):
      r = l()
      self.assertEqual(r.result, 3)

    l = LangFunc('Compute 1 + 2', returns=int, output_transform=lambda x: '3')
    with component.context(
        lm=fake.StaticMapping({
            'Compute 1 + 2': 'three',
        })
    ):
      r = l()
      self.assertEqual(r.result, 3)
    message_transform.MessageTransform.as_structured = prev_as_structured

  def test_nesting(self):
    l = LangFunc(
        'How {{x}}',
        x=LangFunc('are {{y}}', y=LangFunc('you {{n}}')),
        n=1,
    )
    self.assertEqual(l.x.y.n, 1)
    self.assertEqual(l.render(), 'How are you 1')

  def test_subclassing(self):

    class MyPrompt(LangFunc):
      template_str = 'Hello {{x}}, {{y}} and {{z}}'
      x: int = 1
      p: str

      @property
      def z(self):
        return self.p + '!'

    # `p` is required.
    with self.assertRaisesRegex(TypeError, '.* missing 1 required argument'):
      MyPrompt()()

    l = MyPrompt(p='abc')
    self.assertEqual(l.render(y=2), 'Hello 1, 2 and abc!')

  def test_subclassing_with_docstr(self):
    class MyPrompt(LangFunc):
      """My prompt.

      Hello {{x}} and {{y}}
      """

      x: int = 1

    l = MyPrompt()
    self.assertEqual(
        MyPrompt.__schema__.get_field('template_str').default_value,
        'Hello {{x}} and {{y}}',
    )
    self.assertEqual(l.render(y=2), 'Hello 1 and 2')

  def test_subclassing_with_no_template_sign_docstr(self):
    class MyPrompt(LangFunc):
      """My prompt.

      This is a longer version of docstr.

      (THIS IS NOT A TEMPLATE)
      """

      x: int = 1

    self.assertEqual(
        MyPrompt.__schema__.get_field('template_str').default_value,
        pg.MISSING_VALUE,
    )
    with self.assertRaisesRegex(TypeError, '.* missing 1 required argument'):
      MyPrompt(y=2)()

  def test_variables(self):
    class A(LangFunc):
      """A.

      Hello {{x}}{{y}}{{z}}
      """

      x: int

    class B(LangFunc):
      """B.

      hi {{x}}{{p}}{{q}}
      """

      p: str = 'foo'

    b = B(q=LangFunc('There {{i}}{{j}}{{x}}', i=0))
    a = A(x=1, y=b)

    # Test all direct referred variables.
    self.assertEqual(a.vars(), set(['x', 'y', 'z']))
    self.assertEqual(b.vars(), set(['x', 'p', 'q']))
    self.assertEqual(b.q.vars(), set(['x', 'i', 'j']))

    # Test direct referred variables that are specified.
    self.assertEqual(a.vars(specified=True), set(['x', 'y']))
    self.assertEqual(b.vars(specified=True), set(['x', 'p', 'q']))
    self.assertEqual(b.q.vars(specified=True), set(['x', 'i']))

    # Test direct referred variables that are not specified.
    self.assertEqual(a.vars(specified=False), set(['z']))
    self.assertEqual(b.vars(specified=False), set())
    self.assertEqual(b.q.vars(specified=False), set(['j']))

    # Test all referred variables in the closure.
    self.assertEqual(
        a.vars(closure=True), set(['x', 'y', 'z', 'p', 'q', 'i', 'j'])
    )
    self.assertEqual(b.vars(closure=True), set(['x', 'q', 'p', 'i', 'j']))
    self.assertEqual(b.q.vars(closure=True), set(['x', 'i', 'j']))

    # Test all leaf variables.
    self.assertEqual(
        a.vars(closure=True, leaf=True), set(['x', 'z', 'p', 'i', 'j'])
    )
    self.assertEqual(b.vars(closure=True, leaf=True), set(['x', 'p', 'i', 'j']))
    self.assertEqual(b.q.vars(closure=True, leaf=True), set(['x', 'i', 'j']))

    # Test all non-leaf variables.
    self.assertEqual(a.vars(closure=True, leaf=False), set(['y', 'q']))
    self.assertEqual(b.vars(closure=True, leaf=False), set(['q']))
    self.assertEqual(b.q.vars(closure=True, leaf=False), set())

    # Test missing variables.
    self.assertEqual(a.missing_vars, set(['z', 'j']))
    self.assertEqual(b.missing_vars, set(['j']))
    self.assertEqual(b.q.missing_vars, set(['j']))

  def test_cleandoc(self):
    l = LangFunc("""
        Hello
         {{foo}}
           {{bar}}
        """)
    self.assertEqual(l.render(foo='a', bar='b'), 'Hello\n a\n   b')

  def test_custom_typing(self):
    class Foo(component.Component):
      x: str | None
      y: int | None
      z: str
      p: component.Component | None

    d = Foo(z='bar')
    # String template can be assigned to str.
    d.x = LangFunc('Hello, {{y}}')

    d.y = 1
    d.z = LangFunc('Bye, {{y}}')

    # String template can be assigned to Component.
    d.p = LangFunc('Again {{x}}')

    self.assertEqual(d.x.render(), 'Hello, 1')
    self.assertEqual(d.z.render(), 'Bye, 1')
    self.assertEqual(d.p.render(), 'Again Hello, 1')

  def test_render_event(self):
    l = LangFunc(
        'Subject: {{subject}}', subject=LangFunc('The science of {{name}}')
    )

    render_events = []
    render_stacks = []

    class RenderEventHandler(subscription.EventHandler[LangFuncRenderEvent]):

      def on_event(self, event: LangFuncRenderEvent):
        render_events.append(event.lm_input)
        render_stacks.append(event.render_stack)

    x = RenderEventHandler()
    subscription.subscribe(x)
    l.render(name='rocket')
    self.assertEqual(
        render_events,
        ['The science of rocket', 'Subject: The science of rocket'],
    )
    self.assertEqual(render_stacks, [[l], []])
    subscription.unsubscribe(x)
    subscription.subscribe(x, l.subject)

    render_events[:] = []
    render_stacks[:] = []
    l.render(name='ballon')
    self.assertEqual(render_events, ['The science of ballon'])
    self.assertEqual(render_stacks, [[l]])

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

    # Test hybrid event handler.
    class HybridEventHandler(subscription.EventHandler[LangFuncEvent]):

      def __init__(self):
        super().__init__()
        self.num_renders = 0
        self.num_calls = 0

      def on_event(self, event: LangFuncEvent):
        if isinstance(event, LangFuncCallEvent):
          self.num_calls += 1
        elif isinstance(event, LangFuncRenderEvent):
          self.num_renders += 1

    h = HybridEventHandler()
    subscription.subscribe(h)

    with component.context(lm=ExcitedEchoer()):
      l(name='hope')

    self.assertEqual(h.num_calls, 2)
    self.assertEqual(h.num_renders, 4)

  def test_transform(self):
    t = message_transform.Identity() >> LangFunc('hi {{message.text}}')
    i = message.AIMessage('foo')
    with component.context(lm=ExcitedEchoer()):
      r = t.transform(i)
    self.assertEqual(r, message.AIMessage('hi foo!!!'))
    self.assertEqual(r.tags, ['lm-response', 'lm-output', 'transformed'])
    self.assertEqual(
        r.lm_input, message.UserMessage('hi foo', message=pg.Ref(i)))
    self.assertEqual(r.lm_input.tags, ['rendered', 'lm-input'])
    self.assertIs(r.root, i)

    t = LangFunc('hi {{message.text}}', input_path='result')
    with component.context(lm=ExcitedEchoer()):
      r = t.transform(message.AIMessage('abc', result='bar'))
      self.assertEqual(r, message.AIMessage('hi bar!!!'))
      self.assertEqual(r.tags, ['lm-response', 'lm-output', 'transformed'])
      self.assertEqual(
          r.source,
          message.UserMessage('hi bar', message=pg.Ref(r.source.source))
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


class TemplateImplementationTest(unittest.TestCase):
  """Tests for Template."""

  def test_variables(self):
    self.assertEqual(len(_Template('Hello').variables), 0)
    self.assertEqual(
        _Template('Hello {{foo}} {{bar}}').variables,
        set(['foo', 'bar']),
    )

  def test_render(self):
    self.assertEqual(
        _Template('Hello').render(dict()),
        ('Hello', {})
    )

    # Render by dict.
    inputs = dict(foo='a', bar='b')
    self.assertEqual(
        _Template('Hello {{foo}} {{bar}}').render(inputs),
        ('Hello a b', dict(foo='a', bar='b')),
    )

    # Render by getter.
    def var_getter(var_name):
      return inputs.get(var_name, pg.MISSING_VALUE)

    self.assertEqual(
        _Template('Hello {{foo}} {{bar}}').render(var_getter),
        ('Hello a b', dict(foo='a', bar='b')),
    )
    # Bad render.
    with self.assertRaises(ValueError):
      _Template('Hello {{x}}').render(inputs)

    with self.assertRaises(ValueError):
      _Template('Hello {{x}}').render(var_getter)

    with self.assertRaises(ValueError):
      _Template('Hello {{x}}').render(None)

  def test_partial_rendering(self):
    # Test attributes access.
    self.assertEqual(
        _Template('Hello {{x}} {{y}}').render(dict(y=1), allow_partial=True)[0],
        'Hello {{x}} 1'
    )

    self.assertEqual(
        _Template('Hello {{x[0].y.z()}}').render(dict(), allow_partial=True)[0],
        'Hello {{x[0].y.z()}}'
    )

    self.assertEqual(
        _Template('Hello {{f(1, 2, a=\'foo\')}}').render(
            dict(), allow_partial=True)[0],
        'Hello {{f(1, 2, a=\'foo\')}}'
    )

    # Test arithmetic operations.
    self.assertEqual(
        _Template('Hello {{ -x }}').render(dict(), allow_partial=True)[0],
        'Hello {{-x}}'
    )

    self.assertEqual(
        _Template('Hello {{x + 1}}').render(dict(), allow_partial=True)[0],
        'Hello {{x + 1}}'
    )

    self.assertEqual(
        _Template('Hello {{1 + x}}').render(dict(), allow_partial=True)[0],
        'Hello {{1 + x}}')

    self.assertEqual(
        _Template('Hello {{x - 1}}').render(dict(), allow_partial=True)[0],
        'Hello {{x - 1}}')

    self.assertEqual(
        _Template('Hello {{1 - x}}').render(dict(), allow_partial=True)[0],
        'Hello {{1 - x}}')

    self.assertEqual(
        _Template('Hello {{x * 2}}').render(dict(), allow_partial=True)[0],
        'Hello {{x * 2}}')

    self.assertEqual(
        _Template('Hello {{2 * x}}').render(dict(), allow_partial=True)[0],
        'Hello {{2 * x}}')

    self.assertEqual(
        _Template('Hello {{x / 2}}').render(dict(), allow_partial=True)[0],
        'Hello {{x / 2}}')

    self.assertEqual(
        _Template('Hello {{2 / x}}').render(dict(), allow_partial=True)[0],
        'Hello {{2 / x}}')

    self.assertEqual(
        _Template('Hello {{x // 2}}').render(dict(), allow_partial=True)[0],
        'Hello {{x // 2}}')

    self.assertEqual(
        _Template('Hello {{2 // x}}').render(dict(), allow_partial=True)[0],
        'Hello {{2 // x}}')

    self.assertEqual(
        _Template('Hello {{x ** 2}}').render(dict(), allow_partial=True)[0],
        'Hello {{x ** 2}}')

    self.assertEqual(
        _Template('Hello {{2 ** x}}').render(dict(), allow_partial=True)[0],
        'Hello {{2 ** x}}')

    self.assertEqual(
        _Template('Hello {{x % 2}}').render(dict(), allow_partial=True)[0],
        'Hello {{x % 2}}')

    self.assertEqual(
        _Template('Hello {{2 % x}}').render(dict(), allow_partial=True)[0],
        'Hello {{2 % x}}')

    # Test logic operations.
    self.assertEqual(
        _Template('Hello {{x == 1}}').render(dict(), allow_partial=True)[0],
        'Hello {{x == 1}}')

    self.assertEqual(
        _Template('Hello {{1 == x}}').render(dict(), allow_partial=True)[0],
        'Hello {{x == 1}}')

    self.assertEqual(
        _Template('Hello {{x != 1}}').render(dict(), allow_partial=True)[0],
        'Hello {{x != 1}}')

    self.assertEqual(
        _Template('Hello {{1 != x}}').render(dict(), allow_partial=True)[0],
        'Hello {{x != 1}}')

    self.assertEqual(
        _Template('Hello {{x > y}}').render(dict(), allow_partial=True)[0],
        'Hello {{x > y}}')

    self.assertEqual(
        _Template('Hello {{x >= y}}').render(dict(), allow_partial=True)[0],
        'Hello {{x >= y}}')

    self.assertEqual(
        _Template('Hello {{x <= y}}').render(dict(), allow_partial=True)[0],
        'Hello {{x <= y}}')

    self.assertEqual(
        _Template('Hello {{x < y}}').render(dict(), allow_partial=True)[0],
        'Hello {{x < y}}')

    # Test list format.
    self.assertEqual(
        _Template(inspect.cleandoc("""
            {%- for example in examples %}
            {{ example -}}
            {% endfor %}""")).render(dict(), allow_partial=True)[0].strip(),
        inspect.cleandoc("""
            {{examples.item0}}
            {{examples.item1}}
            {{examples.item2}}
            """))

    self.assertEqual(
        _Template(inspect.cleandoc("""
            {%- for k in examples.keys() %}
            {{ k -}}
            {% endfor %}""")).render(dict(), allow_partial=True)[0].strip(),
        inspect.cleandoc("""
            {{examples.key0}}
            {{examples.key1}}
            {{examples.key2}}
            """))

    self.assertEqual(
        _Template(inspect.cleandoc("""
            {%- for v in examples.values() %}
            {{ v -}}
            {% endfor %}""")).render(dict(), allow_partial=True)[0].strip(),
        inspect.cleandoc("""
            {{examples.value0}}
            {{examples.value1}}
            {{examples.value2}}
            """))

    self.assertEqual(
        _Template(inspect.cleandoc("""
            {%- for k, v in examples.items() %}
            {{ k }}: {{ v -}}
            {% endfor %}""")).render(dict(), allow_partial=True)[0].strip(),
        inspect.cleandoc("""
            {{examples.key0}}: {{examples.value0}}
            {{examples.key1}}: {{examples.value1}}
            {{examples.key2}}: {{examples.value2}}
            """))

    # Test len.
    self.assertEqual(
        _Template('Hello {{len(x)}}').render(dict(), allow_partial=True)[0],
        'Hello {{len(x)}}')


if __name__ == '__main__':
  unittest.main()
