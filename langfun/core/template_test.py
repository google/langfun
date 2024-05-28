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
"""Template test."""
import inspect
import unittest

from langfun.core import component
from langfun.core import message as message_lib
from langfun.core import modality
from langfun.core import subscription
from langfun.core.template import Template
from langfun.core.template import TemplateRenderEvent
import pyglove as pg


class BasicTest(unittest.TestCase):

  def test_rebind(self):
    l = Template('Hello')
    self.assertEqual(l._variables, set())
    l.rebind(template_str='Hello {{x}}')
    self.assertEqual(l._variables, set(['x']))

  def test_attrs(self):
    l = Template('Hello {{x}}', x=1)
    self.assertEqual(l.template_str, 'Hello {{x}}')
    self.assertTrue(l.clean)
    self.assertEqual(l.x, 1)

  def test_render_output(self):
    l = Template('Hello {{x}}', x=[1])
    self.assertIsNone(l.render_output)
    self.assertEqual(l.render(), 'Hello `[1]`')
    self.assertEqual(l.render_output, 'Hello `[1]`')

  def test_str(self):
    self.assertEqual(str(Template('Hello')), 'Hello')
    self.assertEqual(str(Template('Hello {{x}}')), 'Hello {{x}}')
    self.assertEqual(str(Template('Hello {{x}}', x=1)), 'Hello 1')

  def test_hash(self):
    l = Template('Hello')
    self.assertEqual(hash(l), hash(l))
    self.assertNotEqual(hash(l), hash(l.clone()))

  def test_eq(self):
    self.assertEqual(Template('Hello'), 'Hello')
    self.assertNotEqual(Template('Hello'), 'Hello\n')

    self.assertEqual(Template('Hello {{x}}', x='World'), 'Hello World')
    self.assertNotEqual(
        Template('Hello {{x}}', x='World'), Template('Hello {{x}}', x='World')
    )

    self.assertTrue(
        pg.eq(
            Template('Hello {{x}}', x='World'),
            Template('Hello {{x}}', x='World'),
        )
    )

    self.assertTrue(
        pg.ne(
            Template('Hello {{x}}', x='World'),
            Template('Hello {{x}}', x='Word'),
        )
    )

  def test_custom_typing(self):
    class Foo(component.Component):
      x: str | None
      y: int | None
      z: str
      p: component.Component | None

    d = Foo(z='bar')
    # String template can be assigned to str.
    d.x = Template('Hello, {{y}}')

    d.y = 1
    d.z = Template('Bye, {{y}}')

    # String template can be assigned to Component.
    d.p = Template('Again {{x}}')

    self.assertEqual(d.x.render(), 'Hello, 1')
    self.assertEqual(d.z.render(), 'Bye, 1')
    self.assertEqual(d.p.render(), 'Again Hello, 1')


class DefinitionTest(unittest.TestCase):

  def test_subclassing(self):

    class MyPrompt(Template):
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

    class MyPrompt(Template):
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

    class MyPrompt(Template):
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

  def test_bad_template(self):
    with self.assertRaisesRegex(ValueError, 'Bad template string.*'):
      Template('{{x=1')


class VarsTest(unittest.TestCase):

  def assert_missing_vars(self, t: Template, missing_vars: set[str]):
    self.assertEqual(t.missing_vars, missing_vars)

  def test_missing_vars(self):
    self.assert_missing_vars(Template('Hello'), set())
    self.assert_missing_vars(Template('Hello {{x}}', x=1), set())
    self.assert_missing_vars(Template('Hello {{x}}'), set(['x']))
    self.assert_missing_vars(
        Template('Hello {{x}}', x=Template('{{y}}')), set(['y'])
    )

  def test_vars(self):
    class A(Template):
      """A.

      Hello {{x}}{{y}}{{z}}
      """

      x: int

    class B(Template):
      """B.

      hi {{x}}{{p}}{{q}}
      """

      p: str = 'foo'

    b = B(q=Template('There {{i}}{{j}}{{x}}', i=0))
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


class RenderTest(unittest.TestCase):

  def test_clean(self):
    l = Template('\n Hello\n ')
    self.assertEqual(l.render(), 'Hello')

    l = Template("""
        Hello
         {{foo}}
           {{bar}}
        """)
    self.assertEqual(l.render(foo='a', bar='b'), 'Hello\n a\n   b')

  def test_no_clean(self):
    l = Template('\n Hello\n ', clean=False)
    self.assertEqual(l.render(), '\n Hello\n ')

  def test_constant_template(self):
    l = Template('Hello')
    self.assertEqual(l.render(), 'Hello')
    self.assertEqual(l.natural_language_format(), 'Hello')

  def test_render_without_call_args(self):
    l = Template(
        'How {{x}}',
        x=Template('are {{y}}', y=Template('you {{n}}')),
        n=1,
    )
    v = l.render()
    self.assertEqual(v, 'How are you 1')
    self.assertIs(v.x, l.x)
    self.assertEqual(v.x.render_output, 'are you 1')
    self.assertIs(v.x.y, l.x.y)
    self.assertEqual(v.x.y.render_output, 'you 1')
    self.assertEqual(v.x.y.n, 1)

  def test_render_with_call_args(self):
    l = Template('Hello {{x}} and {{y}}', x=1, y=Template('{{z}}'), z=3)
    v = l.render(x=2)
    self.assertEqual(v, 'Hello 2 and 3')
    self.assertEqual(v.x, 2)
    self.assertEqual(v.y, '3')
    self.assertEqual(v.y.z, 3)

  def test_render_cache(self):
    class DynamicContent(Template):
      """Random number.

      {{_counter}}
      """

      def _on_bound(self):
        super()._on_bound()
        self._counter = 0

      def render(self, **kwargs):
        self._counter += 1
        return super().render(**kwargs)

    l = Template('{{x}} + {{x}} =', x=DynamicContent())
    self.assertEqual(l.render(), '1 + 1 =')

    l = Template('{{x}} + {{x.render()}} =', x=DynamicContent())
    self.assertEqual(l.render(), '1 + 2 =')

  def test_render_with_modality(self):
    class CustomModality(modality.Modality):
      content: str

      def to_bytes(self):
        return self.content.encode()

    self.assertEqual(
        Template(
            'This is {{ x }} and {{ a }}', x=1, a=CustomModality('foo')
        ).render(),
        'This is 1 and <<[[a]]>>',
    )

  def test_render_with_default(self):

    class Foo(Template):
      """Foo.

      This is {{x}}
      """

    f = Foo(template_str='!{{DEFAULT}}!', x=1)
    self.assertEqual(f.DEFAULT.x, 1)
    self.assertEqual(
        f.render(), '!This is 1!'
    )

    class Bar(Template):
      """Bar.

      {{preamble}}
      {{prompt}}
      """

      preamble: Template = Template('You are a chat bot.')
      prompt: Template = Template('User: hi! {{name}}')

    b = Bar(
        preamble=Template('<h1>{{DEFAULT}}</h1>'),
        prompt=Template('<h2>{{DEFAULT}}</h2>'),
        name='Tom',
    )
    # Test variable access.
    self.assertEqual(
        b.render(),
        inspect.cleandoc("""
            <h1>You are a chat bot.</h1>
            <h2>User: hi! Tom</h2>
            """),
    )

    with self.assertRaisesRegex(ValueError, '`{{ DEFAULT }}` cannot be used'):

      class Baz(Template):  # pylint: disable=unused-variable
        """Baz.

        {{DEFAULT}}
        """

    with self.assertRaisesRegex(
        ValueError, 'The template neither has a default `template_str` nor'
    ):
      Template('{{DEFAULT}}').render()

    d = pg.Dict(x=Template('{{DEFAULT}}'))
    with self.assertRaisesRegex(
        ValueError, 'does not have a default value'
    ):
      _ = d.x.DEFAULT

    class Tes(pg.Object):
      x: str | None = None

    t = Tes(x=Template('{{DEFAULT}}'))
    with self.assertRaisesRegex(
        ValueError, 'is not a `lf.Template` object or str'
    ):
      _ = t.x.DEFAULT

  def test_bad_render(self):
    with self.assertRaises(ValueError):
      Template('Hello {{x}}').render(allow_partial=False)

  def assert_partial(self, t: Template, expected_text: str):
    self.assertEqual(t.render(allow_partial=True), expected_text)

  def test_partial_rendering(self):
    self.assert_partial(Template('Hello {{x}} {{y}}'), 'Hello {{x}} {{y}}')

    self.assert_partial(Template('Hello {{x}} {{y}}', y=1), 'Hello {{x}} 1')

    self.assert_partial(
        Template('Hello {{x[0].y.z()}}'), 'Hello {{x[0].y.z()}}'
    )

    self.assert_partial(
        Template("Hello {{f(1, 2, a='foo')}}"), "Hello {{f(1, 2, a='foo')}}"
    )

    # Test arithmetic operations.
    self.assert_partial(Template('Hello {{ -x }}'), 'Hello {{-x}}')

    self.assert_partial(Template('Hello {{ x + 1 }}'), 'Hello {{x + 1}}')

    self.assert_partial(Template('Hello {{ 1 + x }}'), 'Hello {{1 + x}}')

    self.assert_partial(Template('Hello {{ x - 1 }}'), 'Hello {{x - 1}}')

    self.assert_partial(Template('Hello {{ 1 - x }}'), 'Hello {{1 - x}}')

    self.assert_partial(Template('Hello {{ x * 2 }}'), 'Hello {{x * 2}}')

    self.assert_partial(Template('Hello {{ 2 * x }}'), 'Hello {{2 * x}}')

    self.assert_partial(Template('Hello {{ x / 2 }}'), 'Hello {{x / 2}}')

    self.assert_partial(Template('Hello {{ 2 / x }}'), 'Hello {{2 / x}}')

    self.assert_partial(Template('Hello {{ x // 2 }}'), 'Hello {{x // 2}}')

    self.assert_partial(Template('Hello {{ 2 // x }}'), 'Hello {{2 // x}}')

    self.assert_partial(Template('Hello {{ x ** 2 }}'), 'Hello {{x ** 2}}')

    self.assert_partial(Template('Hello {{ 2 ** x }}'), 'Hello {{2 ** x}}')

    self.assert_partial(Template('Hello {{ x % 2 }}'), 'Hello {{x % 2}}')

    self.assert_partial(Template('Hello {{ 2 % x }}'), 'Hello {{2 % x}}')

    # Test logic operations.
    self.assert_partial(Template('Hello {{ x == 1 }}'), 'Hello {{x == 1}}')

    self.assert_partial(Template('Hello {{ x != 1 }}'), 'Hello {{x != 1}}')

    self.assert_partial(Template('Hello {{ x > y }}'), 'Hello {{x > y}}')

    self.assert_partial(Template('Hello {{ x >= y }}'), 'Hello {{x >= y}}')

    self.assert_partial(Template('Hello {{ x <= y }}'), 'Hello {{x <= y}}')

    self.assert_partial(Template('Hello {{ x < y }}'), 'Hello {{x < y}}')

    # Test list format.
    self.assert_partial(
        Template("""
            {%- for example in examples %}
            {{ example -}}
            {% endfor %}"""),
        inspect.cleandoc("""
            {{examples.item0}}
            {{examples.item1}}
            {{examples.item2}}
            """),
    )

    self.assert_partial(
        Template("""
            {%- for k in examples.keys() %}
            {{ k -}}
            {% endfor %}"""),
        inspect.cleandoc("""
            {{examples.key0}}
            {{examples.key1}}
            {{examples.key2}}
            """),
    )

    self.assert_partial(
        Template("""
            {%- for v in examples.values() %}
            {{ v -}}
            {% endfor %}"""),
        inspect.cleandoc("""
            {{examples.value0}}
            {{examples.value1}}
            {{examples.value2}}
            """),
    )

    self.assert_partial(
        Template("""
            {%- for k, v in examples.items() %}
            {{ k }}: {{ v -}}
            {% endfor %}"""),
        inspect.cleandoc("""
            {{examples.key0}}: {{examples.value0}}
            {{examples.key1}}: {{examples.value1}}
            {{examples.key2}}: {{examples.value2}}
            """),
    )

    # Test len.
    self.assert_partial(Template('Hello {{len(x)}}'), 'Hello {{len(x)}}')

  def test_additional_metadata(self):
    t = Template('hi', metadata_weights=1.0, y=2)
    self.assertEqual(t.render(), message_lib.UserMessage('hi', weights=1.0))

    t = Template('hi')
    with component.context(metadata_weights=1.0, y=2):
      self.assertEqual(t.render(), message_lib.UserMessage('hi', weights=1.0))


class TemplateRenderEventTest(unittest.TestCase):

  def test_render_event(self):
    l = Template(
        'Subject: {{subject}}', subject=Template('The science of {{name}}')
    )

    render_events = []
    render_stacks = []

    class RenderEventHandler(subscription.EventHandler[TemplateRenderEvent]):

      def on_event(self, event: TemplateRenderEvent):
        render_events.append(event.output)
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


if __name__ == '__main__':
  unittest.main()
