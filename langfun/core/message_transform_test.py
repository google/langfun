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
"""Base transform tests."""

import re
import unittest
from langfun.core import message_transform as base
from langfun.core.message import AIMessage
import pyglove as pg


class MessageTransformTest(unittest.TestCase):

  def test_transform(self):
    t = base.Identity() >> base.Lambda(lambda x: AIMessage('hello'))
    i = AIMessage('hi')
    r = t.transform(i)
    self.assertEqual(r, AIMessage('hello'))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])
    self.assertEqual(r.trace(), [i, r])

  def test_input_output_path(self):
    t = base.Lambda(lambda x: x * 2, input_path='text', output_path='text')
    i = AIMessage('hi')
    r = t.transform(i)
    self.assertEqual(r, AIMessage('hi' * 2))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])
    self.assertIsNone(r.source)

    t = base.Lambda(lambda x: x, input_path='non-exist')
    with self.assertRaisesRegex(
        KeyError, 'Input path .* does not exist in the message.'
    ):
      t.transform(AIMessage('hi'))

  def test_make_transform(self):
    self.assertTrue(pg.eq(base.make_transform(int), base.ParseInt()))
    self.assertTrue(pg.eq(base.make_transform(float), base.ParseFloat()))
    self.assertTrue(pg.eq(base.make_transform(bool), base.ParseBool()))
    self.assertTrue(pg.eq(
        base.make_transform(re.compile(r'\d+')), base.Match(r'\d+')))
    x = lambda x: x
    self.assertTrue(pg.eq(base.make_transform(x), base.Lambda(x)))
    with self.assertRaisesRegex(
        ValueError, 'Unsupported expression'
    ):
      base.make_transform(1)

  def test_symbolic_conversion(self):

    class A(pg.Object):
      x: base.MessageTransform

    a = A(lambda x: x)
    self.assertIsInstance(a.x, base.Lambda)


class SaveAsTest(unittest.TestCase):

  def test_transform(self):
    t = base.SaveAs(input_path='result', output_path='x')
    i = AIMessage('hi', result=1)
    r = t.transform(i)
    self.assertIs(r, i)
    self.assertEqual(r, AIMessage('hi', x=1))
    self.assertIsNone(r.source)

    t = base.SaveAs(input_path='result', output_path='x', remove_input=False)
    self.assertEqual(
        t.transform(AIMessage('hi', result=1)), AIMessage('hi', result=1, x=1)
    )


class MultiTransformCompositionTest(unittest.TestCase):

  def assert_topology(
      self,
      cls,
      input_output,
      child_input_outputs,
      expected_input_output,
      expected_child_input_outputs,
  ):
    transforms = [
        base.Identity(input_path=i, output_path=o)
        for i, o in child_input_outputs
    ]
    t = cls(
        input_path=input_output[0],
        output_path=input_output[1],
        transforms=transforms,
    )
    self.assertEqual(t.input_path, expected_input_output[0])
    self.assertEqual(t.output_path, expected_input_output[1])
    self.assertEqual(
        [(c.input_path, c.output_path) for c in t.transforms],
        expected_child_input_outputs,
    )


class SequentialTest(MultiTransformCompositionTest):

  def test_topology(self):
    self.assert_topology(
        base.Sequential,
        input_output=(None, None),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=(None, None),
        expected_child_input_outputs=[(None, None), (None, None), (None, None)],
    )

    self.assert_topology(
        base.Sequential,
        input_output=('text', None),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=('text', None),
        expected_child_input_outputs=[
            ('text', None),
            (None, None),
            (None, None),
        ],
    )

    self.assert_topology(
        base.Sequential,
        input_output=(None, 'result'),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=(None, 'result'),
        expected_child_input_outputs=[
            (None, 'result'),
            ('result', 'result'),
            ('result', 'result'),
        ],
    )

    self.assert_topology(
        base.Sequential,
        input_output=('text', 'result'),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=('text', 'result'),
        expected_child_input_outputs=[
            ('text', 'result'),
            ('result', 'result'),
            ('result', 'result'),
        ],
    )

    self.assert_topology(
        base.Sequential,
        input_output=('text', 'result'),
        child_input_outputs=[
            ('result', 'text'),
            ('text', 'result2'),
            (None, 'result3'),
        ],
        expected_input_output=('result', 'result'),
        expected_child_input_outputs=[
            ('result', 'text'),
            ('text', 'result2'),
            ('result2', 'result3'),
        ],
    )

  def test_transform(self):
    # default, output to result.
    x2 = lambda x: 2 * x
    t = x2 >> base.Lambda(x2) >> x2
    r = t.transform(AIMessage('hi', result=1))
    self.assertEqual(r, AIMessage('hi', result=2**3))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])

    # Take message as the input, output transformed text as the message text.
    y = lambda x: str(x) + '!'
    t = (base.Lambda(y) >> y >> y).as_text()
    r = t.transform(AIMessage('hi'))
    self.assertEqual(r, AIMessage('hi!!!'))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])

    # Take message as the input, output transformed text as metadata.
    t = (base.Lambda(y) >> y >> y).as_metadata('x')
    r = t.transform(AIMessage('hi'))
    self.assertEqual(r, AIMessage('hi', x='hi!!!'))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])

    # Branching outputs.
    t = (base.Lambda(y).as_metadata('x') >> y >> y).as_metadata('y')
    r = t.transform(AIMessage('hi'))
    self.assertEqual(r, AIMessage('hi', x='hi!', y='hi!!!'))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])

    t = base.Lambda(x2)
    self.assertIs(t, t >> None)
    self.assertIs(t, None >> t)


class LogicalOrTest(MultiTransformCompositionTest):

  def test_topology(self):
    self.assert_topology(
        base.LogicalOr,
        input_output=(None, None),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=(None, None),
        expected_child_input_outputs=[(None, None), (None, None), (None, None)],
    )

    self.assert_topology(
        base.LogicalOr,
        input_output=('text', None),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=('text', None),
        expected_child_input_outputs=[
            ('text', None),
            ('text', None),
            ('text', None),
        ],
    )

    self.assert_topology(
        base.LogicalOr,
        input_output=(None, 'text'),
        child_input_outputs=[(None, None), (None, None), (None, None)],
        expected_input_output=(None, 'text'),
        expected_child_input_outputs=[
            (None, 'text'),
            (None, 'text'),
            (None, 'text'),
        ],
    )

    self.assert_topology(
        base.LogicalOr,
        input_output=('text', None),
        child_input_outputs=[('result', None), (None, None), (None, None)],
        expected_input_output=('text', None),
        expected_child_input_outputs=[
            ('result', None),
            ('text', None),
            ('text', None),
        ],
    )

    self.assert_topology(
        base.LogicalOr,
        input_output=(None, 'x'),
        child_input_outputs=[('result', None), ('x', None), (None, None)],
        expected_input_output=(None, 'x'),
        expected_child_input_outputs=[('result', 'x'), ('x', 'x'), (None, 'x')],
    )

    x = lambda x: x
    with self.assertRaisesRegex(
        ValueError,
        'The branches of `LogicalOr` should have the same output path.',
    ):
      _ = base.Lambda(x).as_text() | x | x

  def test_transform(self):
    def x1(unused_x):
      raise ValueError()

    t = (
        x1 | base.Lambda(lambda x: 2 * str(x)) | (lambda x: 4 * str(x))
    ).as_text()
    self.assertEqual(t.transform(AIMessage('a')), 'a' * 2)

    t = (
        base.Lambda(x1)
        | base.Lambda(lambda x: 2 * str(x)) >> (lambda x: 2 * str(x))
        | (lambda x: x)
    ).as_text()
    self.assertEqual(t.transform(AIMessage('a')), AIMessage('a' * 2 * 2))

    t = base.Lambda(x1) | base.Lambda(x1)
    with self.assertRaisesRegex(
        ValueError, 'None of the child transforms has run successfully'):
      t.transform(1)


class RetryTest(unittest.TestCase):

  def test_topology(self):
    x = lambda x: x

    t = base.Lambda(x).retry(2)
    self.assertEqual((t.input_path, t.output_path), (None, None))
    self.assertEqual(
        (t.child_transform.input_path, t.child_transform.output_path),
        (None, None),
    )

    t = base.Lambda(x, input_path='text').retry(2)
    self.assertEqual((t.input_path, t.output_path), ('text', None))
    self.assertEqual(
        (t.child_transform.input_path, t.child_transform.output_path),
        ('text', None),
    )

    t = base.Lambda(x, output_path='result').retry(2)
    self.assertEqual((t.input_path, t.output_path), (None, 'result'))
    self.assertEqual(
        (t.child_transform.input_path, t.child_transform.output_path),
        (None, 'result'),
    )

    t = (
        base.Identity(input_path='x')
        >> base.Lambda(x).retry(2)
        >> base.Identity(output_path='y')
    )
    self.assertEqual(
        [(c.input_path, c.output_path) for c in t.transforms],
        [('x', 'y'), ('y', 'y'), ('y', 'y')],
    )
    self.assertEqual(
        (t[1].child_transform.input_path, t[1].child_transform.output_path),
        ('y', 'y'),
    )

  def test_transform(self):
    def fail_until(calls: int, error_cls=ValueError):
      context = dict(calls=0)

      def fn(x):
        context['calls'] += 1
        if context['calls'] < calls:
          raise error_cls('Intentional failure.')
        return x

      return fn

    t = base.Lambda(fail_until(3)).retry(2)
    with self.assertRaisesRegex(ValueError, '.* failed after 2 attempts'):
      t.transform(AIMessage('hi'))

    t = base.Lambda(fail_until(3)).retry(3)
    r = t.transform(AIMessage('hi'))
    self.assertEqual(r, AIMessage('hi'))
    self.assertEqual(r.tags, [AIMessage.TAG_TRANSFORMED])

    t = base.Lambda(fail_until(2, KeyError)).retry(3, ValueError)
    with self.assertRaisesRegex(KeyError, 'Intentional failure'):
      t.transform(AIMessage('hi'))


class IdentityTest(unittest.TestCase):

  def test_same_instance(self):
    t = base.Identity()
    v = AIMessage('hi')
    self.assertIs(t.transform(v), v)

  def test_copy(self):
    t = base.Identity(copy=True)
    v = AIMessage('hi')
    self.assertIsNot(t.transform(v), v)
    self.assertEqual(t.transform(v), AIMessage('hi'))


class ConversionTest(unittest.TestCase):

  def assert_conversion(self, t, v, expected_v):
    m = AIMessage(v)
    o = t.transform(m)
    self.assertEqual(o, AIMessage(v, result=expected_v))
    self.assertEqual(o.tags, [AIMessage.TAG_TRANSFORMED])

  def test_to_bool(self):
    t = base.Identity(input_path='text').to_bool()
    self.assert_conversion(t, '1', True)
    self.assert_conversion(t, 'yes', True)
    self.assert_conversion(t, 'Yes', True)
    self.assert_conversion(t, 'YES', True)
    self.assert_conversion(t, 'true', True)
    self.assert_conversion(t, 'TRUE', True)
    self.assert_conversion(t, 'True', True)
    self.assert_conversion(t, '0', False)
    self.assert_conversion(t, 'no', False)
    self.assert_conversion(t, 'No', False)
    self.assert_conversion(t, 'NO', False)
    self.assert_conversion(t, 'false', False)
    self.assert_conversion(t, 'False', False)
    self.assert_conversion(t, 'FALSE', False)

    self.assert_conversion(
        base.Identity(input_path='text').to_bool(False), 'abc', False)

    with self.assertRaisesRegex(ValueError, 'Cannot convert .* to bool'):
      t.transform(AIMessage('abc'))

    with self.assertRaisesRegex(
        TypeError, "Metadata 'result' must be a string"):
      base.Identity().to_bool().transform(AIMessage('abc', result=[]))

  def test_to_int(self):
    t = base.Identity().to_int()
    self.assert_conversion(t, '1', 1)
    with self.assertRaisesRegex(ValueError, 'invalid literal for int'):
      t.transform(AIMessage('abc'))

    self.assert_conversion(
        base.Identity(input_path='text').to_int(None), 'abc', None)

  def test_to_float(self):
    t = base.Identity().to_float()
    self.assert_conversion(t, '1.1', 1.1)
    with self.assertRaisesRegex(
        ValueError, 'could not convert string to float'
    ):
      t.transform(AIMessage('abc'))

    self.assert_conversion(
        base.Identity(input_path='text').to_float(None), 'abc', None)

  def test_to_dict(self):
    t = base.Identity().to_dict()
    self.assert_conversion(t, '{"x": 1, "y": [1, 2]}', {'x': 1, 'y': [1, 2]})
    self.assert_conversion(
        base.Identity(input_path='text').to_dict(None), 'abc', None)


class MatchTest(unittest.TestCase):

  def assert_match(self, t, text, expected):
    m = AIMessage(text)
    self.assertEqual(t.transform(m).result, expected)

  def test_single_instance(self):
    t = base.Identity().match(r'(\d+)')
    self.assert_match(t, '1:2:3:4', '1')

  def test_single_instance_no_capture_group(self):
    t = base.Match(r'\d+')
    self.assert_match(t, '1:2:3:4', '1')

  def test_multiple(self):
    t = base.Match(r'(\d+):(\d+)', multiple=True)
    self.assert_match(t, '1:2:3:4', [('1', '2'), ('3', '4')])

  def test_group_alias(self):
    t = base.Match(r'(?P<x>\d+):(?P<y>\d+)', multiple=True)
    self.assert_match(t, '1:2:3:4', [dict(x='1', y='2'), dict(x='3', y='4')])

  def test_no_match(self):
    t = base.Match(r'(\d+)')
    with self.assertRaisesRegex(ValueError, 'No match found'):
      t.transform(AIMessage(''))

    t = base.Match(r'(\d+)', multiple=True)
    with self.assertRaisesRegex(ValueError, 'No match found'):
      t.transform(AIMessage(''))

    t = base.Match(r'(\d+)', default=None)
    self.assertEqual(t.transform(AIMessage('')), AIMessage('', result=None))


class MatchBlockTest(unittest.TestCase):

  def test_inclusive(self):
    t = base.Identity().match_block('<subject>', '</subject>', inclusive=True)
    self.assertEqual(
        t.transform(AIMessage('This is <subject>a test</subject>?')).result,
        '<subject>a test</subject>',
    )

  def test_exclusive(self):
    t = base.MatchBlock('<subject>', '</subject>', inclusive=False)
    self.assertEqual(t.parse('This is <subject>a test</subject>?'), 'a test')

  def test_multiple(self):
    t = base.MatchBlock(
        '<subject>', '</subject>', inclusive=False, multiple=True
    )
    self.assertEqual(
        t.parse('This is <subject>1</subject> and <subject>2</subject>'),
        ['1', '2'],
    )


if __name__ == '__main__':
  unittest.main()
