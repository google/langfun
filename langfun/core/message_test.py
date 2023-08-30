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
"""Tests for message."""

import unittest
from langfun.core import message
import pyglove as pg


class MessageTest(unittest.TestCase):

  def test_basics(self):
    m = message.UserMessage('hi', metadata=dict(x=1), x=2, y=2)
    self.assertEqual(m.metadata, {'x': 2, 'y': 2})
    self.assertEqual(m.sender, 'User')
    self.assertEqual(m.x, 2)
    self.assertEqual(m.y, 2)

    with self.assertRaises(AttributeError):
      _ = m.z
    self.assertEqual(hash(m), hash(m.text))

  def test_source_tracking(self):
    m1 = message.UserMessage('hi')
    m1.tag('lm-input')
    self.assertIsNone(m1.source)
    self.assertIs(m1.root, m1)

    m2 = message.UserMessage('foo', source=m1)
    m2.source = m1
    self.assertIs(m2.source, m1)
    self.assertIs(m2.root, m1)
    m2.tag('lm-response')

    m3 = message.UserMessage('bar', source=m2)
    self.assertIs(m3.source, m2)
    self.assertIs(m3.root, m1)
    m3.tag('transformed')
    m3.tag('lm-output')

    self.assertEqual(
        m3.trace(), [m1, m2, m3],
    )
    self.assertEqual(
        m3.trace('lm-input'), [m1]
    )
    self.assertEqual(
        m3.trace('transformed'), [m3]
    )
    self.assertIs(m2.lm_input, m1)
    self.assertIs(m3.lm_input, m1)
    self.assertEqual(m3.lm_inputs, [m1])
    self.assertIs(m2.lm_response, m2)
    self.assertIs(m3.lm_response, m2)
    self.assertEqual(m3.lm_responses, [m2])
    self.assertIs(m3.lm_output, m3)
    self.assertEqual(m3.lm_outputs, [m3])
    self.assertIsNone(m3.last('non-exist'))

  def test_result(self):
    m = message.UserMessage('hi', x=1, y=2)
    self.assertIsNone(m.result)
    m.result = 1
    self.assertEqual(m.result, 1)

  def test_jsonify(self):
    m = message.UserMessage('hi', result=1)
    self.assertEqual(pg.from_json_str(m.to_json_str()), m)

  def test_get(self):
    m = message.UserMessage('hi', x=1, y=dict(z=[0, 1, 2]))
    self.assertEqual(m.get('text'), 'hi')
    self.assertEqual(m.get('y'), dict(z=[0, 1, 2]))
    self.assertEqual(m.get('y.z'), [0, 1, 2])
    self.assertEqual(m.get('y.z[0]'), 0)
    self.assertIsNone(m.get('p'))
    self.assertEqual(m.get('p', default='foo'), 'foo')

  def test_set(self):
    m = message.UserMessage('hi', metadata=dict(x=1, z=0))
    m.set('text', 'hello')
    m.set('x', 2)
    m.set('y', [0, 1, 2])
    m.set('y[0]', 1)
    m.set('y[2]', pg.MISSING_VALUE)  # delete `y[2]`.
    m.set('z', pg.MISSING_VALUE)  # delete `z`.
    self.assertEqual(
        m, message.UserMessage('hello', metadata=dict(x=2, y=[1, 1]))
    )

  def test_updates(self):
    m = message.UserMessage('hi')
    self.assertFalse(m.modified)
    self.assertFalse(m.has_errors)

    with m.update_scope():
      m.metadata.x = 1
      m.metadata.y = 1
      self.assertTrue(m.modified)
      self.assertEqual(len(m.updates), 2)
      self.assertFalse(m.has_errors)

      with m.update_scope():
        m.metadata.y = 2
        m.metadata.z = 2
        m.errors.append(ValueError('b'))
        self.assertTrue(m.modified)
        self.assertEqual(len(m.updates), 2)
        self.assertTrue(m.has_errors)
        self.assertEqual(len(m.errors), 1)

        with m.update_scope():
          self.assertFalse(m.modified)
          self.assertFalse(m.has_errors)

    self.assertTrue(m.modified)
    self.assertEqual(len(m.updates), 3)
    self.assertTrue(m.has_errors)
    self.assertEqual(len(m.errors), 1)

    m2 = message.UserMessage('hi')
    m2.apply_updates(m.updates)
    self.assertEqual(m, m2)

  def test_user_message(self):
    m = message.UserMessage('hi')
    self.assertEqual(m.text, 'hi')
    self.assertEqual(m.sender, 'User')
    self.assertTrue(m.from_user)
    self.assertFalse(m.from_agent)
    self.assertFalse(m.from_system)
    self.assertFalse(m.from_memory)
    self.assertEqual(str(m), m.text)

    m = message.UserMessage('hi', sender='Tom')
    self.assertEqual(m.sender, 'Tom')
    self.assertEqual(str(m), m.text)

  def test_ai_message(self):
    m = message.AIMessage('hi')
    self.assertEqual(m.text, 'hi')
    self.assertEqual(m.sender, 'AI')
    self.assertFalse(m.from_user)
    self.assertTrue(m.from_agent)
    self.assertFalse(m.from_system)
    self.assertFalse(m.from_memory)
    self.assertEqual(str(m), m.text)

    m = message.AIMessage('hi', sender='Model')
    self.assertEqual(m.sender, 'Model')
    self.assertEqual(str(m), m.text)

  def test_system_message(self):
    m = message.SystemMessage('hi')
    self.assertEqual(m.text, 'hi')
    self.assertEqual(m.sender, 'System')
    self.assertFalse(m.from_user)
    self.assertFalse(m.from_agent)
    self.assertTrue(m.from_system)
    self.assertFalse(m.from_memory)
    self.assertEqual(str(m), m.text)

    m = message.SystemMessage('hi', sender='Environment1')
    self.assertEqual(m.sender, 'Environment1')
    self.assertEqual(str(m), m.text)

  def test_memory_record(self):
    m = message.MemoryRecord('hi')
    self.assertEqual(m.text, 'hi')
    self.assertEqual(m.sender, 'Memory')
    self.assertFalse(m.from_user)
    self.assertFalse(m.from_agent)
    self.assertFalse(m.from_system)
    self.assertTrue(m.from_memory)
    self.assertEqual(str(m), m.text)

    m = message.MemoryRecord('hi', sender="Someone's Memory")
    self.assertEqual(m.sender, 'Someone\'s Memory')
    self.assertEqual(str(m), m.text)


if __name__ == '__main__':
  unittest.main()
