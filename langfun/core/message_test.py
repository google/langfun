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

import inspect
import unittest
from langfun.core import language_model
from langfun.core import message
from langfun.core import modality
import pyglove as pg


class CustomModality(modality.Modality):
  content: str

  def to_bytes(self):
    return self.content.encode()


class MessageTest(unittest.TestCase):

  def test_basics(self):

    class A(pg.Object):
      pass

    d = pg.Dict(x=A())

    m = message.UserMessage(
        'hi',
        metadata=dict(x=1), x=pg.Ref(d.x),
        y=2,
        tags=['lm-input']
    )
    self.assertEqual(m.metadata, {'x': pg.Ref(d.x), 'y': 2})
    self.assertEqual(m.sender, 'User')
    self.assertIs(m.x, d.x)
    self.assertEqual(m.y, 2)
    self.assertTrue(m.has_tag('lm-input'))
    self.assertTrue(m.has_tag(('lm-input', '')))
    self.assertFalse(m.has_tag('lm-response'))

    with self.assertRaises(AttributeError):
      _ = m.z
    self.assertEqual(hash(m), hash(m.text))
    del d

  def test_from_value(self):
    self.assertTrue(
        pg.eq(message.UserMessage.from_value('hi'), message.UserMessage('hi'))
    )
    self.assertTrue(
        pg.eq(
            message.UserMessage.from_value(CustomModality('foo')),
            message.UserMessage('<<[[object]]>>', object=CustomModality('foo')),
        )
    )
    m = message.UserMessage('hi')
    self.assertIs(message.UserMessage.from_value(m), m)

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

    class A(pg.Object):
      p: int

    # Create a symbolic object and assign it to a container, so we could test
    # pg.Ref.
    a = A(1)
    d = pg.Dict(x=a)

    m = message.UserMessage('hi', x=pg.Ref(a), y=dict(z=[0, 1, 2]))
    self.assertEqual(m.get('text'), 'hi')
    self.assertIs(m.get('x'), a)
    self.assertIs(m.get('x.p'), 1)
    self.assertEqual(m.get('y'), dict(z=[0, 1, 2]))
    self.assertEqual(m.get('y.z'), [0, 1, 2])
    self.assertEqual(m.get('y.z[0]'), 0)
    self.assertIsNone(m.get('p'))
    self.assertEqual(m.get('p', default='foo'), 'foo')
    del d

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

  def test_get_modality(self):
    m1 = message.UserMessage(
        'hi, this is a {{img1}} and {{x.img2}}',
        img1=CustomModality('foo'),
        x=dict(img2=pg.Ref(CustomModality('bar'))),
    )
    self.assertIs(m1.get_modality('img1'), m1.img1)
    self.assertIs(m1.get_modality('x.img2'), m1.x.img2)
    self.assertIsNone(m1.get_modality('video'))

    m2 = message.SystemMessage('class Question:\n  image={{img1}}', source=m1)
    self.assertIs(m2.get_modality('img1'), m1.img1)
    # We could get the modality object even it's not directly used by current
    # message.
    self.assertIs(m2.get_modality('x.img2'), m1.x.img2)
    self.assertIsNone(m2.get_modality('video'))

    m3 = message.AIMessage(
        'This is the {{output_image}} based on {{x.img2}}',
        output_image=CustomModality('bar'),
        source=m2,
    )
    self.assertIs(m3.get_modality('x.img2'), m1.x.img2)
    self.assertIs(m3.get_modality('output_image'), m3.output_image)
    self.assertIsNone(m3.get_modality('video'))

  def test_referred_modalities(self):
    m1 = message.UserMessage(
        'hi, this is a <<[[img1]]>> and <<[[x.img2]]>>',
        img1=CustomModality('foo'),
        x=dict(img2=CustomModality('bar')),
    )
    m2 = message.SystemMessage('class Question:\n  image={{img1}}', source=m1)
    m3 = message.AIMessage(
        (
            'This is the <<[[output_image]]>> based on <<[[x.img2]]>>, '
            '{{unknown_var}}'
        ),
        output_image=CustomModality('bar'),
        source=m2,
    )
    self.assertEqual(
        m3.referred_modalities(),
        {
            'output_image': m3.output_image,
            'x.img2': m1.x.img2,
        },
    )

  def test_text_with_modality_hash(self):
    m = message.UserMessage(
        'hi, this is a <<[[img1]]>> and <<[[x.img2]]>>',
        img1=CustomModality('foo'),
        x=dict(img2=CustomModality('bar')),
    )
    self.assertEqual(
        m.text_with_modality_hash,
        (
            'hi, this is a <<[[img1]]>> and <<[[x.img2]]>>'
            '<img1>acbd18db</img1><x.img2>37b51d19</x.img2>'
        )
    )

  def test_chunking(self):
    m = message.UserMessage(
        inspect.cleandoc("""
            Hi, this is <<[[a]]>> and this is {{b}}.
            <<[[x.c]]>> {{something else
            """),
        a=CustomModality('foo'),
        x=dict(c=CustomModality('bar')),
    )
    chunks = m.chunk()
    self.assertTrue(
        pg.eq(
            chunks,
            [
                'Hi, this is',
                CustomModality('foo'),
                'and this is {{b}}.\n',
                CustomModality('bar'),
                '{{something else',
            ],
        )
    )
    self.assertTrue(
        pg.eq(
            message.AIMessage.from_chunks(chunks),
            message.AIMessage(
                inspect.cleandoc("""
                    Hi, this is <<[[obj0]]>> and this is {{b}}.
                    <<[[obj1]]>> {{something else
                    """),
                obj0=pg.Ref(m.a),
                obj1=pg.Ref(m.x.c),
            ),
        )
    )

  def assert_html_content(self, html, expected):
    expected = inspect.cleandoc(expected).strip()
    actual = html.content.strip()
    if actual != expected:
      print(actual)
    self.assertEqual(actual, expected)

  def test_html_style(self):
    self.assertIn(
        inspect.cleandoc(
            """
            /* Langfun Message styles.*/
            [class^="message-"] > details {
                margin: 0px 0px 5px 0px;
                border: 1px solid #EEE;
            }
            .lf-message.summary-title::after {
                content: ' 💬';
            }
            details.pyglove.ai-message {
                border: 1px solid blue;
                color: blue;
            }
            details.pyglove.user-message {
                border: 1px solid green;
                color: green;
            }
            .message-tags {
                margin: 5px 0px 5px 0px;
                font-size: .8em;
            }
            .message-tags > span {
                border-radius: 5px;
                background-color: #CCC;
                padding: 3px;
                margin: 0px 2px 0px 2px;
                color: white;
            }
            .message-text {
                padding: 20px;
                margin: 10px 5px 10px 5px;
                font-style: italic;
                white-space: pre-wrap;
                border: 1px solid #EEE;
                border-radius: 5px;
                background-color: #EEE;
            }
            .modality-in-text {
                display: inline-block;
            }
            .modality-in-text > details.pyglove {
                display: inline-block;
                font-size: 0.8em;
                border: 0;
                background-color: #A6F1A6;
                margin: 0px 5px 0px 5px;
            }
            .message-result {
                color: dodgerblue;
            }
            .message-usage {
                color: orange;
            }
            .message-usage .object-key.str {
                border: 1px solid orange;
                background-color: orange;
                color: white;
            }
            """
        ),
        message.UserMessage('hi').to_html().style_section,
    )

  def test_html_user_message(self):
    self.assert_html_content(
        message.UserMessage(
            'what is a <div>'
        ).to_html(enable_summary_tooltip=False),
        """
        <details open class="pyglove user-message lf-message"><summary><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"></div><div class="message-text">what is a &lt;div&gt;</div><div class="message-metadata"><details open class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><span class="empty-container"></span></div></details></div></div></details>
        """
    )
    self.assert_html_content(
        message.UserMessage(
            'what is this <<[[image]]>>',
            tags=['lm-input'],
            image=CustomModality('bird')
        ).to_html(
            enable_summary_tooltip=False,
            extra_flags=dict(include_message_metadata=False)
        ),
        """
        <details open class="pyglove user-message lf-message"><summary><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-input</span></div><div class="message-text">what is this<div class="modality-in-text"><details class="pyglove custom-modality"><summary><div class="summary-name">image<span class="tooltip">metadata.image</span></div><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><details open class="pyglove str"><summary><div class="summary-name">content<span class="tooltip">metadata.image.content</span></div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;bird&#x27;</span></details></div></details></div></div></div></details>
        """
    )

  def test_html_ai_message(self):
    image = CustomModality('foo')
    user_message = message.UserMessage(
        'What is in this image? <<[[image]]>> this is a test',
        metadata=dict(image=image),
        source=message.UserMessage('User input'),
        tags=['lm-input']
    )
    ai_message = message.AIMessage(
        'My name is Gemini',
        metadata=dict(
            result=pg.Dict(x=1, y=2, z=pg.Dict(a=[12, 323])),
            usage=language_model.LMSamplingUsage(10, 2, 12)
        ),
        tags=['lm-response', 'lm-output'],
        source=user_message,
    )
    self.assert_html_content(
        ai_message.to_html(enable_summary_tooltip=False),
        """
        <details open class="pyglove ai-message lf-message"><summary><div class="summary-title lf-message">AIMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-response</span><span>lm-output</span></div><div class="message-text">My name is Gemini</div><div class="message-result"><details open class="pyglove dict"><summary><div class="summary-name">result<span class="tooltip">metadata.result</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details open class="pyglove int"><summary><div class="summary-name">x<span class="tooltip">metadata.result.x</span></div><div class="summary-title">int</div></summary><span class="simple-value int">1</span></details><details open class="pyglove int"><summary><div class="summary-name">y<span class="tooltip">metadata.result.y</span></div><div class="summary-title">int</div></summary><span class="simple-value int">2</span></details><details class="pyglove dict"><summary><div class="summary-name">z<span class="tooltip">metadata.result.z</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><details class="pyglove list"><summary><div class="summary-name">a<span class="tooltip">metadata.result.z.a</span></div><div class="summary-title">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">metadata.result.z.a[0]</span></td><td><span class="simple-value int">12</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">metadata.result.z.a[1]</span></td><td><span class="simple-value int">323</span></td></tr></table></div></details></div></details></div></details></div><div class="message-usage"><details open class="pyglove lm-sampling-usage"><summary><div class="summary-name">llm usage<span class="tooltip">metadata.usage</span></div><div class="summary-title">LMSamplingUsage(...)</div></summary><div class="complex-value lm-sampling-usage"><table><tr><td><span class="object-key str">prompt_tokens</span><span class="tooltip">metadata.usage.prompt_tokens</span></td><td><span class="simple-value int">10</span></td></tr><tr><td><span class="object-key str">completion_tokens</span><span class="tooltip">metadata.usage.completion_tokens</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key str">total_tokens</span><span class="tooltip">metadata.usage.total_tokens</span></td><td><span class="simple-value int">12</span></td></tr><tr><td><span class="object-key str">num_requests</span><span class="tooltip">metadata.usage.num_requests</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">estimated_cost</span><span class="tooltip">metadata.usage.estimated_cost</span></td><td><span class="simple-value none-type">None</span></td></tr></table></div></details></div><div class="message-metadata"><details open class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><span class="empty-container"></span></div></details></div><details open class="pyglove user-message lf-message"><summary><div class="summary-name lf-message">source<span class="tooltip lf-message">source</span></div><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-input</span></div><div class="message-text">What is in this image?<div class="modality-in-text"><details class="pyglove custom-modality"><summary><div class="summary-name">image<span class="tooltip">source.metadata.image</span></div><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><details open class="pyglove str"><summary><div class="summary-name">content<span class="tooltip">source.metadata.image.content</span></div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;foo&#x27;</span></details></div></details></div>this is a test</div><div class="message-metadata"><details open class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">source.metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><details class="pyglove custom-modality"><summary><div class="summary-name">image<span class="tooltip">source.metadata.image</span></div><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><details open class="pyglove str"><summary><div class="summary-name">content<span class="tooltip">source.metadata.image.content</span></div><div class="summary-title">str</div></summary><span class="simple-value str">&#x27;foo&#x27;</span></details></div></details></div></details></div></div></details></div></details>
        """
    )
    self.assert_html_content(
        ai_message.to_html(
            key_style='label',
            enable_summary_tooltip=False,
            extra_flags=dict(
                collapse_modalities_in_text=False,
                collapse_llm_usage=True,
                collapse_message_result_level=0,
                collapse_message_metadata_level=0,
                collapse_source_message_level=0,
                source_tag=None,
            ),
        ),
        """
        <details open class="pyglove ai-message lf-message"><summary><div class="summary-title lf-message">AIMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-response</span><span>lm-output</span></div><div class="message-text">My name is Gemini</div><div class="message-result"><details class="pyglove dict"><summary><div class="summary-name">result<span class="tooltip">metadata.result</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">x</span><span class="tooltip">metadata.result.x</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">metadata.result.y</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key str">z</span><span class="tooltip">metadata.result.z</span></td><td><details class="pyglove dict"><summary><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">a</span><span class="tooltip">metadata.result.z.a</span></td><td><details class="pyglove list"><summary><div class="summary-title">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">metadata.result.z.a[0]</span></td><td><span class="simple-value int">12</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">metadata.result.z.a[1]</span></td><td><span class="simple-value int">323</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details></div><div class="message-usage"><details class="pyglove lm-sampling-usage"><summary><div class="summary-name">llm usage<span class="tooltip">metadata.usage</span></div><div class="summary-title">LMSamplingUsage(...)</div></summary><div class="complex-value lm-sampling-usage"><table><tr><td><span class="object-key str">prompt_tokens</span><span class="tooltip">metadata.usage.prompt_tokens</span></td><td><span class="simple-value int">10</span></td></tr><tr><td><span class="object-key str">completion_tokens</span><span class="tooltip">metadata.usage.completion_tokens</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key str">total_tokens</span><span class="tooltip">metadata.usage.total_tokens</span></td><td><span class="simple-value int">12</span></td></tr><tr><td><span class="object-key str">num_requests</span><span class="tooltip">metadata.usage.num_requests</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">estimated_cost</span><span class="tooltip">metadata.usage.estimated_cost</span></td><td><span class="simple-value none-type">None</span></td></tr></table></div></details></div><div class="message-metadata"><details class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><span class="empty-container"></span></div></details></div><details class="pyglove user-message lf-message"><summary><div class="summary-name lf-message">source<span class="tooltip lf-message">source</span></div><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-input</span></div><div class="message-text">What is in this image?<div class="modality-in-text"><details open class="pyglove custom-modality"><summary><div class="summary-name">image<span class="tooltip">source.metadata.image</span></div><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><table><tr><td><span class="object-key str">content</span><span class="tooltip">source.metadata.image.content</span></td><td><span class="simple-value str">&#x27;foo&#x27;</span></td></tr></table></div></details></div>this is a test</div><div class="message-metadata"><details class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">source.metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">image</span><span class="tooltip">source.metadata.image</span></td><td><details class="pyglove custom-modality"><summary><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><table><tr><td><span class="object-key str">content</span><span class="tooltip">source.metadata.image.content</span></td><td><span class="simple-value str">&#x27;foo&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></div><details class="pyglove user-message lf-message"><summary><div class="summary-name lf-message">source<span class="tooltip lf-message">source.source</span></div><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"></div><div class="message-text">User input</div><div class="message-metadata"><details class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">source.source.metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><span class="empty-container"></span></div></details></div></div></details></div></details></div></details>
        """
    )
    self.assert_html_content(
        ai_message.to_html(
            key_style='label',
            enable_summary_tooltip=False,
            extra_flags=dict(
                collapse_modalities_in_text=True,
                collapse_llm_usage=False,
                collapse_message_result_level=1,
                collapse_message_metadata_level=1,
                collapse_source_message_level=2,
                source_tag=None,
            ),
        ),
        """
        <details open class="pyglove ai-message lf-message"><summary><div class="summary-title lf-message">AIMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-response</span><span>lm-output</span></div><div class="message-text">My name is Gemini</div><div class="message-result"><details open class="pyglove dict"><summary><div class="summary-name">result<span class="tooltip">metadata.result</span></div><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">x</span><span class="tooltip">metadata.result.x</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">y</span><span class="tooltip">metadata.result.y</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key str">z</span><span class="tooltip">metadata.result.z</span></td><td><details class="pyglove dict"><summary><div class="summary-title">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">a</span><span class="tooltip">metadata.result.z.a</span></td><td><details class="pyglove list"><summary><div class="summary-title">List(...)</div></summary><div class="complex-value list"><table><tr><td><span class="object-key int">0</span><span class="tooltip">metadata.result.z.a[0]</span></td><td><span class="simple-value int">12</span></td></tr><tr><td><span class="object-key int">1</span><span class="tooltip">metadata.result.z.a[1]</span></td><td><span class="simple-value int">323</span></td></tr></table></div></details></td></tr></table></div></details></td></tr></table></div></details></div><div class="message-usage"><details open class="pyglove lm-sampling-usage"><summary><div class="summary-name">llm usage<span class="tooltip">metadata.usage</span></div><div class="summary-title">LMSamplingUsage(...)</div></summary><div class="complex-value lm-sampling-usage"><table><tr><td><span class="object-key str">prompt_tokens</span><span class="tooltip">metadata.usage.prompt_tokens</span></td><td><span class="simple-value int">10</span></td></tr><tr><td><span class="object-key str">completion_tokens</span><span class="tooltip">metadata.usage.completion_tokens</span></td><td><span class="simple-value int">2</span></td></tr><tr><td><span class="object-key str">total_tokens</span><span class="tooltip">metadata.usage.total_tokens</span></td><td><span class="simple-value int">12</span></td></tr><tr><td><span class="object-key str">num_requests</span><span class="tooltip">metadata.usage.num_requests</span></td><td><span class="simple-value int">1</span></td></tr><tr><td><span class="object-key str">estimated_cost</span><span class="tooltip">metadata.usage.estimated_cost</span></td><td><span class="simple-value none-type">None</span></td></tr></table></div></details></div><div class="message-metadata"><details open class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><span class="empty-container"></span></div></details></div><details open class="pyglove user-message lf-message"><summary><div class="summary-name lf-message">source<span class="tooltip lf-message">source</span></div><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"><span>lm-input</span></div><div class="message-text">What is in this image?<div class="modality-in-text"><details class="pyglove custom-modality"><summary><div class="summary-name">image<span class="tooltip">source.metadata.image</span></div><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><table><tr><td><span class="object-key str">content</span><span class="tooltip">source.metadata.image.content</span></td><td><span class="simple-value str">&#x27;foo&#x27;</span></td></tr></table></div></details></div>this is a test</div><div class="message-metadata"><details open class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">source.metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><table><tr><td><span class="object-key str">image</span><span class="tooltip">source.metadata.image</span></td><td><details class="pyglove custom-modality"><summary><div class="summary-title">CustomModality(...)</div></summary><div class="complex-value custom-modality"><table><tr><td><span class="object-key str">content</span><span class="tooltip">source.metadata.image.content</span></td><td><span class="simple-value str">&#x27;foo&#x27;</span></td></tr></table></div></details></td></tr></table></div></details></div><details open class="pyglove user-message lf-message"><summary><div class="summary-name lf-message">source<span class="tooltip lf-message">source.source</span></div><div class="summary-title lf-message">UserMessage(...)</div></summary><div class="complex_value"><div class="message-tags"></div><div class="message-text">User input</div><div class="message-metadata"><details open class="pyglove dict message-metadata"><summary><div class="summary-name message-metadata">metadata<span class="tooltip message-metadata">source.source.metadata</span></div><div class="summary-title message-metadata">Dict(...)</div></summary><div class="complex-value dict"><span class="empty-container"></span></div></details></div></div></details></div></details></div></details>
        """
    )


if __name__ == '__main__':
  unittest.main()
