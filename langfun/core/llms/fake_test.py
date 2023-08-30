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
"""Test Echo LLM."""

import contextlib
import io
import unittest
import langfun.core as lf
from langfun.core.llms.fake import Echo
from langfun.core.llms.fake import StaticMapping
from langfun.core.llms.fake import StaticSequence


class EchoTest(unittest.TestCase):

  def test_sample(self):
    lm = Echo()
    self.assertEqual(
        lm.sample(['hi']), [lf.LMSamplingResult([lf.LMSample('hi', 1.0)])]
    )

  def test_call(self):
    string_io = io.StringIO()
    lm = Echo(debug=True)
    with contextlib.redirect_stdout(string_io):
      self.assertEqual(lm('hi'), 'hi')
    debug_info = string_io.getvalue()
    self.assertIn('[0] LM INFO:', debug_info)
    self.assertIn('[0] PROMPT SENT TO LM:', debug_info)
    self.assertIn('[0] LM RESPONSE', debug_info)


class StaticMappingTest(unittest.TestCase):

  def test_sample(self):
    lm = StaticMapping({
        'Hi': 'Hello',
        'How are you?': 'I am fine, how about you?',
    }, temperature=0.5)
    self.assertEqual(lm.sampling_options.temperature, 0.5)
    self.assertEqual(
        lm.sample(['Hi', 'How are you?']),
        [
            lf.LMSamplingResult([lf.LMSample('Hello', 1.0)]),
            lf.LMSamplingResult([lf.LMSample('I am fine, how about you?', 1.0)])
        ]
    )
    with self.assertRaises(KeyError):
      _ = lm.sample(['I am not in the table.'])


class StaticSequenceTest(unittest.TestCase):

  def test_sample(self):
    lm = StaticSequence([
        'Hello',
        'I am fine, how about you?',
    ], temperature=0.5)
    self.assertEqual(lm.sampling_options.temperature, 0.5)
    self.assertEqual(
        lm.sample(['Hi', 'How are you?']),
        [
            lf.LMSamplingResult([lf.LMSample('Hello', 1.0)]),
            lf.LMSamplingResult([lf.LMSample('I am fine, how about you?', 1.0)])
        ]
    )
    with self.assertRaises(IndexError):
      _ = lm.sample(['No next one.'])


if __name__ == '__main__':
  unittest.main()
