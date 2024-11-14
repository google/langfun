# Copyright 2024 The Langfun Authors
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
"""Tests for base action."""

import unittest

import langfun.core as lf
from langfun.core.agentic import action as action_lib


class SessionTest(unittest.TestCase):

  def test_basics(self):
    test = self

    class Bar(action_lib.Action):

      def call(self, session, **kwargs):
        test.assertIs(session.current_invocation.action, self)
        session.info('Begin Bar')
        return 2

    class Foo(action_lib.Action):
      x: int

      def call(self, session, **kwargs):
        test.assertIs(session.current_invocation.action, self)
        session.info('Begin Foo', x=1)
        return self.x + Bar()(session)

    session = action_lib.Session()
    root = session.root_invocation
    self.assertIsInstance(root.action, action_lib.RootAction)
    self.assertIs(session.current_invocation, session.root_invocation)
    self.assertEqual(Foo(1)(session), 3)
    self.assertEqual(len(session.root_invocation.child_invocations), 1)
    self.assertEqual(len(session.root_invocation.child_invocations[0].logs), 1)
    self.assertEqual(
        len(session.root_invocation.child_invocations[0].child_invocations),
        1
    )
    self.assertEqual(
        len(session.root_invocation
            .child_invocations[0].child_invocations[0].logs),
        1
    )
    self.assertEqual(
        len(session.root_invocation
            .child_invocations[0].child_invocations[0].child_invocations),
        0
    )
    self.assertIs(session.current_invocation, session.root_invocation)
    self.assertIs(session.final_result, 3)
    self.assertIn(
        'invocation-final-result',
        session.to_html().content,
    )

  def test_log(self):
    session = action_lib.Session()
    session.debug('hi', x=1, y=2)
    session.info('hi', x=1, y=2)
    session.warning('hi', x=1, y=2)
    session.error('hi', x=1, y=2)
    session.fatal('hi', x=1, y=2)

  def test_as_message(self):
    session = action_lib.Session()
    self.assertIsInstance(session.as_message(), lf.AIMessage)


if __name__ == '__main__':
  unittest.main()
