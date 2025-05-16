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
from langfun.core.llms import fake
import langfun.core.structured as lf_structured
import pyglove as pg


class Bar(action_lib.Action):
  simulate_action_error: bool = False

  def call(self, session, *, lm, **kwargs):
    assert session.current_action.action is self
    session.info('Begin Bar')
    session.query('bar', lm=lm)
    session.add_metadata(note='bar')
    if self.simulate_action_error:
      raise ValueError('Bar error')
    return 2


class Foo(action_lib.Action):
  x: int
  simulate_action_error: bool = False
  simulate_query_error: bool = False

  def call(self, session, *, lm, **kwargs):
    assert session.current_action.action is self
    with session.track_phase('prepare'):
      session.info('Begin Foo', x=1)
      session.query(
          'foo',
          schema=int if self.simulate_query_error else None,
          lm=lm
      )
    with session.track_queries():
      self.make_additional_query(lm)
    session.add_metadata(note='foo')

    def _sub_task(i):
      session.add_metadata(**{f'subtask_{i}': i})
      return lf_structured.query(f'subtask_{i}', lm=lm)

    for i, output, error in session.concurrent_map(
        _sub_task, range(3), max_workers=2, silence_on_errors=None,
    ):
      assert isinstance(i, int), i
      assert isinstance(output, str), output
      assert error is None, error
    return self.x + Bar(
        simulate_action_error=self.simulate_action_error
    )(session, lm=lm)

  def make_additional_query(self, lm):
    lf_structured.query('additional query', lm=lm)


class ActionInvocationTest(unittest.TestCase):

  def test_basics(self):
    action_invocation = action_lib.ActionInvocation(
        action=Foo(1)
    )
    self.assertEqual(action_invocation.id, '')
    root = action_lib.ActionInvocation(action=action_lib.RootAction())
    root.execution.append(action_invocation)
    self.assertIs(action_invocation.parent_action, root)
    self.assertEqual(action_invocation.id, '/a1')


class ExecutionTraceTest(unittest.TestCase):

  def test_basics(self):
    execution = action_lib.ExecutionTrace()
    self.assertEqual(execution.id, '')

    root = action_lib.ActionInvocation(action=action_lib.RootAction())
    action_invocation = action_lib.ActionInvocation(
        action=Foo(1)
    )
    root.execution.append(action_invocation)
    self.assertEqual(action_invocation.execution.id, '/a1')

    root.execution.reset()
    self.assertFalse(root.execution)


class SessionTest(unittest.TestCase):

  def test_succeeded_trajectory(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1)
    self.assertIsNone(foo.session)
    self.assertIsNone(foo.invocation)
    self.assertIsNone(foo.result)
    self.assertIsNone(foo.metadata)

    session = action_lib.Session(id='agent@1')
    self.assertEqual(session.id, 'agent@1')
    self.assertFalse(session.has_started)
    self.assertFalse(session.has_stopped)

    # Render HTML view to trigger dynamic update during execution.
    _ = session.to_html()

    with session:
      result = foo(session, lm=lm, verbose=True)

    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertEqual(result, 3)
    self.assertIsNone(foo.session)
    self.assertEqual(foo.result, 3)
    self.assertEqual(
        foo.metadata, dict(note='foo', subtask_0=0, subtask_1=1, subtask_2=2)
    )

    self.assertIsInstance(session.root.action, action_lib.RootAction)
    self.assertIs(session.current_action, session.root)

    #
    # Inspecting the root invocation.
    #

    root = session.root
    self.assertIsNone(root.parent_action)
    self.assertEqual(root.id, 'agent@1:')
    self.assertEqual(root.execution.id, 'agent@1:')
    self.assertEqual(len(root.execution), 1)
    self.assertIs(root.execution[0].action, foo)

    self.assertTrue(root.execution.has_started)
    self.assertTrue(root.execution.has_stopped)
    self.assertGreater(root.execution.elapse, 0)
    self.assertEqual(root.result, 3)
    self.assertEqual(
        root.metadata,
        dict(note='foo', subtask_0=0, subtask_1=1, subtask_2=2)
    )

    # The root space should have one action (foo), no queries, and no logs.
    self.assertEqual(len(root.actions), 1)
    self.assertEqual(len(root.queries), 0)
    self.assertEqual(len(root.logs), 0)
    # 1 query from Bar, 2 from Foo and 3 from parallel executions.
    self.assertEqual(len(session.all_queries), 6)
    self.assertEqual(len(root.all_queries), 6)
    # 2 actions: Foo and Bar.
    self.assertEqual(len(session.all_actions), 2)
    self.assertEqual(len(root.all_actions), 2)
    # 1 log from Bar and 1 from Foo.
    self.assertEqual(len(session.all_logs), 2)
    self.assertEqual(len(root.all_logs), 2)
    self.assertIs(session.usage_summary, root.usage_summary)
    self.assertEqual(root.usage_summary.total.num_requests, 6)

    # Inspecting the top-level action (Foo)
    foo_invocation = root.execution[0]
    self.assertIs(foo.invocation, foo_invocation)
    self.assertIs(foo_invocation.parent_action, root)
    self.assertEqual(foo_invocation.id, 'agent@1:/a1')
    self.assertEqual(foo_invocation.execution.id, 'agent@1:/a1')
    self.assertEqual(len(foo_invocation.execution.items), 4)

    # Prepare phase.
    prepare_phase = foo_invocation.execution[0]
    self.assertIsInstance(prepare_phase, action_lib.ExecutionTrace)
    self.assertEqual(prepare_phase.id, 'agent@1:/a1/prepare')
    self.assertEqual(len(prepare_phase.items), 2)
    self.assertTrue(prepare_phase.has_started)
    self.assertTrue(prepare_phase.has_stopped)
    self.assertEqual(prepare_phase.usage_summary.total.num_requests, 1)
    self.assertIsInstance(prepare_phase.items[0], lf.logging.LogEntry)
    self.assertIsInstance(prepare_phase.items[1], lf_structured.QueryInvocation)
    self.assertEqual(prepare_phase.items[1].id, 'agent@1:/a1/prepare/q1')

    # Tracked queries.
    query_invocation = foo_invocation.execution[1]
    self.assertIsInstance(query_invocation, lf_structured.QueryInvocation)
    self.assertEqual(query_invocation.id, 'agent@1:/a1/q2')
    self.assertIs(query_invocation.lm, lm)
    self.assertEqual(
        foo_invocation.execution.indexof(
            query_invocation, lf_structured.QueryInvocation
        ),
        1
    )
    self.assertEqual(
        root.execution.indexof(
            query_invocation, lf_structured.QueryInvocation
        ),
        -1
    )

    # Tracked parallel executions.
    parallel_executions = foo_invocation.execution[2]
    self.assertEqual(parallel_executions.id, 'agent@1:/a1/p1')
    self.assertIsInstance(parallel_executions, action_lib.ParallelExecutions)
    self.assertEqual(len(parallel_executions), 3)
    self.assertEqual(parallel_executions[0].id, 'agent@1:/a1/p1/b1')
    self.assertEqual(parallel_executions[1].id, 'agent@1:/a1/p1/b2')
    self.assertEqual(parallel_executions[2].id, 'agent@1:/a1/p1/b3')
    self.assertEqual(len(parallel_executions[0].queries), 1)
    self.assertEqual(len(parallel_executions[1].queries), 1)
    self.assertEqual(len(parallel_executions[2].queries), 1)

    # Invocation to Bar.
    bar_invocation = foo_invocation.execution[3]
    self.assertIs(bar_invocation.parent_action, foo_invocation)
    self.assertEqual(bar_invocation.id, 'agent@1:/a1/a1')
    self.assertIsInstance(bar_invocation, action_lib.ActionInvocation)
    self.assertIsInstance(bar_invocation.action, Bar)
    self.assertEqual(bar_invocation.result, 2)
    self.assertEqual(bar_invocation.metadata, dict(note='bar'))
    self.assertEqual(len(bar_invocation.execution.items), 2)

    # Save to HTML
    self.assertIn('result', session.to_html().content)

    # Save session to JSON
    json_str = session.to_json_str(save_ref_value=True)
    self.assertIsInstance(pg.from_json_str(json_str), action_lib.Session)

  def test_failed_action(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1, simulate_action_error=True)
    with self.assertRaisesRegex(ValueError, 'Bar error'):
      foo(lm=lm)

    session = foo.session
    self.assertIsNotNone(session)
    self.assertIsInstance(session.root.action, action_lib.RootAction)
    self.assertIs(session.current_action, session.root)

    # Inspecting the root invocation.
    root = session.root
    self.assertRegex(root.id, 'agent@.*:')
    self.assertTrue(root.has_error)
    foo_invocation = root.execution[0]
    self.assertIsInstance(foo_invocation, action_lib.ActionInvocation)
    self.assertTrue(foo_invocation.has_error)
    bar_invocation = foo_invocation.execution[3]
    self.assertIsInstance(bar_invocation, action_lib.ActionInvocation)
    self.assertTrue(bar_invocation.has_error)

    # Save to HTML
    self.assertIn('error', session.to_html().content)

  def test_failed_query(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1, simulate_query_error=True)
    with self.assertRaisesRegex(lf_structured.MappingError, 'SyntaxError'):
      foo(lm=lm)

    session = foo.session
    self.assertIsNotNone(session)
    self.assertIsInstance(session.root.action, action_lib.RootAction)
    self.assertIs(session.current_action, session.root)

    # Inspecting the root invocation.
    root = session.root
    self.assertRegex(root.id, 'agent@.*:')
    self.assertTrue(root.has_error)
    foo_invocation = root.execution[0]
    self.assertIsInstance(foo_invocation, action_lib.ActionInvocation)
    self.assertTrue(foo_invocation.has_error)
    self.assertEqual(len(foo_invocation.execution.items), 3)

  def test_succeeded_with_implicit_session(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1)
    foo(lm=lm, verbose=True)
    session = foo.session
    self.assertIsNotNone(session)
    self.assertIsInstance(session.root.action, action_lib.RootAction)
    self.assertIs(session.current_action, session.root)
    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertEqual(session.final_result, 3)
    self.assertFalse(session.root.has_error)
    self.assertEqual(session.root.metadata, {})

  def test_failed_with_implicit_session(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1, simulate_action_error=True)
    with self.assertRaisesRegex(ValueError, 'Bar error'):
      foo(lm=lm)
    session = foo.session
    self.assertIsNotNone(session)
    self.assertIsInstance(session.root.action, action_lib.RootAction)
    self.assertIs(session.current_action, session.root)
    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertTrue(session.has_error)
    self.assertIsInstance(session.final_error, pg.ErrorInfo)
    self.assertIn('Bar error', str(session.root.error))

  def test_succeeded_with_explicit_session(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1)
    self.assertIsNone(foo.session)
    self.assertIsNone(foo.result)
    self.assertIsNone(foo.metadata)

    session = action_lib.Session(id='agent@1')
    self.assertEqual(session.id, 'agent@1')
    self.assertFalse(session.has_started)
    self.assertFalse(session.has_stopped)

    with session:
      result = foo(session, lm=lm, verbose=True)

    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertEqual(result, 3)
    self.assertIsNone(foo.session)
    self.assertEqual(foo.result, 3)
    self.assertEqual(
        foo.metadata, dict(note='foo', subtask_0=0, subtask_1=1, subtask_2=2)
    )
    self.assertIs(session.final_result, foo.result)
    self.assertFalse(session.has_error)

  def test_succeeded_with_explicit_session_start_end(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1)
    self.assertIsNone(foo.session)
    self.assertIsNone(foo.result)
    self.assertIsNone(foo.metadata)

    session = action_lib.Session(id='agent@1')
    self.assertEqual(session.id, 'agent@1')
    self.assertFalse(session.has_started)
    self.assertFalse(session.has_stopped)

    session.start()
    result = foo(session, lm=lm, verbose=True)
    session.end(result)

    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertEqual(result, 3)
    self.assertIsNone(foo.session)
    self.assertEqual(foo.result, 3)
    self.assertEqual(
        foo.metadata, dict(note='foo', subtask_0=0, subtask_1=1, subtask_2=2)
    )
    self.assertIs(session.final_result, foo.result)
    self.assertFalse(session.has_error)

  def test_failed_with_explicit_session(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1, simulate_action_error=True)
    session = action_lib.Session(id='agent@1')
    with self.assertRaisesRegex(ValueError, 'Bar error'):
      with session:
        foo(session, lm=lm, verbose=True)
    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertTrue(session.has_error)
    self.assertIsNone(session.final_result)
    self.assertIsInstance(session.root.error, pg.ErrorInfo)
    self.assertIn('Bar error', str(session.root.error))

  def test_failed_with_explicit_session_without_start(self):
    lm = fake.StaticResponse('lm response')
    foo = Foo(1, simulate_action_error=True)
    session = action_lib.Session(id='agent@1')
    with self.assertRaisesRegex(ValueError, 'Please call `Session.start'):
      foo(session, lm=lm, verbose=True)

  def test_succeed_with_multiple_actions(self):
    lm = fake.StaticResponse('lm response')
    with action_lib.Session() as session:
      x = Bar()(session, lm=lm)
      y = Bar()(session, lm=lm)
      self.assertTrue(session.has_started)
      self.assertFalse(session.has_stopped)
      session.add_metadata(note='root metadata')
      session.end(x + y)

    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertEqual(session.final_result, 2 + 2)
    self.assertEqual(len(session.root.execution), 2)
    self.assertEqual(session.root.metadata, dict(note='root metadata'))

  def test_failed_with_multiple_actions(self):
    lm = fake.StaticResponse('lm response')
    with self.assertRaisesRegex(ValueError, 'Bar error'):
      with action_lib.Session() as session:
        x = Bar()(session, lm=lm)
        y = Bar(simulate_action_error=True)(session, lm=lm)
        session.end(x + y)

    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertTrue(session.has_error)
    self.assertIsInstance(session.root.error, pg.ErrorInfo)
    self.assertEqual(len(session.root.execution), 3)
    self.assertEqual(len(session.root.actions), 2)
    self.assertEqual(len(session.root.logs), 1)
    self.assertFalse(session.root.execution[0].has_error)
    self.assertTrue(session.root.execution[1].has_error)

  def test_log(self):
    session = action_lib.Session()
    session.debug('hi', x=1, y=2)
    session.info('hi', x=1, y=2, for_action=session.root)
    session.warning('hi', x=1, y=2, for_action=session.root.action)
    session.error('hi', x=1, y=2)
    session.fatal('hi', x=1, y=2)

  def test_as_message(self):
    session = action_lib.Session()
    self.assertIn('agent@', session.id)
    self.assertIsInstance(session.as_message(), lf.AIMessage)


if __name__ == '__main__':
  unittest.main()
