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

import asyncio
import os
import tempfile
import time
from typing import Any
import unittest

import langfun.core as lf
from langfun.core.agentic import action as action_lib
from langfun.core.llms import fake
import langfun.core.structured as lf_structured
import pyglove as pg


class Bar(action_lib.Action):
  simulate_action_error: bool = False
  simulate_execution_time: float = 0

  def call(self, session, *, lm, **kwargs):
    assert session.current_action.action is self
    session.info('Begin Bar')
    time.sleep(self.simulate_execution_time)
    session.query('bar', lm=lm)
    session.add_metadata(note='bar')
    session.update_progress('Query completed')
    if self.simulate_action_error:
      raise ValueError('Bar error')
    return 2 + pg.contextual_value('baz', 0)


class Foo(action_lib.Action):
  x: int
  simulate_action_error: bool = False
  simulate_query_error: bool = False
  simulate_execution_time: list[float] = [0, 0, 0, 0]
  max_bar_execution_time: float | None = None

  def call(self, session, *, lm, **kwargs):
    assert session.current_action.action is self
    with session.track_phase('prepare'):
      session.info('Begin Foo', x=1)
      time.sleep(self.simulate_execution_time[0])
      Bar()(session, lm=lm)
      session.query(
          'foo',
          schema=int if self.simulate_query_error else None,
          lm=lm
      )
    with session.track_queries():
      time.sleep(self.simulate_execution_time[1])
      self.make_additional_query(lm)
    session.add_metadata(note='foo')

    def _sub_task(i):
      session.add_metadata(**{f'subtask_{i}': i})
      time.sleep(self.simulate_execution_time[2])
      Bar()(session, lm=lm)
      return lf_structured.query(f'subtask_{i}', lm=lm)

    self._state = []
    for i, output, error in session.concurrent_map(
        _sub_task,
        range(3),
        max_workers=2,
        ordered=True,
        silence_on_errors=None,
    ):
      assert isinstance(i, int), i
      assert isinstance(output, str), output
      assert error is None, error
      self._state.append(i)
    return self.x + Bar(
        simulate_action_error=self.simulate_action_error,
        simulate_execution_time=self.simulate_execution_time[3]
    )(session, lm=lm, max_execution_time=self.max_bar_execution_time)

  def make_additional_query(self, lm):
    lf_structured.query('additional query', lm=lm)


class ExecutionUnitPositionTest(unittest.TestCase):

  def test_basics(self):
    pos1 = action_lib.ExecutionUnit.Position(None, 0)
    self.assertEqual(repr(pos1), 'Position(0)')
    self.assertEqual(str(pos1), '')
    self.assertIsNone(pos1.parent)
    self.assertEqual(pos1.index, 0)
    self.assertEqual(pos1.indices(), (0,))
    self.assertEqual(pos1, (0,))
    self.assertEqual(pos1, '')
    self.assertEqual(pos1, action_lib.ExecutionUnit.Position(None, 0))
    self.assertNotEqual(pos1, 1)
    self.assertNotEqual(pos1, (1,))
    self.assertNotEqual(pos1, action_lib.ExecutionUnit.Position(None, 1))

    pos2 = action_lib.ExecutionUnit.Position(pos1, 0)
    self.assertEqual(repr(pos2), 'Position(0, 0)')
    self.assertEqual(str(pos2), '1')
    self.assertEqual(pos2, '1')
    self.assertEqual(pos2.parent, pos1)
    self.assertEqual(pos2.index, 0)
    self.assertEqual(pos2.indices(), (0, 0))
    self.assertNotEqual(pos1, pos2)
    self.assertLess(pos1, pos2)
    self.assertGreater(pos2, pos1)
    self.assertEqual(
        hash(pos2),
        hash(
            action_lib.ExecutionUnit.Position(
                action_lib.ExecutionUnit.Position(None, 0), 0
            )
        )
    )

    pos3 = action_lib.ExecutionUnit.Position(pos2, 0)
    self.assertEqual(str(pos3), '1.1')
    self.assertEqual(pos3, '1.1')
    self.assertEqual(pos3.parent, pos2)
    self.assertEqual(pos3.index, 0)
    self.assertEqual(pos3.indices(), (0, 0, 0))
    self.assertEqual(pos3.to_str(separator='>'), '1>1')


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
    action_invocation = action_lib.ActionInvocation(action=Foo(1))
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
    self.assertIsNone(foo.state)
    self.assertIsNone(foo.result)
    self.assertIsNone(foo.metadata)

    session = action_lib.Session(id='agent@1', verbose=True)
    self.assertEqual(session.id, 'agent@1')
    self.assertFalse(session.has_started)
    self.assertFalse(session.has_stopped)

    # Render HTML view to trigger dynamic update during execution.
    _ = session.to_html()

    with session:
      result = foo(session, lm=lm)

    self.assertTrue(session.has_started)
    self.assertTrue(session.has_stopped)
    self.assertEqual(result, 3)
    self.assertIsNone(foo.session)
    self.assertEqual(foo.state, [0, 1, 2])
    self.assertIs(foo.invocation.state, foo.state)
    self.assertEqual(foo.result, 3)
    self.assertEqual(
        foo.metadata, dict(note='foo', subtask_0=0, subtask_1=1, subtask_2=2)
    )

    self.assertIsInstance(session.root.action, action_lib.RootAction)
    self.assertIs(session.current_action, session.root)
    self.assertIs(session.metadata, session.root.metadata)

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
    self.assertEqual(len(root.execution_units), 1)
    self.assertEqual(len(root.actions), 1)
    self.assertEqual(len(root.queries), 0)
    self.assertEqual(len(root.logs), 0)
    # 2 query from Bar, 2 from Foo and 2 * 3 from parallel executions.
    self.assertEqual(len(session.all_queries), 10)
    self.assertEqual(len(root.all_queries), 10)
    # 6 actions: Foo and 2 Bar, and 3 Bar from parallel executions.
    self.assertEqual(len(session.all_actions), 6)
    self.assertEqual(
        [str(a.position) for a in session.all_actions],
        ['1', '1.1', '1.2.1.1', '1.2.2.1', '1.2.3.1', '1.3']
    )
    self.assertEqual(len(root.all_actions), 6)
    # 1 log from Bar and 1 from Foo and 3 from Bar in parallel executions.
    self.assertEqual(len(session.all_logs), 6)
    self.assertEqual(len(root.all_logs), 6)
    self.assertIs(session.usage_summary, root.usage_summary)
    self.assertEqual(root.usage_summary.total.num_requests, 10)

    # Inspecting the top-level action (Foo)
    foo_invocation = root.execution[0]
    self.assertIs(foo.invocation, foo_invocation)
    self.assertIs(foo_invocation.parent_action, root)
    self.assertEqual(foo_invocation.id, 'agent@1:/a1')
    self.assertEqual(foo_invocation.execution.id, 'agent@1:/a1')
    self.assertEqual(len(foo_invocation.execution.items), 4)

    # Prepare phase.
    prepare_phase = foo_invocation.execution[0]
    self.assertIsNone(prepare_phase.position)
    self.assertIsInstance(prepare_phase, action_lib.ExecutionTrace)
    self.assertEqual(prepare_phase.id, 'agent@1:/a1/prepare')
    self.assertEqual(len(prepare_phase.items), 3)
    self.assertTrue(prepare_phase.has_started)
    self.assertTrue(prepare_phase.has_stopped)
    self.assertEqual(prepare_phase.usage_summary.total.num_requests, 2)
    self.assertIsInstance(prepare_phase.items[0], lf.logging.LogEntry)
    self.assertIsInstance(prepare_phase.items[1], action_lib.ActionInvocation)
    self.assertIs(prepare_phase.items[1].parent_execution_unit, foo_invocation)
    self.assertEqual(prepare_phase.items[1].id, 'agent@1:/a1/prepare/a1')
    self.assertIsInstance(prepare_phase.items[2], lf_structured.QueryInvocation)
    self.assertEqual(prepare_phase.items[2].id, 'agent@1:/a1/prepare/q1')

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
    # root (0) > foo (0) > parallel executions (1)
    self.assertEqual(parallel_executions.position, (0, 0, 1))
    self.assertEqual(parallel_executions.id, 'agent@1:/a1/p1')
    self.assertIsInstance(parallel_executions, action_lib.ParallelExecutions)
    self.assertIs(
        parallel_executions.all_actions[0].parent_execution_unit,
        parallel_executions
    )
    self.assertIs(
        parallel_executions.all_actions[0].parent_action,
        foo_invocation
    )
    self.assertEqual(len(parallel_executions), 3)
    self.assertEqual(parallel_executions[0].id, 'agent@1:/a1/p1/b1')
    self.assertEqual(parallel_executions[1].id, 'agent@1:/a1/p1/b2')
    self.assertEqual(parallel_executions[2].id, 'agent@1:/a1/p1/b3')
    self.assertEqual(len(parallel_executions[0].execution_units), 1)
    self.assertEqual(len(parallel_executions[1].execution_units), 1)
    self.assertEqual(len(parallel_executions[2].execution_units), 1)
    self.assertEqual(len(parallel_executions[0].queries), 1)
    self.assertEqual(len(parallel_executions[0].all_queries), 2)
    self.assertEqual(len(parallel_executions[1].queries), 1)
    self.assertEqual(len(parallel_executions[1].all_queries), 2)
    self.assertEqual(len(parallel_executions[2].queries), 1)
    self.assertEqual(len(parallel_executions[2].all_queries), 2)
    self.assertEqual(len(parallel_executions.execution_units), 0)
    self.assertEqual(len(parallel_executions.actions), 0)
    self.assertEqual(len(parallel_executions.queries), 0)
    self.assertEqual(len(parallel_executions.logs), 0)
    self.assertEqual(len(parallel_executions.all_actions), 3)
    self.assertEqual(len(parallel_executions.all_queries), 6)
    self.assertEqual(len(parallel_executions.all_logs), 3)

    # Invocation to Bar.
    bar_invocation = foo_invocation.execution[3]
    self.assertIs(bar_invocation.parent_action, foo_invocation)
    self.assertIs(bar_invocation.parent_execution_unit, foo_invocation)
    self.assertEqual(bar_invocation.id, 'agent@1:/a1/a5')
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

  def test_acall(self):
    bar = Bar()
    with lf.context(baz=1):
      r = bar.acall(lm=fake.StaticResponse('lm response'))
      self.assertEqual(asyncio.run(r), 3)

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
    result = foo(session, lm=lm)
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
        foo(session, lm=lm)
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
      foo(session, lm=lm)

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

  def test_max_execution_time(self):
    lm = fake.StaticResponse('lm response')
    bar = Bar(simulate_execution_time=1)
    with self.assertRaisesRegex(
        action_lib.ActionTimeoutError,
        'Action .*Bar.*has exceeded .* 0.5 seconds'
    ):
      bar(lm=lm, max_execution_time=0.5)

    foo = Foo(1, simulate_execution_time=[0, 0, 0, 1])
    with self.assertRaisesRegex(
        action_lib.ActionTimeoutError,
        'Action .*Foo.* has exceeded .* 0.5 seconds'
    ):
      foo(lm=lm, max_execution_time=0.5)

    # Timeout within concurrent_map.
    foo = Foo(1, simulate_execution_time=[0, 0, 1, 0])
    with self.assertRaisesRegex(
        action_lib.ActionTimeoutError,
        'Action .*Foo.* has exceeded .* 0.5 seconds'
    ):
      foo(lm=lm, max_execution_time=0.5)

    # Timeout within bar.
    foo = Foo(
        1, simulate_execution_time=[0, 0, 0, 1], max_bar_execution_time=0.5
    )
    with self.assertRaisesRegex(
        action_lib.ActionTimeoutError,
        'Action .*Bar.* has exceeded .* 0.5 seconds'
    ):
      foo(lm=lm)

    # Timeout within bar, however the effective max_execution_time of bar is the
    # remaining time of the parent action as it's smaller (0.5 < 1).
    foo = Foo(
        1, simulate_execution_time=[0, 0.5, 0, 1.0], max_bar_execution_time=1.0
    )
    with self.assertRaisesRegex(
        action_lib.ActionTimeoutError,
        'Action .*Foo.* has exceeded .*1.0 seconds'
    ):
      foo(lm=lm, max_execution_time=1.0)

  def test_event_handler(self):

    class MyActionHandler(pg.Object, action_lib.SessionEventHandler):
      def _on_bound(self):
        super()._on_bound()
        self.progresses = []

      def on_session_start(self, session):
        session.add_metadata(progresses=pg.Ref(self.progresses))

      def on_action_progress(self, session, action, title, **kwargs):
        self.progresses.append((action.id, title))

    handler = MyActionHandler()
    self.assertIs(handler.get(MyActionHandler), handler)
    self.assertIsNone(handler.get(action_lib.SessionLogging))

    handler_chain = action_lib.SessionEventHandlerChain(
        handlers=[handler, action_lib.SessionLogging()]
    )
    self.assertIs(handler_chain.get(MyActionHandler), handler)
    self.assertIs(
        handler_chain.get(action_lib.SessionLogging),
        handler_chain.handlers[1]
    )

    session = action_lib.Session(
        id='agent@1',
        event_handler=handler_chain
    )
    bar = Bar()
    with session:
      bar(session, lm=fake.StaticResponse('lm response'))
      session.update_progress('Trajectory completed')

    self.assertIs(session.metadata['progresses'], handler.progresses)
    self.assertEqual(handler.progresses, [
        ('agent@1:/a1', 'Query completed'),
        ('agent@1:', 'Trajectory completed'),
    ])

  def test_clone(self):
    event_handler = action_lib.SessionLogging()
    session = action_lib.Session(event_handler=event_handler)
    other = session.clone()
    self.assertIsNot(session, other)
    self.assertIs(other.event_handler, event_handler)

    other = session.clone(deep=True)
    self.assertIsNot(session, other)
    self.assertIsNot(other.event_handler, session.event_handler)

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

  def test_query_with_track_if(self):
    lm = fake.StaticResponse('lm response')
    session = action_lib.Session()

    # Render session to trigger javascript updates to the HTML when
    # operating on the session.
    _ = session.to_html()
    with session:
      # This query will succeed.
      session.query(
          'prompt1',
          schema=None,
          lm=lm,
          track_if=lambda q: not q.has_error,
          default=None)
      # This query will fail during parsing.
      session.query(
          'prompt2',
          schema=int,
          lm=lm,
          track_if=lambda q: not q.has_error,
          default=None)
    self.assertEqual(len(session.root.queries), 1)
    self.assertIsNone(session.root.queries[0].error)


class CheckpointTest(unittest.TestCase):
  """Tests for action checkpointing functionality."""

  def test_action_checkpoint_serialization(self):
    """Round-trip via to_json_str/from_json_str preserves all fields."""
    checkpoint = action_lib.ActionCheckpoint(
        action_type='test.module.TestAction',
        state={'key': 'value', 'count': 42},
        step=5,
    )
    json_str = checkpoint.to_json_str()
    restored = pg.from_json_str(json_str)
    self.assertEqual(restored.action_type, 'test.module.TestAction')
    self.assertEqual(restored.state, {'key': 'value', 'count': 42})
    self.assertEqual(restored.step, 5)

  def test_file_checkpointer_save_load(self):
    """Save then load returns equal checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'checkpoint.json')
      checkpointer = action_lib.FileCheckpointer(path=path)
      checkpoint = action_lib.ActionCheckpoint(
          action_type='test.TestAction',
          state={'items': [1, 2, 3]},
          step=10,
      )
      checkpointer.save(checkpoint)
      loaded = checkpointer.load()
      self.assertIsNotNone(loaded)
      self.assertEqual(loaded.action_type, checkpoint.action_type)
      self.assertEqual(loaded.state, checkpoint.state)
      self.assertEqual(loaded.step, checkpoint.step)

  def test_file_checkpointer_missing_file(self):
    """load() returns None for missing path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'nonexistent.json')
      checkpointer = action_lib.FileCheckpointer(path=path)
      self.assertIsNone(checkpointer.load())

  def test_file_checkpointer_creates_dirs(self):
    """Nested path dirs created via pg.io.mkdirs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'nested', 'deep', 'checkpoint.json')
      checkpointer = action_lib.FileCheckpointer(path=path)
      checkpoint = action_lib.ActionCheckpoint(
          action_type='test.TestAction',
          state={},
          step=0,
      )
      checkpointer.save(checkpoint)
      self.assertTrue(os.path.exists(path))

  def test_action_default_on_checkpoint(self):
    """Default on_checkpoint returns ActionCheckpoint with empty state."""
    action = Bar()
    checkpoint = action.on_checkpoint()
    self.assertIsInstance(checkpoint, action_lib.ActionCheckpoint)
    self.assertIn('Bar', checkpoint.action_type)
    self.assertEqual(checkpoint.state, {})
    self.assertEqual(checkpoint.step, 0)

  def test_action_default_on_restore_validates_type(self):
    """Mismatched action_type raises ValueError."""
    action = Bar()
    wrong_checkpoint = action_lib.ActionCheckpoint(
        action_type='wrong.module.WrongAction',
        state={},
        step=0,
    )
    with self.assertRaises(ValueError) as ctx:
      action.on_restore(wrong_checkpoint)
    self.assertIn('Checkpoint type mismatch', str(ctx.exception))

  def test_custom_action_checkpoint_hooks(self):
    """Subclass state persists across checkpoint/restore."""

    class StatefulAction(action_lib.Action):

      def _on_bound(self):
        super()._on_bound()
        self._counter = 0
        self._items = []

      def on_checkpoint(self):
        return action_lib.ActionCheckpoint(
            action_type=self.checkpoint_type_id,
            state={'counter': self._counter, 'items': self._items},
            step=self._counter,
        )

      def on_restore(self, checkpoint):
        super().on_restore(checkpoint)
        self._counter = checkpoint.state.get('counter', 0)
        self._items = checkpoint.state.get('items', [])

      def call(self, session, **kwargs):
        self._counter += 1
        self._items.append(f'item_{self._counter}')
        return self._counter

    action1 = StatefulAction()
    action1._counter = 5
    action1._items = ['a', 'b', 'c']
    checkpoint = action1.on_checkpoint()

    action2 = StatefulAction()
    self.assertEqual(action2._counter, 0)
    self.assertEqual(action2._items, [])

    action2.on_restore(checkpoint)
    self.assertEqual(action2._counter, 5)
    self.assertEqual(action2._items, ['a', 'b', 'c'])

  def test_checkpoint_with_pg_object_state(self):
    """State containing pg.Object instances serializes correctly."""

    class TestData(pg.Object):
      name: str
      value: int

    checkpoint = action_lib.ActionCheckpoint(
        action_type='test.TestAction',
        state={'data': TestData(name='test', value=42)},
        step=1,
    )
    json_str = checkpoint.to_json_str()
    restored = pg.from_json_str(json_str)
    self.assertIsInstance(restored.state['data'], TestData)
    self.assertEqual(restored.state['data'].name, 'test')
    self.assertEqual(restored.state['data'].value, 42)

  def test_restore_from_checkpoint_convenience(self):
    """Creates action and restores state in one call."""

    class TestAction(action_lib.Action):
      x: int

      def _on_bound(self):
        super()._on_bound()
        self._state_value = 0

      def on_checkpoint(self):
        return action_lib.ActionCheckpoint(
            action_type=self.checkpoint_type_id,
            state={'state_value': self._state_value},
            step=self._state_value,
        )

      def on_restore(self, checkpoint):
        super().on_restore(checkpoint)
        self._state_value = checkpoint.state.get('state_value', 0)

      def call(self, session, **kwargs):
        return self.x

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'checkpoint.json')
      checkpointer = action_lib.FileCheckpointer(path=path)

      # Save a checkpoint with some state
      original = TestAction(x=10)
      original._state_value = 99
      checkpointer.save(original.on_checkpoint())

      # Use convenience method to restore
      restored, checkpoint = TestAction.restore_from_checkpoint(
          checkpointer, x=10
      )
      self.assertEqual(restored._state_value, 99)
      self.assertIsNotNone(checkpoint)
      self.assertEqual(checkpoint.step, 99)

      # Test with no existing checkpoint
      checkpointer2 = action_lib.FileCheckpointer(
          path=os.path.join(tmpdir, 'new.json')
      )
      fresh, checkpoint = TestAction.restore_from_checkpoint(
          checkpointer2, x=20
      )
      self.assertEqual(fresh._state_value, 0)  # Default from _on_bound
      self.assertIsNone(checkpoint)

  def test_agent_checkpoint_round_trip_with_pg_objects(self):
    """Mimics LangfunAgent pattern: list of pg.Objects in state round-trips."""

    class Step(pg.Object):
      step: int
      thoughts: str
      results: Any

    class AgentLikeAction(action_lib.Action):
      _CKPT_KEY_STEPS = 'steps'
      _CKPT_KEY_START_INDEX = 'start_index'
      _CKPT_KEY_FILE_DIR = 'file_dir'

      def _on_bound(self):
        super()._on_bound()
        self._steps = []
        self._start_index = 0
        self._restored = False

      def on_checkpoint(self):
        return action_lib.ActionCheckpoint(
            action_type=self.checkpoint_type_id,
            state={
                self._CKPT_KEY_STEPS: self._steps,
                self._CKPT_KEY_START_INDEX: self._start_index,
                self._CKPT_KEY_FILE_DIR: '/tmp/files',
            },
            step=len(self._steps),
        )

      def on_restore(self, checkpoint):
        super().on_restore(checkpoint)
        self._steps = checkpoint.state.get(self._CKPT_KEY_STEPS, [])
        self._start_index = checkpoint.state.get(self._CKPT_KEY_START_INDEX, 0)
        self._restored = True

      def call(self, session, **kwargs):
        return None

    original = AgentLikeAction()
    original._steps = [
        Step(step=0, thoughts='thinking', results='found it'),
        Step(step=1, thoughts='analyzing', results={'key': 'value'}),
    ]
    original._start_index = 1

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'agent_checkpoint.json')
      checkpointer = action_lib.FileCheckpointer(path=path)

      checkpointer.save(original.on_checkpoint())
      checkpoint = checkpointer.load()
      self.assertIsNotNone(checkpoint)
      self.assertEqual(checkpoint.step, 2)
      self.assertEqual(
          checkpoint.state[AgentLikeAction._CKPT_KEY_FILE_DIR], '/tmp/files'
      )

      restored = AgentLikeAction()
      self.assertFalse(restored._restored)
      restored.on_restore(checkpoint)
      self.assertTrue(restored._restored)
      self.assertEqual(len(restored._steps), 2)
      self.assertIsInstance(restored._steps[0], Step)
      self.assertEqual(restored._steps[0].step, 0)
      self.assertEqual(restored._steps[0].thoughts, 'thinking')
      self.assertEqual(restored._steps[1].results, {'key': 'value'})
      self.assertEqual(restored._start_index, 1)


if __name__ == '__main__':
  unittest.main()
