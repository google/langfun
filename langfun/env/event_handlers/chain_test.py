# Copyright 2025 The Langfun Authors
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

import unittest
from langfun.env import interface
from langfun.env.event_handlers import chain


class MockEventHandler(interface.EventHandler):

  def __init__(self):
    self.calls = []

  def _record_call(self, method_name, *args, **kwargs):
    self.calls.append((method_name, args, kwargs))

  def on_environment_starting(self, environment):
    self._record_call('on_environment_starting', environment)

  def on_environment_shutting_down(self, environment, offline_duration):
    self._record_call(
        'on_environment_shutting_down', environment, offline_duration
    )

  def on_environment_start(self, environment, duration, error):
    self._record_call('on_environment_start', environment, duration, error)

  def on_environment_housekeep(
      self, environment, counter, duration, error, **kwargs
  ):
    self._record_call(
        'on_environment_housekeep',
        environment, counter, duration, error, **kwargs
    )

  def on_environment_shutdown(self, environment, duration, lifetime, error):
    self._record_call(
        'on_environment_shutdown', environment, duration, lifetime, error
    )

  def on_sandbox_start(self, sandbox, duration, error):
    self._record_call('on_sandbox_start', sandbox, duration, error)

  def on_sandbox_status_change(
      self, sandbox, old_status, new_status, span
  ):
    self._record_call(
        'on_sandbox_status_change',
        sandbox,
        old_status,
        new_status,
        span,
    )

  def on_sandbox_shutdown(
      self, sandbox, duration, lifetime, error
  ):
    self._record_call('on_sandbox_shutdown', sandbox, duration, lifetime, error)

  def on_sandbox_session_start(self, sandbox, session_id, duration, error):
    self._record_call(
        'on_sandbox_session_start', sandbox, session_id, duration, error
    )

  def on_sandbox_session_end(
      self, sandbox, session_id, duration, lifetime, error
  ):
    self._record_call(
        'on_sandbox_session_end',
        sandbox,
        session_id,
        duration,
        lifetime,
        error,
    )

  def on_sandbox_activity(
      self,
      name,
      sandbox,
      session_id,
      duration,
      error,
      **kwargs,
  ):
    self._record_call(
        'on_sandbox_activity',
        name,
        sandbox,
        session_id,
        duration,
        error,
        **kwargs,
    )

  def on_sandbox_housekeep(self, sandbox, counter, duration, error, **kwargs):
    self._record_call(
        'on_sandbox_housekeep',
        sandbox,
        counter,
        duration,
        error,
        **kwargs,
    )

  def on_feature_setup(self, feature, duration, error):
    self._record_call('on_feature_setup', feature, duration, error)

  def on_feature_teardown(self, feature, duration, error):
    self._record_call('on_feature_teardown', feature, duration, error)

  def on_feature_setup_session(self, feature, session_id, duration, error):
    self._record_call(
        'on_feature_setup_session',
        feature,
        session_id,
        duration,
        error,
    )

  def on_feature_teardown_session(self, feature, session_id, duration, error):
    self._record_call(
        'on_feature_teardown_session',
        feature,
        session_id,
        duration,
        error,
    )

  def on_feature_activity(
      self, name, feature, session_id, duration, error, **kwargs,
  ):
    self._record_call(
        'on_feature_activity',
        name,
        feature,
        session_id,
        duration,
        error,
        **kwargs,
    )

  def on_feature_housekeep(
      self, feature, counter, duration, error, **kwargs
  ):
    self._record_call(
        'on_feature_housekeep',
        feature,
        counter,
        duration,
        error,
        **kwargs,
    )


class EventHandlerChainTest(unittest.TestCase):

  def test_chain(self):
    handler1 = MockEventHandler()
    handler2 = MockEventHandler()
    chain_handler = chain.EventHandlerChain([handler1, handler2])

    env = object()
    sandbox = object()
    feature = object()

    chain_handler.on_environment_starting(env)
    chain_handler.on_environment_shutting_down(env, 1.0)
    chain_handler.on_environment_start(env, 2.0, None)
    chain_handler.on_environment_housekeep(env, 1, 3.0, None, a=1)
    chain_handler.on_environment_shutdown(env, 4.0, 5.0, None)
    chain_handler.on_sandbox_start(sandbox, 6.0, None)
    chain_handler.on_sandbox_status_change(sandbox, 'old', 'new', 7.0)
    chain_handler.on_sandbox_shutdown(sandbox, 8.0, 9.0, None)
    chain_handler.on_sandbox_session_start(sandbox, 's2', 16.0, None)
    chain_handler.on_sandbox_session_end(sandbox, 's2', 17.0, 18.0, None)
    chain_handler.on_sandbox_activity('act', sandbox, 's2', 19.0, None, d=4)
    chain_handler.on_sandbox_housekeep(sandbox, 2, 10.0, None, b=2)
    chain_handler.on_feature_setup(feature, 11.0, None)
    chain_handler.on_feature_teardown(feature, 12.0, None)
    chain_handler.on_feature_setup_session(feature, 's1', 13.0, None)
    chain_handler.on_feature_teardown_session(feature, 's1', 14.0, None)
    chain_handler.on_feature_activity('act', feature, 's1', 16.0, None, d=5)
    chain_handler.on_feature_housekeep(feature, 3, 15.0, None, c=3)

    self.assertEqual(handler1.calls, handler2.calls)
    self.assertEqual(
        handler1.calls,
        [
            ('on_environment_starting', (env,), {}),
            ('on_environment_shutting_down', (env, 1.0), {}),
            ('on_environment_start', (env, 2.0, None), {}),
            ('on_environment_housekeep', (env, 1, 3.0, None), {'a': 1}),
            ('on_environment_shutdown', (env, 4.0, 5.0, None), {}),
            ('on_sandbox_start', (sandbox, 6.0, None), {}),
            ('on_sandbox_status_change', (sandbox, 'old', 'new', 7.0), {}),
            ('on_sandbox_shutdown', (sandbox, 8.0, 9.0, None), {}),
            ('on_sandbox_session_start', (sandbox, 's2', 16.0, None), {}),
            ('on_sandbox_session_end', (sandbox, 's2', 17.0, 18.0, None), {}),
            (
                'on_sandbox_activity',
                ('act', sandbox, 's2', 19.0, None),
                {'d': 4},
            ),
            ('on_sandbox_housekeep', (sandbox, 2, 10.0, None), {'b': 2}),
            ('on_feature_setup', (feature, 11.0, None), {}),
            ('on_feature_teardown', (feature, 12.0, None), {}),
            ('on_feature_setup_session', (feature, 's1', 13.0, None), {}),
            ('on_feature_teardown_session', (feature, 's1', 14.0, None), {}),
            (
                'on_feature_activity',
                ('act', feature, 's1', 16.0, None),
                {'d': 5}
            ),
            ('on_feature_housekeep', (feature, 3, 15.0, None), {'c': 3}),
        ],
    )

  def test_add_remove(self):
    handler1 = MockEventHandler()
    handler2 = MockEventHandler()
    chain_handler = chain.EventHandlerChain([handler1])
    chain_handler.add(handler2)
    env = object()
    chain_handler.on_environment_starting(env)
    self.assertEqual(handler1.calls, [('on_environment_starting', (env,), {})])
    self.assertEqual(handler2.calls, [('on_environment_starting', (env,), {})])

    chain_handler.remove(handler1)
    chain_handler.on_environment_start(env, 1.0, None)
    self.assertEqual(handler1.calls, [('on_environment_starting', (env,), {})])
    self.assertEqual(
        handler2.calls,
        [
            ('on_environment_starting', (env,), {}),
            ('on_environment_start', (env, 1.0, None), {}),
        ],
    )


if __name__ == '__main__':
  unittest.main()
