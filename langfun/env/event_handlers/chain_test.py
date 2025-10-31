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

  def on_sandbox_start(self, environment, sandbox, duration, error):
    self._record_call('on_sandbox_start', environment, sandbox, duration, error)

  def on_sandbox_status_change(
      self, environment, sandbox, old_status, new_status, span
  ):
    self._record_call(
        'on_sandbox_status_change',
        environment,
        sandbox,
        old_status,
        new_status,
        span,
    )

  def on_sandbox_shutdown(
      self, environment, sandbox, duration, lifetime, error
  ):
    self._record_call(
        'on_sandbox_shutdown', environment, sandbox, duration, lifetime, error
    )

  def on_sandbox_housekeep(
      self, environment, sandbox, counter, duration, error, **kwargs
  ):
    self._record_call(
        'on_sandbox_housekeep',
        environment,
        sandbox,
        counter,
        duration,
        error,
        **kwargs,
    )

  def on_feature_setup(self, environment, sandbox, feature, duration, error):
    self._record_call(
        'on_feature_setup', environment, sandbox, feature, duration, error
    )

  def on_feature_teardown(self, environment, sandbox, feature, duration, error):
    self._record_call(
        'on_feature_teardown', environment, sandbox, feature, duration, error
    )

  def on_feature_setup_session(
      self, environment, sandbox, feature, session_id, duration, error
  ):
    self._record_call(
        'on_feature_setup_session',
        environment,
        sandbox,
        feature,
        session_id,
        duration,
        error,
    )

  def on_feature_teardown_session(
      self, environment, sandbox, feature, session_id, duration, error
  ):
    self._record_call(
        'on_feature_teardown_session',
        environment,
        sandbox,
        feature,
        session_id,
        duration,
        error,
    )

  def on_feature_housekeep(
      self, environment, sandbox, feature, counter, duration, error, **kwargs
  ):
    self._record_call(
        'on_feature_housekeep',
        environment,
        sandbox,
        feature,
        counter,
        duration,
        error,
        **kwargs,
    )

  def on_session_start(
      self, environment, sandbox, session_id, duration, error
  ):
    self._record_call(
        'on_session_start', environment, sandbox, session_id, duration, error
    )

  def on_session_end(
      self, environment, sandbox, session_id, duration, lifetime, error
  ):
    self._record_call(
        'on_session_end',
        environment,
        sandbox,
        session_id,
        duration,
        lifetime,
        error,
    )

  def on_sandbox_activity(
      self,
      name,
      environment,
      sandbox,
      feature,
      session_id,
      duration,
      error,
      **kwargs,
  ):
    self._record_call(
        'on_sandbox_activity',
        name,
        environment,
        sandbox,
        feature,
        session_id,
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
    chain_handler.on_sandbox_start(env, sandbox, 6.0, None)
    chain_handler.on_sandbox_status_change(env, sandbox, 'old', 'new', 7.0)
    chain_handler.on_sandbox_shutdown(env, sandbox, 8.0, 9.0, None)
    chain_handler.on_sandbox_housekeep(env, sandbox, 2, 10.0, None, b=2)
    chain_handler.on_feature_setup(env, sandbox, feature, 11.0, None)
    chain_handler.on_feature_teardown(env, sandbox, feature, 12.0, None)
    chain_handler.on_feature_setup_session(
        env, sandbox, feature, 's1', 13.0, None
    )
    chain_handler.on_feature_teardown_session(
        env, sandbox, feature, 's1', 14.0, None
    )
    chain_handler.on_feature_housekeep(
        env, sandbox, feature, 3, 15.0, None, c=3
    )
    chain_handler.on_session_start(env, sandbox, 's2', 16.0, None)
    chain_handler.on_session_end(env, sandbox, 's2', 17.0, 18.0, None)
    chain_handler.on_sandbox_activity(
        'act', env, sandbox, feature, 's2', 19.0, None, d=4
    )

    self.assertEqual(handler1.calls, handler2.calls)
    self.assertEqual(
        handler1.calls,
        [
            ('on_environment_starting', (env,), {}),
            ('on_environment_shutting_down', (env, 1.0), {}),
            ('on_environment_start', (env, 2.0, None), {}),
            ('on_environment_housekeep', (env, 1, 3.0, None), {'a': 1}),
            ('on_environment_shutdown', (env, 4.0, 5.0, None), {}),
            ('on_sandbox_start', (env, sandbox, 6.0, None), {}),
            ('on_sandbox_status_change', (env, sandbox, 'old', 'new', 7.0), {}),
            ('on_sandbox_shutdown', (env, sandbox, 8.0, 9.0, None), {}),
            ('on_sandbox_housekeep', (env, sandbox, 2, 10.0, None), {'b': 2}),
            ('on_feature_setup', (env, sandbox, feature, 11.0, None), {}),
            ('on_feature_teardown', (env, sandbox, feature, 12.0, None), {}),
            (
                'on_feature_setup_session',
                (env, sandbox, feature, 's1', 13.0, None),
                {},
            ),
            (
                'on_feature_teardown_session',
                (env, sandbox, feature, 's1', 14.0, None),
                {},
            ),
            (
                'on_feature_housekeep',
                (env, sandbox, feature, 3, 15.0, None),
                {'c': 3},
            ),
            ('on_session_start', (env, sandbox, 's2', 16.0, None), {}),
            ('on_session_end', (env, sandbox, 's2', 17.0, 18.0, None), {}),
            (
                'on_sandbox_activity',
                ('act', env, sandbox, feature, 's2', 19.0, None),
                {'d': 4},
            ),
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
