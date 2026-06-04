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
import time
import unittest
from unittest import mock

from langfun.env import base_feature
from langfun.env import environment
from langfun.env import interface
from langfun.env import test_utils

TestingNonSandboxBasedFeature = test_utils.TestingNonSandboxBasedFeature
TestingEventHandler = test_utils.TestingEventHandler


class NonSandboxBasedFeatureTests(unittest.TestCase):

  def test_basics(self):
    feature = TestingNonSandboxBasedFeature()
    event_handler = TestingEventHandler(
        log_session_setup=True,
        log_feature_setup=True,
        log_sandbox_status=True
    )
    env = environment.Environment(
        id='testing-env',
        features={'test_feature': feature},
        event_handler=event_handler,
    )
    self.assertFalse(env.all_online)
    self.assertEqual(len(list(env.features.values())), 1)
    with env:
      self.assertTrue(env.all_online)
      with env.test_feature('session1') as feature:
        self.assertIsNone(feature.sandbox)
        self.assertEqual(feature.session_id, 'session1')

    self.assertEqual(
        event_handler.logs,
        [
            '[testing-env/test_feature] feature setup',
            '[testing-env] environment started',
            '[testing-env/test_feature@session1] feature setup session',
            '[testing-env/test_feature@session1] feature teardown session',
            '[testing-env/test_feature] feature teardown',
            '[testing-env] environment shutdown'
        ]
    )

  def test_feature_setup_error(self):
    event_handler = TestingEventHandler(
        log_session_setup=True,
        log_feature_setup=True,
        log_sandbox_status=True
    )
    env = environment.Environment(
        id='testing-env',
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                simulate_setup_error=ValueError
            )
        },
        event_handler=event_handler,
    )
    with self.assertRaises(interface.FeatureSetupError):
      with env:
        pass
    self.assertEqual(
        event_handler.logs,
        [
            '[testing-env/test_feature] feature setup with FeatureSetupError',
            '[testing-env] environment started with FeatureSetupError',
            '[testing-env/test_feature] feature teardown',
            '[testing-env] environment shutdown'
        ]
    )

  def test_feature_teardown_error(self):
    event_handler = TestingEventHandler(
        log_session_setup=True,
        log_feature_setup=True,
        log_sandbox_status=True
    )
    env = environment.Environment(
        id='testing-env',
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                simulate_teardown_error=ValueError
            )
        },
        event_handler=event_handler,
    )
    with env:
      pass
    self.assertEqual(
        event_handler.logs,
        [
            '[testing-env/test_feature] feature setup',
            '[testing-env] environment started',
            (
                '[testing-env/test_feature] feature teardown with '
                'FeatureTeardownError'
            ),
            '[testing-env] environment shutdown'
        ]
    )

  def test_feature_setup_session_error(self):
    event_handler = TestingEventHandler(
        log_session_setup=True,
        log_feature_setup=True,
        log_sandbox_status=True
    )
    env = environment.Environment(
        id='testing-env',
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                simulate_setup_session_error=ValueError
            )
        },
        event_handler=event_handler,
    )
    with env:
      with self.assertRaises(ValueError):
        with env.test_feature('session1'):
          pass
    self.assertEqual(
        event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env/test_feature] feature setup',
            '[testing-env] environment started',
            '[testing-env/test_feature@session1] feature setup session with ValueError',
            '[testing-env/test_feature@session1] feature teardown session',
            '[testing-env/test_feature] feature teardown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )

  def test_feature_teardown_session_error(self):
    event_handler = TestingEventHandler(
        log_session_setup=True,
        log_feature_setup=True,
        log_sandbox_status=True
    )
    env = environment.Environment(
        id='testing-env',
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                simulate_teardown_session_error=ValueError
            )
        },
        event_handler=event_handler,
    )
    with env:
      with env.test_feature('session1'):
        pass
    self.assertEqual(
        event_handler.logs,
        [
            # pylint: disable=line-too-long
            '[testing-env/test_feature] feature setup',
            '[testing-env] environment started',
            '[testing-env/test_feature@session1] feature setup session',
            '[testing-env/test_feature@session1] feature teardown session with ValueError',
            '[testing-env/test_feature] feature teardown',
            '[testing-env] environment shutdown',
            # pylint: enable=line-too-long
        ]
    )

  def test_feature_housekeeping(self):
    event_handler = TestingEventHandler(
        log_sandbox_status=False,
        log_feature_setup=False,
        log_housekeep=True
    )
    feature = TestingNonSandboxBasedFeature(housekeep_interval=0.1)
    env = environment.Environment(
        id='testing-env',
        features={'test_feature': feature},
        event_handler=event_handler,
    )
    with env:
      feature.wait_for_housekeeping()
    self.assertIn(
        '[testing-env/test_feature] feature housekeeping 0',
        event_handler.logs
    )

  def test_feature_housekeeping_error(self):
    event_handler = TestingEventHandler(
        log_sandbox_status=False,
        log_feature_setup=False,
        log_housekeep=True
    )
    feature = TestingNonSandboxBasedFeature(
        simulate_housekeep_error=ValueError,
        housekeep_interval=0.1,
    )
    env = environment.Environment(
        id='testing-env',
        features={'test_feature': feature},
        event_handler=event_handler,
        outage_grace_period=0.0,
    )
    with env:
      feature.wait_for_housekeeping()
      while env.all_online:
        time.sleep(0.01)
      self.assertFalse(env.all_online)
    self.assertIn(
        '[testing-env/test_feature] feature housekeeping 0 with ValueError',
        event_handler.logs
    )

  def test_root_dir_propagation(self):
    feature = TestingNonSandboxBasedFeature()
    env = environment.Environment(
        id='testing-env',
        features={'test_feature': feature},
        root_dir='/env/root',
    )
    with env:
      self.assertEqual(feature.root_dir, '/env/root')

  def test_working_dir_unbound(self):
    feature = TestingNonSandboxBasedFeature()
    self.assertIsNone(feature.working_dir)

  def test_offline_duration(self):
    feature = TestingNonSandboxBasedFeature()
    self.assertEqual(feature.offline_duration, 0.0)

    # Simulate offline state
    feature._offline_start_time = time.time() - 5.0
    self.assertGreaterEqual(feature.offline_duration, 5.0)


class SandboxBasedFeatureTests(unittest.TestCase):

  def test_sandbox_based_feature_properties(self):
    class TestingSandboxBasedFeature(base_feature.BaseFeature):
      is_sandbox_based: bool = True

    mock_sandbox = mock.MagicMock()
    mock_sandbox.is_online = True
    mock_sandbox.working_dir = '/mock/sandbox/dir'

    with (
        mock.patch.object(
            TestingSandboxBasedFeature,
            'sandbox',
            new_callable=mock.PropertyMock,
        ) as mock_sandbox_property,
        mock.patch.object(
            TestingSandboxBasedFeature,
            'name',
            new_callable=mock.PropertyMock,
        ) as mock_name_property,
    ):
      mock_sandbox_property.return_value = mock_sandbox
      mock_name_property.return_value = 'TestingSandboxBasedFeature'
      feature = TestingSandboxBasedFeature()

      self.assertTrue(feature.is_online)
      self.assertEqual(
          feature.working_dir,
          '/mock/sandbox/dir/TestingSandboxBasedFeature'
      )

if __name__ == '__main__':
  unittest.main()

