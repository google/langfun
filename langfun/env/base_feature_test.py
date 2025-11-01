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

from langfun.env import test_utils

TestingEnvironment = test_utils.TestingEnvironment
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
    env = TestingEnvironment(
        image_ids=[],
        features={'test_feature': feature},
        event_handler=event_handler,
    )
    self.assertFalse(env.is_online)
    self.assertEqual(len(list(env.non_sandbox_based_features())), 1)
    with env:
      self.assertTrue(env.is_online)
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
    env = TestingEnvironment(
        image_ids=[],
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                simulate_setup_error=ValueError
            )
        },
        event_handler=event_handler,
    )
    with self.assertRaises(ValueError):
      with env:
        pass
    self.assertEqual(
        event_handler.logs,
        [
            '[testing-env/test_feature] feature setup with ValueError',
            '[testing-env] environment started with ValueError',
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
    env = TestingEnvironment(
        image_ids=[],
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
            '[testing-env/test_feature] feature teardown with ValueError',
            '[testing-env] environment shutdown'
        ]
    )

  def test_feature_setup_session_error(self):
    event_handler = TestingEventHandler(
        log_session_setup=True,
        log_feature_setup=True,
        log_sandbox_status=True
    )
    env = TestingEnvironment(
        image_ids=[],
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
    env = TestingEnvironment(
        image_ids=[],
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
    env = TestingEnvironment(
        image_ids=[],
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                housekeep_interval=0.1
            )
        },
        event_handler=event_handler,
        housekeep_interval=0.2
    )
    with env:
      env.wait_for_housekeeping()
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
    env = TestingEnvironment(
        image_ids=[],
        features={
            'test_feature': TestingNonSandboxBasedFeature(
                simulate_housekeep_error=ValueError,
                housekeep_interval=0.1
            )
        },
        event_handler=event_handler,
        housekeep_interval=0.2
    )
    with env:
      env.wait_for_housekeeping()
      self.assertFalse(env.is_online)
    self.assertIn(
        '[testing-env/test_feature] feature housekeeping 0 with ValueError',
        event_handler.logs
    )


if __name__ == '__main__':
  unittest.main()

