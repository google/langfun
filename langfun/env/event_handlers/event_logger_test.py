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

import contextlib
import io
from typing import Iterator
import unittest

from langfun.env import interface
from langfun.env import test_utils
from langfun.env.event_handlers import event_logger as event_logger_lib
import pyglove as pg


class EventLoggerTest(unittest.TestCase):

  LOGGER_CLS = event_logger_lib.EventLogger

  @contextlib.contextmanager
  def _capture_logs(self, test_name: str) -> Iterator[io.StringIO]:
    try:
      stream = io.StringIO()
      with pg.logging.redirect_stream(
          stream, name=f'{self.LOGGER_CLS.__name__}.{test_name}'
      ):
        yield stream
    finally:
      pass

  def _test_logger(
      self,
      *,
      test_name: str,
      expected_substrings: list[str],
      unexpected_substrings: list[str],
      error_only: bool = False,
      regex: str | None = None,
      sandbox_status: bool = True,
      session_status: bool = True,
      feature_status: bool = True,
      housekeep_status: bool = True,
      stats_report_interval: float = 1.0,
  ):
    event_logger = self.LOGGER_CLS(
        error_only=error_only,
        regex=regex,
        sandbox_status=sandbox_status,
        session_status=session_status,
        feature_status=feature_status,
        housekeep_status=housekeep_status,
        stats_report_interval=stats_report_interval,
    )
    env = test_utils.TestingEnvironment(
        features={
            'test_feature1': test_utils.TestingFeature(housekeep_interval=0),
            'test_feature2': test_utils.TestingFeature(housekeep_interval=None),
        },
        pool_size=2,
        outage_grace_period=0,
        outage_retry_interval=0,
        housekeep_interval=1.0,
        sandbox_keepalive_interval=1.0,
        event_handler=event_logger,
    )
    with self._capture_logs(test_name) as stream:
      with env:
        with env.sandbox(session_id='session1') as sb:
          self.assertEqual(sb.test_feature1.num_shell_calls(), 4)

        with self.assertRaises(interface.SandboxStateError):
          with env.sandbox(session_id='session2') as sb:
            sb.shell('echo "bar"', raise_error=RuntimeError)

    stdout = stream.getvalue()
    for substring in expected_substrings:
      self.assertIn(substring, stdout)
    for substring in unexpected_substrings:
      self.assertNotIn(substring, stdout)

  def test_all_flags_on(self):
    return self._test_logger(
        test_name='test_all_flags_on',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutdown',
            'environment housekeeping',
            'environment stats',
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'sandbox housekeeping',
            'feature setup complete',
            'feature teardown complete',
            '/test_feature1] feature housekeeping',
            'sandbox session started',
            'sandbox session ended',
            'sandbox call \'shell\'',
            'RuntimeError',
        ],
        unexpected_substrings=[
            '/test_feature2] feature housekeeping',
        ],
    )

  def test_error_only(self):
    return self._test_logger(
        test_name='test_error_only',
        expected_substrings=[
            'session ended',
            'call \'shell\'',
            'RuntimeError',
        ],
        unexpected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment housekeeping',
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'feature setup complete',
            'feature teardown complete',
            'session started',
        ],
        error_only=True,
    )

  def test_regex(self):
    return self._test_logger(
        test_name='test_regex',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment housekeeping',
        ],
        unexpected_substrings=[
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'feature setup complete',
            'feature teardown complete',
            'sandbox session started',
            'sandbox session ended',
            'sandbox call \'shell\'',
            'RuntimeError',
        ],
        regex='.*environment.*',
    )

  def test_sandbox_status_off(self):
    return self._test_logger(
        test_name='test_sandbox_status_off',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment housekeeping',
            'feature setup complete',
            'feature teardown complete',
            'feature housekeeping',
            'sandbox session started',
            'sandbox session ended',
            'sandbox call \'shell\'',
            'RuntimeError',
        ],
        unexpected_substrings=[
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'sandbox housekeeping',
        ],
        sandbox_status=False,
    )

  def test_feature_status_off(self):
    return self._test_logger(
        test_name='test_feature_status_off',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment housekeeping',
            'environment stats',
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'sandbox housekeeping',
            'sandbox session started',
            'sandbox session ended',
            'sandbox call \'shell\'',
            'RuntimeError',
        ],
        unexpected_substrings=[
            'feature setup complete',
            'feature teardown complete',
            'feature housekeeping',
        ],
        feature_status=False,
    )

  def test_session_status_off(self):
    return self._test_logger(
        test_name='test_session_status_off',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment stats',
            'environment housekeeping',
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'sandbox housekeeping',
            'feature setup complete',
            'feature teardown complete',
            'feature housekeeping',
            'sandbox call \'shell\'',
            'RuntimeError',
        ],
        unexpected_substrings=[
            'sandbox session started',
            'sandbox session ended',
        ],
        session_status=False,
    )

  def test_housekeep_status_off(self):
    return self._test_logger(
        test_name='test_housekeep_status_off',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment stats',
            'sandbox started',
            '-> acquired',
            'sandbox shutdown',
            'feature setup complete',
            'feature teardown complete',
            'sandbox session started',
            'sandbox session ended',
            'sandbox call \'shell\'',
            'RuntimeError',
        ],
        unexpected_substrings=[
            'environment housekeeping',
            'sandbox housekeeping',
            'feature housekeeping',
        ],
        housekeep_status=False,
    )

  def test_no_stats_report(self):
    return self._test_logger(
        test_name='test_housekeep_status_off',
        expected_substrings=[
            'environment starting',
            'environment started',
            'environment shutting down',
            'environment shutdown',
            'environment housekeeping',
        ],
        unexpected_substrings=[
            'environment stats',
        ],
        stats_report_interval=None,
    )


class ConsoleEventLoggerTest(EventLoggerTest):

  LOGGER_CLS = event_logger_lib.ConsoleEventLogger

  @contextlib.contextmanager
  def _capture_logs(self, test_name: str) -> Iterator[io.StringIO]:
    try:
      stream = io.StringIO()
      with contextlib.redirect_stdout(stream):
        yield stream
    finally:
      pass


if __name__ == '__main__':
  unittest.main()
