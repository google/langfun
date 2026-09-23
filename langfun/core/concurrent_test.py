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
"""Tests for langfun.core.concurrent."""

import collections
from concurrent import futures
import contextlib
import gc
import io
import sys
import threading
import time
from typing import Any
import unittest
from unittest import mock
import weakref

from langfun.core import component
from langfun.core import concurrent
from langfun.core import language_model as lm_lib
import pyglove as pg


class A(component.Component):
  x: int = 1
  y: int = component.contextual()


class RetryErrorTest(unittest.TestCase):

  def test_basics(self):
    def foo():
      pass

    retry_error = concurrent.RetryError(
        foo, [ValueError('abc'), ValueError('def'), ValueError('ghi')], [1, 2]
    )

    self.assertEqual(retry_error.attempts, 3)
    self.assertRegex(
        repr(retry_error),
        r'RetryError\(func=.*, errors=.*, wait_intervals=\[1, 2\]\)',
    )
    self.assertRegex(
        str(retry_error),
        (
            r'Calling .* failed after 3 attempts \(wait time: 1, 2 seconds\). '
            r'Last error: ghi'
        ),
    )

  def test_eq(self):
    f = lambda x: x
    e = ValueError()
    self.assertEqual(
        concurrent.RetryError(f, [e, e], [1]),
        concurrent.RetryError(f, [e, e], [1]),
    )
    self.assertNotEqual(
        concurrent.RetryError(f, [e, e], [1]),
        1,
    )
    self.assertNotEqual(
        concurrent.RetryError(f, [e, e], [1]),
        concurrent.RetryError(f, [e, e], [2]),
    )
    # Test hashing.
    self.assertEqual(
        hash(concurrent.RetryError(f, [e, e], [1])),
        hash(concurrent.RetryError(f, [e, e], [1])),
    )
    self.assertNotEqual(
        hash(concurrent.RetryError(f, [e, e], [1])),
        hash(concurrent.RetryError(f, [e, e], [2])),
    )


class RetryTest(unittest.TestCase):

  def assert_retry(self, func, expected_attempts, expected_wait_intervals):
    with pg.catch_errors(concurrent.RetryError) as error_context:
      func()

    self.assertIsNotNone(error_context.error)
    self.assertEqual(error_context.error.attempts, expected_attempts)
    self.assertEqual(
        error_context.error.wait_intervals, expected_wait_intervals
    )

  def test_retry_with_static_interval(self):
    def foo():
      raise ValueError('Intentional error.')

    foo_with_retry = concurrent.with_retry(
        foo,
        ValueError,
        max_attempts=3,
        retry_interval=1,
        exponential_backoff=False,
    )
    self.assert_retry(foo_with_retry, 3, [1, 1])

  def test_retry_with_interval_range(self):
    def foo():
      raise ValueError('Intentional error.')

    foo_with_retry = concurrent.with_retry(
        foo,
        ValueError,
        max_attempts=3,
        retry_interval=(1, 5),
        exponential_backoff=False,
        seed=1,
    )
    self.assert_retry(foo_with_retry, 3, [2, 5])

  def test_retry_with_exponential_backoff(self):
    def foo():
      raise ValueError('Intentional error.')

    foo_with_retry = concurrent.with_retry(
        foo, ValueError, max_attempts=4, retry_interval=1,
    )
    self.assert_retry(foo_with_retry, 4, [1, 2, 4])

  def test_retry_with_max_retry_interval(self):
    def foo():
      raise ValueError('Intentional error.')

    foo_with_retry = concurrent.with_retry(
        foo, ValueError, max_attempts=4, retry_interval=1, max_retry_interval=3,
    )
    self.assert_retry(foo_with_retry, 4, [1, 2, 3])

  def test_retry_with_uncaught_exception(self):
    def foo():
      raise ValueError('Intentional error.')

    foo_with_retry = concurrent.with_retry(
        foo, KeyError, max_attempts=4, retry_interval=1
    )

    with self.assertRaises(ValueError):
      foo_with_retry()

  def test_retry_with_job(self):
    count = 0

    def foo():
      nonlocal count
      count += 1
      if count < 3:
        raise ValueError('Foo temporary error.')
      return 'Success'

    job = concurrent.Job(
        foo,
        retry_on_errors=ValueError,
        retry_interval=1,
    )
    job()
    self.assertEqual(job.result, 'Success')
    self.assertEqual(
        [retry_entry.wait_interval for retry_entry in job.retry_entries],
        [0, 1, 2],
    )
    self.assertIsInstance(job.retry_entries[0].error, ValueError)
    self.assertIsInstance(job.retry_entries[1].error, ValueError)
    self.assertIsNone(job.retry_entries[2].error)


# Hang guard only: a correct `with_hedging` satisfies every case below in
# milliseconds, so this limit is reached only when something deadlocks.
_HEDGE_TIMEOUT = 30.0

# A `hedge_after` no test can outlive, used to prove that no hedge fires.
_NEVER_HEDGE = 60.0

# A `hedge_after` short enough that a hedge fires promptly.
_HEDGE_SOON = 0.01

# Window for asserting that something never happens. At `_HEDGE_SOON` it spans
# 50 re-issue boundaries.
_NEGATIVE_WINDOW = 0.5


class _Attempt:
  """A single execution of the function wrapped by `with_hedging`."""

  def __init__(self, ordinal: int, args, kwargs):
    self.ordinal = ordinal
    self.thread = threading.current_thread()
    self.daemon = self.thread.daemon
    self.args = args
    self.kwargs = kwargs
    self.completed = False


class _AttemptLog:
  """Thread-safe log of the attempts made by a hedged call."""

  def __init__(self):
    self._lock = threading.Lock()
    self._started = {}
    self.records = []

  def record(self, args=(), kwargs=None) -> _Attempt:
    """Logs the calling thread as a new attempt and returns its record."""
    with self._lock:
      attempt = _Attempt(len(self.records) + 1, args, kwargs or {})
      self.records.append(attempt)
    self._event(attempt.ordinal).set()
    return attempt

  def started(self, ordinal: int, timeout: float) -> bool:
    """Returns True if attempt `ordinal` started within `timeout` seconds."""
    return self._event(ordinal).wait(timeout)

  def _event(self, ordinal: int) -> threading.Event:
    with self._lock:
      event = self._started.get(ordinal)
      if event is None:
        event = threading.Event()
        self._started[ordinal] = event
      return event

  def __len__(self) -> int:
    with self._lock:
      return len(self.records)


class _FakeLM(lm_lib.LanguageModel):
  """An LM that echoes the prompt, so usage accounting is real but hermetic."""

  def _sample(self, prompts: list[Any]) -> list[lm_lib.LMSamplingResult]:
    return [
        lm_lib.LMSamplingResult(
            [lm_lib.LMSample(response=prompt.text, score=1.0)],
            usage=lm_lib.LMSamplingUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        )
        for prompt in prompts
    ]

  @property
  def model_info(self) -> lm_lib.ModelInfo:
    return lm_lib.ModelInfo(model_id='fake-lm')


class HedgingTest(unittest.TestCase):
  """Tests for `with_hedging`.

  These tests are event-driven rather than sleep-based: attempts park on
  `threading.Event` objects that the test releases, and every wait carries a
  timeout so that a hang fails loudly instead of stalling the test runner.
  """

  def setUp(self):
    super().setUp()
    self.attempts = _AttemptLog()
    self.hedges = []
    # Released on teardown, so attempts parked for the duration of a test can
    # exit and hand back their thread quota.
    self.blocked = threading.Event()
    self.addCleanup(self.blocked.set)
    # Give each test its own thread quota, so attempts still winding down from
    # an earlier test cannot influence this one.
    # pylint: disable=protected-access
    quota = mock.patch.object(
        concurrent,
        '_hedge_thread_quota',
        threading.Semaphore(concurrent._MAX_HEDGE_THREADS),
    )
    # pylint: enable=protected-access
    quota.start()
    self.addCleanup(quota.stop)

  def park(self, event: threading.Event | None = None) -> None:
    """Parks the calling attempt until the test releases it."""
    (event or self.blocked).wait(timeout=_HEDGE_TIMEOUT)

  def call_async(self, func) -> futures.Future[Any]:
    """Calls `func()` on a helper thread, so that the test can drive events."""
    future = futures.Future()

    def _run():
      try:
        future.set_result(func())
      except BaseException as e:  # pylint: disable=broad-except
        future.set_exception(e)

    threading.Thread(target=_run, daemon=True).start()
    return future

  def fail_thread_start_after(
      self, healthy_starts: int, refused: threading.Event | None = None
  ):
    """Returns a patch making `Thread.start` raise after `healthy_starts`."""
    real_thread = threading.Thread
    lock = threading.Lock()
    created = [0]

    def refuse_to_start():
      if refused is not None:
        refused.set()
      raise RuntimeError("can't start new thread")

    def make_thread(*args, target=None, **kwargs):
      thread = real_thread(*args, target=target, **kwargs)
      with lock:
        created[0] += 1
        healthy = created[0] <= healthy_starts
      if not healthy:
        thread.start = mock.Mock(side_effect=refuse_to_start)
      return thread

    return mock.patch.object(threading, 'Thread', side_effect=make_thread)

  def test_fast_call_costs_one_request(self):
    def func():
      return self.attempts.record().ordinal

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_NEVER_HEDGE,
        max_hedges=5,
        on_hedge=self.hedges.append,
    )
    # A call answered under the threshold costs exactly one request, however
    # large the hedge budget is.
    self.assertEqual(hedged(), 1)
    self.assertEqual(len(self.attempts), 1)
    self.assertEqual(self.hedges, [])

  def test_hedge_is_issued_once_the_threshold_elapses(self):
    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park()
        attempt.completed = True
        return 'primary'
      return 'hedge-1'

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=1,
        on_hedge=self.hedges.append,
    )
    self.assertEqual(hedged(), 'hedge-1')
    self.assertEqual(len(self.attempts), 2)
    self.assertEqual(self.hedges, [1])
    self.assertFalse(self.attempts.records[0].completed)

  def test_hedge_is_reissued_at_every_boundary(self):
    def func():
      attempt = self.attempts.record()
      if attempt.ordinal <= 3:
        self.park()
        attempt.completed = True
        return 'parked'
      return 'answered'

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=3,
        on_hedge=self.hedges.append,
    )
    # Each replica is an independent draw, so re-issuing at every boundary is
    # what makes the probability of a slow call decay with the budget.
    self.assertEqual(hedged(), 'answered')
    self.assertEqual(len(self.attempts), 4)
    self.assertEqual(self.hedges, [1, 2, 3])
    for attempt in self.attempts.records[:3]:
      self.assertFalse(attempt.completed)

  def test_hedge_budget_is_a_hard_ceiling(self):
    primary_release = threading.Event()
    self.addCleanup(primary_release.set)

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park(primary_release)
        return 'primary'
      self.park()
      return 'hedge-1'

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=1,
        on_hedge=self.hedges.append,
    )
    result = self.call_async(hedged)
    self.assertTrue(self.attempts.started(2, _HEDGE_TIMEOUT))
    # Both attempts stay outstanding well past the next boundary, and no
    # further replica is issued once the budget is spent.
    self.assertFalse(self.attempts.started(3, _NEGATIVE_WINDOW))
    self.assertEqual(len(self.attempts), 2)
    primary_release.set()
    self.assertEqual(result.result(timeout=_HEDGE_TIMEOUT), 'primary')
    self.assertEqual(len(self.attempts), 2)
    self.assertEqual(self.hedges, [1])

  def test_exhausted_budget_still_answers(self):
    primary_release = threading.Event()
    self.addCleanup(primary_release.set)

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park(primary_release)
        return 'primary'
      self.park()
      return 'hedge-1'

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1
    )
    result = self.call_async(hedged)
    self.assertTrue(self.attempts.started(2, _HEDGE_TIMEOUT))
    # The budget is a ceiling on replicas, not a deadline: with it spent the
    # wrapper keeps waiting on what is in flight instead of giving up.
    self.assertFalse(result.done())
    primary_release.set()
    self.assertEqual(result.result(timeout=_HEDGE_TIMEOUT), 'primary')
    self.assertEqual(len(self.attempts), 2)

  def test_late_replica_wins_over_running_attempts(self):
    answers = [object(), object(), object()]

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal <= 2:
        self.park()
        attempt.completed = True
      return answers[attempt.ordinal - 1]

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=2
    )
    # The last replica answers while the earlier attempts are still running.
    self.assertIs(hedged(), answers[2])
    self.assertEqual(len(self.attempts), 3)
    for attempt in self.attempts.records[:2]:
      self.assertFalse(attempt.completed)

  def test_failing_replica_does_not_abort_a_live_call(self):
    primary_release = threading.Event()
    self.addCleanup(primary_release.set)
    replica_failed = threading.Event()

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park(primary_release)
        return 'primary'
      replica_failed.set()
      raise ValueError('replica failed')

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=1,
        on_hedge=self.hedges.append,
    )
    result = self.call_async(hedged)
    self.assertTrue(replica_failed.wait(_HEDGE_TIMEOUT))
    primary_release.set()
    # First to succeed wins, not first to finish: a fast failure must not turn
    # a good reply into an error.
    self.assertEqual(result.result(timeout=_HEDGE_TIMEOUT), 'primary')
    self.assertEqual(len(self.attempts), 2)
    self.assertEqual(self.hedges, [1])

  def test_all_failures_raise_the_primary_error(self):
    primary_release = threading.Event()
    self.addCleanup(primary_release.set)
    replica_failed = threading.Event()
    failures = []

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park(primary_release)
        failures.append('primary')
        raise KeyError('primary failed')
      failures.append('replica')
      replica_failed.set()
      raise ValueError('replica failed')

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1
    )
    result = self.call_async(hedged)
    self.assertTrue(replica_failed.wait(_HEDGE_TIMEOUT))
    primary_release.set()
    # Failure semantics match the unwrapped call, so the error raised is the
    # primary's even though it is not the first one seen.
    with self.assertRaisesRegex(KeyError, 'primary failed'):
      result.result(timeout=_HEDGE_TIMEOUT)
    self.assertEqual(failures, ['replica', 'primary'])

  def test_non_positive_hedge_after_runs_inline(self):
    for hedge_after in [0, -1]:
      with self.subTest(hedge_after=hedge_after):
        self.attempts = _AttemptLog()
        hedged = concurrent.with_hedging(
            lambda: self.attempts.record().ordinal,
            hedge_after=hedge_after,
            max_hedges=5,
            on_hedge=self.hedges.append,
        )
        self.assertEqual(hedged(), 1)
        self.assertEqual(len(self.attempts), 1)
        self.assertIs(
            self.attempts.records[0].thread, threading.current_thread()
        )
    self.assertEqual(self.hedges, [])

  def test_non_positive_max_hedges_runs_inline(self):
    for max_hedges in [0, -1]:
      with self.subTest(max_hedges=max_hedges):
        self.attempts = _AttemptLog()
        hedged = concurrent.with_hedging(
            lambda: self.attempts.record().ordinal,
            hedge_after=_HEDGE_SOON,
            max_hedges=max_hedges,
            on_hedge=self.hedges.append,
        )
        self.assertEqual(hedged(), 1)
        self.assertEqual(len(self.attempts), 1)
        self.assertIs(
            self.attempts.records[0].thread, threading.current_thread()
        )
    self.assertEqual(self.hedges, [])

  def test_arguments_reach_every_replica(self):
    payload = ['shared']

    def func(*args, **kwargs):
      attempt = self.attempts.record(args, kwargs)
      if attempt.ordinal <= 2:
        self.park()
        return None
      return 'answered'

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=2
    )
    self.assertEqual(hedged(payload, 2, key='value'), 'answered')
    self.assertEqual(len(self.attempts), 3)
    for attempt in self.attempts.records:
      self.assertEqual(attempt.args, (payload, 2))
      self.assertEqual(attempt.kwargs, {'key': 'value'})
      # Replicas share one set of argument objects, they are not copied.
      self.assertIs(attempt.args[0], payload)

  def test_composes_with_with_retry(self):
    lock = threading.Lock()
    calls_per_thread = collections.defaultdict(int)

    def flaky():
      attempt = self.attempts.record()
      with lock:
        calls_per_thread[attempt.thread] += 1
        rung = calls_per_thread[attempt.thread]
      if rung == 1:
        raise ValueError('first rung of this replica ladder')
      if attempt.thread is self.attempts.records[0].thread:
        self.park()
        return 'primary'
      return 'answered'

    hedged = concurrent.with_hedging(
        concurrent.with_retry(
            flaky, ValueError, max_attempts=2, retry_interval=0
        ),
        hedge_after=_HEDGE_SOON,
        max_hedges=1,
    )
    # Hedging outside retry gives each replica its own retry ladder, and still
    # bounds the call while the other ladder is parked on its second rung.
    self.assertEqual(hedged(), 'answered')
    self.assertEqual(len(self.attempts), 4)
    self.assertEqual(len(calls_per_thread), 2)
    self.assertEqual(sorted(calls_per_thread.values()), [2, 2])

  def test_replicas_run_on_daemon_threads(self):
    caller = threading.current_thread()

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park()
        return 'primary'
      return 'hedge-1'

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1
    )
    self.assertEqual(hedged(), 'hedge-1')
    self.assertEqual(len(self.attempts), 2)
    for attempt in self.attempts.records:
      # A thread pool would register an atexit hook joining its workers, which
      # would hold the process open for as long as the abandoned attempt runs.
      self.assertTrue(attempt.daemon)
      self.assertIsNot(attempt.thread, caller)

  def test_contextual_override_reaches_every_attempt(self):
    observed = {}

    def func():
      attempt = self.attempts.record()
      observed[attempt.ordinal] = (A(1).y, component.context_value('y'))
      if attempt.ordinal == 1:
        self.park()
        return 'primary'
      return 'hedge-1'

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1
    )
    with component.context(y=7):
      self.assertEqual(hedged(), 'hedge-1')
    # Overrides are per-thread, so they reach the attempts only because they
    # are captured on the caller's thread when the call starts.
    self.assertEqual(observed, {1: (7, 7), 2: (7, 7)})

  def test_override_attrs_reaches_every_attempt(self):
    observed = {}

    def func():
      attempt = self.attempts.record()
      override = component.get_contextual_override('x')
      observed[attempt.ordinal] = (A(1).x, override.override_attrs)
      if attempt.ordinal == 1:
        self.park()
        return 'primary'
      return 'hedge-1'

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1
    )
    # `x` already has a value, so only `override_attrs` replaces it, and the
    # flag itself has to survive the thread hop for that to happen.
    with component.context(x=5, override_attrs=True):
      self.assertEqual(hedged(), 'hedge-1')
    self.assertEqual(observed, {1: (5, True), 2: (5, True)})

  def test_saturated_thread_ceiling_skips_the_hedge(self):
    primary_release = threading.Event()
    self.addCleanup(primary_release.set)

    def func():
      self.attempts.record()
      self.park(primary_release)
      return 'primary'

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=3,
        on_hedge=self.hedges.append,
    )
    with mock.patch.object(
        concurrent, '_hedge_thread_quota', threading.Semaphore(0)
    ):
      result = self.call_async(hedged)
      self.assertTrue(self.attempts.started(1, _HEDGE_TIMEOUT))
      # Hedging is an optimization: a saturated ceiling skips the duplicate
      # rather than making the primary wait for a permit.
      self.assertFalse(self.attempts.started(2, _NEGATIVE_WINDOW))
      self.assertEqual(self.hedges, [])
      primary_release.set()
      self.assertEqual(result.result(timeout=_HEDGE_TIMEOUT), 'primary')
    self.assertEqual(len(self.attempts), 1)

  def test_hedge_thread_start_failure_skips_the_hedge(self):
    permits = 2
    quota = threading.Semaphore(permits)
    refused = threading.Event()
    waited = []

    def func():
      self.attempts.record()
      waited.append(refused.wait(_HEDGE_TIMEOUT))
      return 'primary'

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=1,
        on_hedge=self.hedges.append,
    )
    with mock.patch.object(concurrent, '_hedge_thread_quota', quota):
      # The first thread is the primary; every later one refuses to start.
      with self.fail_thread_start_after(1, refused):
        self.assertEqual(hedged(), 'primary')
    self.assertEqual(waited, [True])
    self.assertEqual(len(self.attempts), 1)
    self.assertEqual(self.hedges, [])
    # A permit taken for a replica that never started is handed back, so the
    # ceiling does not leak away over time.
    for _ in range(permits):
      self.assertTrue(quota.acquire(blocking=False))
    self.assertFalse(quota.acquire(blocking=False))

  def test_primary_thread_start_failure_runs_inline(self):
    caller = threading.get_ident()
    ran_on = []

    def func():
      self.attempts.record()
      ran_on.append(threading.get_ident())
      return 'primary'

    hedged = concurrent.with_hedging(
        func,
        hedge_after=_HEDGE_SOON,
        max_hedges=1,
        on_hedge=self.hedges.append,
    )
    # Hedging is an optimization, so it must never make a call worse: when not
    # even the primary can get a thread, the call runs inline on the caller's
    # thread instead of failing a call that would have succeeded unwrapped.
    with self.fail_thread_start_after(0):
      self.assertEqual(hedged(), 'primary')
    self.assertEqual(ran_on, [caller])
    self.assertEqual(len(self.attempts), 1)
    self.assertEqual(self.hedges, [])

  def test_on_hedge_error_does_not_break_the_call(self):
    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park()
        return 'primary'
      return 'hedge-1'

    def on_hedge(ordinal: int) -> None:
      self.hedges.append(ordinal)
      raise RuntimeError('on_hedge failed')

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1, on_hedge=on_hedge
    )
    # `on_hedge` is observability-only, so a callback that raises must neither
    # fail a call that would have succeeded unwrapped nor withdraw the replica
    # it was notified about. The failure is logged instead of surfaced.
    with self.assertLogs(pg.logging.get_logger(), level='WARNING') as logs:
      self.assertEqual(hedged(), 'hedge-1')
    self.assertEqual(len(self.attempts), 2)
    self.assertEqual(self.hedges, [1])
    self.assertIn('on_hedge failed', '\n'.join(logs.output))

  def test_usage_tracking_and_context_reach_the_winning_replica(self):
    caller = threading.get_ident()
    lm = _FakeLM()
    observed = {}
    # Held until the assertions are done, so the replica is the only attempt
    # that can win and the primary's value can never be mistaken for it.
    primary_release = threading.Event()
    self.addCleanup(primary_release.set)

    def func():
      attempt = self.attempts.record()
      if attempt.ordinal == 1:
        self.park(primary_release)
        return 'primary'
      observed['override'] = (A(1).y, component.context_value('y'))
      observed['thread'] = threading.get_ident()
      return lm('hedge-1').text

    hedged = concurrent.with_hedging(
        func, hedge_after=_HEDGE_SOON, max_hedges=1
    )
    # `track_usages` installs its tracker as a thread-local override, exactly
    # like `y`, so usage is accounted only because the caller's overrides are
    # captured when the call starts and replayed on the replica's thread.
    with lm_lib.track_usages() as usages:
      with component.context(y=7):
        self.assertEqual(hedged(), 'hedge-1')

    # One request, from the replica: the parked primary never reached the LM.
    self.assertEqual(
        usages.uncached.breakdown,
        {'fake-lm': lm_lib.LMSamplingUsage(10, 20, 30, 0, 1)},
    )
    self.assertEqual(observed['override'], (7, 7))
    self.assertNotEqual(observed['thread'], caller)
    self.assertEqual(len(self.attempts), 2)

    # Drain the primary here rather than at teardown, so that the assertion
    # below proves the abandoned attempt actually unwinds.
    primary_release.set()
    primary = self.attempts.records[0].thread
    primary.join(timeout=_HEDGE_TIMEOUT)
    self.assertFalse(primary.is_alive())


class ConcurrentExecuteTest(unittest.TestCase):

  def test_concurrent_execute(self):
    def fun(a):
      return a.x * a.y

    with component.context(y=2):
      self.assertEqual(concurrent.concurrent_execute(fun, [A(1), A(2)]), [2, 4])

  def test_concurrent_execute_with_a_single_worker(self):
    def fun(a):
      return a.x * a.y

    with component.context(y=2):
      self.assertEqual(
          concurrent.concurrent_execute(fun, [A(1), A(2)], max_workers=1),
          [2, 4],
      )

  def test_concurrent_execute_with_external_executor(self):
    def fun(a):
      return a.x * a.y

    executor = futures.ThreadPoolExecutor(max_workers=2)
    with component.context(y=2):
      self.assertEqual(
          concurrent.concurrent_execute(fun, [A(1), A(2)], executor=executor),
          [2, 4])

    # Making sure the executor could be reused.
    with component.context(y=2):
      self.assertEqual(
          concurrent.concurrent_execute(fun, [A(2), A(4)]), [4, 8])


class ProgressTest(unittest.TestCase):

  def test_progress(self):
    p = concurrent.Progress(total=10)
    self.assertEqual(p.total, 10)
    self.assertEqual(p.succeeded, 0)
    self.assertEqual(p.failed, 0)
    self.assertEqual(p.completed, 0)
    self.assertEqual(p.success_rate, 0)
    self.assertEqual(p.failure_rate, 0)
    self.assertEqual(p.avg_duration, 0)

    def fun(x):
      time.sleep(x)
      return x

    def fun2(unused_x):
      raise ValueError('Intentional error.')

    job1 = concurrent.Job(fun, (1,))
    job2 = concurrent.Job(fun2, (2,))
    job1()
    job2()

    p.update(job1)
    self.assertEqual(p.succeeded, 1)
    self.assertEqual(p.failed, 0)
    self.assertEqual(p.completed, 1)
    self.assertEqual(p.success_rate, 1)
    self.assertEqual(p.failure_rate, 0)
    self.assertGreater(p.avg_duration, 0.5)
    self.assertIs(p.job, job1)
    self.assertIsNone(p.last_error)

    p.update(job2)
    self.assertEqual(p.succeeded, 1)
    self.assertEqual(p.failed, 1)
    self.assertEqual(p.completed, 2)
    self.assertEqual(p.success_rate, 0.5)
    self.assertEqual(p.failure_rate, 0.5)
    self.assertIs(p.job, job2)
    self.assertIs(p.last_error, job2.error)


class ProgressControlTest(unittest.TestCase):

  def test_noop(self):
    concurrent.progress_bar = None
    ctrl = concurrent._progress_control(100, 'noop', 'blue', None)
    self.assertIsInstance(ctrl, concurrent._NoopProgressControl)
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      ctrl.update(1)
      ctrl.refresh()
      sys.stderr.flush()
    self.assertEqual(string_io.getvalue(), '')
    concurrent.progress_bar = 'tqdm'

  def test_console(self):
    concurrent.progress_bar = 'console'
    ctrl = concurrent._progress_control(100, 'foo', 'blue', None)
    self.assertIsInstance(ctrl, concurrent._ConsoleProgressControl)
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      ctrl.set_status('bar')
      ctrl.update(10)
      ctrl.refresh()
      sys.stderr.flush()
    self.assertEqual(
        string_io.getvalue(),
        '\x1b[1m\x1b[31mfoo\x1b[0m: \x1b[34m10% (10/100)\x1b[0m : bar\n'
    )
    concurrent.progress_bar = 'tqdm'

  def test_tqdm(self):
    concurrent.progress_bar = 'tqdm'
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      ctrl = concurrent._progress_control(100, 'foo', 'blue', None)
      self.assertIsInstance(ctrl, concurrent._TqdmProgressControl)
      ctrl.update(10)
      ctrl.refresh()
      sys.stderr.flush()
    self.assertIn('10/100', string_io.getvalue())

    tqdm = concurrent.tqdm
    concurrent.tqdm = None
    with self.assertRaisesRegex(RuntimeError, 'install package "tqdm"'):
      _ = concurrent._progress_control(100, 'foo', 'blue', None)
    concurrent.tqdm = tqdm

  def test_unsupported(self):
    concurrent.progress_bar = 'unknown'
    with self.assertRaisesRegex(ValueError, 'Unsupported progress bar type'):
      _ = concurrent._progress_control(100, 'foo', 'blue', None)
    concurrent.progress_bar = 'tqdm'


class ProgressBarTest(unittest.TestCase):

  def test_multithread_support(self):
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      bar_id = concurrent.ProgressBar.install(None, 5)
      def fun(x):
        del x
        concurrent.ProgressBar.update(bar_id, 1, status=None)

      for _ in concurrent.concurrent_execute(fun, range(5)):
        concurrent.ProgressBar.refresh()
      concurrent.ProgressBar.uninstall(bar_id)
      sys.stderr.flush()
    output_str = string_io.getvalue()
    self.assertIn('100%', output_str)
    self.assertIn('5/5', output_str)

  def test_report(self):
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      bar_id = concurrent.ProgressBar.install(None, 4)
      concurrent.ProgressBar.update(bar_id, 1, status=None)
      concurrent.ProgressBar.update(bar_id, 1, status='hello')
      concurrent.ProgressBar.update(bar_id, color='green')
      concurrent.ProgressBar.update(bar_id, 2, status=dict(x=1))
      with self.assertRaisesRegex(ValueError, 'Unsupported status'):
        concurrent.ProgressBar.update(bar_id, 0, status=1)
      concurrent.ProgressBar.uninstall(bar_id)
      sys.stderr.flush()
    time.sleep(1)
    self.assertIn('1/4', string_io.getvalue())
    # TODO(daiyip): Re-enable once flakiness is fixed.
    # self.assertIn('2/4', string_io.getvalue())
    # self.assertIn('hello', string_io.getvalue())
    # self.assertNotIn('3/4', string_io.getvalue())
    # self.assertIn('4/4', string_io.getvalue())
    # self.assertIn('x=1', string_io.getvalue())


class ConcurrentMapTest(unittest.TestCase):
  def test_concurrent_map_raise_on_error(self):
    error = ValueError()

    def fun(x):
      time.sleep(x)
      if x == 2:
        raise error
      return x**2

    with component.context(y=2):
      it = concurrent.concurrent_map(fun, [1, 2, 3], silence_on_errors=KeyError)
      self.assertEqual(next(it), (1, 1, None))
      with self.assertRaises(ValueError):
        _ = next(it)

    # Once error has raisen. The next call to `next` will raise StopIteration.
    with self.assertRaises(StopIteration):
      next(it)

  def test_concurrent_map_retry_on_error(self):
    error = ValueError()
    call_time = collections.defaultdict(int)

    def fun(x):
      call_time[x] += 1
      if call_time[x] >= x:
        return x ** 2
      raise error

    with component.context(y=2):
      self.assertEqual(
          set(
              concurrent.concurrent_map(
                  fun,
                  [1, 2, 3],
                  retry_on_errors=ValueError,
                  retry_interval=1,
                  silence_on_errors=concurrent.RetryError,
                  max_attempts=2,
              )
          ),
          set([
              (1, 1, None),
              (2, 4, None),
              (
                  3,
                  pg.MISSING_VALUE,
                  concurrent.RetryError(fun, [error, error], [1]),
              ),
          ]),
      )

  def test_concurrent_map_silence_on_errors(self):
    error = ValueError()

    def fun(x):
      if x == 2:
        raise error
      return x**2

    with component.context(y=2):
      self.assertEqual(
          set(
              concurrent.concurrent_map(
                  fun, [1, 2, 3], silence_on_errors=ValueError
              )
          ),
          set([
              (1, 1, None),
              (2, pg.MISSING_VALUE, error),
              (3, 9, None),
          ]),
      )

  def test_concurrent_map_with_async_complete(self):
    def fun(x):
      time.sleep(3 - x)
      return x

    with component.context(y=2):
      self.assertEqual(
          list(concurrent.concurrent_map(fun, [1, 2, 3])),
          [
              (3, 3, None),
              (2, 2, None),
              (1, 1, None),
          ],
      )

  def test_concurrent_map_with_ordering(self):
    def fun(x):
      time.sleep(3 - x)
      return x

    with component.context(y=2):
      self.assertEqual(
          list(concurrent.concurrent_map(fun, [1, 2, 3], ordered=True)),
          [
              (1, 1, None),
              (2, 2, None),
              (3, 3, None),
          ],
      )

  def test_concurrent_map_with_external_executor(self):
    def fun(x):
      return x

    executor = futures.ThreadPoolExecutor(max_workers=2)
    self.assertEqual(
        list(concurrent.concurrent_map(
            fun, [1, 2, 3], executor=executor, ordered=True)),
        [
            (1, 1, None),
            (2, 2, None),
            (3, 3, None),
        ],
    )
    self.assertEqual(
        list(concurrent.concurrent_map(
            fun, [4, 5, 6], executor=executor, ordered=True)),
        [
            (4, 4, None),
            (5, 5, None),
            (6, 6, None),
        ],
    )

  def test_concurrent_map_with_order_and_raise_on_errors(self):
    error = ValueError()

    def fun(x):
      if x == 2:
        raise error
      return x**2

    with component.context(y=2):
      it = concurrent.concurrent_map(
          fun, [1, 2, 3], ordered=True, silence_on_errors=KeyError)
      self.assertEqual(next(it)[1], 1)

      with self.assertRaises(ValueError):
        _ = next(it)

  def test_concurrent_map_with_order_and_timeout(self):
    def fun(x):
      time.sleep(3 - x)
      return x

    self.assertEqual(
        [
            (i, o)
            for i, o, _ in concurrent.concurrent_map(
                fun, [-1, 2, 3], ordered=True, timeout=1.5
            )
        ],
        [
            (-1, pg.MISSING_VALUE),
            (2, 2),
            (3, 3),
        ],
    )

  def test_concurent_map_unordered_with_timeout(self):
    def fun(x):
      time.sleep(x)
      return x

    self.assertEqual(
        [
            (i, o)
            for i, o, _ in concurrent.concurrent_map(
                fun, [5, 2, 1, 4], timeout=3
            )
        ],
        [
            (1, 1),
            (2, 2),
            (5, pg.MISSING_VALUE),
            (4, pg.MISSING_VALUE),
        ],
    )
    with self.assertRaises(TimeoutError):
      next(concurrent.concurrent_map(
          fun, [5, 3], timeout=1, silence_on_errors=None))

  def test_concurent_map_unordered_with_timeout_less_worker(self):
    def fun(x):
      time.sleep(x)
      return x

    self.assertEqual(
        [
            (i, o)
            for i, o, _ in concurrent.concurrent_map(
                fun, [5, 2, 1, 6], timeout=3, max_workers=1
            )
        ],
        [
            (5, pg.MISSING_VALUE),
            (2, 2),
            (1, 1),
            (6, pg.MISSING_VALUE),
        ],
    )

  def test_concurrent_map_with_showing_progress(self):
    def fun(x):
      with pg.timeit('foo'):
        if x == 2:
          raise ValueError('Intentional error.')
        time.sleep(x)
        return x

    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      output = sorted([
          (i, o) for i, o, _ in concurrent.concurrent_map(
              fun, [1, 2, 3], timeout=1.5, max_workers=1, show_progress=True
          )
      ], key=lambda x: x[0])
      sys.stderr.flush()

    self.assertEqual(   # pylint: disable=g-generic-assert
        output,
        [
            (1, 1),
            (2, pg.MISSING_VALUE),
            (3, pg.MISSING_VALUE),
        ],
    )
    output = string_io.getvalue()
    self.assertIn('100%', output)

  def test_concurrent_map_with_showing_progress_and_status_fn(self):
    def fun(x):
      if x == 2:
        raise ValueError('Intentional error.')
      time.sleep(x)
      return x

    bar_id = concurrent.ProgressBar.install(None, 3)
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      output = sorted([
          (i, o) for i, o, _ in concurrent.concurrent_map(
              fun, [1, 2, 3], timeout=1.5, max_workers=1,
              show_progress=bar_id, status_fn=lambda p: dict(x=1, y=1)
          )
      ], key=lambda x: x[0])
      sys.stderr.flush()

    self.assertEqual(  # pylint: disable=g-generic-assert
        output,
        [
            (1, 1),
            (2, pg.MISSING_VALUE),
            (3, pg.MISSING_VALUE),
        ],
    )
    concurrent.ProgressBar.uninstall(bar_id)
    concurrent.ProgressBar.refresh()
    self.assertIn('100%', string_io.getvalue())

  def test_concurrent_map_max_duration_ordered(self):
    """max_duration bounds total wall-clock time for ordered concurrent_map."""

    def slow(x):
      time.sleep(x)
      return x

    t0 = time.time()
    results = list(
        concurrent.concurrent_map(
            slow,
            [1, 1, 10, 10],
            ordered=True,
            max_duration=3.0,
            max_workers=2,
        )
    )
    elapsed = time.time() - t0
    # Should complete in ~3s, not 10+.
    self.assertLess(elapsed, 6.0)
    # First two (1s each) should succeed, last two canceled.
    succeeded = [r for _, r, e in results if e is None]
    canceled = [e for _, _, e in results if isinstance(e, TimeoutError)]
    self.assertEqual(len(succeeded), 2)
    self.assertEqual(len(canceled), 2)

  def test_concurrent_map_max_duration_unordered(self):
    """max_duration bounds total wall-clock time for unordered concurrent_map."""

    def slow(x):
      time.sleep(x)
      return x

    t0 = time.time()
    results = list(
        concurrent.concurrent_map(
            slow,
            [1, 10, 1, 10],
            max_duration=3.0,
            max_workers=4,
        )
    )
    elapsed = time.time() - t0
    self.assertLess(elapsed, 6.0)
    succeeded = [r for _, r, e in results if e is None]
    canceled = [e for _, _, e in results if isinstance(e, TimeoutError)]
    self.assertGreaterEqual(len(succeeded), 2)
    self.assertGreaterEqual(len(canceled), 2)

  def test_concurrent_map_max_duration_none_default(self):
    """max_duration=None preserves backward-compatible behavior."""

    def fast(x):
      return x * 2

    results = list(
        concurrent.concurrent_map(
            fast,
            [1, 2, 3],
            max_duration=None,
        )
    )
    self.assertEqual(
        set(results),
        {(1, 2, None), (2, 4, None), (3, 6, None)},
    )

  def test_concurrent_map_max_duration_raises_when_not_silenced(self):
    """max_duration raises TimeoutError when silence_on_errors=None."""

    def slow(x):
      time.sleep(x)
      return x

    with self.assertRaises(TimeoutError):
      list(
          concurrent.concurrent_map(
              slow,
              [10],
              max_duration=1.0,
              silence_on_errors=None,
              max_workers=1,
          )
      )

  def test_concurrent_map_max_duration_unordered_race_requeues_completed(self):
    """Race: future done between as_completed() and done() check is re-queued.

    Regression test for daiyip's review (concurrent.py:935-938).
    Without the fix, mark_canceled would overwrite a valid result with
    TimeoutError — the job would have BOTH result and error set, corrupting
    the output. The fix gates mark_canceled behind `not future.done()` and
    re-queues completed futures for normal processing.

    This test mocks as_completed to deterministically force the race:
    on the first poll, as_completed yields nothing (so the fast future
    is NOT in completed_batch), but future.done() returns True because
    the task already completed. The deadline check then hits the else
    branch, re-queueing the future. On the next poll, as_completed
    correctly yields it with its valid result.
    """
    gate = threading.Event()

    def task(x):
      if x == 'fast':
        return 'result_ok'
      gate.wait(timeout=30)
      return 'never'

    call_count = [0]
    original_as_completed = futures.as_completed

    def rigged_as_completed(fs, timeout=None):
      call_count[0] += 1
      if call_count[0] == 1:
        # First poll: yield nothing. The 'fast' future IS done but won't
        # be in completed_batch, forcing the remaining_futures loop to
        # encounter a done future during deadline processing.
        time.sleep(0.05)
        return iter([])
      return original_as_completed(fs, timeout=timeout)

    try:
      with mock.patch.object(
          futures, 'as_completed', rigged_as_completed
      ):
        results = list(
            concurrent.concurrent_map(
                task,
                ['fast', 'slow'],
                max_duration=0.001,
                max_workers=2,
            )
        )
    finally:
      gate.set()

    results_dict = {inp: (r, e) for inp, r, e in results}

    # 'fast' completed — result MUST be preserved, not overwritten.
    r, e = results_dict['fast']
    self.assertIsNone(
        e,
        f'Race condition bug: completed result overwritten with {e}',
    )
    self.assertEqual(r, 'result_ok')

    # 'slow' canceled with TimeoutError.
    _, e = results_dict['slow']
    self.assertIsInstance(e, TimeoutError)

  def test_concurrent_map_max_duration_no_result_corruption_invariant(self):
    """Adversarial stress test: no result is ever corrupted at boundaries.

    Runs tasks that complete at various times near the deadline.
    The invariant: if error is None, the result must be the correct
    computed value. If error is set, it must be TimeoutError.
    """

    def boundary_task(x):
      time.sleep(x * 0.05)
      return x * 100

    for trial in range(3):
      results = list(
          concurrent.concurrent_map(
              boundary_task,
              list(range(1, 11)),
              max_duration=0.25,
              max_workers=10,
          )
      )
      for inp, result, error in results:
        if error is None:
          self.assertEqual(
              result,
              inp * 100,
              f'Trial {trial}: input={inp} expected {inp * 100}, '
              f'got {result}',
          )
        else:
          self.assertIsInstance(error, TimeoutError)

  def test_concurrent_map_max_duration_near_zero_cancels_all(self):
    """Near-zero max_duration cancels all items immediately."""

    def task(x):
      time.sleep(1)
      return x

    results = list(
        concurrent.concurrent_map(
            task,
            [1, 2, 3],
            max_duration=0.001,
            max_workers=3,
        )
    )
    errors = [e for _, _, e in results if e is not None]
    self.assertGreater(len(errors), 0)
    for e in errors:
      self.assertIsInstance(e, TimeoutError)
      self.assertIn('max_duration', str(e))

  def test_concurrent_map_max_duration_with_per_item_timeout(self):
    """max_duration and timeout interact correctly.

    Per-item timeout fires for individual slow items; max_duration acts
    as a wall-clock ceiling independent of per-item timing.
    """

    def task(x):
      time.sleep(x)
      return x

    results = list(
        concurrent.concurrent_map(
            task,
            [1, 5, 1, 5],
            timeout=3,
            max_duration=10.0,
            max_workers=4,
        )
    )
    succeeded = [(i, r) for i, r, e in results if e is None]
    timed_out = [(i, e) for i, _, e in results if e is not None]
    self.assertGreater(len(succeeded), 0, 'Fast items should succeed')
    self.assertGreater(len(timed_out), 0, 'Slow items should timeout')
    for _, e in timed_out:
      self.assertIsInstance(e, TimeoutError)

  def test_concurrent_map_max_duration_all_complete_before_deadline(self):
    """When all items finish before deadline, no errors occur."""

    def fast(x):
      return x ** 2

    results = list(
        concurrent.concurrent_map(
            fast,
            [1, 2, 3, 4, 5],
            max_duration=60.0,
            max_workers=5,
        )
    )
    self.assertEqual(len(results), 5)
    for _, _, error in results:
      self.assertIsNone(error)

  def test_concurrent_map_max_duration_ordered_error_message_format(self):
    """Ordered path: error message includes the max_duration value."""

    def slow(x):
      time.sleep(10)
      return x

    results = list(
        concurrent.concurrent_map(
            slow,
            [1],
            max_duration=1.0,
            ordered=True,
            max_workers=1,
        )
    )
    _, _, error = results[0]
    self.assertIsInstance(error, TimeoutError)
    self.assertIn('max_duration=1.0', str(error))


class ExecutorPoolTest(unittest.TestCase):

  def test_pool(self):
    pool = concurrent.ExecutorPool()
    executor1 = futures.ThreadPoolExecutor()
    self.assertIs(pool.executor_from(executor1), executor1)

    executor2 = pool.executor_from('executor2', max_workers=1)
    self.assertIsInstance(executor2, futures.ThreadPoolExecutor)
    self.assertIs(pool.get('executor2'), executor2)
    self.assertEqual(pool.resource_ids, ['executor2'])

    executor3 = pool.executor_from(None, max_workers=1)
    self.assertIsInstance(executor3, futures.ThreadPoolExecutor)
    self.assertEqual(pool.resource_ids, ['executor2'])

    with self.assertRaises(ValueError):
      pool.executor_from(1)


def _settle(baseline: int, max_delta: int = 2, budget_s: float = 2.0) -> int:
  deadline = time.monotonic() + budget_s
  while time.monotonic() < deadline:
    n = threading.active_count()
    if n <= baseline + max_delta:
      return n
    time.sleep(0.05)
  return threading.active_count()


class ShutdownWaitOnTimeoutTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    gc.collect()
    self._baseline = threading.active_count()

  def tearDown(self):
    super().tearDown()
    gc.collect()

  def test_baseline_thread_cleanup_with_wait_true(self):
    def slow(_):
      time.sleep(5.0)
      return None

    t0 = time.monotonic()
    for _ in range(5):
      try:
        list(
            concurrent.concurrent_map(
                slow,
                [1],
                max_workers=1,
                timeout=0.5,
                silence_on_errors=TimeoutError,
                wait_timeout_threads_on_shutdown=True,
            )
        )
      except TimeoutError:
        pass
      n = _settle(self._baseline)
      self.assertLessEqual(
          n, self._baseline + 1, f'leak after iter: {n} vs {self._baseline}'
      )
    self.assertLess(time.monotonic() - t0, 5 * (0.5 + 5.0 + 1.0))

  def test_backward_compat_default_false_still_leaks(self):
    def slow(_):
      time.sleep(5.0)
      return None

    t0 = time.monotonic()
    for _ in range(3):
      try:
        list(
            concurrent.concurrent_map(
                slow,
                [1],
                max_workers=1,
                timeout=0.5,
                silence_on_errors=TimeoutError,
            )
        )
      except TimeoutError:
        pass
    elapsed = time.monotonic() - t0
    self.assertGreater(
        threading.active_count(),
        self._baseline,
        'default=False should preserve leak signature',
    )
    self.assertLess(elapsed, 3 * (0.5 + 1.0))

  def test_worker_ignoring_cancellation_bounded_shutdown(self):
    stop = threading.Event()

    def cooperative(_):
      while not stop.is_set():
        time.sleep(0.05)

    ex = futures.ThreadPoolExecutor(max_workers=1)
    ex.submit(cooperative, 0)
    time.sleep(0.1)
    shutdown_done = threading.Event()

    def do_shutdown():
      ex.shutdown(wait=True)
      shutdown_done.set()

    t = threading.Thread(target=do_shutdown)
    t.start()
    t.join(timeout=2.0)
    self.assertTrue(t.is_alive())
    self.assertFalse(shutdown_done.is_set())
    stop.set()
    t.join(timeout=2.0)
    self.assertFalse(t.is_alive())
    self.assertTrue(shutdown_done.is_set())
    _settle(self._baseline)

  def test_worker_raises_during_shutdown(self):
    class BoomError(RuntimeError):
      pass

    def raiser(_):
      raise BoomError('teardown')

    with self.assertRaises(BoomError):
      list(
          concurrent.concurrent_map(
              raiser,
              [1],
              max_workers=1,
              silence_on_errors=None,
              wait_timeout_threads_on_shutdown=True,
          )
      )
    n = _settle(self._baseline)
    self.assertLessEqual(n, self._baseline + 1)
    out = list(
        concurrent.concurrent_map(
            lambda x: x * 2,
            [1, 2, 3],
            max_workers=2,
            wait_timeout_threads_on_shutdown=True,
        )
    )
    self.assertEqual(len(out), 3)

  def test_concurrent_shutdown_calls(self):
    ex = futures.ThreadPoolExecutor(max_workers=2)
    for _ in range(2):
      ex.submit(lambda: time.sleep(0.05))
    t0 = time.monotonic()

    def call_shutdown():
      ex.shutdown(wait=True)

    t1 = threading.Thread(target=call_shutdown)
    t2 = threading.Thread(target=call_shutdown)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    self.assertFalse(t1.is_alive() or t2.is_alive(), 'deadlock')
    self.assertLess(time.monotonic() - t0, 5.0)
    _settle(self._baseline)

  def test_nested_executor_pools(self):
    def inner(_):
      try:
        list(
            concurrent.concurrent_map(
                lambda x: time.sleep(5.0),
                [1],
                max_workers=1,
                timeout=0.3,
                silence_on_errors=TimeoutError,
                wait_timeout_threads_on_shutdown=True,
            )
        )
      except TimeoutError:
        pass
      return 'inner-done'

    t0 = time.monotonic()
    out = list(
        concurrent.concurrent_map(
            inner, [1, 2], max_workers=2, wait_timeout_threads_on_shutdown=True
        )
    )
    self.assertEqual(len(out), 2)
    self.assertLess(time.monotonic() - t0, 30.0)
    n = _settle(self._baseline)
    self.assertLessEqual(n, self._baseline + 2)

  def test_high_fanout_stress_50_workers(self):
    def mixed(i):
      time.sleep(5.0 if i % 2 == 0 else 0.1)
      return i

    t0 = time.monotonic()
    try:
      list(
          concurrent.concurrent_map(
              mixed,
              list(range(50)),
              max_workers=50,
              timeout=0.5,
              silence_on_errors=TimeoutError,
              wait_timeout_threads_on_shutdown=True,
          )
      )
    except TimeoutError:
      pass
    self.assertLess(time.monotonic() - t0, 10.0)
    n = _settle(self._baseline, max_delta=2, budget_s=6.0)
    self.assertLessEqual(
        n, self._baseline + 1, f'leak after stress: {n} vs {self._baseline}'
    )

  # (h)
  def test_num_attempts_retry_interaction(self):
    """Simulate Gemini retry loop: 10 attempts each timing out, bounded threads."""

    def slow(_):
      time.sleep(5.0)
      return None

    max_workers = 2
    for _ in range(10):
      try:
        list(
            concurrent.concurrent_map(
                slow,
                [1],
                max_workers=max_workers,
                timeout=0.2,
                silence_on_errors=TimeoutError,
                wait_timeout_threads_on_shutdown=True,
            )
        )
      except TimeoutError:
        pass
      n = threading.active_count()
      self.assertLessEqual(
          n,
          self._baseline + max_workers,
          f'unbounded growth: {n} vs baseline {self._baseline}',
      )
    final = _settle(self._baseline)
    self.assertLessEqual(final, self._baseline + 1)

  def test_resource_fault_injection_on_shutdown(self):
    ex = futures.ThreadPoolExecutor(max_workers=1)
    real_shutdown = ex.shutdown
    calls = {'n': 0}

    def flaky(*a, **kw):
      calls['n'] += 1
      if calls['n'] == 1:
        raise OSError('injected')
      return real_shutdown(*a, **kw)

    ex.submit(lambda: time.sleep(0.05))
    with mock.patch.object(ex, 'shutdown', side_effect=flaky):
      with self.assertRaises(OSError):
        ex.shutdown(wait=True)
    real_shutdown(wait=True)
    n = _settle(self._baseline)
    self.assertLessEqual(n, self._baseline + 1)

  def test_executor_gc_after_shutdown(self):
    ex = futures.ThreadPoolExecutor(max_workers=1)
    list(ex.map(lambda x: x, [1, 2, 3]))
    ex.shutdown(wait=True)
    ref = weakref.ref(ex)
    del ex
    gc.collect()
    self.assertIsNone(ref(), 'executor not collected — strong ref leaked')


if __name__ == '__main__':
  unittest.main()
