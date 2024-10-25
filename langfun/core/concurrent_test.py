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
import io
import time
import unittest
from langfun.core import component
from langfun.core import concurrent
import pyglove as pg


class A(component.Component):
  x: int = 1
  y: int = component.contextual()


class WithContextAccessTest(unittest.TestCase):

  def test_context_access(self):
    inputs = [A(1), A(2)]
    with futures.ThreadPoolExecutor() as executor:
      with component.context(y=3):
        self.assertEqual(
            list(
                executor.map(
                    concurrent.with_context_access(lambda x: x.y), inputs
                )
            ),
            [3, 3],
        )


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


class WithRetryTest(unittest.TestCase):

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

    job1 = concurrent.Job(fun, 1)
    job2 = concurrent.Job(fun2, 2)
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
    self.assertIn('1/4', string_io.getvalue())
    self.assertIn('2/4', string_io.getvalue())
    self.assertIn('hello', string_io.getvalue())
    self.assertNotIn('3/4', string_io.getvalue())
    self.assertIn('4/4', string_io.getvalue())
    self.assertIn('x=1', string_io.getvalue())


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
    self.assertIn('TimeIt=foo (', output)

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

    self.assertEqual(  # pylint: disable=g-generic-assert
        output,
        [
            (1, 1),
            (2, pg.MISSING_VALUE),
            (3, pg.MISSING_VALUE),
        ],
    )
    concurrent.ProgressBar.uninstall(bar_id)
    self.assertIn('100%', string_io.getvalue())


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


if __name__ == '__main__':
  unittest.main()
