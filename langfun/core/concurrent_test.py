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
        foo, ValueError, max_attempts=4, retry_interval=1
    )
    self.assert_retry(foo_with_retry, 4, [1, 2, 4])

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


class ConcurrentMapTest(unittest.TestCase):
  def test_concurrent_map_raise_on_error(self):
    error = ValueError()

    def fun(x):
      time.sleep(x)
      if x == 2:
        raise error
      return x**2

    with component.context(y=2):
      it = concurrent.concurrent_map(fun, [1, 2, 3])
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

  def test_concurrent_map_with_order_and_raise_on_errors(self):
    error = ValueError()

    def fun(x):
      if x == 2:
        raise error
      return x**2

    with component.context(y=2):
      it = concurrent.concurrent_map(fun, [1, 2, 3], ordered=True)
      self.assertEqual(next(it)[1], 1)

      with self.assertRaises(ValueError):
        _ = next(it)

  def test_concurrent_map_with_timeout(self):
    def fun(x):
      time.sleep(3 - x)
      return x

    with component.context(y=0.9):
      self.assertEqual(
          [
              (i, o)
              for i, o, _ in concurrent.concurrent_map(
                  fun, [1, 2, 3], ordered=True, timeout=2
              )
          ],
          [
              (1, pg.MISSING_VALUE),
              (2, pg.MISSING_VALUE),
              (3, 3),
          ],
      )


if __name__ == '__main__':
  unittest.main()
