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

import time
import unittest
from langfun.core import component
from langfun.core import concurrent
import pyglove as pg


class A(component.Component):
  x: int = 1
  y: int = component.contextual()


class ConcurrentTest(unittest.TestCase):

  def test_concurrent_execute(self):
    def fun(a):
      return a.x * a.y

    with component.context(y=2):
      self.assertEqual(concurrent.concurrent_execute(fun, [A(1), A(2)]), [2, 4])

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
