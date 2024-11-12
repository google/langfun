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
import math
import unittest

from langfun.core.eval.v2 import metric_values
import pyglove as pg


class RateTest(unittest.TestCase):

  def test_basic(self):
    rate = metric_values.Rate()
    self.assertEqual(rate.total, 0)
    self.assertTrue(math.isnan(float(rate)))
    self.assertEqual(pg.format(rate), 'n/a')
    rate.increment_total()
    self.assertEqual(rate.total, 1)
    self.assertEqual(float(rate), 0.0)
    rate.add(1, 1.0, 1.0)
    self.assertEqual(float(rate), 1.0)
    self.assertEqual(pg.format(rate, verbose=False), '100.0%')
    self.assertEqual(pg.format(rate, verbose=True), '100.0% (1/1)')
    self.assertEqual(
        rate.data_points, [metric_values.MetricValue.DataPoint(1, 1.0, 1.0)]
    )
    self.assertEqual(rate, 1.0)
    self.assertGreater(rate, 0.5)
    self.assertLess(rate, 1.5)
    self.assertEqual(
        rate,
        metric_values.Rate(
            [metric_values.MetricValue.DataPoint(1, 1.0, 1.0)], 1
        )
    )
    self.assertGreater(rate, metric_values.Rate([], 1))
    self.assertLess(metric_values.Rate([], 1), rate)

    rate.reset()
    self.assertEqual(rate.total, 0)
    self.assertTrue(math.isnan(float(rate)))


class AverageTest(unittest.TestCase):

  def test_basic(self):
    average = metric_values.Average()
    self.assertEqual(average.total, 0)
    self.assertTrue(math.isnan(float(average)))
    self.assertEqual(pg.format(average, verbose=False), 'n/a')
    average.add(1, 1.0, 0.5, increment_total=True)
    average.add(1, 0.0, 1.0, increment_total=True)
    self.assertEqual(average.total, 2)
    self.assertEqual(float(average), 0.25)
    self.assertEqual(pg.format(average, verbose=False), '0.250')
    self.assertEqual(pg.format(average, verbose=True), '0.250 (2/2)')
    self.assertEqual(
        average.data_points,
        [
            metric_values.MetricValue.DataPoint(1, 1.0, 0.5),
            metric_values.MetricValue.DataPoint(1, 0.0, 1.0),
        ]
    )
    average.reset()
    self.assertEqual(average.total, 0)


if __name__ == '__main__':
  unittest.main()
