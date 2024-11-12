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
import time
import unittest

from langfun.core.eval.v2 import progress as progress_lib
import pyglove as pg

Progress = progress_lib.Progress


class ProgressTest(unittest.TestCase):

  def test_basic(self):
    p = Progress()
    self.assertFalse(p.is_started)
    self.assertFalse(p.is_stopped)
    self.assertFalse(p.is_completed)
    self.assertFalse(p.is_skipped)
    self.assertFalse(p.is_failed)
    self.assertEqual(p.num_completed, 0)
    self.assertIsNone(p.elapse)
    self.assertIsNone(p.start_time_str)
    self.assertIsNone(p.stop_time_str)
    self.assertTrue(math.isnan(float(p)))

    p.start(10)
    self.assertEqual(p.num_total, 10)
    self.assertTrue(p.is_started)
    self.assertFalse(p.is_stopped)
    self.assertIsNotNone(p.start_time_str)
    self.assertIsNotNone(p.elapse)
    self.assertEqual(float(p), 0.0)

    with pg.views.html.controls.HtmlControl.track_scripts() as scripts:
      p.increment_processed()
      p.increment_failed()
      p.increment_skipped()
    # Does not triggger scripts as progress is not rendered yet.
    self.assertEqual(len(scripts), 0)
    self.assertEqual(p.num_completed, 3)
    self.assertIn(
        '3/10',
        p.to_html(extra_flags=dict(interactive=True)).content,
    )
    with pg.views.html.controls.HtmlControl.track_scripts() as scripts:
      p.increment_processed()
      p.increment_failed()
      p.increment_skipped()
    self.assertEqual(len(scripts), 24)
    self.assertEqual(p.num_completed, 6)
    self.assertIn(
        '6/10',
        p.to_html(extra_flags=dict(interactive=False)).content,
    )
    p.update_execution_summary(
        dict(
            evaluate=pg.object_utils.TimeIt.Status(name='evaluate', elapse=1.0)
        )
    )
    p.stop()
    elapse1 = p.elapse
    time.sleep(0.1)
    self.assertEqual(p.elapse, elapse1)
    self.assertTrue(p.is_stopped)
    self.assertIsNotNone(p.stop_time_str)


if __name__ == '__main__':
  unittest.main()
