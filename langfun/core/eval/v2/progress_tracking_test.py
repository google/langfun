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
import contextlib
import io
import os
import tempfile
import unittest

from langfun.core import console as lf_console
from langfun.core.eval.v2 import eval_test_helper
from langfun.core.eval.v2 import progress_tracking    # pylint: disable=unused-import
from langfun.core.eval.v2 import runners as runners_lib  # pylint: disable=unused-import
import pyglove as pg


class HtmlProgressTrackerTest(unittest.TestCase):

  def test_track_progress(self):
    result = pg.Dict()
    def display(x):
      result['view'] = x.to_html()

    lf_console._notebook = pg.Dict(
        display=display
    )
    root_dir = os.path.join(tempfile.gettempdir(), 'test_html_progress_tracker')
    experiment = eval_test_helper.test_experiment()
    _ = experiment.run(root_dir, 'new', plugins=[])
    self.assertIsInstance(result['view'], pg.Html)
    lf_console._notebook = None


class TqdmProgressTrackerTest(unittest.TestCase):

  def test_basic(self):
    root_dir = os.path.join(tempfile.gettempdir(), 'test_tqdm_progress_tracker')
    experiment = eval_test_helper.test_experiment()
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      _ = experiment.run(root_dir, 'new', plugins=[])
    self.assertIn('All: 100%', string_io.getvalue())

  def test_with_example_ids(self):
    root_dir = os.path.join(
        tempfile.gettempdir(), 'test_tqdm_progress_tracker_with_example_ids'
    )
    experiment = eval_test_helper.test_experiment()
    string_io = io.StringIO()
    with contextlib.redirect_stderr(string_io):
      _ = experiment.run(root_dir, 'new', example_ids=[1], plugins=[])
    self.assertIn('All: 100%', string_io.getvalue())


if __name__ == '__main__':
  unittest.main()
