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
import unittest

from langfun.core.eval.v2 import example as example_lib
from langfun.core.eval.v2 import metrics
import pyglove as pg

Example = example_lib.Example


class MatchTest(unittest.TestCase):

  def test_basic(self):
    m = metrics.Match()  # pylint: disable=invalid-name
    self.assertEqual(
        m.audit(Example(id=1, input=pg.Dict(groundtruth=1), output=1)),
        dict(match=True)
    )
    self.assertEqual(
        m.audit(Example(id=2, input=pg.Dict(groundtruth=1), output=2)),
        dict(mismatch=True)
    )
    self.assertEqual(
        m.audit(
            Example(
                id=3,
                input=pg.Dict(groundtruth=1),
                error=pg.symbolic.ErrorInfo(
                    tag='ValueError',
                    description='Bad input.',
                    stacktrace='...',
                )
            )
        ),
        dict(error='ValueError')
    )
    self.assertEqual(
        m.audit(
            Example(
                id=3,
                input=pg.Dict(groundtruth=1),
                error=pg.symbolic.ErrorInfo(
                    tag='MappingError.CodeError',
                    description='Bad input.',
                    stacktrace='...',
                )
            )
        ),
        dict(error='MappingError.CodeError')
    )
    self.assertEqual(m.matches, 0.25)
    self.assertEqual(m.mismatches, 0.25)
    self.assertEqual(m.oop_errors, 0.25)
    self.assertEqual(m.non_oop_errors, 0.25)

    self.assertEqual(m.values(), [
        m.matches,
        m.mismatches,
        m.oop_errors,
        m.non_oop_errors
    ])
    m.reset()
    self.assertEqual(len(m.matches.data_points), 0)
    self.assertEqual(len(m.mismatches.data_points), 0)
    self.assertEqual(len(m.oop_errors.data_points), 0)
    self.assertEqual(len(m.non_oop_errors.data_points), 0)

  def test_bad_case(self):
    m = metrics.Match()  # pylint: disable=invalid-name
    with self.assertRaisesRegex(ValueError, '`groundtruth` is not present'):
      m.audit(Example(id=1, input=pg.Dict(x=1), output=1))

  def test_custom_metadata(self):

    class MyMatch(metrics.Match):
      def match(self, example_input, output):
        return example_input.x == output, dict(x=example_input.x)

    m = MyMatch()  # pylint: disable=invalid-name
    self.assertEqual(
        m.audit(Example(id=1, input=pg.Dict(x=1), output=1)),
        dict(match=True, x=1)
    )
    self.assertEqual(m.matches, 1.0)

  def test_html_view(self):
    m = metrics.Match()  # pylint: disable=invalid-name
    m.audit(Example(id=1, input=pg.Dict(groundtruth=1), output=1))
    self.assertIn(
        '100.0%',
        m.to_html().content,
    )
    with pg.views.html.controls.HtmlControl.track_scripts() as scripts:
      m.audit(Example(id=2, input=pg.Dict(groundtruth=1), output=2))
      self.assertEqual(len(scripts), 12)


class ScoreTest(unittest.TestCase):

  def test_basic(self):

    class MyScore(metrics.Score):

      def score(self, example_input, output) -> float:
        return example_input.x * output

    m = MyScore()  # pylint: disable=invalid-name
    self.assertEqual(
        m.audit(Example(id=1, input=pg.Dict(x=1), output=1)),
        dict(score=1 * 1)
    )
    self.assertEqual(
        m.audit(Example(id=2, input=pg.Dict(x=2), output=2)),
        dict(score=2 * 2)
    )
    self.assertEqual(
        m.audit(
            Example(
                id=3,
                input=pg.Dict(x=1),
                error=pg.symbolic.ErrorInfo(
                    tag='ValueError',
                    description='Bad input.',
                    stacktrace='...',
                )
            )
        ),
        dict(error='ValueError')
    )
    self.assertEqual(
        m.audit(
            Example(
                id=3,
                input=pg.Dict(x=1),
                error=pg.symbolic.ErrorInfo(
                    tag='MappingError.CodeError',
                    description='Bad input.',
                    stacktrace='...',
                )
            )
        ),
        dict(error='MappingError.CodeError')
    )
    self.assertEqual(m.average_score, 2.5)
    self.assertEqual(m.oop_errors, 0.25)
    self.assertEqual(m.non_oop_errors, 0.25)

    self.assertEqual(m.values(), [
        m.average_score,
        m.oop_errors,
        m.non_oop_errors
    ])
    m.reset()
    self.assertEqual(len(m.average_score.data_points), 0)
    self.assertEqual(len(m.oop_errors.data_points), 0)
    self.assertEqual(len(m.non_oop_errors.data_points), 0)

  def test_custom_metadata(self):

    class MyScore(metrics.Score):

      def score(self, example_input, output):
        return example_input.x * output, dict(x=example_input.x)

    m = MyScore()  # pylint: disable=invalid-name
    self.assertEqual(
        m.audit(Example(id=1, input=pg.Dict(x=1), output=1)),
        dict(score=1 * 1, x=1)
    )
    self.assertEqual(m.average_score, 1.0)

  def test_html_view(self):

    class MyScore(metrics.Score):

      def score(self, example_input, output) -> float:
        return example_input.x * output

    m = MyScore()  # pylint: disable=invalid-name
    m.audit(Example(id=1, input=pg.Dict(x=1), output=2))
    self.assertIn(
        '2.000',
        m.to_html().content,
    )
    with pg.views.html.controls.HtmlControl.track_scripts() as scripts:
      m.audit(Example(id=2, input=pg.Dict(x=1), output=2))
      self.assertEqual(len(scripts), 9)


if __name__ == '__main__':
  unittest.main()
