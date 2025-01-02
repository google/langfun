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
"""Scoring evaluation."""

import abc
import io
import os
from typing import Any
import langfun.core as lf
from langfun.core.eval import base
import pyglove as pg


class Scoring(base.Evaluation):
  """Base class of evaluations by scoring."""

  # CONSTANTS.
  SCORED_JSON = 'scored.json'
  SCORED_HTML = 'scored.html'

  @property
  def scored(self) -> list[tuple[Any, Any, float, lf.Message]]:
    """Returns a list of (example, structured output, score, output message)."""
    return self._scored

  @property
  def num_scored(self) -> int:
    """Returns the number of scored examples."""
    return len(self._scored)

  @property
  def score_rate(self) -> float:
    """Returns the score rate."""
    if self.num_completed == 0:
      return 0.0
    return self.num_scored / self.num_completed

  @property
  def scored_link(self) -> str:
    """Returns the matches page."""
    return self.link(os.path.join(self.dir, Scoring.SCORED_HTML))

  @property
  def avg_score(self) -> float:
    if self.num_scored == 0:
      return 0
    return sum([i[2] for i in self._scored]) / self.num_scored

  def _reset(self) -> None:
    super()._reset()
    self._scored = []

  def audit_processed(
      self, example_idx: int, example: Any, output: Any, message: lf.Message,
      dryrun: bool = False
  ) -> None:
    del example_idx
    score = self.score(example, output)

    if dryrun:
      lf.console.write('')
      lf.console.write(
          str(score),
          title='SCORE',
          color='blue',
      )
    self._scored.append((example, output, score, message))

  @abc.abstractmethod
  def score(self, example: Any, output: Any) -> float:
    """Scores the output against its input example."""

  def _eval_status(self, progress: lf.concurrent.Progress) -> dict[str, Any]:
    del progress
    return {
        'Average Score': {self.avg_score},
        'Scored': '%.3f%% (%d/%d)' % (
            self.score_rate * 100,
            self.num_scored,
            self.num_completed,
        ),
        'Failed': '%.3f%% (%d/%d)' % (
            self.failure_rate * 100,
            self.num_failures,
            self.num_completed,
        ),
    }

  def _completion_status(self, run_status: str) -> str:
    assert self.result is not None
    m = self.result.metrics
    return (
        'COMPLETED(%s): AvgScore=%f Scored=%.3f%% (%d/%d) '
        'Failures=%.3f%% (%d/%d)'
    ) % (
        run_status,
        m.avg_score,
        m.score_rate * 100,
        m.num_scored,
        m.total,
        m.failure_rate * 100,
        m.failures,
        m.total,
    )

  def finalize(self) -> pg.Dict:
    result = super().finalize()
    result.metrics.update(
        num_scored=self.num_scored,
        score_rate=self.score_rate,
        avg_score=self.avg_score,
    )
    return result

  def save(
      self, definition: bool = True, result: bool = True, report: bool = True
  ) -> None:
    super().save(definition, result, report)

    if result:
      # Save scored.
      pg.save(
          [
              # We force the output to be dict as its type may be defined
              # within functors which could be deserialized.
              pg.Dict(input=input, output=output, score=score)
              for input, output, score, _ in self.scored
          ],
          os.path.join(self.dir, Scoring.SCORED_JSON),
      )

    if report:
      pg.save(
          self._html([self._render_result, self._render_scored]),
          os.path.join(self.dir, Scoring.SCORED_HTML),
          file_format='txt',
      )

  def _render_result_header(self, s: io.StringIO):
    super()._render_result_header(s)
    s.write('<td>Avg Score</td>')
    s.write('<td>Scored</td>')

  def _render_result_row(self, s: io.StringIO):
    super()._render_result_row(s)
    s.write(
        '<td><span style="color:blue">%.3f</span></td>' % self.avg_score
    )
    s.write(
        '<td><span style="color:red">%s</span>%s</td>'
        % (
            '%.3f%% ' % (self.score_rate * 100),
            '<a href="%s">(%d/%d)</a>'
            % (self.scored_link, self.num_scored, self.num_completed),
        )
    )

  def _render_summary_metrics(self, s: io.StringIO) -> None:
    """Renders metrics in HTML."""
    assert self.result is not None
    m = self.result.metrics
    self._render_link(
        s,
        'Average score (%d/%d)' % (m.num_scored, m.total),
        '%.3f (%.3f%%)' % (m.avg_score, m.score_rate * 100),
        'color:green',
        lambda: self.scored_link,
    )
    s.write(' | ')
    super()._render_summary_metrics(s)

  def _render_scored(self, s: io.StringIO) -> None:
    """Formats the matched cases into html."""
    s.write('<h2> Scored </h2>')
    s.write('<div style="white-space:pre">\n')
    s.write(
        '<table style="border: 1px solid;">'
        '<tr class="header">'
        '<td>No.</td><td>Input</td><td>Output</td><td>Score</td>'
        '<td>Prompt/Response Chain</td>'
        '</tr>'
    )
    for i, (example, output, score, message) in enumerate(self.scored):
      bgcolor = 'white' if i % 2 == 0 else '#DDDDDD'
      s.write(f'<tr style="background-color: {bgcolor}"><td>{i + 1}</td>')
      input_str = pg.Html.escape(
          pg.format(example, verbose=False, max_bytes_len=32)
      )
      s.write(f'<td style="color:green;white-space:pre-wrap">{input_str}</td>')
      output_str = pg.Html.escape(
          pg.format(output, verbose=False, max_bytes_len=32)
      )
      s.write(f'<td style="color:blue;white-space:pre-wrap">{output_str}</td>')
      s.write(f'<td style="color:magenta;white-space:pre-wrap">{score}</td>')
      s.write('<td>')
      self._render_message(message, s)
      s.write('</td>')
      s.write('</tr>')
    s.write('</table></div>')
