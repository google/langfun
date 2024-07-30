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
"""Match-based evaluation."""

import abc
import io
import os
from typing import Any
import langfun.core as lf
from langfun.core.eval import base
import pyglove as pg


class Matching(base.Evaluation):
  """Base class for matching-based evaluation."""

  # CONSTANTS.
  MATCHES_JSON = 'matches.json'
  MISMATCHES_JSON = 'mismatches.json'

  MATCHES_HTML = 'matches.html'
  MISMATCHES_HTML = 'mismatches.html'

  @abc.abstractmethod
  def groundtruth(self, example: Any) -> Any:
    """Returns the groundtruth from an input example."""

  @abc.abstractmethod
  def answer(self, output: Any, example: Any) -> Any:
    """Returns the answer from the structure output."""

  @property
  def matches(self) -> list[tuple[Any, Any, lf.Message]]:
    """Returns the matches examples, outputs and the output messages."""
    return self._matches

  @property
  def num_matches(self) -> int:
    """Returns the number of matches between the answers and groundtruths."""
    return len(self.matches)

  @property
  def match_rate(self) -> float:
    if self.num_completed == 0:
      return 0.0
    return self.num_matches / self.num_completed

  @property
  def mismatches(self) -> list[tuple[Any, Any, lf.Message]]:
    """Returns the mismatches examples, outputs and output messages."""
    return self._mismatches

  @property
  def num_mismatches(self) -> int:
    """Returns the number of mismatches between the answers and groundtruths."""
    return len(self.mismatches)

  @property
  def mismatch_rate(self) -> float:
    if self.num_completed == 0:
      return 0.0
    return self.num_mismatches / self.num_completed

  @property
  def matches_link(self) -> str:
    """Returns the matches page."""
    return self.link(os.path.join(self.dir, Matching.MATCHES_HTML))

  @property
  def mismatches_link(self) -> str:
    return self.link(os.path.join(self.dir, Matching.MISMATCHES_HTML))

  def _reset(self) -> None:
    super()._reset()
    self._matches = []
    self._mismatches = []

  def audit_processed(
      self, example: Any, output: Any, message: lf.Message, dryrun: bool = False
  ) -> None:
    groundtruth = self.groundtruth(example)
    answer = self.answer(output, example)

    if dryrun:
      lf.console.write('')
      lf.console.write(
          str(groundtruth),
          title='GROUDTRUTH',
          color='green',
      )
      lf.console.write('')
      lf.console.write(
          str(answer),
          title='ANSWER',
          color='blue',
      )

    if self.match(answer, groundtruth):
      self._matches.append((example, output, message))
    else:
      self._mismatches.append((example, output, message))

  def match(self, answer: Any, groundtruth: Any) -> bool:
    """Matches answer against the groundtruth. Subclasses can override."""
    return pg.eq(answer, groundtruth)

  def _status(self, progress: lf.concurrent.Progress) -> dict[str, Any]:
    del progress
    return {
        'Model': self.lm.model_id,
        'Matches': '%s (%d/%d)' % (
            self._format_rate(self.match_rate),
            self.num_matches,
            self.num_completed,
        ),
        'Mismatches': '%s (%d/%d)' % (
            self._format_rate(self.mismatch_rate),
            self.num_mismatches,
            self.num_completed,
        ),
        'Failed': '%s (%d/%d)' % (
            self._format_rate(self.failure_rate),
            self.num_failures,
            self.num_completed,
        ),
    }

  def _completion_status(self, run_status: str) -> str:
    assert self.result is not None
    m = self.result.metrics
    return (
        'COMPLETED(%s):'
        ' Matches=%s (%d/%d)'
        ' Mismatches=%s (%d/%d)'
        ' Failures=%s (%d/%d)'
    ) % (
        run_status,
        self._format_rate(m.match_rate),
        m.num_matches,
        m.total,
        self._format_rate(m.mismatch_rate),
        m.num_mismatches,
        m.total,
        self._format_rate(m.failure_rate),
        m.failures,
        m.total,
    )

  def finalize(self) -> pg.Dict:
    result = super().finalize()
    result.metrics.update(
        num_matches=self.num_matches,
        match_rate=self.match_rate,
        num_mismatches=self.num_mismatches,
        mismatch_rate=self.mismatch_rate,
    )
    return result

  def save(
      self, definition: bool = True, result: bool = True, report: bool = True
  ) -> None:
    super().save(definition, result, report)

    if result:
      # Save matches.
      pg.save(
          [
              pg.Dict(input=input, output=output)
              for input, output, _ in self.matches
          ],
          os.path.join(self.dir, Matching.MATCHES_JSON),
          # We force the input and output to be dict so it does not depend on
          # the downstream to serialize.
          force_dict=True,
      )

      # Save mismatches.
      pg.save(
          [
              # We force the output to be dict as its type may be defined
              # within functors which could be deserialized.
              pg.Dict(input=input, output=output)
              for input, output, _ in self.mismatches
          ],
          os.path.join(self.dir, Matching.MISMATCHES_JSON),
          # We force the input and output to be dict so it does not depend on
          # the downstream to serialize.
          force_dict=True,
      )

    if report:
      pg.save(
          self._html([self._render_result, self._render_matches]),
          os.path.join(self.dir, Matching.MATCHES_HTML),
          file_format='txt',
      )
      pg.save(
          self._html([self._render_result, self._render_mismatches]),
          os.path.join(self.dir, Matching.MISMATCHES_HTML),
          file_format='txt',
      )

  def _render_result_header(self, s: io.StringIO):
    super()._render_result_header(s)
    s.write('<td>Mismatches</td>')
    s.write('<td>Matches</td>')

  def _render_result_row(self, s: io.StringIO):
    super()._render_result_row(s)
    s.write(
        '<td><span style="color:orange">%s</span>%s</td>'
        % (
            self._format_rate(self.mismatch_rate),
            '<a href="%s">(%d/%d)</a>'
            % (self.mismatches_link, self.num_mismatches, self.num_completed),
        )
    )
    s.write(
        '<td><span style="color:green">%s</span>%s</td>'
        % (
            self._format_rate(self.match_rate),
            '<a href="%s">(%d/%d)</a>'
            % (self.matches_link, self.num_matches, self.num_completed),
        )
    )

  def _render_summary_metrics(self, s: io.StringIO) -> None:
    """Renders metrics in HTML."""
    assert self.result is not None
    m = self.result.metrics
    self._render_link(
        s,
        'Matches (%d/%d)' % (m.num_matches, m.total),
        self._format_rate(m.match_rate),
        'color:green',
        lambda: self.matches_link,
    )
    s.write(' | ')
    self._render_link(
        s,
        'Mismatches (%d/%d)' % (m.num_mismatches, m.total),
        self._format_rate(m.mismatch_rate),
        'color:orange',
        lambda: self.mismatches_link,
    )
    s.write(' | ')
    super()._render_summary_metrics(s)

  def _render_matches(self, s: io.StringIO) -> None:
    """Formats the matched cases into html."""
    s.write('<h2> Matches (Correct) </h2>')
    s.write('<div style="white-space:pre">\n')
    s.write(
        '<table style="border: 1px solid;">'
        '<tr class="header">'
        '<td>No.</td><td>Input</td><td>Output</td>'
        '<td>Prompt/Response Chain</td>'
        '</tr>'
    )
    for i, (example, output, message) in enumerate(self.matches):
      bgcolor = 'white' if i % 2 == 0 else '#DDDDDD'
      s.write(f'<tr style="background-color: {bgcolor}"><td>{i + 1}</td>')
      input_str = pg.format(example, verbose=False, max_bytes_len=32)
      s.write(f'<td style="color:green;white-space:pre-wrap">{input_str}</td>')
      output_str = pg.format(output, verbose=False, max_bytes_len=32)
      s.write(f'<td style="color:blue;white-space:pre-wrap">{output_str}</td>')
      s.write('<td>')
      self._render_message(message, s)
      s.write('</td>')
      s.write('</tr>')
    s.write('</table></div>')

  def _render_mismatches(self, s: io.StringIO) -> None:
    """Formats the mismatched cases into html."""
    s.write('<h2> Mismatches (Incorrect) </h2>')
    s.write('<div style="white-space:pre">\n')
    s.write(
        '<table style="border: 1px solid;">'
        '<tr class="header">'
        '<td>No.</td><td>Input</td><td>Output</td>'
        '<td>Prompt/Response Chain</td>'
        '</tr>'
    )

    for i, (example, output, message) in enumerate(self.mismatches):
      bgcolor = 'white' if i % 2 == 0 else '#DDDDDD'
      s.write(f'<tr style="background-color: {bgcolor}"><td>{i + 1}</td>')
      input_str = pg.format(example, verbose=False, max_bytes_len=32)
      s.write(f'<td style="color:green;white-space:pre-wrap">{input_str}</td>')
      output_str = pg.format(output, verbose=False, max_bytes_len=32)
      s.write(
          f'<td style="color:magenta;white-space:pre-wrap">{output_str}</td>'
      )
      s.write('<td>')
      self._render_message(message, s)
      s.write('</td>')
      s.write('</tr>')
    s.write('</table></div>')
