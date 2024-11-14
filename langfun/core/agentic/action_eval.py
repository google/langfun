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
"""Evaluation (v1) for Langfun agentic actions."""

import io
import os
from typing import Annotated, Any

import langfun.core as lf
from langfun.core import eval as lf_eval
from langfun.core.agentic import action as action_lib
import pyglove as pg


class ActionEval(lf.eval.v2.Evaluation):
  """Agent evaluation."""

  action_args: Annotated[
      dict[str, Any],
      'Arguments to call the action.'
  ] = {}

  def process(self, example: pg.Dict) -> tuple[str, dict[str, Any]]:
    action = example.action
    session = action_lib.Session()
    with lf.logging.use_log_level('fatal'):
      action(session=session, **self.action_args)
    return session.final_result, dict(session=session)


#
# TODO(daiyip): Remove V1 once V2 is fully launched.
#


@pg.functor()
def _dummy_schema():
  return int


class ExampleView(pg.Object):
  id: int
  input: Any
  output: Any
  error: str | None = None


class ActionEvalV1(lf_eval.Matching):
  """Base class for action evaluations.

  The input function should returns a list of pg.Dict, with `action` and
  `groundtruth` fields.
  """
  # We override the schema and prompt to dummy values since they are not used.
  schema_fn = _dummy_schema()
  prompt = '<unused>'

  def process(self, example: pg.Dict, **kwargs):
    action = example.action
    session = action_lib.Session()
    action(session=session, lm=self.lm, **kwargs)
    return session.as_message()

  def answer(self, output: Any, example: pg.Dict) -> Any:
    return output

  def groundtruth(self, example: Any) -> Any:
    return example.groundtruth

  def audit(
      self,
      example_idx: int,
      example: Any,
      message: lf.Message | None,
      error: Exception | None = None,
      dryrun: bool = False,
  ):
    super().audit(example_idx, example, message, error, dryrun)
    # Write each example to HTML.
    if not dryrun and self.dir:
      def _save_html():
        ExampleView(
            example_idx,
            example,
            None if message is None else message.result,
            error
        ).to_html(
            collapse_level=None,
            enable_summary_tooltip=False,
        ).save(
            os.path.join(self.dir, f'example_{example_idx}.html')
        )
      # Write HTML in a separate thread to avoid blocking the main thread.
      lf.concurrent.get_executor(
          'background_eval_io', max_workers=16
      ).submit(_save_html)

  def _render_mismatches(self, s: io.StringIO) -> None:
    s.write('<h2> Mismatches (Incorrect) </h2>')
    first_url = None
    mismatched_ids = sorted([
        example_idx for example_idx, *_ in self.mismatches
    ])
    for example_idx in mismatched_ids:
      url = os.path.join(self.dir, f'example_{example_idx}.html')
      if first_url is None:
        first_url = url
      s.write(
          f'<a href="{url}" style="margin-right: 10px" target="example_view">'
          f'{example_idx}</a> '
      )
    if first_url:
      s.write(
          '<iframe style="border:0;width:100%;height:100%" name="example_view"'
          f'src="{first_url}" title="Example View"></iframe>'
      )
    else:
      s.write('No mismatches found.')

  def _render_matches(self, s: io.StringIO) -> None:
    s.write('<h2> Matches (correct) </h2>')
    first_url = None
    matched_ids = sorted([
        example_idx for example_idx, *_ in self.matches
    ])
    for example_idx in matched_ids:
      url = os.path.join(self.dir, f'example_{example_idx}.html')
      if first_url is None:
        first_url = url
      s.write(
          f'<a href="{url}" style="margin-right: 10px">{example_idx}</a> '
      )
    if first_url:
      s.write(
          '<iframe style="border:0;width:100%;height:100%" name="example_view"'
          f'src="{first_url}" title="Example View"></iframe>'
      )
    else:
      s.write('No matches found.')
