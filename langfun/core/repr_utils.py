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
"""Helpers for implementing _repr_xxx_ methods."""

import collections
import contextlib
import html
import io
from typing import Any, Callable, Iterator

from langfun.core import component
import pyglove as pg


class Html(pg.Object):
  """A HTML adapter for rendering."""
  content: str

  def _repr_html_(self) -> str:
    return self.content


@contextlib.contextmanager
def share_parts() -> Iterator[dict[str, int]]:
  """Context manager for defining the context (scope) of shared content.

  Under the context manager, call to `lf.write_shared` with the same content
  will be written only once. This is useful for writing shared content such as
  shared style and script sections in the HTML.

  Example:
    ```
    class Foo(pg.Object):
      def _repr_html_(self) -> str:
        s = io.StringIO()
        lf.repr_utils.write_shared_part(s, '<style>..</style>')
        lf.repr_utils.write_shared_part(s, '<script>..</script>')
        return s.getvalue()

    with lf.repr_utils.share_parts() as share_parts:
      # The <style> and <script> section will be written only once.
      lf.console.display(Foo())
      lf.console.display(Foo())

    # Assert that the shared content is attempted to be written twice.
    assert share_parts['<style>..</style>'] == 2
    ```

  Yields:
    A dictionary mapping the shared content to the number of times it is
    attempted to be written.
  """
  context = component.context_value(
      '__shared_parts__', collections.defaultdict(int)
  )
  with component.context(__shared_parts__=context):
    try:
      yield context
    finally:
      pass


def write_maybe_shared(s: io.StringIO, content: str) -> bool:
  """Writes a maybe shared part to an string stream.

  Args:
    s: The string stream to write to.
    content: A maybe shared content to write.

  Returns:
    True if the content is written to the string. False if the content is
    already written under the same share context.
  """
  context = component.context_value('__shared_parts__', None)
  if context is None:
    s.write(content)
    return True
  written = content in context
  if not written:
    s.write(content)
  context[content] += 1
  return not written


def html_repr(
    value: Any,
    item_color: Callable[
        [str, str],
        tuple[
            str | None,   # Label text color
            str | None,   # Label background color
            str | None,   # Value text color
            str | None,   # Value background color
        ]
    ] | None = None    # pylint: disable=bad-whitespace
) -> str:
  """Writes a list of key-value pairs to an string stream.

  Args:
    value: A value to be rendered in HTML.
    item_color: A function that takes the key and value and returns a tuple
      of four strings, the text color and background color of the label and
      value respectively. If None, a default color scheme will be used.

  Returns:
    The HTML representation of the value.
  """
  s = io.StringIO()
  s.write('<div style="padding-left: 20px; margin-top: 10px">')
  s.write('<table style="border-top: 1px solid #EEEEEE;">')
  item_color = item_color or (lambda k, v: (None, '#F1C40F', None, None))

  for k, v in pg.object_utils.flatten(value).items():
    if isinstance(v, pg.Ref):
      v = v.value
    if hasattr(v, '_repr_html_'):
      cs = v._repr_html_()  # pylint: disable=protected-access
    else:
      cs = f'<span style="white-space: pre-wrap">{html.escape(str(v))}</span>'

    key_color, key_bg_color, value_color, value_bg_color = item_color(k, v)
    key_span = html_round_text(
        k,
        text_color=key_color,
        background_color=key_bg_color,
        margin_bottom='0px'
    )
    value_color_style = f'color: {value_color};' if value_color else ''
    value_bg_color_style = (
        f'background-color: {value_bg_color};' if value_bg_color else ''
    )
    s.write(
        '<tr>'
        '<td style="padding: 5px; vertical-align: top; '
        f'border-bottom: 1px solid #EEEEEE">{key_span}</td>'
        '<td style="padding: 15px 5px 5px 5px; vertical-align: top; '
        'border-bottom: 1px solid #EEEEEE;'
        f'{value_color_style}{value_bg_color_style}">{cs}</td></tr>'
    )
  s.write('</table></div>')
  return s.getvalue()


def html_round_text(
    text: str,
    *,
    text_color: str = 'black',
    background_color: str = '#EEEEEE',
    display: str = 'inline-block',
    margin_top: str = '5px',
    margin_bottom: str = '5px',
    whitespace: str = 'pre-wrap') -> str:
  """Renders a HTML span with rounded corners."""
  color_style = f'color: {text_color};' if text_color else ''
  bg_color_style = (
      f'background-color: {background_color};' if background_color else ''
  )
  return (
      f'<span style="{color_style}{bg_color_style}'
      f'display:{display}; border-radius:10px; padding:5px; '
      f'margin-top: {margin_top}; margin-bottom: {margin_bottom}; '
      f'white-space: {whitespace}">{text}</span>'
  )
