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
import io
from typing import Iterator

from langfun.core import component


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
