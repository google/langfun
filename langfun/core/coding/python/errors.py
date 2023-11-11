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
"""Python code errors."""

import io
import sys
import textwrap
import traceback
import langfun.core as lf


class CodeError(RuntimeError):
  """Python code error."""

  def __init__(
      self,
      code: str,
      cause: Exception,
      ):
    self.code = code
    self.cause = cause

    # Figure out the starting and ending line numbers of the erratic code.
    lineno = None
    end_lineno = None
    if isinstance(cause, SyntaxError):
      lineno = cause.lineno
      end_lineno = cause.end_lineno
    elif not isinstance(cause, TimeoutError):
      tb = sys.exc_info()[2]
      frames = traceback.extract_tb(tb, limit=5)
      for f in frames:
        if not f.filename or f.filename == '<string>':
          lineno = f.lineno
          end_lineno = lineno
          break
    self.lineno = lineno
    self.end_lineno = end_lineno

  def __str__(self):
    return self.format(include_complete_code=True)

  def code_lines(self, start_line: int, end_line: int):
    """Returns code lines ."""
    return '\n'.join(self.code.split('\n')[start_line:end_line])

  def format(self, include_complete_code: bool = True):
    """Formats the code error."""
    r = io.StringIO()
    error_message = str(self.cause).rstrip()
    if 'line' not in error_message and self.lineno is not None:
      error_message += f' (<unknown>, line {self.lineno})'
    r.write(
        lf.colored(
            f'{self.cause.__class__.__name__}: {error_message}', 'magenta'))

    if self.lineno is not None:
      r.write('\n\n')
      r.write(textwrap.indent(
          lf.colored(
              self.code_lines(self.lineno - 1, self.end_lineno), 'magenta'),
          ' ' * 2
      ))
      r.write('\n')

    if include_complete_code:
      r.write('\n')
      r.write(lf.colored('[Generated Code]', 'green', styles=['bold']))
      r.write('\n\n')
      r.write(lf.colored('  ```python\n', 'green'))
      r.write(textwrap.indent(
          lf.colored(self.code, 'green'),
          ' ' * 2
      ))
      r.write(lf.colored('\n  ```\n', 'green'))
    return r.getvalue()


class SerializationError(RuntimeError):
  """Object serialization error."""

  def __init__(self, message: str | None, cause: Exception):
    self.message = message
    self.cause = cause

  def __str__(self):
    r = io.StringIO()
    cause_message = str(self.cause).rstrip()
    if self.message:
      r.write(lf.colored(self.message, 'magenta'))
      r.write('\n\n')
    r.write(
        lf.colored(
            f'{self.cause.__class__.__name__}: {cause_message}', 'magenta'
        )
    )
    return r.getvalue()
