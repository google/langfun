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
"""Python code parsing."""

import inspect
import io
import re


_ID_REGEX = re.compile('^[a-zA-Z_\\-]*$')


def clean(code_text: str) -> str:
  """Cleans up Python code.

  LLM may generate code with markdown annotations, as well as minor syntax
  errors. This function removes such annotations and fixes minor syntax errors
  without extra LLM calls.

  Args:
    code_text: The code text to clean up.

  Returns:
    The cleaned up code text.
  """
  # TODO(daiyip): Deal with markdown in docstrings.
  code = io.StringIO()
  quote_char = None
  in_code = False
  i = 0
  in_comment = False
  while i < len(code_text):
    c = code_text[i]
    # Detect code block separator (```).
    if (not in_comment
        and quote_char is None
        and c == '`'
        and code_text[i:i + 3] == '```'):
      in_code = not in_code
      if in_code:
        i += 3
        continue
      else:
        break

    # Detect string literal boundary.
    if (in_code
        and not in_comment
        and c in ('\'', '"')
        and i > 0
        and code_text[i - 1] != '\\'):
      # Handle ''' and """.
      if code_text[i: i + 3] == c * 3:
        c = c * 3
        i += 2

      if quote_char is None:
        quote_char = c
      elif quote_char == c:
        # NOTE(daiyip): at times, LM forgets to escape quotes inside a string.
        # Thus we do some smart checking here to automatically correct such
        # case. This logic here is pretty involved in handling special cases.
        # We might want to revisit them later.

        # Peek forward to see if it could be a valid string.
        nt, nnt_start = _next_token(code_text, i + 1)
        if (len(c) == 3
            or nt in (',', '[', ']', '}', ')', '+', '*', '%', '\n', ':')):
          end_quote = True
        elif nt == ' ':
          # Detect if . could be a method invocation.
          # NOTE(daiyip): 'in' and 'not in' might have false positives. But
          # given the chance is low, we do not complicate the reasoning logic
          # for now.
          nnt, _ = _next_token(code_text, nnt_start, skip_whitespace=True)
          end_quote = nnt in ('+', '*', '%', '#', '[', 'in', 'not', ':')
        elif nt == '.':
          # Detect if . could be method invocation on string.
          nnt, nnnt_start = _next_token(code_text, nnt_start)
          nnnt, _ = _next_token(code_text, nnnt_start)
          end_quote = nnt.isidentifier() and nnnt == '('
        else:
          end_quote = False

        if end_quote:
          quote_char = None
        else:
          c = f'\\{c}'
    # Detect comment.
    elif c == '#' and quote_char is None:
      in_comment = True
    # Detect end-of-comment.
    elif c == '\n':
      # NOTE(daiyip): deal with cases that LM forgot to escape linebreaks
      # within strings.
      if quote_char is not None:
        # Only add \\ for ' and " (other than ''' and """).
        if len(quote_char) == 1:
          c = '\\n'
      else:
        in_comment = False

    if in_code:
      code.write(c)

    i += 1

  code = code.getvalue()
  if code:
    pos = code.find('\n')
    # Strip markdown code type. E.g. ```python
    if pos > 0 and _ID_REGEX.match(code[:pos]):
      code = code[pos:]
  else:
    # Maybe-code that resides not within a code markdown block.
    # Adding '\n' makes inspect.cleandoc to make right adjustment.
    code = '\n' + code_text
  return inspect.cleandoc(code).strip()


def _next_token(
    text: str,
    start: int = 0,
    skip_whitespace: bool = False
    ) -> tuple[str, int]:
  """Find the next token in a string with a start position."""
  token_start = start
  if skip_whitespace:
    while token_start < len(text) and text[token_start] in ' \t':
      token_start += 1
  token_end = token_start + 1
  if text[token_start].isalpha():
    while (token_end < len(text)
           and text[token_end].isalpha() or text[token_end] in '_'):
      token_end += 1
  return text[token_start:token_end], token_end
