# Copyright 2025 The Langfun Authors
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
r"""The `verbatim` and `select` features of the Python prompting protocol.

Under the `python` protocol every `str` field is transported as a Python string
literal, so an LM that emits code (a shell script, a Python program, a file
body) must double-escape every `\n`, `\t`, `\"` and `\\` that belongs to the
payload. LMs do this inconsistently, and a single missed escape silently mutates
the payload (typically a `\n` inside a regex becomes a real newline, which then
breaks the embedded program).

Under `protocol='python+verbatim'`, the object keeps Python syntax, but every
`str` field marked `dict(verbatim=True)` travels out of band:

    ```python
    output = TerminalNextStep(
        think_step_by_step='Write the report.',
        next_step=BashCode(bash_code=BASH_CODE, timeout_secs=600),
    )
    ```

    <<<LF:BLOCK BASH_CODE 7f3a>>>
    python3 -c 'print("done\n")'
    <<<LF:END BASH_CODE 7f3a>>>

A block name is just a Python identifier bound in the globals of the existing
`structure_from_python()` exec, so nothing downstream needs to change. A schema
that marks no field gets the plain `python` prompt.

`select:1.0`, a separate feature, parses the fence that builds the output
object when the LM also echoes the class definitions; callers usually ask for
both, as in `protocol='python+verbatim+select'`.
"""

import ast
import builtins
import functools
import hashlib
import inspect
import io
import re
import tokenize
from typing import Any, Callable, NamedTuple

from langfun.core.coding.python import parsing
from langfun.core.structured.schema import base
from langfun.core.structured.schema import python as python_lib
import pyglove as pg

# Field metadata key that moves a `str` field into a block under the `verbatim`
# feature, e.g. `Annotated[str, 'Bash code.', dict(verbatim=True)]`.
VERBATIM_METADATA_KEY = 'verbatim'

# The words and numbers of a lowercased block name, e.g. `code_2`.
_WORD_RE = re.compile(r'[a-z]+|[0-9]+')


def _name_words(name: str) -> list[str]:
  """Returns the lowercase words and numbers of a block name."""
  return _WORD_RE.findall(name.lower())


class VerbatimBlockError(ValueError):
  """Raised when the LM response misuses the verbatim block channel."""


def is_verbatim_field(field: pg.typing.Field) -> bool:
  """Returns True if a field is marked to travel in a block."""
  return isinstance(field.value, pg.typing.Str) and bool(
      field.metadata.get(VERBATIM_METADATA_KEY)
  )


# Blocks are written `<<<LF:BLOCK NAME token>>>` ... `<<<LF:END NAME token>>>`.
#
# Parsing is deliberately more permissive than rendering, because LMs fuse the
# keyword into the name (`<<<LF:BLOCK_1 7f3a>>>`), drop the name from the
# closing fence (`<<<LF:END 7f3a>>>`, `<<<LF:END_1 7f3a>>>`), use a token with
# punctuation in it, append the closing fence to the last payload line
# (`EOF<<<LF:END CODE 7f3a>>>`), garble its end (`<<<LF:END CODE 7f3a>>`,
# `<<LF:END CODE 7f3a>>>`, `<<<LF:END CODE 7f3a>>>.`), or use CRLF line breaks.
# None of those should cost a whole agent step.
#
# Every block of a response carries the same token, so an opening fence with the
# token, whatever its name, starts the next block. Before it, the block ends at
# a closing fence that carries the token and names the block or nothing, or at
# its fence fused to a line; fences without the token count only if there is
# none (see `_find_close`), so a payload may quote them. If another of these
# fences, with or without the token, follows the one that would end a block, a
# payload that quotes its own closing fence looks like prose that repeats it, so
# the parser rejects an output object that uses the block instead of guessing.
_OPEN_RE = re.compile(
    r'^([ \t]*)<<<LF:(BLOCK[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)'
    r'(?:[ \t]+([^\s>]+))?[ \t]*>>>[ \t\r]*$'
)
# A line that starts like a closing fence is one, whatever follows its `>`. The
# text after `END`, up to a `>`, may name the block, carry the token, both, or
# neither. One that names another block, or no block and another token, does
# not end the block; one that names it with another token counts as a fence
# without the token.
_CLOSE_RE = re.compile(r'[ \t]*<<<?LF:END([^>\n]*)')
# The kinds of lines that can end a block (see `_closing_kind`), by rank.
_CLOSE_RANKS = (('token', 'fused'), ('name', 'bare'), ('other',))


def _match_open(line: str) -> re.Match[str] | None:
  """Returns the opening-fence match for `line`, or None.

  Args:
    line: A line of the response.

  Returns:
    A match that spans the whole line, with the indentation, the block name and
    the token (if any) as groups 1, 3 and 4.
  """
  m = _OPEN_RE.match(line)
  # Without the `BLOCK` keyword, `<<<LF:END …>>>` and `<<<LF:END_1 …>>>` would
  # read as opening fences. `<<<LF:BLOCK_1 …>>>` is a real (fused) opener.
  if m is None or (not m.group(2) and m.group(3).upper().startswith('END')):
    return None
  return m


class _Close(NamedTuple):
  """Where a block ends; see `_find_close`."""

  line: int
  # The payload text before a closing fence fused to the end of `line`.
  fused: str | None
  # Why it is unclear where the block ends, as a clause after its name, or None.
  unclear: str | None


def _find_close(
    lines: list[str], start: int, opening: re.Match[str]
) -> _Close | None:
  """Returns where the block of `opening` ends, or None if it does not.

  It ends at the first line of the best rank in `_CLOSE_RANKS`, before the
  next opening fence with its token. Past an opening fence with another token,
  which the payload may quote, only a fence with the token counts: another one
  may close the quoted block. Lines of lower rank before that line are payload.
  It is unclear where the block ends if:
  * the lines of the best rank name another block or token;
  * a later line of the first two ranks follows that one with text between
    them (between two lines of the best rank, a line of lower rank is text);
  * that one is fused to a line of text, and another line of the first two
    ranks follows it, or an earlier line without the token names the block
    (that one may be the end, and the fused one a quote in prose).
  A later line that names another block or token does not count; only blank
  lines between them are a stutter. An unclear block ends at the last of those
  lines, so that none of it is left in the code (in a response without
  Markdown fences, neither is code between them).

  Args:
    lines: The lines of the response.
    start: The line after the opening fence.
    opening: The opening-fence match.

  Returns:
    Where the block ends, or None if it has no closing fence.
  """
  name, token = opening.group(3), opening.group(4)
  kinds: dict[int, str] = {}
  other_token = next_block = None
  for j in range(start, len(lines)):
    reopened = _match_open(lines[j])
    if reopened is not None:
      # An opener with this token starts the next block. One with another
      # token is payload, e.g. a block that quotes its own opening fence.
      if reopened.group(4) == token:
        break
      if next_block is None:
        next_block = j
      if other_token is None and reopened.group(3) == name:
        other_token = j
      continue
    kind = _closing_kind(lines[j], name, token)
    if kind is None:
      continue
    if next_block is not None and kind in ('name', 'bare', 'other'):
      # Past a quoted opener, a fence without the token may close the quoted
      # block, unless it names this block and no quoted block has its name.
      if kind != 'name' or other_token is not None:
        continue
    kinds[j] = kind
  for rank in _CLOSE_RANKS:
    ends = [j for j, kind in kinds.items() if kind in rank]
    if ends:
      break
  else:
    return None

  def fused_text(j: int) -> str | None:
    return (
        _fused_payload(lines[j], name, token) if kinds[j] == 'fused' else None
    )

  first, unclear = ends[0], None
  # A later line could end the block too, with or without the token; one that
  # names another block or token counts only if the block has no other kind.
  counted = rank if rank == ('other',) else _CLOSE_RANKS[0] + _CLOSE_RANKS[1]
  candidates = [first] + [j for j in kinds if j > first and kinds[j] in counted]
  best = [j for j in candidates if kinds[j] in rank]
  # Between two lines of the best rank, a line of lower rank is text too.
  pairs = [*zip(candidates, candidates[1:]), *zip(best, best[1:])]
  if rank == ('other',):
    unclear = (
        f'has no closing fence of its own: {lines[first].strip()!r} names '
        'another block or token'
    )
  elif any(
      any(lines[k].strip() for k in range(a + 1, b))
      or (fused_text(b) or '').strip()
      for a, b in pairs
  ):
    unclear = f'has {len(candidates)} closing fences with text between them'
  elif fused_text(first) is not None and any(
      kinds[j] == 'name' for j in kinds if j < first
  ):
    # A closing fence without the token may be the real end, and the fused one
    # a quote at the end of a line of prose.
    unclear = f'has {len(candidates) + 1} closing fences with text between them'
  elif fused_text(first) is not None and len(candidates) > 1:
    # The text before a fused fence may end with a quote of that fence.
    unclear = (
        f'has {len(candidates)} closing fences, the first of them after text '
        'on its line'
    )
  end = candidates[-1] if unclear else first
  return _Close(end, fused_text(end), unclear)


def _closing_kind(line: str, name: str, token: str | None) -> str | None:
  """Returns how `line` could end the block `name`, or None if it cannot.

  Args:
    line: A line after the block's opening fence.
    name: The block name.
    token: The opening fence's token.

  Returns:
    'token' for a closing fence with the token that names the block or
    nothing, 'fused' for the block's fence with the token at the end of a
    payload line, 'name' or 'bare' for a closing fence without the token that
    names the block or nothing, 'other' for one that names another block or
    token, or None.
  """
  closing = _CLOSE_RE.match(line)
  if closing is None:
    return 'fused' if _fused_payload(line, name, token) is not None else None
  label = closing.group(1)
  words = label.split()
  if label[:1].strip():
    # The end of the name fused to the keyword, as in `<<<LF:END_1 7f3a>>>`.
    named = ('_' + name).endswith(words.pop(0))
    if not named:
      return 'other'
  else:
    named = re.search(rf'\b{re.escape(name)}\b', label) is not None
  if named:
    return 'token' if token in words else 'name'
  if not words:
    return 'bare'
  return 'token' if words == [token] else 'other'


def _fused_payload(line: str, name: str, token: str | None) -> str | None:
  """Returns the payload text before this block's closing fence on `line`.

  Args:
    line: A line that is not a whole closing fence.
    name: The block name.
    token: The opening fence's token.

  Returns:
    The text before `<<<LF:END name token>>>` if `line` ends with it, else
    None. Without a token there is nothing to match the fence by.
  """
  if token is None:
    return None
  m = re.fullmatch(
      rf'(.*)<<<LF:END[ \t]+{re.escape(name)}[ \t]+{re.escape(token)}'
      r'[ \t]*>>>[ \t\r]*',
      line,
  )
  return m.group(1) if m else None


def _closes(line: str, name: str, token: str | None) -> bool:
  """Returns whether `line` is a closing fence that names the block and token."""
  closing = _CLOSE_RE.match(line)
  if closing is None:
    return _fused_payload(line, name, token) is not None
  return bool(re.search(rf'\b{re.escape(name)}\b', closing.group(1))) and (
      token is None or token in closing.group(1).split()
  )


def _quoted_block(
    lines: list[str], start: int, opening: re.Match[str]
) -> tuple[str, int] | None:
  """Returns the block that a block without a closing fence quotes, if any.

  The next opening fence with a block's token ends its search for a closing
  fence. If the block's own closing fence comes after that opening fence, the
  block most likely quotes the next block, fences and all, and its text is cut
  short.

  Args:
    lines: The lines of the response.
    start: The line after the opening fence.
    opening: The opening-fence match of a block that has no closing fence.

  Returns:
    The name of the first block with the same token that opens inside the block
    and the line of the block's own closing fence after it, or None if there is
    no such closing fence before the block is opened again.
  """
  name, token = opening.group(3), opening.group(4)
  quoted = None
  for j in range(start, len(lines)):
    reopened = _match_open(lines[j])
    if reopened is not None and reopened.group(4) == token:
      if reopened.group(3) == name:
        return None
      quoted = quoted or reopened.group(3)
    elif quoted is not None and _closes(lines[j], name, token):
      return quoted, j
  return None


def _describe(name: str) -> str:
  """Returns a one-line example of the block syntax, for error messages."""
  return f'`<<<LF:BLOCK {name}>>> ... <<<LF:END {name}>>>`'


# A block named `BLOCK…`/`END…` makes the opening fence read as a stutter
# (`<<<LF:BLOCK BLOCK_1 …>>>`), which LMs deduplicate to `<<<LF:BLOCK_1 …>>>`.
_RESERVED_NAME_PREFIXES = ('BLOCK', 'END')


def _is_reserved_name(name: str) -> bool:
  return name.upper().startswith(_RESERVED_NAME_PREFIXES)


def block_repr(name: str, content: str, token: str = '') -> str:
  """Returns the wire form of a single verbatim block."""
  suffix = f' {token}' if token else ''
  return (
      f'<<<LF:BLOCK {name}{suffix}>>>\n{content}\n<<<LF:END {name}{suffix}>>>'
  )


def extract_blocks(text: str) -> tuple[dict[str, str], str]:
  """Splits an LM response into verbatim blocks and the remaining text.

  A block starts at a line that is exactly an opening fence and ends at the
  closing line chosen by `_find_close`. The block value is the text strictly
  between the newline that ends the opening fence and the newline that starts
  the closing fence (or the closing fence itself, if it was appended to the last
  payload line): no stripping and no trailing newline. If the opening fence ends
  with a CRLF, the newline before the closing fence is a CRLF too.

  If the opening fence is indented (LMs sometimes indent a whole response) and
  every non-blank body line starts with that indentation, it is removed from
  them. Otherwise the body is kept as written.

  A line inside a string literal of the Python code is never a fence. The code
  is in the Markdown fences that hold Python or, if the response has no fence,
  anywhere outside the blocks.

  A block whose closing fence comes only after the opening fence of another
  block with its token quotes that block: its text ends at that opening fence,
  and the rest of it, up to its closing fence, is dropped. A block with two
  closing fences and text between them ends at the last one, as does a block
  whose only closing fences name another block or token. It is unclear where
  such blocks end, and the parser rejects an output object that uses one.

  Args:
    text: The raw LM response.

  Returns:
    A tuple of (blocks, remainder), where `remainder` is `text` with every block
    (fences included) removed.

  Raises:
    VerbatimBlockError: If a block name is defined twice with different text.
  """
  blocks, remainder, _, _ = _extract_blocks_leniently(text)
  return blocks, remainder


def _extract_blocks_leniently(
    text: str,
) -> tuple[dict[str, str], str, str | None, dict[str, str]]:
  """Implements `extract_blocks`; also returns the blocks that may be cut short.

  Args:
    text: The raw LM response.

  Returns:
    A tuple of (blocks, remainder, cut_off, unclear), where `cut_off` names the
    block that has no closing fence and runs to the end of the response, if any,
    and `unclear` maps each block where it is unclear where it ends to why, as a
    clause that follows its name.
  """
  blocks, remainder, unclosed, unclear = _extract_blocks(text, lenient=True)
  # An unterminated fence swallowed everything: the response most likely just
  # mentions the fence syntax in prose, so keep it as text instead.
  if unclosed and not remainder.strip():
    blocks, remainder, unclosed, unclear = _extract_blocks(text, lenient=False)
  cut_off = [name for name, at_end in unclosed.items() if at_end]
  return blocks, remainder, cut_off[0] if cut_off else None, unclear


def _extract_blocks(
    text: str, lenient: bool
) -> tuple[dict[str, str], str, dict[str, bool], dict[str, str]]:
  """Implements `extract_blocks`; `lenient` allows a missing closing fence.

  Args:
    text: The raw LM response.
    lenient: Whether a block without a closing fence runs to the next block, or
      to the end of the response.

  Returns:
    A tuple of (blocks, remainder, unclosed, unclear), where `unclosed` maps
    the name of every block without a closing fence to whether it runs to the
    end, and `unclear` maps each block where it is unclear where it ends to
    why (see `extract_blocks`).
  """
  lines = text.split('\n')
  code = _CodeScan(lines)
  blocks: dict[str, str] = {}
  kept: list[str] = []
  unclosed: dict[str, bool] = {}
  unclear: dict[str, str] = {}
  quote_end = -1  # The closing fence of the last block that quotes a block.
  i = 0
  while i < len(lines):
    opening = _match_open(lines[i])
    if opening is None and i <= quote_end:
      i += 1
      continue
    if opening is None:
      end = code.next_line(i)
      kept.extend(lines[i:end])
      i = end
      continue
    name = opening.group(3)
    close = _find_close(lines, i + 1, opening)
    if close is None:
      if not lenient:
        kept.append(lines[i])
        i += 1
        continue
      # Recover from a missing closing fence: run to the next block or EOF.
      end = next(
          (j for j in range(i + 1, len(lines)) if _match_open(lines[j])),
          len(lines),
      )
      unclosed[name] = end == len(lines)
      quote = _quoted_block(lines, i + 1, opening)
      if quote is not None:
        # Its text ends at the quoted block, but the rest of it, up to its own
        # closing fence, is no part of the remainder either.
        quoted, quote_end = quote
        unclear[name] = (
            f'contains the opening fence of block {quoted!r} with the same'
            ' token'
        )
      _add_block(blocks, name, _block_text(lines[i + 1 : end], opening))
      i = end
      continue
    body = lines[i + 1 : close.line]
    if close.fused is not None:
      body.append(close.fused)
    _add_block(blocks, name, _block_text(body, opening))
    if close.unclear is not None:
      unclear[name] = close.unclear
    i = close.line + 1
  return blocks, '\n'.join(kept), unclosed, unclear


def _add_block(blocks: dict[str, str], name: str, text: str) -> None:
  if blocks.get(name, text) != text:
    raise VerbatimBlockError(
        f'Verbatim block {name!r} is defined twice with different text. '
        'Define each block once.'
    )
  blocks[name] = text


def _block_text(body: list[str], opening: re.Match[str]) -> str:
  indent = opening.group(1)
  if all(line.startswith(indent) for line in body if line.strip()):
    body = [line.removeprefix(indent) for line in body]
  text = '\n'.join(body)
  # With CRLF line breaks, the last CR belongs to the closing fence's newline.
  return text.removesuffix('\r') if opening.string.endswith('\r') else text


# A Markdown fence line; an info string may follow the backticks.
_MD_FENCE_RE = re.compile(r'[ \t]*```')
# The info strings of the Markdown fences that hold Python code.
_PYTHON_FENCES = frozenset(['', 'py', 'python'])
# What starts a string literal or a comment; only triple quotes span lines.
_LITERAL_START_RE = re.compile(r"'''|\"\"\"|'|\"|#")
# The rest of a string literal, up to its closing quote, by opening quote.
_LITERAL_END_RE = {
    quote: re.compile(r'(?:\\.|[^\\])*?' + quote)
    for quote in ("'''", '"""', "'", '"')
}


class _CodeScan:
  """Follows the Markdown fences and string literals of a response, by line."""

  def __init__(self, lines: list[str]):
    self._lines = lines
    # Without a fence, the whole response is code, as for `parsing.clean`.
    self._fenced = any(map(_MD_FENCE_RE.match, lines))
    self._fence: str | None = None  # The info string of the open fence.

  def next_line(self, i: int) -> int:
    """Returns the first line after `lines[i]` that is outside string literals.

    Args:
      i: A line outside blocks and string literals, after every line that was
        passed before.
    """
    line = self._lines[i]
    if self._fenced and _MD_FENCE_RE.match(line):
      opens = self._fence is None
      self._fence = line.strip('` \t\r').lower() if opens else None
    elif not self._fenced or self._fence in _PYTHON_FENCES:
      return _skip_literals(self._lines, i)
    return i + 1


def _skip_literals(lines: list[str], i: int) -> int:
  """Returns the first line after the string literals that `lines[i]` opens.

  Args:
    lines: Lines of Python code.
    i: A line that starts outside string literals.

  Returns:
    The line after the one that closes the last string literal opened on
    `lines[i]`. A literal that is never closed is broken code and is ignored.
  """
  line, pos = lines[i], 0
  while True:
    start = _LITERAL_START_RE.search(line, pos)
    if start is None or start.group() == '#':
      return i + 1
    quote = start.group()
    end = _LITERAL_END_RE[quote].match(line, start.end())
    j = i
    while end is None and len(quote) == 3 and j + 1 < len(lines):
      j += 1
      end = _LITERAL_END_RE[quote].match(lines[j])
    if end is None:
      return i + 1
    line, pos, i = lines[j], end.end(), j


class _BlockRef(pg.Object, pg.typing.CustomTyping):
  """A placeholder that renders as a bare block name in Python source."""

  name: str

  def custom_apply(self, *args, **kwargs) -> tuple[bool, Any]:
    del args, kwargs
    return (False, self)

  def format(self, *args, **kwargs) -> str:
    del args, kwargs
    return self.name


# The block instructions of `verbatim:1.0`, which also follow the `lf.parse`
# preamble and a caller's. Frozen, like all of its prompt text: changing it
# takes a new feature version.
_BLOCK_INSTRUCTIONS = inspect.cleandoc(r"""
    A field marked `(verbatim field)` MUST NOT be given a Python string literal. Assign it a block name and put its text in a block placed AFTER the ```python fence. A block is copied byte for byte: escape nothing, indent nothing, and do not wrap it in a fence. Name the block after its field, in upper case. Every block in a response carries the same short random token that you pick, in both of its fences:

      <<<LF:BLOCK THE_BLOCK_NAME token>>>
      the exact text
      <<<LF:END THE_BLOCK_NAME token>>>
    """)

# The `lf.query` preamble of `verbatim:1.0`: the first line of the one of
# `python:2.0`, the only base it supports, then this text instead of its `1 + 1`
# example.
_QUERY_PREAMBLE_BODY = '\n\n'.join([
    inspect.cleandoc(r"""
    Emit exactly ONE ```python fence, containing only the `output = ...` assignment. Never repeat the class definitions: they are given to you above.
    """),
    _BLOCK_INSTRUCTIONS,
    inspect.cleandoc(r"""
    {{ input_title }}:
      Write a script that generates a report.

    {{ schema_title }}:
      Answer

      ```python
      class Script:
        # (verbatim field: do not write a string literal; assign a block name, e.g. `code=CODE`, and define that block below)
        code: str

      class Answer:
        note: str
        script: Script
      ```

    {{ output_title }}:
      ```python
      output = Answer(
        note='A heredoc keeps the Python source unescaped.',
        script=Script(code=CODE)
      )
      ```

      <<<LF:BLOCK CODE 7f3a>>>
      cat > /tmp/report.py <<'EOF'
      with open('/tmp/report.md', 'w') as f:
          f.write("# Title\n\nDone.\n")
      EOF
      python3 /tmp/report.py
      <<<LF:END CODE 7f3a>>>
    """),
])


class _Verbatim(base._ProtocolFeature):  # pylint: disable=protected-access
  """`verbatim`: the text of verbatim fields travels in blocks after the object.

  It applies to schemas that mark a field with `dict(verbatim=True)`; for any
  other schema a query is the same as without it.
  """

  NAME = 'verbatim'
  VERSION = '1.0'
  DEFAULT = True
  BASES = {'python': ('2.0',)}
  # Before `select`, which keeps only the fence and so would drop the blocks.
  RANK = 1

  def applies_to(self, schema: base.Schema | None) -> bool:
    if schema is None:
      return False
    for cls in schema.class_dependencies(include_base_classes=False):
      if any(map(is_verbatim_field, pg.schema(cls).fields.values())):
        return True
    return False

  def query_preamble(self, base_preamble: str) -> str:
    return base_preamble.partition('\n\n')[0] + '\n\n' + _QUERY_PREAMBLE_BODY

  def parse_instructions(self) -> str:
    return _BLOCK_INSTRUCTIONS

  def field_hint(
      self, field: pg.typing.Field, schema: base.Schema
  ) -> str | None:
    if not is_verbatim_field(field):
      return None
    name = _field_block_name(field, _taken_names(schema))
    return (
        '(verbatim field: do not write a string literal; assign a block name, '
        f'e.g. `{field.key}={name}`, and define that block below)'
    )

  def render_value(
      self,
      value: Any,
      schema: base.Schema | None,
      render: Callable[[Any], str],
  ) -> str:
    value, blocks = _to_block_refs(value, schema)
    object_code = render(value)
    if not blocks:
      return object_code
    token = _choose_token(object_code, blocks)
    return '\n\n'.join(
        [object_code]
        + [block_repr(name, content, token) for name, content in blocks.items()]
    )

  def parse_hooks(self, schema: base.Schema | None) -> '_VerbatimParse':
    return _VerbatimParse(_taken_names(schema))


class _Select(base._ProtocolFeature):  # pylint: disable=protected-access
  """`select`: parses the fence that builds the output object.

  LMs sometimes echo the class definitions of the schema before the object.
  This feature adds no prompt text.
  """

  NAME = 'select'
  VERSION = '1.0'
  DEFAULT = True
  BASES = {'python': None}
  RANK = 2

  def parse_hooks(self, schema: base.Schema | None) -> '_SelectParse':
    classes = schema.class_dependencies() if schema is not None else []
    return _SelectParse([c.__name__ for c in classes])


class _SelectParse(base._ParseHooks):  # pylint: disable=protected-access
  """Picks the code that builds the output object out of a response."""

  def __init__(self, class_names: list[str]):
    self._class_names = class_names

  def before_exec(self, text: str) -> str:
    return python_lib.select_object_code(text, class_names=self._class_names)


class _VerbatimParse(base._ParseHooks):  # pylint: disable=protected-access
  """Cuts the blocks out of a response, then checks and binds them."""

  def __init__(self, taken: frozenset[str]):
    # The names that the prompt's block names avoid (see `_taken_names`).
    self._taken = taken
    self._blocks: dict[str, str] = {}
    self._cut_off: str | None = None
    # Blocks whose text may be cut short, with that text and the error to raise
    # if the output object uses one.
    self._incomplete: dict[str, tuple[str, VerbatimBlockError]] = {}
    self._text = ''
    self._code = ''
    self._defined: dict[str, Any] = {}

  def before_exec(self, text: str) -> str:
    self._blocks, remainder, self._cut_off, unclear = _extract_blocks_leniently(
        text
    )
    errors: dict[str, VerbatimBlockError] = {}
    if self._cut_off is not None:
      errors[self._cut_off] = self._cut_off_error(self._cut_off)
    for name, why in unclear.items():
      errors[name] = self._unclear_error(name, why)
    self._incomplete = {n: (self._blocks[n], e) for n, e in errors.items()}
    return remainder

  def check_code(self, code: str, global_vars: dict[str, Any]) -> None:
    self._text = code
    # Check exactly the code that will run: prose outside the fence is dropped.
    self._code = parsing.clean(code)
    self._defined = dict(global_vars)
    # A block never shadows a name that the code may use otherwise, e.g. `int`.
    taken = set(global_vars).union(dir(builtins))
    self._resolve_block_references(global_vars, taken)
    global_vars.update(
        {n: t for n, t in self._blocks.items() if n not in taken}
    )

  def after_exec(self, value: Any, final_code: str) -> Any:
    autofixed = final_code != self._text
    value = _inline_quoted_block_names(
        value,
        self._blocks,
        parsing.clean(final_code) if autofixed else self._code,
        self._defined,
        self._taken,
    )
    allowed = None
    if autofixed:
      # Autofix rewrote the code without seeing the blocks, so it must not
      # bring in text that neither the response nor the schema determines.
      allowed = set(self._blocks.values())
      for text in (self._code, self._text):
        allowed.update(_string_values(text, self._blocks))

    def _check(field: pg.typing.Field, text: str) -> str:
      for incomplete, error in self._incomplete.values():
        if text == incomplete:
          raise error
      if (
          allowed is not None
          and text not in allowed
          and text != field.default_value
      ):
        key, block = str(field.key), _field_block_name(field, self._taken)
        raise VerbatimBlockError(
            f'Autofix set verbatim field {key!r} to text that is in no block '
            'or string literal of the response, so it may be invented. Define '
            f'it as {_describe(block)} after the Python object.'
        )
      return text

    return _rebind_verbatim_strs(value, _check)

  def _resolve_block_references(
      self, defined: dict[str, Any], taken: set[str]
  ) -> None:
    """Reconciles the block names the LM referenced with the ones it defined.

    Runs *before* execution on purpose: the autofix loop only ever sees the
    Python snippet, with the blocks already stripped out, so letting a missing
    block reach it would invite the correction LM to invent a payload.

    Args:
      defined: Names that will be available at execution time.
      taken: Names that a block cannot be bound to, as the code may use them
        otherwise.

    Raises:
      VerbatimBlockError: If a referenced block is missing, cut short or cannot
        be bound, or if misnamed blocks cannot be matched to their references.
    """
    blocks, cut_off = self._blocks, self._cut_off
    refs, loaded, stored, _ = _referenced_names(self._code, defined)
    clash = next((n for n in refs if n in blocks and n in taken), None)
    if clash is not None:
      name = f'VERBATIM_{clash}'
      raise VerbatimBlockError(
          f'Verbatim block {clash!r} has the name of a class or other name '
          'that the Python code may use, so the output object cannot refer to '
          f'it. Rename the block and its reference, e.g. to {name!r}: '
          f'{_describe(name)}.'
      )
    undefined = [
        n
        for n in refs
        if n not in stored and n not in defined and n not in blocks
    ]
    unused = [n for n in blocks if n not in loaded]
    if len(undefined) > len(unused):
      missing = undefined[len(unused)]
      message = (
          f'Verbatim block {missing!r} is referenced by the output object but '
          f'not defined. Defined blocks: {sorted(blocks.keys())}. Define it as '
          f'{_describe(missing)} after the Python object.'
      )
      if cut_off is not None:
        message += (
            f' Block {cut_off!r} has no closing fence, so the response may '
            'have been cut off.'
        )
      raise VerbatimBlockError(message)
    renames = _match_renames(
        undefined, unused, [n for n in blocks if n in loaded]
    )
    if renames is None:
      raise VerbatimBlockError(
          f'Verbatim blocks {undefined} are referenced by the output object '
          f'but not defined, and blocks {unused} are defined but not '
          'referenced. Name each block as the output object refers to it.'
      )
    referenced = [n for n in self._incomplete if n in loaded]
    for name, block in renames.items():
      pg.logging.warning(
          f'Verbatim block {block!r} was renamed to {name!r}: the output '
          'object references a block that was not defined under that name.'
      )
      blocks[name] = blocks.pop(block)
      if block in self._incomplete:
        referenced.append(block)
    if referenced:
      raise self._incomplete[referenced[0]][1]

  def _cut_off_error(self, name: str) -> VerbatimBlockError:
    return VerbatimBlockError(
        f'Verbatim block {name!r} has no closing fence and the response ends '
        'inside it, so the response was probably cut off. Define it as '
        f'{_describe(name)}.'
    )

  def _unclear_error(self, name: str, why: str) -> VerbatimBlockError:
    return VerbatimBlockError(
        f'Verbatim block {name!r} {why}, so it is unclear where {name!r} '
        "ends. Pick a token that occurs in no block's text, use it in every "
        f'fence, and close each block with its own name: {_describe(name)}.'
    )


def _rebind_verbatim_strs(
    value: Any, transform: Callable[[pg.typing.Field, str], Any]
) -> Any:
  """Rebinds every `str` leaf that sits in a verbatim field.

  Args:
    value: The value to walk: a symbolic value, or a list, tuple or dict of
      values, as `exec` builds for `list[...]` and `dict[...]` output types.
      Anything else is returned unchanged.
    transform: Called with (field, current value); its result replaces the leaf.
      Return the current value to leave the leaf alone.

  Returns:
    `value`, rebound in place.
  """
  if not isinstance(value, pg.Symbolic):
    if isinstance(value, (list, tuple, dict)):
      for item in value.values() if isinstance(value, dict) else value:
        _rebind_verbatim_strs(item, transform)
    return value

  def _transform(k: pg.KeyPath, v: Any, p: Any) -> Any:
    del p
    if not isinstance(v, str) or not k.keys:
      return v
    parent = _parent_of(value, k)
    if not isinstance(parent, pg.Object):
      return v
    field = pg.schema(parent.__class__).get_field(str(k.key))
    if field is None or not is_verbatim_field(field):
      return v
    return transform(field, v)

  return value.rebind(_transform, raise_on_no_change=False)


def _to_block_refs(
    value: Any, schema: base.Schema | None = None
) -> tuple[Any, dict[str, str]]:
  """Replaces verbatim `str` leaves with block references.

  Args:
    value: The value to render.
    schema: The schema of `value`, if known.

  Returns:
    A tuple of (a copy of `value` with block references, the blocks by name).
  """
  blocks: dict[str, str] = {}
  if not isinstance(value, (pg.Symbolic, list, tuple, dict)):
    return value, blocks
  # On demand: most values have no verbatim field.
  taken = functools.cache(lambda: _taken_names(schema, value))

  def _to_ref(field: pg.typing.Field, content: str) -> Any:
    name = _unique_block_name(_field_block_name(field, taken()), blocks)
    blocks[name] = content
    return _BlockRef(name)

  return _rebind_verbatim_strs(pg.clone(value, deep=True), _to_ref), blocks


def _value_classes(value: Any) -> list[type[Any]]:
  """Returns the classes that the code of `value` may use.

  Those are the classes of its objects, and the dependencies of the outermost
  ones, which parsing binds for a schema that `value` is an instance of.

  Args:
    value: A value that `_rebind_verbatim_strs` walks.
  """
  classes = []

  def _visit(k: pg.KeyPath, v: Any, p: Any) -> pg.TraverseAction:
    del k, p
    if isinstance(v, pg.Object):
      classes.append(type(v))
    elif isinstance(v, tuple):  # `pg.traverse` does not enter tuples.
      for item in v:
        pg.traverse(item, _visit)
    return pg.TraverseAction.ENTER

  pg.traverse(value, _visit)
  # `class_dependencies` takes a tuple for specs, not values.
  roots = list(value) if isinstance(value, tuple) else value
  try:
    declared = base.class_dependencies(roots)  # pyrefly: ignore[bad-argument-type]
  except TypeError:
    # `pg` cannot walk some classes, e.g. `lf.Template`; a schema with one
    # cannot be parsed, so there are no classes for parsing to bind.
    declared = []
  return classes + declared


def _inline_quoted_block_names(
    value: Any,
    blocks: dict[str, str],
    code: str,
    defined: dict[str, Any],
    taken: frozenset[str],
) -> Any:
  """Expands a verbatim field whose value is a quoted block name.

  LMs occasionally write `bash_code="BASH_CODE"` instead of the bare identifier.
  Only a string literal that the code passes for the field (by its keyword
  anywhere in the code, or positionally to any call: the call that built the
  value is not tracked) is such a name, never text that came from a block, and
  it only names a block of that field: one named after the field (`BASH_CODE` or
  `BASH_CODE_2` for `bash_code`), or one that no other field claims, i.e. that
  the code does not read by name and that is not named after another verbatim
  field. A quoted name derived from the field that names no block means the
  block is missing. Other literals are kept, e.g. a short answer like `KMT2D`,
  unless the response contradicts them: blocks named after the field that the
  code does not use, none of which holds the literal's text.

  Args:
    value: The parsed value.
    blocks: Blocks found in the response.
    code: The code that built `value`.
    defined: Names available at execution time, besides the blocks.
    taken: The names that the prompt's block names avoid (see `_taken_names`).

  Returns:
    `value` with quoted block names replaced by their content.

  Raises:
    VerbatimBlockError: If a field is set to a quoted name derived from it, and
      no block has that name, or to a literal that the response contradicts.
  """
  _, loaded, _, quoted = _referenced_names(code, defined)
  used = loaded.union(text for _, text in quoted)
  fields = []
  for cls in defined.values():
    if inspect.isclass(cls) and issubclass(cls, pg.Object):
      fields.extend(filter(is_verbatim_field, pg.schema(cls).fields.values()))

  def _inline(field: pg.typing.Field, text: str) -> str:
    key = str(field.key)
    if (key, text) not in quoted and (None, text) not in quoted:
      return text
    own = _is_field_block_name(field, text, taken)
    if text in blocks:
      claimed = text in loaded or any(
          str(f.key) != key and _is_field_block_name(f, text, taken)
          for f in fields
      )
      if own or not claimed:
        return blocks[text]
    elif own:
      raise VerbatimBlockError(
          f'Verbatim field {key!r} is set to {text!r}, which names no block. '
          f'Defined blocks: {sorted(blocks)}. Define it as '
          f'{_describe(text)} after the Python object.'
      )
    # LMs sometimes write a payload twice, as a literal and as a block.
    unused = sorted(
        n
        for n in blocks
        if n not in used and _is_field_block_name(field, n, taken)
    )
    if unused and all(blocks[n] != text for n in unused):
      raise VerbatimBlockError(
          f'Verbatim field {key!r} is set to a string literal, but blocks '
          f'{unused} named after it are defined and not used, and none of '
          'them holds that text. Refer to the intended block by its name, '
          f'e.g. `{key}={unused[0]}`.'
      )
    return text

  return _rebind_verbatim_strs(value, _inline)


def _is_field_block_name(
    field: pg.typing.Field, text: str, taken: frozenset[str]
) -> bool:
  """Returns whether `text` is a block name the prompt derives from `field`."""
  name = _field_block_name(field, taken)
  return re.fullmatch(rf'{re.escape(name)}(?:_[0-9]+)?', text) is not None


def _parent_of(root: Any, path: pg.KeyPath) -> Any:
  node = root
  for key in path.keys[:-1]:
    node = node.sym_getattr(key) if isinstance(node, pg.Object) else node[key]
  return node


def _taken_names(
    schema: base.Schema | None, value: Any = None
) -> frozenset[str]:
  """Returns the names that the code may use otherwise than for a block.

  Those are the builtins, `STRUCTURE_GLOBALS`, the classes that parsing binds
  for `schema` and, when rendering `value`, the classes that its code may use.

  Args:
    schema: The schema of the prompt, the response or `value`, if known.
    value: The value to render, if any.
  """
  classes = schema.class_dependencies() if schema is not None else []
  if value is not None:
    classes += _value_classes(value)
  return frozenset(dir(builtins)).union(
      python_lib.STRUCTURE_GLOBALS, (c.__name__ for c in classes)
  )


def _field_block_name(field: pg.typing.Field, taken: frozenset[str]) -> str:
  """Returns the block name that the prompt suggests for a verbatim field.

  It is the upper-cased field name, with a `VERBATIM_` prefix if the code may
  use that name otherwise (e.g. class `SQL` for field `sql`, a class `DB`
  elsewhere in the schema for field `db`, or `UNKNOWN`), or if it would stutter
  with the fence keyword.

  Args:
    field: A verbatim field.
    taken: The names that the code may use otherwise (see `_taken_names`).
  """
  name = str(field.key).upper()
  if _is_reserved_name(name) or name in taken:
    name = f'VERBATIM_{name}'
  return name


def _unique_block_name(base_name: str, taken: dict[str, str]) -> str:
  if base_name not in taken:
    return base_name
  i = 2
  while f'{base_name}_{i}' in taken:
    i += 1
  return f'{base_name}_{i}'


def _choose_token(object_code: str, blocks: dict[str, str]) -> str:
  """Returns a short token that does not occur anywhere in the response."""
  haystack = object_code + '\n' + '\n'.join(blocks.values())
  digest = hashlib.sha256(haystack.encode('utf-8')).hexdigest()
  for i in range(0, len(digest) - 4, 4):
    token = digest[i : i + 4]
    if token not in haystack:
      return token
  return digest  # Practically unreachable.


# The prompt asks the LM to name a block after its field, in upper case.
_BLOCK_NAME_RE = re.compile(r'[A-Z][A-Z0-9_]*')


def _match_renames(
    undefined: list[str], unused: list[str], used: list[str]
) -> dict[str, str] | None:
  """Matches undefined block references to unused blocks, by name.

  A lone undefined reference takes a lone unused block, unless the block agrees
  with a used block instead: then it most likely repeats that block's payload,
  and the payload of the reference is missing. Nor does it if the block only
  shares the name stem of a used block: the response numbers those blocks
  differently from their references, so even the used one may be off by one
  (references `CODE_1`, `CODE_2` and blocks `CODE_2`, `CODE_3`). Otherwise each
  reference needs
  exactly one unused block that agrees with it, and with no other reference, by
  ordinal or name stem (see `_names_agree`). Source order alone is never
  trusted: a wrong guess silently swaps two payloads.

  Args:
    undefined: Block names that the code reads but the response does not define.
    unused: Blocks that the response defines but the code does not read.
    used: Blocks that the response defines and the code reads.

  Returns:
    A dict from each undefined name to the block to rename to it, or None if
    the match is ambiguous.
  """
  if len(undefined) == 1 and len(unused) == 1:
    name, block = undefined[0], unused[0]
    if not _names_agree(name, block) and any(
        _names_agree(b, block) or _name_stem(b) == _name_stem(block)
        for b in used
    ):
      return None
    return {name: block}
  renames = {}
  for name in undefined:
    blocks = [b for b in unused if _names_agree(name, b)]
    if len(blocks) != 1:
      return None
    if [n for n in undefined if _names_agree(n, blocks[0])] != [name]:
      return None
    renames[name] = blocks[0]
  return renames


def _names_agree(a: str, b: str) -> bool:
  """Returns True if two block names share their ordinals or name stem.

  E.g. `CONTENT_2` agrees with `BLOCK_2` but not with `CONTENT_1`, and
  `OLD_CODE` agrees with `OLD` but not with `NEW`.

  Args:
    a: A block name.
    b: Another block name.
  """
  ordinals_a = [w for w in _name_words(a) if w.isdigit()]
  ordinals_b = [w for w in _name_words(b) if w.isdigit()]
  if ordinals_a and ordinals_b:
    return ordinals_a == ordinals_b
  stem_a, stem_b = _name_stem(a), _name_stem(b)
  return bool(stem_a and stem_b) and (stem_a <= stem_b or stem_b <= stem_a)


def _name_stem(name: str) -> set[str]:
  """Returns the words of a block name without its ordinals."""
  return {w for w in _name_words(name) if not w.isdigit()}


def _string_values(code: str, names: dict[str, str]) -> set[str]:
  """Returns the strings that `code` spells out, even if it is not valid Python.

  That is each string literal, each run of adjacent literals (Python
  concatenates them), and each `+` chain of those and of `names`, e.g.
  `'cd /x && ' + SCRIPT`. Other expressions are not evaluated, an f-string
  such as `f'cd /x && {SCRIPT}'` included.

  Args:
    code: Python code, possibly broken.
    names: The values of the names that a chain may use.
  """
  # One (kind, text) per token: a literal (s), a name (n), `+`, or other (x).
  items: list[tuple[str, str]] = []
  try:
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
      if token.type in (tokenize.NL, tokenize.COMMENT):
        continue
      value = None
      if token.type == tokenize.STRING:
        try:
          value = ast.literal_eval(token.string)
        except (SyntaxError, ValueError):
          pass
      if isinstance(value, str):
        items.append(('s', value))
      elif token.type == tokenize.NAME and token.string in names:
        items.append(('n', names[token.string]))
      else:
        items.append(('+' if token.exact_type == tokenize.PLUS else 'x', ''))
  except (tokenize.TokenError, SyntaxError):
    pass
  kinds = ''.join(kind for kind, _ in items)
  values = {text for kind, text in items if kind == 's'}
  for pattern in (r's{2,}', r'(?:s+|n)(?:\+(?:s+|n))+'):
    for m in re.finditer(pattern, kinds):
      values.add(''.join(text for _, text in items[m.start() : m.end()]))
  return values


def _referenced_names(
    code: str, defined: dict[str, Any]
) -> tuple[list[str], set[str], set[str], set[tuple[str | None, str]]]:
  """Returns the block references, names and string arguments of a snippet.

  A block holds the value of a verbatim field, so only a bare name passed as an
  argument can refer to one. For a field of a known class, the field decides.
  Otherwise (an unknown class or field, or a positional argument) only an
  upper-case name counts, as the prompt asks for. Any other unbound name, e.g.
  a misspelled class or constant, is left to execution, where autofix can
  repair it.

  Args:
    code: The Python code that will be executed.
    defined: Names available at execution time. The classes among them tell
      which keyword arguments are verbatim fields.

  Returns:
    The block references in source order, the names read, the names bound by
    `code` or built in, and the (keyword, or None if positional, value) of each
    string literal passed as an argument. All empty if `code` is not valid
    Python.
  """
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return [], set(), set(), set()
  refs: list[ast.Name] = []
  loaded, stored, quoted = set(), set(dir(builtins)), set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Name):
      if isinstance(node.ctx, ast.Load):
        loaded.add(node.id)
      else:
        stored.add(node.id)
    elif isinstance(node, ast.Call):
      func = node.func
      cls = defined.get(func.id) if isinstance(func, ast.Name) else None
      args: list[tuple[str | None, ast.expr]] = [(None, a) for a in node.args]
      args.extend((k.arg, k.value) for k in node.keywords if k.arg)
      for key, value in args:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
          quoted.add((key, value.value))
        if not isinstance(value, ast.Name):
          continue
        field = _field_of(cls, key)
        if field is None:
          is_ref = bool(_BLOCK_NAME_RE.fullmatch(value.id))
        else:
          is_ref = is_verbatim_field(field)
        if is_ref:
          refs.append(value)
  refs.sort(key=lambda n: (n.lineno, n.col_offset))
  return list(dict.fromkeys(n.id for n in refs)), loaded, stored, quoted


def _field_of(cls: Any, key: str | None) -> pg.typing.Field | None:
  """Returns field `key` of `cls` if it is a `pg.Object` class, or None."""
  if key is None or not (inspect.isclass(cls) and issubclass(cls, pg.Object)):
    return None
  return pg.schema(cls).get_field(key)
