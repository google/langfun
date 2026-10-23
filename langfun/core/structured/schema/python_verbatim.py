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
r"""Python prompting protocol with a verbatim out-of-band channel for text.

Under the `python` protocol every `str` field is transported as a Python string
literal, so an LM that emits code (a shell script, a Python program, a file
body) must double-escape every `\n`, `\t`, `\"` and `\\` that belongs to the
payload. LMs do this inconsistently, and a single missed escape silently mutates
the payload (typically a `\n` inside a regex becomes a real newline, which then
breaks the embedded program).

The `python-verbatim` protocol keeps Python object syntax for the *structure*
but moves such fields out of band:

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
`structure_from_python()` exec, so nothing downstream needs to change.
"""

import abc
import ast
import builtins
import hashlib
import inspect
import io
import re
import tokenize
from typing import Any, Callable, ClassVar, Type

import langfun.core as lf
from langfun.core.coding.python import parsing
from langfun.core.structured.schema import base
from langfun.core.structured.schema import python as python_lib
import pyglove as pg

# Field metadata key for opting a `str` field in (True) or out (False) of the
# verbatim channel, e.g. `Annotated[str, 'Bash code.', dict(verbatim=True)]`.
VERBATIM_METADATA_KEY = 'verbatim'

# A `str` field whose name contains one of these words uses the verbatim
# channel unless `VERBATIM_METADATA_KEY` says otherwise.
AUTO_VERBATIM_NAMES = frozenset(
    ['code', 'script', 'content', 'source', 'patch', 'diff', 'command']
)

# Splits both `snake_case` and `CamelCase` field names into lowercase words, so
# auto-detection works for schemas that mirror an external tool's naming (e.g.
# `CodeContent`, `ReplacementContent`, `CommandLine`).
_WORD_RE = re.compile(r'[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+')


def field_name_words(name: str) -> list[str]:
  """Returns the lowercase words of a snake_case or CamelCase identifier."""
  return [w.lower() for w in _WORD_RE.findall(name)]


class VerbatimBlockError(ValueError):
  """Raised when the LM response misuses the verbatim block channel."""


def is_verbatim_field(field: pg.typing.Field) -> bool:
  """Returns True if a field should be transported through a block."""
  if not isinstance(field.value, pg.typing.Str):
    return False
  marked = field.metadata.get(VERBATIM_METADATA_KEY)
  if marked is not None:
    return bool(marked)
  return bool(
      AUTO_VERBATIM_NAMES.intersection(field_name_words(str(field.key)))
  )


class BlockSyntax(metaclass=abc.ABCMeta):
  """Wire syntax for a verbatim block."""

  @abc.abstractmethod
  def match_open(self, line: str) -> re.Match[str] | None:
    """Returns the opening-fence match for `line`, or None.

    The match spans the whole line and captures its indentation as group 1.
    """

  @abc.abstractmethod
  def name_of(self, opening: re.Match[str]) -> str:
    """Returns the block name of an opening-fence match."""

  @abc.abstractmethod
  def render(self, name: str, content: str, token: str) -> str:
    """Returns the wire form of one block."""

  @abc.abstractmethod
  def find_close(
      self, lines: list[str], start: int, opening: re.Match[str]
  ) -> int | None:
    """Returns the index of the closing line for `opening`, or None."""

  @abc.abstractmethod
  def describe(self, name: str) -> str:
    """Returns a one-line example of the syntax, for error messages."""


class _AngleSyntax(BlockSyntax):
  """`<<<LF:BLOCK NAME token>>>` ... `<<<LF:END NAME token>>>`.

  Parsing is deliberately more permissive than rendering, because LMs fuse the
  keyword into the name (`<<<LF:BLOCK_1 7f3a>>>`), drop the name from the
  closing fence (`<<<LF:END 7f3a>>>`, `<<<LF:END_1 7f3a>>>`), or use a token
  with punctuation in it. None of those should cost a whole agent step.

  Among the closing fences that name a block, one that repeats the opening
  fence's token wins, so a stray or quoted closing fence can neither cut the
  block short nor extend it.
  """

  OPEN_RE = re.compile(
      r'^([ \t]*)<<<LF:(BLOCK[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)'
      r'(?:[ \t]+([^\s>]+))?[ \t]*>>>[ \t]*$'
  )
  # Anything on the line after `END` is advisory: it may name the block, carry
  # the token, both, or neither.
  _CLOSE_RE = re.compile(r'^[ \t]*<<<LF:END([^>\n]*)>>>[ \t]*$')

  def match_open(self, line: str) -> re.Match[str] | None:
    m = self.OPEN_RE.match(line)
    # Without the `BLOCK` keyword, `<<<LF:END …>>>` and `<<<LF:END_1 …>>>` would
    # read as opening fences. `<<<LF:BLOCK_1 …>>>` is a real (fused) opener.
    if m is None or (not m.group(2) and m.group(3).upper().startswith('END')):
      return None
    return m

  def name_of(self, opening: re.Match[str]) -> str:
    return opening.group(3)

  def render(self, name: str, content: str, token: str) -> str:
    suffix = f' {token}' if token else ''
    return (
        f'<<<LF:BLOCK {name}{suffix}>>>\n{content}\n<<<LF:END {name}{suffix}>>>'
    )

  def find_close(
      self, lines: list[str], start: int, opening: re.Match[str]
  ) -> int | None:
    name, token = self.name_of(opening), opening.group(4)
    named, tokened, any_close = [], [], []
    for j in range(start, len(lines)):
      reopened = self.match_open(lines[j])
      if reopened is not None and self.name_of(reopened) == name:
        break
      closing = self._CLOSE_RE.match(lines[j])
      if closing is None:
        continue
      any_close.append(j)
      if re.search(rf'\b{re.escape(name)}\b', closing.group(1)):
        named.append(j)
        if token in closing.group(1).split():
          tokened.append(j)
    # A named closing fence is unambiguous up to the next block of the same
    # name, and one that repeats the token surely closes this block. The last
    # one wins, so a payload may quote its own terminator.
    if named:
      return (tokened or named)[-1]
    # Otherwise the fence does not say what it closes: stay inside this block.
    stop = next(
        (j for j in range(start, len(lines)) if self.match_open(lines[j])),
        len(lines),
    )
    unnamed = [j for j in any_close if j < stop]
    return unnamed[-1] if unnamed else None

  def describe(self, name: str) -> str:
    return f'`<<<LF:BLOCK {name}>>> ... <<<LF:END {name}>>>`'


ANGLE = _AngleSyntax()


# A block named `BLOCK…`/`END…` makes the opening fence read as a stutter
# (`<<<LF:BLOCK BLOCK_1 …>>>`), which LMs deduplicate to `<<<LF:BLOCK_1 …>>>`.
_RESERVED_NAME_PREFIXES = ('BLOCK', 'END')


def _is_reserved_name(name: str) -> bool:
  return name.upper().startswith(_RESERVED_NAME_PREFIXES)


def block_repr(
    name: str, content: str, token: str = '', syntax: BlockSyntax = ANGLE
) -> str:
  """Returns the wire form of a single verbatim block."""
  return syntax.render(name, content, token)


def extract_blocks(
    text: str, syntax: BlockSyntax = ANGLE
) -> tuple[dict[str, str], str]:
  """Splits an LM response into verbatim blocks and the remaining text.

  A block starts at a line that is exactly an opening fence and ends at the
  closing line chosen by `syntax`. The block value is the text strictly between
  the newline that ends the opening fence and the newline that starts the
  closing fence: no stripping and no trailing newline.

  If the opening fence is indented (LMs sometimes indent a whole response), that
  exact prefix is removed from every body line that carries it.

  Args:
    text: The raw LM response.
    syntax: The block syntax to look for.

  Returns:
    A tuple of (blocks, remainder), where `remainder` is `text` with every block
    (fences included) removed.

  Raises:
    VerbatimBlockError: If a block name is defined twice with different text.
  """
  blocks, remainder, _ = _extract_blocks_leniently(text, syntax)
  return blocks, remainder


def _extract_blocks_leniently(
    text: str, syntax: BlockSyntax
) -> tuple[dict[str, str], str, str | None]:
  """Implements `extract_blocks`; also returns the block cut off at the end.

  Args:
    text: The raw LM response.
    syntax: The block syntax to look for.

  Returns:
    A tuple of (blocks, remainder, cut_off), where `cut_off` names the block
    that has no closing fence and runs to the end of the response, if any.
  """
  blocks, remainder, unclosed = _extract_blocks(text, syntax, lenient=True)
  # An unterminated fence swallowed everything: the response most likely just
  # mentions the fence syntax in prose, so keep it as text instead.
  if unclosed and not remainder.strip():
    blocks, remainder, unclosed = _extract_blocks(text, syntax, lenient=False)
  cut_off = [name for name, at_end in unclosed.items() if at_end]
  return blocks, remainder, cut_off[0] if cut_off else None


def _extract_blocks(
    text: str, syntax: BlockSyntax, lenient: bool
) -> tuple[dict[str, str], str, dict[str, bool]]:
  """Implements `extract_blocks`; `lenient` allows a missing closing fence.

  Args:
    text: The raw LM response.
    syntax: The block syntax to look for.
    lenient: Whether a block without a closing fence runs to the next block, or
      to the end of the response.

  Returns:
    A tuple of (blocks, remainder, unclosed), where `unclosed` maps the name of
    every block without a closing fence to whether it runs to the end.
  """
  lines = text.split('\n')
  blocks: dict[str, str] = {}
  kept: list[str] = []
  unclosed: dict[str, bool] = {}
  i = 0
  while i < len(lines):
    opening = syntax.match_open(lines[i])
    if opening is None:
      kept.append(lines[i])
      i += 1
      continue
    indent, name = opening.group(1), syntax.name_of(opening)
    end = syntax.find_close(lines, i + 1, opening)
    if end is None:
      if not lenient:
        kept.append(lines[i])
        i += 1
        continue
      # Recover from a missing closing fence: run to the next block or EOF.
      end = next(
          (j for j in range(i + 1, len(lines)) if syntax.match_open(lines[j])),
          len(lines),
      )
      unclosed[name] = end == len(lines)
      _add_block(blocks, name, _dedent(lines[i + 1 : end], indent))
      i = end
      continue
    _add_block(blocks, name, _dedent(lines[i + 1 : end], indent))
    i = end + 1
  return blocks, '\n'.join(kept), unclosed


def _add_block(blocks: dict[str, str], name: str, text: str) -> None:
  if blocks.get(name, text) != text:
    raise VerbatimBlockError(
        f'Verbatim block {name!r} is defined twice with different text. '
        'Define each block once.'
    )
  blocks[name] = text


def _dedent(lines: list[str], indent: str) -> str:
  if not indent:
    return '\n'.join(lines)
  return '\n'.join(
      line[len(indent) :] if line.startswith(indent) else line.lstrip()
      for line in lines
  )


class _BlockRef(pg.Object, pg.typing.CustomTyping):
  """A placeholder that renders as a bare block name in Python source."""

  name: str

  def custom_apply(self, *args, **kwargs) -> tuple[bool, Any]:
    del args, kwargs
    return (False, self)

  def format(self, *args, **kwargs) -> str:
    del args, kwargs
    return self.name


class PythonVerbatimPromptingProtocol(python_lib.PythonPromptingProtocol):
  """Python prompting protocol with an out-of-band channel for text fields."""

  NAME = 'python-verbatim'
  SYNTAX: ClassVar[BlockSyntax] = ANGLE

  def class_definitions(
      self,
      schema: base.Schema,
      additional_dependencies: list[Type[Any]] | None = None,
      **kwargs,
  ) -> str | None:
    kwargs.setdefault('field_hint', _verbatim_field_hint)
    return super().class_definitions(
        schema, additional_dependencies=additional_dependencies, **kwargs
    )

  def value_repr(
      self,
      value: Any,
      schema: base.Schema | None = None,
      *,
      markdown: bool = True,
      assign_to_var: str | None = None,
      **kwargs,
  ) -> str:
    if inspect.isclass(value) or isinstance(value, lf.Template):
      return super().value_repr(
          value,
          schema,
          markdown=markdown,
          assign_to_var=assign_to_var,
          **kwargs,
      )
    value, blocks = _to_block_refs(value)
    object_code = super().value_repr(
        value, schema, markdown=markdown, assign_to_var=assign_to_var, **kwargs
    )
    if not blocks:
      return object_code
    token = _choose_token(object_code, blocks)
    return '\n\n'.join(
        [object_code]
        + [
            self.SYNTAX.render(name, content, token)
            for name, content in blocks.items()
        ]
    )

  def parse_value(
      self,
      text: str,
      schema: base.Schema | None = None,
      *,
      additional_context: dict[str, Any] | None = None,
      **kwargs,
  ) -> Any:
    blocks, remainder, cut_off = _extract_blocks_leniently(text, self.SYNTAX)
    classes = schema.class_dependencies() if schema is not None else []
    remainder = python_lib.select_object_code(
        remainder, class_names=[c.__name__ for c in classes]
    )
    context: dict[str, Any] = {
        **(additional_context or {}),
        **{c.__name__: c for c in classes},
        **python_lib.STRUCTURE_GLOBALS,
    }
    cut_off_text = blocks[cut_off] if cut_off is not None else None
    # Check exactly the code that will run: prose outside the fence is dropped.
    code = parsing.clean(remainder)
    self._resolve_block_references(code, blocks, context, cut_off)
    context.update(blocks)
    value, final_code = super().parse_value(
        remainder,
        schema,
        additional_context=context,
        returns_code=True,
        **kwargs,
    )
    value = _inline_quoted_block_names(value, blocks)
    literals = None
    if final_code != remainder:
      # Autofix rewrote the code without seeing the blocks, so it must not
      # bring in text from anywhere else.
      literals = set(blocks.values())
      literals.update(_string_literals(code), _string_literals(remainder))

    def _check(field: str, text: str) -> str:
      if cut_off is not None and text == cut_off_text:
        raise self._cut_off_error(cut_off)
      if literals is not None and text not in literals:
        block = _unique_block_name(field.upper(), {})
        raise VerbatimBlockError(
            f'Autofix set verbatim field {field!r} to text that is in no block '
            'or string literal of the response, so it may be invented. Define '
            f'it as {self.SYNTAX.describe(block)} after the Python object.'
        )
      return text

    return _rebind_verbatim_strs(value, _check)

  def _resolve_block_references(
      self,
      code: str,
      blocks: dict[str, str],
      defined: dict[str, Any],
      cut_off: str | None,
  ) -> None:
    """Reconciles the block names the LM referenced with the ones it defined.

    Runs *before* execution on purpose: the autofix loop only ever sees the
    Python snippet, with the blocks already stripped out, so letting a missing
    block reach it would invite the correction LM to invent a payload.

    Args:
      code: The Python code that will be executed.
      blocks: Blocks found in the response; renamed in place to match `code`.
      defined: Names that will be available at execution time.
      cut_off: The block that runs to the end of the response, if any.

    Raises:
      VerbatimBlockError: If a referenced block is missing or cut off, or if
        misnamed blocks cannot be matched to their references.
    """
    refs, loaded, stored = _referenced_names(code, defined)
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
          f'{self.SYNTAX.describe(missing)} after the Python object.'
      )
      if cut_off is not None:
        message += (
            f' Block {cut_off!r} has no closing fence, so the response may '
            'have been cut off.'
        )
      raise VerbatimBlockError(message)
    renames = _match_renames(undefined, unused)
    if renames is None:
      raise VerbatimBlockError(
          f'Verbatim blocks {undefined} are referenced by the output object '
          f'but not defined, and blocks {unused} are defined but not '
          'referenced. Name each block as the output object refers to it.'
      )
    referenced = cut_off in loaded
    for name, block in renames.items():
      pg.logging.warning(
          f'Verbatim block {block!r} was renamed to {name!r}: the output '
          'object references a block that was not defined under that name.'
      )
      blocks[name] = blocks.pop(block)
      referenced |= block == cut_off
    if cut_off is not None and referenced:
      raise self._cut_off_error(cut_off)

  def _cut_off_error(self, name: str) -> VerbatimBlockError:
    return VerbatimBlockError(
        f'Verbatim block {name!r} has no closing fence and the response ends '
        'inside it, so the response was probably cut off. Define it as '
        f'{self.SYNTAX.describe(name)}.'
    )


def _verbatim_field_hint(field: pg.typing.Field) -> str | None:
  if not is_verbatim_field(field):
    return None
  name = _unique_block_name(str(field.key).upper(), {})
  return (
      '(verbatim field: do not write a string literal; assign a block name, '
      f'e.g. `{field.key}={name}`, and define that block below)'
  )


def _rebind_verbatim_strs(
    value: Any, transform: Callable[[str, str], Any]
) -> Any:
  """Rebinds every `str` leaf that sits in a verbatim field.

  Args:
    value: The (symbolic) value to walk. Returned unchanged if not symbolic.
    transform: Called with (field name, current value); its result replaces the
      leaf. Return the current value to leave the leaf alone.

  Returns:
    `value`, rebound in place.
  """
  if not isinstance(value, pg.Symbolic):
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
    return transform(str(k.key), v)

  return value.rebind(_transform, raise_on_no_change=False)


def _to_block_refs(value: Any) -> tuple[Any, dict[str, str]]:
  """Replaces verbatim `str` leaves with block references."""
  blocks: dict[str, str] = {}
  if not isinstance(value, pg.Symbolic):
    return value, blocks

  def _to_ref(field_name: str, content: str) -> Any:
    name = _unique_block_name(field_name.upper(), blocks)
    blocks[name] = content
    return _BlockRef(name)

  return _rebind_verbatim_strs(pg.clone(value, deep=True), _to_ref), blocks


def _inline_quoted_block_names(value: Any, blocks: dict[str, str]) -> Any:
  """Expands a verbatim field whose value is a quoted block name.

  LMs occasionally write `bash_code="BASH_CODE"` instead of the bare identifier.
  A verbatim payload is never plausibly equal to a block name, so substituting
  is unambiguous.

  Args:
    value: The parsed value.
    blocks: Blocks found in the response.

  Returns:
    `value` with quoted block names replaced by their content.
  """
  if not blocks:
    return value
  return _rebind_verbatim_strs(value, lambda _, v: blocks.get(v, v))


def _parent_of(root: Any, path: pg.KeyPath) -> Any:
  node = root
  for key in path.keys[:-1]:
    node = node.sym_getattr(key) if isinstance(node, pg.Object) else node[key]
  return node


def _unique_block_name(base_name: str, taken: dict[str, str]) -> str:
  if _is_reserved_name(base_name):
    base_name = f'VERBATIM_{base_name}'
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
    undefined: list[str], unused: list[str]
) -> dict[str, str] | None:
  """Matches undefined block references to unused blocks, by name.

  A lone undefined reference takes a lone unused block. Otherwise each
  reference needs exactly one unused block that agrees with it, and with no
  other reference, by ordinal or name stem (see `_names_agree`). Source order
  alone is never trusted: a wrong guess silently swaps two payloads.

  Args:
    undefined: Block names that the code reads but the response does not define.
    unused: Blocks that the response defines but the code does not read.

  Returns:
    A dict from each undefined name to the block to rename to it, or None if
    the match is ambiguous.
  """
  if len(undefined) == 1 and len(unused) == 1:
    return {undefined[0]: unused[0]}
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
  words_a, words_b = field_name_words(a), field_name_words(b)
  ordinals_a = [w for w in words_a if w.isdigit()]
  ordinals_b = [w for w in words_b if w.isdigit()]
  if ordinals_a and ordinals_b:
    return ordinals_a == ordinals_b
  stem_a = {w for w in words_a if not w.isdigit()}
  stem_b = {w for w in words_b if not w.isdigit()}
  return bool(stem_a and stem_b) and (stem_a <= stem_b or stem_b <= stem_a)


def _string_literals(code: str) -> set[str]:
  """Returns the string literals in `code`, even if it is not valid Python.

  Adjacent literals also count as one, as Python concatenates them.

  Args:
    code: Python code, possibly broken.
  """
  runs: list[list[str]] = [[]]
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
        runs[-1].append(value)
      elif runs[-1]:
        runs.append([])
  except (tokenize.TokenError, SyntaxError):
    pass
  literals = set().union(*runs)
  literals.update(''.join(run) for run in runs if len(run) > 1)
  return literals


def _referenced_names(
    code: str, defined: dict[str, Any]
) -> tuple[list[str], set[str], set[str]]:
  """Returns (block references, names read, names bound) in a Python snippet.

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
    The block references in source order, the names read, and the names
    bound by `code` or built in. All empty if `code` is not valid Python.
  """
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return [], set(), set()
  refs: list[ast.Name] = []
  loaded, stored = set(), set(dir(builtins))
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
  return list(dict.fromkeys(n.id for n in refs)), loaded, stored


def _field_of(cls: Any, key: str | None) -> pg.typing.Field | None:
  """Returns field `key` of `cls` if it is a `pg.Object` class, or None."""
  if key is None or not (inspect.isclass(cls) and issubclass(cls, pg.Object)):
    return None
  return pg.schema(cls).get_field(key)


#
# Detection of escape-eating in payloads that were inlined anyway.
#

_HEREDOC_RE = re.compile(
    r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1[^\n]*\n(.*?)^\t*\2$",
    re.DOTALL | re.MULTILINE,
)
_PYTHON_HINT_RE = re.compile(
    r'^\s*(import |from \S+ import |def |class |print\(|with open)', re.M
)


def diagnose_escape_damage(code: str) -> str | None:
  """Returns a diagnostic if `code` embeds Python damaged by escape eating.

  A payload that reached us through a Python string literal loses any escape
  the LM failed to double. The signature is an embedded Python program that
  fails to parse with an unterminated string literal.

  Args:
    code: A code payload, e.g. a bash script with a Python heredoc.

  Returns:
    A human readable diagnostic, or None if no damage was detected.
  """
  for match in _HEREDOC_RE.finditer(code):
    body = match.group(3)
    if not _PYTHON_HINT_RE.search(body):
      continue
    try:
      ast.parse(body)
    except SyntaxError as e:
      if e.msg and 'unterminated' in e.msg and 'string literal' in e.msg:
        return (
            f'Embedded Python in heredoc {match.group(2)!r} has an '
            f'{e.msg} at line {e.lineno}. This is the signature of an escape '
            'sequence that was consumed by the enclosing Python string '
            "literal: write the payload with the 'python-verbatim' protocol "
            'so it is transported without escaping.'
        )
    except (ValueError, MemoryError, RecursionError):
      continue
  return None
