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
"""Tests for the `verbatim` and `select` protocol features."""

import inspect
import textwrap
from typing import Annotated, Any
import unittest
from unittest import mock

import langfun.core as lf
from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import parsing
from langfun.core.structured import querying
from langfun.core.structured.schema import base
from langfun.core.structured.schema import python_verbatim
import pyglove as pg

# pylint: disable=line-too-long

PROTOCOL = 'python+verbatim+select'
RECORDED = 'python:2.0+verbatim:1.0+select:1.0'


def _protocol():
  return base.PromptingProtocol.from_name(PROTOCOL)


class BashCode(pg.Object):
  """Bash code to run."""

  bash_code: Annotated[str, 'Bash code to execute.', dict(verbatim=True)]
  timeout_secs: int = 60


class Note(pg.Object):
  note: str


class NextStep(pg.Object):
  """The next step."""

  think_step_by_step: str
  next_step: BashCode | Note | None = None
  learnings: list[str] | None = None


class FileEdit(pg.Object):
  """Replaces text in a file."""

  file_path: Annotated[str, 'ABSOLUTE path of the file.']
  new_text: Annotated[str, None, dict(verbatim=True)]
  old_text: Annotated[
      str | None, 'Exact text to replace.', dict(verbatim=True)
  ] = None
  block_content: Annotated[
      str | None, 'Appended as is.', dict(verbatim=True)
  ] = None


class Edits(pg.Object):
  reasoning: str
  edits: list[FileEdit]


class Task(pg.Object):
  """A task with a payload."""

  title: str
  payload: Annotated[str, 'Text to process.', dict(verbatim=True)]


class Plain(pg.Object):
  answer: int
  note: str


class Job(pg.Object):
  setup: BashCode
  main_code: Annotated[str, None, dict(verbatim=True)]


# `Conn.db` is named like `DB`, another class of `Setup`.
class Conn(pg.Object):
  db: Annotated[str, None, dict(verbatim=True)]


class DB(pg.Object):
  name: str


class Setup(pg.Object):
  conn: Conn
  target: DB


# Payloads that break naive delimiters: markdown fences, heredoc terminators,
# triple quotes, backslashes and the block fences themselves.
ADVERSARIAL = [
    '',
    'x',
    'line1\nline2',
    'trailing newline\n',
    '\n\nleading blank lines',
    'tab\tand backslash \\ and quote " and \'',
    're.split(r"[.!?\\n]+", text)',
    '```python\nprint("hi")\n```',
    "cat <<'EOF' > a.py\nprint('EOF')\nEOF",
    '"""triple quoted"""',
    'mentions <<<LF:END X 1234>>> inline',
    '<<<LF:BLOCK Y>>>\nnested\n<<<LF:END Y>>>',
    '<<<LF:END BASH_CODE>>>',
    '   indented\n\tand tabbed   ',
    'unicode: \u00e9\u4e2d\U0001f600',
]


# Payloads made of markdown code fences, which a block must carry untouched.
BACKTICK_ADVERSARIAL = [
    '```',
    '````\nnested\n````',
    '```python\nprint(1)\n```\ntail',
    'a\n```\n',
]


class FieldDetectionTest(unittest.TestCase):

  def _field(self, cls, name):
    return pg.schema(cls).get_field(name)

  def test_only_marked_str_fields_are_verbatim(self):
    class A(pg.Object):
      body: Annotated[str, 'Body.', dict(verbatim=True)]
      maybe: Annotated[str | None, None, dict(verbatim=True)] = None
      bash_code: str
      opted_out: Annotated[str, None, dict(verbatim=False)]
      count: Annotated[int, None, dict(verbatim=True)] = 0

    for name in ['body', 'maybe']:
      self.assertTrue(
          python_verbatim.is_verbatim_field(self._field(A, name)), name
      )
    # Names never count, nor marks on fields that are not `str`.
    for name in ['bash_code', 'opted_out', 'count']:
      self.assertFalse(
          python_verbatim.is_verbatim_field(self._field(A, name)), name
      )


class ExtractBlocksTest(unittest.TestCase):

  def test_basic(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'output = X(a=A)\n\n<<<LF:BLOCK A 7f3a>>>\nhello\nworld\n'
        '<<<LF:END A 7f3a>>>\n'
    )
    self.assertEqual(blocks, {'A': 'hello\nworld'})
    self.assertEqual(remainder, 'output = X(a=A)\n\n')

  def test_no_token(self):
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A>>>\nhi\n<<<LF:END A>>>'
    )
    self.assertEqual(blocks, {'A': 'hi'})

  def test_empty_block(self):
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A>>>\n<<<LF:END A>>>'
    )
    self.assertEqual(blocks, {'A': ''})

  def test_trailing_newline_is_explicit(self):
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A>>>\nhi\n\n<<<LF:END A>>>'
    )
    self.assertEqual(blocks, {'A': 'hi\n'})

  def test_payload_containing_closing_fence(self):
    # Inside a line, or without the token; see `ClosingFenceSearchTest`.
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A 7f3a>>>\nsee <<<LF:END A 7f3a>>> here\n<<<LF:END A>>>\n'
        'real content\n<<<LF:END A 7f3a>>>'
    )
    self.assertEqual(
        blocks,
        {'A': 'see <<<LF:END A 7f3a>>> here\n<<<LF:END A>>>\nreal content'},
    )

  def test_payload_containing_opening_fence(self):
    # With another token; see `ClosingFenceSearchTest`.
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A 7f3a>>>\n<<<LF:BLOCK B>>>\nx\n<<<LF:END A 7f3a>>>'
    )
    self.assertEqual(blocks, {'A': '<<<LF:BLOCK B>>>\nx'})

  def test_multiple_blocks(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\n<<<LF:BLOCK A 1>>>\na\n<<<LF:END A 1>>>\n'
        '<<<LF:BLOCK B 1>>>\nb\n<<<LF:END B 1>>>\ntail'
    )
    self.assertEqual(blocks, {'A': 'a', 'B': 'b'})
    self.assertEqual(remainder, 'obj\ntail')

  def test_blocks_before_object(self):
    blocks, remainder = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A>>>\na\n<<<LF:END A>>>\noutput = X(a=A)'
    )
    self.assertEqual(blocks, {'A': 'a'})
    self.assertEqual(remainder, 'output = X(a=A)')

  def test_mismatched_token_is_tolerated(self):
    # It names the block, so it counts as a closing fence without the token.
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A 7f3a>>>\nx\n<<<LF:END A beef>>>'
    )
    self.assertEqual(blocks, {'A': 'x'})

  def test_missing_closing_fence_runs_to_next_block_or_eof(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\n<<<LF:BLOCK A>>>\na1\na2\n<<<LF:BLOCK B>>>\nb\n<<<LF:END B>>>'
    )
    self.assertEqual(blocks, {'A': 'a1\na2', 'B': 'b'})
    self.assertEqual(remainder, 'obj')

  def test_prose_mentioning_a_fence_is_not_a_block(self):
    text = '<<<LF:BLOCK A>>>\nonly prose here'
    blocks, remainder = python_verbatim.extract_blocks(text)
    self.assertEqual(blocks, {})
    self.assertEqual(remainder, text)

  def test_block_defined_twice_with_different_text_raises(self):
    # E.g. the LM corrects itself by emitting the block again.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, "'MAIN_CODE' is defined twice"
    ):
      python_verbatim.extract_blocks(
          '<<<LF:BLOCK MAIN_CODE 1a2b>>>\nprint("draft")\n'
          '<<<LF:END MAIN_CODE 1a2b>>>\n'
          'Correction, use this instead:\n'
          '<<<LF:BLOCK MAIN_CODE 1a2b>>>\nprint("final")\n'
          '<<<LF:END MAIN_CODE 1a2b>>>'
      )

  def test_block_defined_twice_with_the_same_text(self):
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A>>>\nx\n<<<LF:END A>>>\n'
        '<<<LF:BLOCK A>>>\nx\n<<<LF:END A>>>'
    )
    self.assertEqual(blocks, {'A': 'x'})

  def test_closing_fence_with_the_opening_token_wins(self):
    blocks, remainder = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A 7f3a>>>\nx\n<<<LF:END A 7f3a>>>\n<<<LF:END A ffff>>>'
    )
    self.assertEqual(blocks, {'A': 'x'})
    self.assertEqual(remainder, '<<<LF:END A ffff>>>')

  def test_fence_must_be_the_whole_line(self):
    text = 'x = 1  # <<<LF:BLOCK A>>>\ny = 2'
    blocks, remainder = python_verbatim.extract_blocks(text)
    self.assertEqual(blocks, {})
    self.assertEqual(remainder, text)

  # Fence malformations observed in production responses.
  def test_keyword_fused_into_the_name(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\n<<<LF:BLOCK_1 5a9b>>>\necho hi\n<<<LF:END 5a9b>>>'
    )
    self.assertEqual(blocks, {'BLOCK_1': 'echo hi'})
    self.assertEqual(remainder, 'obj')

  def test_closing_fence_fuses_or_drops_the_name(self):
    for closing in [
        '<<<LF:END_1 cdf1>>>',
        '<<<LF:END cdf1>>>',
        '<<<LF:END BLOCK_1 cdf1>>>',
        '<<<LF:END>>>',
    ]:
      blocks, _ = python_verbatim.extract_blocks(
          f'<<<LF:BLOCK BLOCK_1 cdf1>>>\necho hi\n{closing}'
      )
      self.assertEqual(blocks, {'BLOCK_1': 'echo hi'}, closing)

  def test_token_may_contain_punctuation(self):
    for token in ['random_1234', 'v1.2', 'a-b']:
      blocks, _ = python_verbatim.extract_blocks(
          f'<<<LF:BLOCK CODE_SCRIPT_719a {token}>>>\nx\n'
          f'<<<LF:END CODE_SCRIPT_719a {token}>>>'
      )
      self.assertEqual(blocks, {'CODE_SCRIPT_719a': 'x'}, token)

  def test_a_closing_fence_is_not_read_as_an_opening_fence(self):
    text = '<<<LF:END 5a9b>>>'
    blocks, remainder = python_verbatim.extract_blocks(text)
    self.assertEqual(blocks, {})
    self.assertEqual(remainder, text)

  def test_a_block_may_still_be_named_endpoint(self):
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK ENDPOINT 1>>>\nx\n<<<LF:END ENDPOINT 1>>>'
    )
    self.assertEqual(blocks, {'ENDPOINT': 'x'})

  def test_garbled_closing_fence_still_closes_the_block(self):
    # Before another block, the block would run to it, fence included; at the
    # end of the response, it would read as cut off.
    def parse(blocks):
      return _protocol().parse_value(
          'output = Job(\n'
          '    setup=BashCode(bash_code=BASH_CODE), main_code=MAIN_CODE)\n'
          + blocks,
          base.Schema.from_value(Job),
      )

    main = '<<<LF:BLOCK MAIN_CODE t1>>>\necho main\n<<<LF:END MAIN_CODE t1>>>'
    for closing in (
        '<<<LF:END BASH_CODE t1>>>\r',
        '<<<LF:END BASH_CODE t1>>',
        '<<LF:END BASH_CODE t1>>>',
        '<<<LF:END BASH_CODE t1>>>.',
        '<<<LF:END BASH_CODE t1>>>`',
        '<<<LF:END BASH_CODE t1>>>\u00a0',
        '<<<LF:END BASH_CODE t1>>>\u8eaf',  # Observed.
        '<<<LF:END BASH_CODE>>> and more',
    ):
      bash = f'<<<LF:BLOCK BASH_CODE t1>>>\necho setup\n{closing}'
      for blocks in (f'{bash}\n{main}', f'{main}\n{bash}'):
        with self.subTest(blocks=blocks):
          value = parse(blocks)
          self.assertEqual(value.setup.bash_code, 'echo setup')
          self.assertEqual(value.main_code, 'echo main')

  def test_crlf_line_breaks(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\r\n<<<LF:BLOCK A t1>>>\r\na1\r\na2\r\n<<<LF:END A t1>>>\r\n'
        '<<<LF:BLOCK B t1>>>\r\nb\r\n\r\n<<<LF:END B t1>>>\r\n'
    )
    # The payload keeps its line breaks, but not the one before the closer.
    self.assertEqual(blocks, {'A': 'a1\r\na2', 'B': 'b\r\n'})
    self.assertEqual(remainder, 'obj\r\n')
    value = _protocol().parse_value(
        'output = BashCode(bash_code=BASH_CODE)\r\n\r\n'
        '<<<LF:BLOCK BASH_CODE t1>>>\r\necho hi\r\n'
        '<<<LF:END BASH_CODE t1>>>\r\n',
        base.Schema.from_value(BashCode),
    )
    self.assertEqual(value.bash_code, 'echo hi')

  def test_fences_in_a_string_literal_of_the_code_are_text(self):
    # Rule 12: an inline literal parses as under `python`, block syntax and all.
    text = (
        "cat > doc.md <<'X'\n"
        '<<<LF:BLOCK DOC t9>>>\nhello\n<<<LF:END DOC t9>>>\n'
        'X'
    )
    code = f"output = BashCode(bash_code='''{text}''')"
    schema = base.Schema.from_value(BashCode)
    for protocol in (PROTOCOL, 'python'):
      for response in (code, f'```python\n{code}\n```'):
        with self.subTest(protocol=protocol, response=response):
          value = base.PromptingProtocol.from_name(protocol).parse_value(
              response, schema
          )
          self.assertEqual(value.bash_code, text)
    # The blocks after the object are still blocks.
    value = _protocol().parse_value(
        f"output = NextStep(think_step_by_step='''{text}''',\n"
        '    next_step=BashCode(bash_code=BASH_CODE))\n'
        '<<<LF:BLOCK BASH_CODE t9>>>\necho hi\n<<<LF:END BASH_CODE t9>>>',
        base.Schema.from_value(NextStep),
    )
    self.assertEqual(value.think_step_by_step, text)
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_only_string_literals_of_the_code_hide_fences(self):
    block = '<<<LF:BLOCK A t1>>>\nx\n<<<LF:END A t1>>>'
    for text in (
        # A ``` line in a literal does not end the code fence.
        (
            '```python\noutput = X(a=A, b="""\n```\n'
            f'<<<LF:BLOCK A t1>>>\ny\n<<<LF:END A t1>>>\n""")\n```\n{block}'
        ),
        # Outside literals, a block in the code fence is still a block.
        f'```python\noutput = X(a=A)\n{block}\n```',
        # Quotes in comments and escaped quotes start no literal.
        f"output = X(a=A)  # '''\nb = 'it\\'''s'\n{block}\n'''",
        # Nor does a literal that is never closed, nor a one-line one.
        f"output = X(a=A, b='''x)\n{block}",
        f"output = X(a=A, b='x)\n{block}\n'",
        # Text outside the Python fences is not code.
        f"Use ''' here.\n```python\noutput = X(a=A)\n```\n{block}\n'''",
        f"```bash\necho '''\n```\n{block}\n'''",
    ):
      with self.subTest(text=text):
        self.assertEqual(python_verbatim.extract_blocks(text)[0], {'A': 'x'})

  def test_indented_fences_are_dedented_by_the_opening_indent(self):
    blocks, remainder = python_verbatim.extract_blocks(
        '    obj\n'
        '    <<<LF:BLOCK A>>>\n'
        '    def f():\n'
        '        return 1\n'
        '\n'
        '    <<<LF:END A>>>'
    )
    self.assertEqual(blocks, {'A': 'def f():\n    return 1\n'})
    self.assertEqual(remainder, '    obj')

  def test_block_round_trip_on_adversarial_payloads(self):
    for payload in ADVERSARIAL + BACKTICK_ADVERSARIAL:
      token = python_verbatim._choose_token('', {'A': payload})  # pylint: disable=protected-access
      text = python_verbatim.block_repr('A', payload, token)
      blocks, remainder = python_verbatim.extract_blocks(text)
      self.assertEqual(blocks, {'A': payload}, repr(payload))
      self.assertEqual(remainder, '', repr(payload))


class ClosingFenceSearchTest(unittest.TestCase):
  """Rule 4: where the search for a block's closing fence stops."""

  SCHEMA = base.Schema.from_value(BashCode)

  def _parse_wire(self, payload, token='t1'):
    fence = f' {token}' if token else ''
    return _protocol().parse_value(
        'output = BashCode(bash_code=BASH_CODE)\n\n'
        f'<<<LF:BLOCK BASH_CODE{fence}>>>\n{payload}\n'
        f'<<<LF:END BASH_CODE{fence}>>>',
        self.SCHEMA,
    ).bash_code

  def _round_trip(self, payload):
    protocol = _protocol()
    text = protocol.value_repr(
        BashCode(bash_code=payload), assign_to_var='output'
    )
    return protocol.parse_value(text, self.SCHEMA).bash_code

  def test_payload_may_quote_an_opener_of_its_name_with_another_token(self):
    for payload in (
        # The opener and closer of a quoted block, without and with a token.
        (
            'echo start\n<<<LF:BLOCK BASH_CODE>>>\ninner\n<<<LF:END'
            ' BASH_CODE>>>\necho end'
        ),
        (
            'echo start\n<<<LF:BLOCK BASH_CODE t2>>>\ninner\n<<<LF:END'
            ' BASH_CODE t2>>>\necho end'
        ),
        # Prose about the syntax.
        'Write blocks like this:\n<<<LF:BLOCK BASH_CODE>>>',
        # Snapshot 2 cut this one short at the quoted opener, silently.
        'X\n<<<LF:BLOCK BASH_CODE>>>\nX',
        'a\n<<<LF:BLOCK BASH_CODE t2>>>\nb\n<<<LF:BLOCK BASH_CODE t3>>>\nc',
        # Unchanged: another block with another token, or a closing fence only.
        (
            'echo start\n<<<LF:BLOCK OTHER t2>>>\ninner\n<<<LF:END OTHER'
            ' t2>>>\necho end'
        ),
        'echo a\n<<<LF:END BASH_CODE>>>\necho b',
    ):
      with self.subTest(payload=payload):
        self.assertEqual(self._parse_wire(payload), payload)
        self.assertEqual(self._round_trip(payload), payload)
    # Past a quoted opener, a fence without the token closes the quoted block,
    # unless it names this block (and no quoted block has its name).
    payload = (
        'echo start\n<<<LF:BLOCK OTHER t2>>>\ninner\n<<<LF:END>>>\necho end'
    )
    self.assertEqual(self._parse_wire(payload, token=''), payload)
    for payload, token in (
        ('a\n<<<LF:BLOCK OTHER t2>>>\nb\n<<<LF:END BASH_CODE>>>\nc', ''),
        ('a\n<<<LF:BLOCK OTHER t2>>>\nb<<<LF:END BASH_CODE t1>>>\nc', 't1'),
    ):
      with self.subTest(payload=payload):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, '2 closing fences with text'
        ):
          self._parse_wire(payload, token=token)

  def test_fences_without_the_token_are_payload(self):
    # A closing fence with the token outranks them, named or not.
    payload = (
        'echo a\n<<<LF:END BASH_CODE>>>\n<<<LF:END>>>\n'
        '<<<LF:END BASH_CODE t2>>>\necho b'
    )
    for closing in (
        '<<<LF:END BASH_CODE t1>>>',
        '<<<LF:END t1>>>',
        '<<<LF:END_CODE t1>>>',
    ):
      with self.subTest(closing=closing):
        value = _protocol().parse_value(
            'output = BashCode(bash_code=BASH_CODE)\n\n'
            f'<<<LF:BLOCK BASH_CODE t1>>>\n{payload}\n{closing}',
            self.SCHEMA,
        )
        self.assertEqual(value.bash_code, payload)

  def test_an_opener_with_the_same_token_starts_the_next_block(self):
    payload = (
        'echo start\n<<<LF:BLOCK BASH_CODE t1>>>\ninner\n'
        '<<<LF:END BASH_CODE t1>>>\necho end'
    )
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, 'defined twice'
    ):
      self._parse_wire(payload)
    # `value_repr` picks a token that the payload does not contain.
    self.assertEqual(self._round_trip(payload), payload)
    # Without tokens, an opener of the same name always starts a new block.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, 'defined twice'
    ):
      self._parse_wire(
          'echo start\n<<<LF:BLOCK BASH_CODE>>>\ninner\n'
          '<<<LF:END BASH_CODE>>>\necho end',
          token='',
      )

  def test_any_opener_with_the_token_starts_the_next_block(self):
    # Every block of a response carries the same token, so a later block that
    # quotes this block's closing fence does not extend it.
    for fence in (' t1', ''):
      response = (
          'output = BashCode(bash_code=BASH_CODE)\n\n'
          f'<<<LF:BLOCK BASH_CODE{fence}>>>\necho setup\n'
          f'<<<LF:END BASH_CODE{fence}>>>\n\n'
          f'<<<LF:BLOCK NOTES{fence}>>>\nBASH_CODE ends with:\n'
          f'<<<LF:END BASH_CODE{fence}>>>\n<<<LF:END NOTES{fence}>>>'
      )
      with self.subTest(fence=fence):
        self.assertEqual(
            _protocol().parse_value(response, self.SCHEMA).bash_code,
            'echo setup',
        )
        self.assertEqual(
            python_verbatim.extract_blocks(response)[0]['NOTES'],
            f'BASH_CODE ends with:\n<<<LF:END BASH_CODE{fence}>>>',
        )

  def test_a_stutter_of_the_closing_fence_is_dropped(self):
    # Only blank lines between them: the block ends at the first one.
    bash = '<<<LF:BLOCK BASH_CODE t1>>>\necho setup\n<<<LF:END BASH_CODE t1>>>'
    for blocks in (
        f'{bash}\n<<<LF:END BASH_CODE t1>>>',
        f'{bash}\n\n<<<LF:END t1>>>\n',
        f'{bash}\n<<<LF:END BASH_CODE>>>',
        f'{bash[:-1]}\n<<<LF:END BASH_CODE t1>>>',
        f'{bash}\n<<<LF:END BASH_CODE t1>>>'.replace(' t1', ''),
    ):
      with self.subTest(blocks=blocks):
        value = _protocol().parse_value(
            '```python\noutput = BashCode(bash_code=BASH_CODE)\n```\n\n'
            + blocks,
            self.SCHEMA,
        )
        self.assertEqual(value.bash_code, 'echo setup')

  def test_closing_fences_with_text_between_them_are_unclear(self):
    # A payload that quotes its own closing fence, with the token of the
    # prompt's example, which LMs copy. It reads like prose that repeats the
    # closing fence after the block, so neither is guessed.
    payload = (
        "cat > notes.md <<'EOF'\nBlocks end with a line like:\n"
        '<<<LF:END BASH_CODE 7f3a>>>\nEOF\nwc -l notes.md'
    )
    wire = (
        f'<<<LF:BLOCK BASH_CODE 7f3a>>>\n{payload}\n<<<LF:END BASH_CODE 7f3a>>>'
    )
    bash = '<<<LF:BLOCK BASH_CODE t1>>>\necho setup\n<<<LF:END BASH_CODE t1>>>'
    notes = '<<<LF:BLOCK NOTES t1>>>\nnotes\n<<<LF:END NOTES t1>>>'
    quote = 'It ends with:\n<<<LF:END BASH_CODE t1>>>'
    error = (
        "Verbatim block 'BASH_CODE' has 2 closing fences with text between "
        "them, so it is unclear where 'BASH_CODE' ends"
    )
    for blocks, refs in (
        (wire, ('BASH_CODE', "'BASH_CODE'", 'CODE_1')),
        (f'{bash}\n\n{quote}\n\n{notes}', ('BASH_CODE',)),
        (f'{notes}\n\n{bash}\n\n{quote}', ('BASH_CODE',)),
        (f'{bash}\n{quote}'.replace(' t1', ''), ('BASH_CODE',)),
        (f'{bash}\nIt ends with:\n<<<LF:END t1>>>', ('BASH_CODE',)),
        (
            (
                '<<<LF:BLOCK BASH_CODE t1>>>\necho a\n<<<LF:END t1>>>\necho b\n'
                '<<<LF:END t1>>>'
            ),
            ('BASH_CODE',),
        ),
    ):
      for ref in refs:
        with self.subTest(blocks=blocks, ref=ref):
          with self.assertRaisesRegex(
              python_verbatim.VerbatimBlockError, error
          ):
            _protocol().parse_value(
                f'```python\noutput = BashCode(bash_code={ref})\n```\n\n'
                + blocks,
                self.SCHEMA,
            )
    # It ends at the last one, and an unused one is no error.
    self.assertEqual(
        python_verbatim.extract_blocks(f'obj\n{wire}\ntail'),
        ({'BASH_CODE': payload}, 'obj\ntail'),
    )
    value = _protocol().parse_value(
        "output = BashCode(bash_code='echo hi')\n\n"
        + wire.replace('BLOCK BASH_CODE', 'BLOCK EXAMPLE').replace(
            'END BASH_CODE', 'END EXAMPLE'
        ),
        self.SCHEMA,
    )
    self.assertEqual(value.bash_code, 'echo hi')
    # As the error advises, a token that the payload lacks parses it exactly.
    self.assertEqual(self._parse_wire(payload, token='k9'), payload)
    self.assertEqual(self._round_trip(payload), payload)

  def test_a_later_closing_fence_without_the_token_makes_the_end_unclear(self):
    # A payload line with the token, or with the fence fused to it, would
    # outrank a real closing fence without the token.
    quoted = (
        "cat > notes.md <<'EOF'\nBlocks end with a line like:\n"
        '<<<LF:END BASH_CODE 7f3a>>>\nEOF\nwc -l notes.md'
    )
    fused = (
        '<<<LF:BLOCK BASH_CODE t1>>>\necho a<<<LF:END BASH_CODE t1>>>\necho b\n'
        '<<<LF:END BASH_CODE>>>'
    )
    for blocks, n in (
        (f'<<<LF:BLOCK BASH_CODE 7f3a>>>\n{quoted}\n<<<LF:END BASH_CODE>>>', 2),
        (fused, 2),
        # A payload that ends with fences of mixed rank: between the two with
        # the token, the one without it is text.
        (
            (
                '<<<LF:BLOCK BASH_CODE t1>>>\necho start\n<<<LF:END BASH_CODE'
                ' t1>>>\n<<<LF:END BASH_CODE>>>\n<<<LF:END BASH_CODE t1>>>'
            ),
            3,
        ),
    ):
      with self.subTest(blocks=blocks):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError,
            f"'BASH_CODE' has {n} closing fences with text between them",
        ):
          _protocol().parse_value(
              f'output = BashCode(bash_code=BASH_CODE)\n\n{blocks}',
              self.SCHEMA,
          )
    # It ends at the later one.
    self.assertEqual(
        python_verbatim.extract_blocks(fused)[0],
        {'BASH_CODE': 'echo a<<<LF:END BASH_CODE t1>>>\necho b'},
    )
    # Fences without the token before the real closing fence are payload.
    self.assertEqual(
        self._parse_wire('echo a\n<<<LF:END BASH_CODE>>>\necho b'),
        'echo a\n<<<LF:END BASH_CODE>>>\necho b',
    )
    # A fused one after a closing fence without the token: that one may be the
    # end, and the fused one a quote in prose; or the first is quoted in the
    # payload, and the fused one ends it. Either way, it is unclear.
    for blocks in (
        (
            'echo hi\n<<<LF:END BASH_CODE>>>\nThe block ends with '
            '<<<LF:END BASH_CODE t1>>>'
        ),
        "cat > f <<'X'\n<<<LF:END BASH_CODE>>>\nX<<<LF:END BASH_CODE t1>>>",
    ):
      with self.subTest(blocks=blocks):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError,
            "'BASH_CODE' has 2 closing fences with text between them",
        ):
          _protocol().parse_value(
              'output = BashCode(bash_code=BASH_CODE)\n\n'
              f'<<<LF:BLOCK BASH_CODE t1>>>\n{blocks}',
              self.SCHEMA,
          )
    # A later closing fence of another block does not count.
    value = _protocol().parse_value(
        '```python\noutput = BashCode(bash_code=BASH_CODE)\n```\n\n'
        '<<<LF:BLOCK BASH_CODE t1>>>\necho a\n<<<LF:END BASH_CODE t1>>>\n'
        'echo b\n<<<LF:END NOTES t1>>>',
        self.SCHEMA,
    )
    self.assertEqual(value.bash_code, 'echo a')

  def test_a_renamed_unclear_block_raises_before_autofix(self):
    # Autofix sees no blocks, so it is not asked to repair code that uses one,
    # even when the code would fail for another reason.
    lm = fake.StaticResponse(
        '```python\nCorrectedCode(corrected_code=%r)\n```'
        % 'output = BashCode(bash_code=CODE_1, timeout_secs=60)'
    )
    with querying.track_queries() as queries:
      with self.assertRaisesRegex(
          python_verbatim.VerbatimBlockError,
          "'BASH_CODE' has 2 closing fences with text between them",
      ):
        _protocol().parse_value(
            'output = BashCode(bash_code=CODE_1, timeout_secs=sixty)\n\n'
            '<<<LF:BLOCK BASH_CODE t1>>>\necho a\n<<<LF:END BASH_CODE t1>>>\n'
            'echo b\n<<<LF:END BASH_CODE t1>>>',
            self.SCHEMA,
            autofix=1,
            autofix_lm=lm,
        )
    self.assertEqual(queries, [])

  def test_a_closing_fence_of_another_block_does_not_end_it(self):
    # A parser test that quotes fences of other blocks and tokens.
    payload = (
        'def test():\n    text = """\n<<<LF:END OTHER t1>>>\n<<<LF:END t2>>>\n'
        '<<<LF:ENDPOINT x>>>\n"""'
    )
    for closing in (
        '<<<LF:END t1>>>',
        '<<<LF:END BASH_CODE t1>>>',
        '<<<LF:END>>>',
        '<<<LF:END BASH_CODE>>>',
    ):
      with self.subTest(closing=closing):
        value = _protocol().parse_value(
            'output = BashCode(bash_code=BASH_CODE)\n\n'
            f'<<<LF:BLOCK BASH_CODE t1>>>\n{payload}\n{closing}',
            self.SCHEMA,
        )
        self.assertEqual(value.bash_code, payload)
    # If it is the only one, it is unclear where the block ends: using the
    # block raises, and an unused one is no error.
    blocks = '<<<LF:BLOCK BASH_CODE t1>>>\necho setup\n<<<LF:END NOTES t1>>>'
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        "Verbatim block 'BASH_CODE' has no closing fence of its own: "
        "'<<<LF:END NOTES t1>>>' names another block or token, so it is "
        "unclear where 'BASH_CODE' ends",
    ):
      _protocol().parse_value(
          f'output = BashCode(bash_code=BASH_CODE)\n\n{blocks}', self.SCHEMA
      )
    self.assertEqual(
        python_verbatim.extract_blocks(blocks)[0], {'BASH_CODE': 'echo setup'}
    )
    # With several of them, it ends at the last one.
    self.assertEqual(
        python_verbatim.extract_blocks(
            f'{blocks}\nmore\n<<<LF:END OTHER t1>>>'
        )[0],
        {'BASH_CODE': 'echo setup\n<<<LF:END NOTES t1>>>\nmore'},
    )
    value = _protocol().parse_value(
        "output = BashCode(bash_code='echo hi')\n\n"
        + blocks.replace('BASH_CODE', 'EXAMPLE'),
        self.SCHEMA,
    )
    self.assertEqual(value.bash_code, 'echo hi')

  def test_a_block_that_quotes_a_block_with_its_token_is_not_used(self):
    payload = (
        'echo start\n<<<LF:BLOCK OTHER t1>>>\ninner\n<<<LF:END OTHER t1>>>\n'
        'echo end'
    )
    # `value_repr` picks a token that the payload does not contain.
    self.assertEqual(self._round_trip(payload), payload)
    # With the token of the payload, BASH_CODE ends at the quoted opening fence
    # and the rest of it, up to its closing fence, is dropped.
    response = (
        '{code}\n\n'
        f'<<<LF:BLOCK BASH_CODE t1>>>\n{payload}\n<<<LF:END BASH_CODE t1>>>'
    )
    self.assertEqual(
        python_verbatim.extract_blocks(response.format(code='obj')),
        ({'BASH_CODE': 'echo start', 'OTHER': 'inner'}, 'obj\n'),
    )
    # Using it raises, however it is referenced.
    for ref in ('BASH_CODE', "'BASH_CODE'", 'CODE_1'):
      with self.subTest(ref=ref):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError,
            "Verbatim block 'BASH_CODE' contains the opening fence of block "
            "'OTHER' with the same token, so it is unclear where 'BASH_CODE' "
            'ends',
        ):
          _protocol().parse_value(
              response.format(code=f'output = BashCode(bash_code={ref})'),
              self.SCHEMA,
          )
    # An unused one, e.g. an example in prose, is no error. It is named after
    # no field: an unused BASH_CODE would contradict the literal.
    value = _protocol().parse_value(
        "output = BashCode(bash_code='echo hi')\n\n"
        f'<<<LF:BLOCK EXAMPLE t1>>>\n{payload}\n<<<LF:END EXAMPLE t1>>>',
        self.SCHEMA,
    )
    self.assertEqual(value.bash_code, 'echo hi')

  def test_past_another_token_only_the_token_closes_the_block(self):
    # The first block lost its closing fence. The closing fence of the second
    # does not close the first, so the two are not merged into one payload.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, 'defined twice'
    ):
      _protocol().parse_value(
          'output = BashCode(bash_code=BASH_CODE)\n\n'
          '<<<LF:BLOCK BASH_CODE ab12>>>\necho one\n'
          '<<<LF:BLOCK BASH_CODE cd34>>>\necho two\n'
          '<<<LF:END BASH_CODE cd34>>>',
          self.SCHEMA,
      )


class IndentedFenceTest(unittest.TestCase):
  """Rule 5: an indented opening fence dedents only a uniformly indented body."""

  def _extract(self, indent, body_lines):
    blocks, _ = python_verbatim.extract_blocks(
        f'{indent}<<<LF:BLOCK FILE_CONTENT t1>>>\n'
        + '\n'.join(body_lines)
        + f'\n{indent}<<<LF:END FILE_CONTENT t1>>>'
    )
    return blocks['FILE_CONTENT']

  def test_uniformly_indented_body_is_dedented(self):
    self.assertEqual(
        self._extract('  ', ['  all:', '  \techo hi', '    nested', '']),
        'all:\n\techo hi\n  nested\n',
    )
    python = 'def f(x):\n    if x:\n        return 1\n    return 2'
    self.assertEqual(
        self._extract('  ', ['  ' + l for l in python.split('\n')]), python
    )
    # Blank lines need not carry the indentation.
    self.assertEqual(
        self._extract('  ', ['  a', '', ' ', '   ', '  b']), 'a\n\n \n \nb'
    )

  def test_other_bodies_are_kept_as_written(self):
    for indent, body in (
        ('  ', 'all:\n\techo hi'),  # A Makefile at column 0.
        ('\t', 'all:\n\techo hi'),  # Under a tab-indented fence.
        ('    ', '    a\n  b\nc'),  # Less indented lines.
        ('  ', 'def f(x):\n    if x:\n        return 1\n    return 2'),
        ('    ', 'def f(x):\n  if x:\n    return 1\n  return 2'),
        ('  ', '  x = 1\n\ty = 2\n z = 3\n\n   \n    w = 4'),
        # One line lacks the indentation, so the rest keeps it too, even the
        # heredoc terminator.
        ('  ', "  cat > Makefile <<'EOF'\n  all:\n\techo hi\n  EOF"),
    ):
      with self.subTest(indent=indent, body=body):
        self.assertEqual(self._extract(indent, body.split('\n')), body)

  def test_recorded_body_at_column_0_is_kept(self):
    # The shape of the only indented fence among 33,729 recorded replies (TBS
    # ex40): the fences are indented by 2, the body is at column 0.
    script = (
        "cat <<'EOF' > /workspace/bench.py\n"
        'def run(graph):\n'
        '    for u in graph:\n'
        '        push(u)\n'
        'EOF\n'
        'python3 /workspace/bench.py'
    )
    value = _protocol().parse_value(
        '  ```python\n'
        'output = BashCode(bash_code=BASH_CODE, timeout_secs=180)\n'
        '```\n\n'
        f'  <<<LF:BLOCK BASH_CODE 9c4e>>>\n{script}\n'
        '<<<LF:END BASH_CODE 9c4e>>>',
        base.Schema.from_value(BashCode),
    )
    self.assertEqual(value.bash_code, script)

  def test_makefile_control_is_kept(self):
    # The synthetic control: the recipe line starts with a tab instead of the
    # indentation, so every line keeps it, even the heredoc terminator.
    script = "  cat > Makefile <<'EOF'\n  all:\n\techo hi\n  EOF"
    value = _protocol().parse_value(
        '```python\n'
        'output = BashCode(bash_code=BASH_CODE, timeout_secs=60)\n'
        '```\n\n'
        f'  <<<LF:BLOCK BASH_CODE ab12>>>\n{script}\n'
        '  <<<LF:END BASH_CODE ab12>>>',
        base.Schema.from_value(BashCode),
    )
    self.assertEqual(value.bash_code, script)


class FusedClosingFenceTest(unittest.TestCase):
  """A closing fence appended to the last payload line closes the block."""

  def test_fused_fence_before_the_next_block(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\n'
        '<<<LF:BLOCK CONTENT 9a4c>>>\n'
        'Norovirus\n'
        'Norwalk virus<<<LF:END CONTENT 9a4c>>>\n'
        '\n'
        '<<<LF:BLOCK BASH_CODE 9a4c>>>\n'
        'cat /tmp/submission.txt\n'
        '<<<LF:END BASH_CODE 9a4c>>>'
    )
    self.assertEqual(
        blocks,
        {
            'CONTENT': 'Norovirus\nNorwalk virus',
            'BASH_CODE': 'cat /tmp/submission.txt',
        },
    )
    self.assertEqual(remainder, 'obj\n')

  def test_fused_fence_at_the_end_of_the_response(self):
    value = _protocol().parse_value(
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE))\n'
        '<<<LF:BLOCK BASH_CODE 8f2b>>>\n'
        "cat > a.txt <<'EOF'\n"
        'x\n'
        'EOF<<<LF:END BASH_CODE 8f2b>>>  ',
        base.Schema.from_value(NextStep),
    )
    self.assertEqual(
        value.next_step.bash_code, "cat > a.txt <<'EOF'\nx\nEOF"
    )

  def test_only_this_blocks_name_and_token_close_it(self):
    for line in (
        'x<<<LF:END A ffff>>>',
        'x<<<LF:END B 7f3a>>>',
        'x<<<LF:END 7f3a>>>',
        'x<<<LF:END A 7f3a>>> and more',
    ):
      with self.subTest(line=line):
        blocks, _ = python_verbatim.extract_blocks(
            f'<<<LF:BLOCK A 7f3a>>>\n{line}\n<<<LF:END A 7f3a>>>'
        )
        self.assertEqual(blocks, {'A': line})

  def test_a_fused_fence_and_a_later_whole_line_fence(self):
    # Even right after it: the line before may end with a quote of the fence,
    # as in a payload about the blocks.
    text = 'The block ends with <<<LF:END BASH_CODE 7f3a>>>'
    after_text = 'closing fences, the first of them after text'
    for blocks, error in (
        (f'{text}\n<<<LF:END BASH_CODE 7f3a>>>', after_text),
        (f'{text}\n\n<<<LF:END BASH_CODE>>>', after_text),
        (
            'x<<<LF:END BASH_CODE 7f3a>>>\ny\n<<<LF:END BASH_CODE 7f3a>>>',
            'closing fences with text between them',
        ),
    ):
      with self.subTest(blocks=blocks):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, f"'BASH_CODE' has 2 {error}"
        ):
          _protocol().parse_value(
              'output = BashCode(bash_code=BASH_CODE)\n'
              f'<<<LF:BLOCK BASH_CODE 7f3a>>>\n{blocks}',
              base.Schema.from_value(BashCode),
          )
    # It ends at the last one.
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A 7f3a>>>\nThe block ends with <<<LF:END A 7f3a>>>\n'
        '<<<LF:END A 7f3a>>>'
    )
    self.assertEqual(blocks, {'A': 'The block ends with <<<LF:END A 7f3a>>>'})

  def test_fused_fence_in_the_next_block_does_not_count(self):
    blocks, _ = python_verbatim.extract_blocks(
        'obj\n'
        '<<<LF:BLOCK A 7f3a>>>\n'
        'x\n'
        '<<<LF:BLOCK B 7f3a>>>\n'
        'y<<<LF:END A 7f3a>>>\n'
        '<<<LF:END B 7f3a>>>'
    )
    self.assertEqual(blocks, {'A': 'x', 'B': 'y<<<LF:END A 7f3a>>>'})

  def test_closing_fence_quoted_in_the_next_block_does_not_extend_it(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\n'
        '<<<LF:BLOCK CONTENT 9a4c>>>\n'
        'Norwalk virus<<<LF:END CONTENT 9a4c>>>\n'
        '\n'
        '<<<LF:BLOCK BASH_CODE 9a4c>>>\n'
        "cat > fences.txt <<'EOF'\n"
        '<<<LF:END CONTENT 9a4c>>>\n'
        'EOF\n'
        '<<<LF:END BASH_CODE 9a4c>>>'
    )
    self.assertEqual(
        blocks,
        {
            'CONTENT': 'Norwalk virus',
            'BASH_CODE': (
                "cat > fences.txt <<'EOF'\n<<<LF:END CONTENT 9a4c>>>\nEOF"
            ),
        },
    )
    self.assertEqual(remainder, 'obj\n')

  def test_two_fused_fences_with_text_between_them_are_unclear(self):
    blocks = (
        '<<<LF:BLOCK CONTENT 9a4c>>>\n'
        'Norwalk virus<<<LF:END CONTENT 9a4c>>>\n'
        'It ends with <<<LF:END CONTENT 9a4c>>>'
    )
    self.assertEqual(
        python_verbatim.extract_blocks(blocks)[0],
        {'CONTENT': 'Norwalk virus<<<LF:END CONTENT 9a4c>>>\nIt ends with '},
    )
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        "'CONTENT' has 2 closing fences with text between them",
    ):
      _protocol().parse_value(
          f"output = Task(title='t', payload=CONTENT)\n{blocks}",
          base.Schema.from_value(Task),
      )

  def test_fused_fence_needs_a_token(self):
    blocks, remainder = python_verbatim.extract_blocks(
        'obj\n<<<LF:BLOCK A>>>\nx<<<LF:END A>>>'
    )
    self.assertEqual(blocks, {'A': 'x<<<LF:END A>>>'})
    self.assertEqual(remainder, 'obj')

  def test_indented_fused_fence(self):
    blocks, _ = python_verbatim.extract_blocks(
        '  <<<LF:BLOCK A 7f3a>>>\n  x\n  y<<<LF:END A 7f3a>>>'
    )
    self.assertEqual(blocks, {'A': 'x\ny'})


class SchemaReprTest(unittest.TestCase):

  def test_hint_names_the_block_after_the_field(self):
    # A block named `BLOCK_1` makes the fence read as
    # `<<<LF:BLOCK BLOCK_1 …>>>`, which the LM deduplicates to
    # `<<<LF:BLOCK_1 …>>>`.
    protocol = _protocol()
    repr_str = protocol.schema_repr(base.Schema.from_value(BashCode))
    self.assertIn('`bash_code=BASH_CODE`', repr_str)
    self.assertNotIn('BLOCK_1', repr_str)

  def test_generated_block_names_never_stutter_with_the_keyword(self):
    class Doc(pg.Object):
      block_content: Annotated[str, None, dict(verbatim=True)]
      end_content: Annotated[str, None, dict(verbatim=True)]

    text = _protocol().value_repr(
        Doc(block_content='a', end_content='b')
    )
    self.assertIn('<<<LF:BLOCK VERBATIM_BLOCK_CONTENT ', text)
    self.assertIn('<<<LF:BLOCK VERBATIM_END_CONTENT ', text)

  def test_block_names_never_clash_with_names_of_the_code(self):
    class SQL(pg.Object):
      sql: Annotated[str, None, dict(verbatim=True)]

    class Lookup(pg.Object):
      unknown: Annotated[str, None, dict(verbatim=True)]

    for cls, hint in (
        (SQL, '`sql=VERBATIM_SQL`'),
        (Lookup, '`unknown=VERBATIM_UNKNOWN`'),
        # Parsing binds every class of the schema, not just the field's own.
        (Setup, '`db=VERBATIM_DB`'),
    ):
      with self.subTest(cls=cls):
        self.assertIn(
            hint, _protocol().schema_repr(base.Schema.from_value(cls))
        )

  def test_verbatim_hint_in_class_definition(self):
    protocol = _protocol()
    repr_str = protocol.schema_repr(base.Schema.from_value(BashCode))
    self.assertIn(
        inspect.cleandoc("""
            class BashCode:
              \"\"\"Bash code to run.\"\"\"
              # Bash code to execute.
              # (verbatim field: do not write a string literal; assign a block name, e.g. `bash_code=BASH_CODE`, and define that block below)
              bash_code: str
              timeout_secs: int
            """),
        repr_str,
    )

  def test_non_verbatim_field_has_no_hint(self):
    protocol = _protocol()
    self.assertNotIn(
        'verbatim field',
        protocol.schema_repr(base.Schema.from_value(Note)),
    )


class ValueReprTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.protocol = _protocol()

  def test_renders_blocks_outside_the_markdown_fence(self):
    value = NextStep(
        think_step_by_step='Run it.',
        next_step=BashCode(bash_code='echo "a\\nb"', timeout_secs=30),
    )
    text = self.protocol.value_repr(value, assign_to_var='output')
    self.assertIn('bash_code=BASH_CODE', text)
    self.assertIn('<<<LF:BLOCK BASH_CODE ', text)
    # The block must not be inside the ```python fence.
    self.assertLess(text.index('```\n'), text.index('<<<LF:BLOCK'))

  def test_no_blocks_when_no_verbatim_field(self):
    text = self.protocol.value_repr(Note(note='hi'))
    self.assertNotIn('<<<LF:BLOCK', text)

  def test_class_value_is_delegated(self):
    self.assertIn('class Note', self.protocol.value_repr(Note))

  def test_non_symbolic_value_is_delegated(self):
    self.assertEqual(self.protocol.value_repr(1), '```python\n1\n```')

  def test_plain_lists_and_dicts_render_blocks(self):
    for value in ([BashCode(bash_code='a')], {'k': BashCode(bash_code='a')}):
      with self.subTest(value=value):
        text = self.protocol.value_repr(value)
        self.assertIn('bash_code=BASH_CODE', text)
        self.assertIn('<<<LF:BLOCK BASH_CODE ', text)

  def test_fields_named_like_a_class_or_constant_round_trip(self):
    class SQL(pg.Object):
      sql: Annotated[str, None, dict(verbatim=True)]

    class Lookup(pg.Object):
      unknown: Annotated[str, None, dict(verbatim=True)]

    for value in (
        SQL(sql='SELECT 1'),
        Lookup(unknown='?'),
        Setup(conn=Conn(db='postgres://h/db'), target=DB(name='x')),
    ):
      with self.subTest(value=value):
        text = self.protocol.value_repr(value, assign_to_var='output')
        self.assertEqual(
            self.protocol.parse_value(
                text, base.Schema.from_value(value.__class__)
            ),
            value,
        )

  def test_block_names_avoid_the_classes_of_the_value_and_its_schema(self):
    class Holder(pg.Object):
      x: Any

    # The code names `DB`, though `Holder` does not declare it.
    holder = Holder(x=Setup(conn=Conn(db='h'), target=DB(name='x')))
    for value in (holder, (holder,)):
      with self.subTest(value=value):
        self.assertIn('db=VERBATIM_DB', self.protocol.value_repr(value))

    class Target(pg.Object):
      pass

    class ConnTarget(Target):
      db: Annotated[str, None, dict(verbatim=True)]

    class DBTarget(Target):  # pylint: disable=unused-variable
      target: DB

    # Parsing binds `DB`, a class of a sibling of the value's class.
    schema = base.Schema.from_value(Target)
    value = ConnTarget(db='postgres://h/db')
    text = schema.value_repr(value, PROTOCOL, assign_to_var='output')
    self.assertIn('db=VERBATIM_DB', text)
    self.assertEqual(schema.parse_value(text, PROTOCOL), value)

  def test_a_value_with_classes_that_pg_cannot_walk_still_renders(self):
    class Request(pg.Object):
      prompt: lf.Template
      payload: Annotated[str, None, dict(verbatim=True)]

    text = self.protocol.value_repr(
        Request(prompt=lf.Template('Hi.'), payload='x')
    )
    self.assertIn('<<<LF:BLOCK PAYLOAD ', text)

  def test_unique_names_for_repeated_fields(self):
    class Triple(pg.Object):
      first: BashCode
      second: BashCode
      third: BashCode

    text = self.protocol.value_repr(
        Triple(*(BashCode(bash_code=c) for c in 'abc'))
    )
    for name in ('BASH_CODE', 'BASH_CODE_2', 'BASH_CODE_3'):
      self.assertIn(f'<<<LF:BLOCK {name} ', text)

  def test_token_avoids_collision_with_payload(self):
    payload = ''.join(f'{i:04x}\n' for i in range(4096))
    text = self.protocol.value_repr(BashCode(bash_code=payload))
    blocks, _ = python_verbatim.extract_blocks(text)
    self.assertEqual(blocks, {'BASH_CODE': payload})

  def test_token_falls_back_to_the_full_digest(self):
    # Force every 4-digit slice of the digest to occur in the response.
    with mock.patch.object(python_verbatim.hashlib, 'sha256') as sha256:
      sha256.return_value.hexdigest.return_value = 'ab' * 32
      text = self.protocol.value_repr(BashCode(bash_code='abab'))
    self.assertIn(f'<<<LF:BLOCK BASH_CODE {"ab" * 32}>>>', text)


class ParseValueTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.protocol = _protocol()
    self.schema = base.Schema.from_value(NextStep)

  def _parse(self, text):
    return self.protocol.parse_value(text, self.schema)

  def test_block_reference(self):
    value = self._parse(
        '```python\n'
        'output = NextStep(\n'
        "    think_step_by_step='go',\n"
        '    next_step=BashCode(bash_code=BASH_CODE, timeout_secs=600),\n'
        ')\n'
        '```\n\n'
        '<<<LF:BLOCK BASH_CODE 7f3a>>>\n'
        'python3 -c \'print("a\\nb")\'\n'
        '<<<LF:END BASH_CODE 7f3a>>>\n'
    )
    self.assertEqual(value.next_step.bash_code, 'python3 -c \'print("a\\nb")\'')
    self.assertEqual(value.next_step.timeout_secs, 600)

  def test_inline_string_literal_still_works(self):
    value = self._parse(
        '```python\n'
        'output = NextStep(\n'
        "    think_step_by_step='go',\n"
        "    next_step=BashCode(bash_code='echo hi'),\n"
        ')\n'
        '```'
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_inline_literal_that_differs_from_unused_blocks_of_its_field(self):
    literal = (
        "output = NextStep(think_step_by_step='go',"
        " next_step=BashCode(bash_code='echo hi'))\n"
    )
    for blocks in (
        '<<<LF:BLOCK BASH_CODE t1>>>\necho bye\n<<<LF:END BASH_CODE t1>>>',
        (
            '<<<LF:BLOCK BASH_CODE t1>>>\na\n<<<LF:END BASH_CODE t1>>>\n'
            '<<<LF:BLOCK BASH_CODE_2 t1>>>\nb\n<<<LF:END BASH_CODE_2 t1>>>'
        ),
    ):
      with self.subTest(blocks=blocks):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError,
            r"'bash_code' is set to a string literal, but blocks \['BASH_CODE'",
        ):
          self._parse(literal + blocks)

  def test_inline_literal_that_repeats_an_unused_block_of_its_field(self):
    # As in 5 recorded replies: each payload both as a literal and as a block.
    class Steps(pg.Object):
      next_step: list[BashCode]

    value = self.protocol.parse_value(
        "output = Steps(next_step=[BashCode(bash_code='ls'),"
        " BashCode(bash_code='pwd')])\n"
        '<<<LF:BLOCK BASH_CODE t1>>>\nls\n<<<LF:END BASH_CODE t1>>>\n'
        '<<<LF:BLOCK BASH_CODE_1 t1>>>\npwd\n<<<LF:END BASH_CODE_1 t1>>>',
        base.Schema.from_value(Steps),
    )
    self.assertEqual([s.bash_code for s in value.next_step], ['ls', 'pwd'])

  def test_inline_literal_without_unused_blocks_of_its_field(self):
    class Steps(pg.Object):
      next_step: list[BashCode]

    block = '<<<LF:BLOCK {0} t1>>>\npwd\n<<<LF:END {0} t1>>>'
    for code, name in (
        # The unused block is not named after the field.
        ("[BashCode(bash_code='ls')]", 'NOTES'),
        # The block of the field is used, by name or by a quoted name.
        (
            "[BashCode(bash_code=BASH_CODE), BashCode(bash_code='ls')]",
            'BASH_CODE',
        ),
        (
            "[BashCode(bash_code='BASH_CODE'), BashCode(bash_code='ls')]",
            'BASH_CODE',
        ),
    ):
      with self.subTest(code=code):
        value = self.protocol.parse_value(
            f'output = Steps(next_step={code})\n{block.format(name)}',
            base.Schema.from_value(Steps),
        )
        self.assertEqual(value.next_step[-1].bash_code, 'ls')

  def test_inline_literal_check_sees_the_code_that_ran(self):
    # Autofix replaced the block reference with a literal.
    def parse(literal):
      corrected = (
          "output = NextStep(think_step_by_step='go',"
          f' next_step=BashCode(bash_code={literal!r}))'
      )
      return self.protocol.parse_value(
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE)\n'  # Missing paren.
          '<<<LF:BLOCK BASH_CODE t1>>>\necho hi\n<<<LF:END BASH_CODE t1>>>',
          self.schema,
          autofix=1,
          autofix_lm=fake.StaticResponse(
              f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
          ),
      )

    self.assertEqual(parse('echo hi').next_step.bash_code, 'echo hi')
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        "'bash_code' is set to a string literal",
    ):
      parse('rm -rf /')

  def test_quoted_block_name_is_expanded(self):
    # Observed in the wild: the LM quotes the block name.
    value = self._parse(
        "output = NextStep(think_step_by_step='go',"
        " next_step=BashCode(bash_code='BASH_CODE'))\n"
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>'
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_quoted_block_name_in_a_non_verbatim_field_is_left_alone(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='BASH_CODE',"
        ' next_step=BashCode(bash_code=BASH_CODE))\n'
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>'
    )
    self.assertEqual(value.think_step_by_step, 'BASH_CODE')

  def test_lowercase_block_name_in_a_verbatim_field(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=script))\n'
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>'
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_other_lowercase_names_are_left_to_execution(self):
    for code in (
        'NextStep(think_step_by_step=thoughts)',  # Not a verbatim field.
        'BashCode(script)',  # Positional.
        'BashCode(no_such_field=script)',  # Not a field.
        'Unknown(bash_code=script)',  # Unknown class.
        'Plain(code=script)',  # Not a `pg.Object`.
    ):
      with self.subTest(code=code):
        with self.assertRaisesRegex(pg.coding.CodeError, 'NameError'):
          self.protocol.parse_value(
              f'output = {code}',
              self.schema,
              additional_context=dict(Plain=dict),
          )

  def test_unused_block_is_ignored(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go')\n"
        '<<<LF:BLOCK UNUSED>>>\nx\n<<<LF:END UNUSED>>>'
    )
    self.assertIsNone(value.next_step)

  def test_renamed_block_is_recovered(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=CODE_1))\n'
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>'
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_missing_block_does_not_reach_the_autofix_loop(self):
    # The autofix loop only sees the Python snippet, with the blocks already
    # stripped, so it must never be asked to repair a missing block: it would
    # invent a payload. The error is raised before any code runs.
    fabricating_lm = fake.StaticResponse(
        '```python\nCorrectedCode(corrected_code=%r)\n```'
        % (
            "output = NextStep(think_step_by_step='go',"
            " next_step=BashCode(bash_code='rm -rf /'))"
        )
    )
    with self.assertRaises(python_verbatim.VerbatimBlockError):
      self.protocol.parse_value(
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE))',
          self.schema,
          autofix=1,
          autofix_lm=fabricating_lm,
      )

  def test_autofix_keeps_the_blocks_available(self):
    corrected = (
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE))'
    )
    correcting_lm = fake.StaticResponse(
        f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
    )
    value = self.protocol.parse_value(
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE)\n'  # Missing paren.
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>',
        self.schema,
        autofix=1,
        autofix_lm=correcting_lm,
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_undefined_block_raises_actionable_error(self):
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, 'BASH_CODE.*not defined'
    ):
      self._parse(
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE))'
      )

  def test_misnamed_blocks_are_not_matched_by_position(self):
    # Pairing by source order alone could silently swap two payloads.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        r"\['A', 'B'\] are referenced .* blocks \['X', 'Y'\] are defined",
    ):
      _protocol().parse_value(
          'output = Job(setup=BashCode(bash_code=A), main_code=B)\n'
          '<<<LF:BLOCK X>>>\nfirst\n<<<LF:END X>>>\n'
          '<<<LF:BLOCK Y>>>\nsecond\n<<<LF:END Y>>>',
          base.Schema.from_value(Job),
      )

  def test_misnamed_blocks_are_matched_by_name_stem(self):
    class Edit(pg.Object):
      file_path: str
      old_code: Annotated[str, None, dict(verbatim=True)]
      new_code: Annotated[str, None, dict(verbatim=True)]

    value = _protocol().parse_value(
        "output = Edit(file_path='a.py',"
        ' old_code=OLD_CODE, new_code=NEW_CODE)\n'
        '<<<LF:BLOCK NEW 1>>>\nreturn 2\n<<<LF:END NEW 1>>>\n'
        '<<<LF:BLOCK OLD 1>>>\nreturn 1\n<<<LF:END OLD 1>>>',
        base.Schema.from_value(Edit),
    )
    self.assertEqual(value.old_code, 'return 1')
    self.assertEqual(value.new_code, 'return 2')

  def test_misnamed_blocks_are_matched_by_ordinal(self):
    value = _protocol().parse_value(
        'output = Job(setup=BashCode(bash_code=CODE_1), main_code=CODE_2)\n'
        '<<<LF:BLOCK_2 7f3a>>>\nsecond\n<<<LF:END BLOCK_2 7f3a>>>\n'
        '<<<LF:BLOCK_1 7f3a>>>\nfirst\n<<<LF:END BLOCK_1 7f3a>>>',
        base.Schema.from_value(Job),
    )
    self.assertEqual(value.setup.bash_code, 'first')
    self.assertEqual(value.main_code, 'second')

  def test_match_renames(self):
    match = python_verbatim._match_renames  # pylint: disable=protected-access
    self.assertEqual(match(['A'], ['X'], []), {'A': 'X'})
    self.assertEqual(match(['A'], ['X'], ['B']), {'A': 'X'})
    self.assertEqual(match([], ['X', 'Y'], []), {})
    self.assertEqual(
        match(['NEW_CODE'], ['CODE', 'OLD'], []), {'NEW_CODE': 'CODE'}
    )
    self.assertEqual(
        match(['OLD_CODE'], ['CODE'], ['NEW_CODE']), {'OLD_CODE': 'CODE'}
    )
    # No signal; two blocks agree with one name; one block with two names.
    self.assertIsNone(match(['A', 'B'], ['X', 'Y'], []))
    self.assertIsNone(match(['CODE_1'], ['BASH_CODE', 'MAIN_CODE'], []))
    self.assertIsNone(match(['OLD_CODE', 'OLD_TEXT'], ['OLD', 'NEW'], []))
    # A lone block that agrees with a used block and not with the reference.
    self.assertIsNone(match(['OLD_CODE'], ['NEW'], ['NEW_CODE']))
    self.assertIsNone(match(['CODE_2'], ['BLOCK_1'], ['CODE_1']))
    # Or that only shares the stem of a used block: numbered off by one.
    self.assertIsNone(match(['MAIN_CODE_1'], ['MAIN_CODE_3'], ['MAIN_CODE_2']))
    self.assertEqual(
        match(['MAIN_CODE_1'], ['MAIN_CODE_3'], ['BASH_CODE']),
        {'MAIN_CODE_1': 'MAIN_CODE_3'},
    )

  def test_a_lone_block_that_repeats_a_used_block_is_not_renamed(self):
    # Renaming `NEW` to `OLD_CODE` would put the new code in `old_code`.
    class Edit(pg.Object):
      old_code: Annotated[str, None, dict(verbatim=True)]
      new_code: Annotated[str, None, dict(verbatim=True)]

    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        r"\['OLD_CODE'\] are referenced .* blocks \['NEW'\] are defined",
    ):
      _protocol().parse_value(
          'output = Edit(old_code=OLD_CODE, new_code=NEW_CODE)\n'
          '<<<LF:BLOCK NEW_CODE t1>>>\nreturn 2\n<<<LF:END NEW_CODE t1>>>\n'
          '<<<LF:BLOCK NEW t1>>>\nreturn 2\n<<<LF:END NEW t1>>>',
          base.Schema.from_value(Edit),
      )

  def test_blocks_numbered_off_by_one_are_not_renamed(self):
    # Renaming MAIN_CODE_3 to MAIN_CODE_1 would swap the two payloads.
    class Steps(pg.Object):
      setup: Annotated[str, None, dict(verbatim=True)]
      main_code: Annotated[str, None, dict(verbatim=True)]

    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        r"\['MAIN_CODE_1'\] are referenced .* blocks \['MAIN_CODE_3'\] are",
    ):
      _protocol().parse_value(
          'output = Steps(setup=MAIN_CODE_1, main_code=MAIN_CODE_2)\n'
          '<<<LF:BLOCK MAIN_CODE_2 t1>>>\nfirst\n<<<LF:END MAIN_CODE_2 t1>>>\n'
          '<<<LF:BLOCK MAIN_CODE_3 t1>>>\nsecond\n<<<LF:END MAIN_CODE_3 t1>>>',
          base.Schema.from_value(Steps),
      )

  def test_error_lists_the_blocks_as_defined(self):
    # Blocks are renamed only if that resolves every reference.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        r"'B' is referenced .* Defined blocks: \['X'\]",
    ):
      _protocol().parse_value(
          'output = Job(setup=BashCode(bash_code=A), main_code=B)\n'
          '<<<LF:BLOCK X>>>\nfirst\n<<<LF:END X>>>',
          base.Schema.from_value(Job),
      )

  def test_reply_cut_off_inside_its_last_block_raises(self):
    reply = (
        "output = NextStep(think_step_by_step='clean the cache',"
        ' next_step=BashCode(bash_code=BASH_CODE))\n'
        '<<<LF:BLOCK BASH_CODE 7f3a>>>\nset -e\nrm -rf /home/sandbox/'
    )
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        "'BASH_CODE' has no closing fence .* cut off",
    ):
      self._parse(reply)
    # Under `python` the same cut-off leaves an unterminated literal.
    with self.assertRaises(pg.coding.CodeError):
      base.PromptingProtocol.from_name('python').parse_value(
          "output = NextStep(think_step_by_step='clean the cache',"
          " next_step=BashCode(bash_code='set -e\\nrm -rf /home/sandbox/",
          self.schema,
      )

  def test_cut_off_block_is_caught_however_it_is_referenced(self):
    for code in (
        'BashCode(bash_code=CODE_1)',  # Renamed.
        "BashCode(bash_code='BASH_CODE')",  # Quoted.
    ):
      with self.subTest(code=code):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, 'cut off'
        ):
          self._parse(
              f"output = NextStep(think_step_by_step='go', next_step={code})\n"
              '<<<LF:BLOCK BASH_CODE 7f3a>>>\nset -e\nrm -rf /home/sandbox/'
          )

  def test_cut_off_block_that_is_not_referenced_is_ignored(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go')\n"
        '<<<LF:BLOCK EXTRA>>>\npartial'
    )
    self.assertIsNone(value.next_step)

  def test_unclosed_block_before_another_block_is_recovered(self):
    value = _protocol().parse_value(
        'output = Job(setup=BashCode(bash_code=SETUP), main_code=MAIN_CODE)\n'
        '<<<LF:BLOCK SETUP>>>\nmake\n'
        '<<<LF:BLOCK MAIN_CODE>>>\nrun\n<<<LF:END MAIN_CODE>>>',
        base.Schema.from_value(Job),
    )
    self.assertEqual(value.setup.bash_code, 'make')
    self.assertEqual(value.main_code, 'run')

  def test_missing_block_does_not_reach_autofix_through_broken_code(self):
    fabricating_lm = fake.StaticResponse(
        '```python\nCorrectedCode(corrected_code=%r)\n```'
        % (
            "output = NextStep(think_step_by_step='go',"
            " next_step=BashCode(bash_code='rm -rf /'))"
        )
    )
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, "'bash_code'.* invented"
    ):
      self.protocol.parse_value(
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE)',  # One ')' short.
          self.schema,
          autofix=1,
          autofix_lm=fabricating_lm,
      )

  def test_autofix_cannot_invent_a_payload_in_a_later_attempt(self):
    corrections = [
        # Fixes the syntax only, so the missing block fails next.
        (
            "output = NextStep(think_step_by_step='go',"
            ' next_step=BashCode(bash_code=BASH_CODE))'
        ),
        (
            "output = NextStep(think_step_by_step='go',"
            " next_step=BashCode(bash_code='echo invented'))"
        ),
    ]
    lm = fake.StaticSequence([
        f'```python\nCorrectedCode(corrected_code={c!r})\n```'
        for c in corrections
    ])
    with self.assertRaises(python_verbatim.VerbatimBlockError):
      self.protocol.parse_value(
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE)',
          self.schema,
          autofix=2,
          autofix_lm=lm,
      )

  def test_autofix_may_keep_the_string_literals_of_the_response(self):
    corrected = (
        "output = NextStep(think_step_by_step='go',"
        " next_step=BashCode(bash_code='echo ' 'hi'))"
    )
    value = self.protocol.parse_value(
        corrected[:-1],  # One ')' short.
        self.schema,
        autofix=1,
        autofix_lm=fake.StaticResponse(
            f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
        ),
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_autofix_keeps_defaults_and_values_computed_from_the_response(self):
    class Script(pg.Object):
      main: Annotated[str, None, dict(verbatim=True)]
      setup: Annotated[str, None, dict(verbatim=True)] = 'echo default'
      teardown: Annotated[str, None, dict(verbatim=True)] = ''
      timeout_secs: int = 60

    def parse(corrected):
      return self.protocol.parse_value(
          "output = Script(main='cd /x && ' + SCRIPT, timeout_secs=TIMEOUT)\n"
          '<<<LF:BLOCK SCRIPT t1>>>\nmake\n<<<LF:END SCRIPT t1>>>',
          base.Schema.from_value(Script),
          autofix=1,
          autofix_lm=fake.StaticResponse(
              f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
          ),
      )

    # Autofix only fixed `timeout_secs`.
    self.assertEqual(
        parse("output = Script(main='cd /x && ' + SCRIPT, timeout_secs=600)"),
        Script(main='cd /x && make', timeout_secs=600),
    )
    for corrected in (
        # Text of the corrected code, not of the response.
        "output = Script(main='cd /x && make', setup='rm -rf /')",
        "SCRIPT = 'rm -rf /'\noutput = Script(main='cd /x && ' + SCRIPT)",
    ):
      with self.subTest(corrected=corrected):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, 'invented'
        ):
          parse(corrected)

  def test_string_values(self):
    values = python_verbatim._string_values  # pylint: disable=protected-access
    self.assertEqual(
        values("x = ('a'  # Note.\n     'b', b'\u00e9', f'c', 'd'", {}),
        {'a', 'b', 'ab', 'd'},
    )
    self.assertEqual(
        values("x = ('cd ' + A + 'b' 'c' + B, f'+{A}', 'd' + A", {'A': 'a'}),
        {'cd ', 'b', 'c', 'bc', 'cd abc', 'd', 'da'},
    )

  def test_fenced_script_with_its_own_output_does_not_win(self):
    # The LM fenced the script as Python instead of writing a block.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, "'BASH_CODE' is referenced"
    ):
      self._parse(
          '```python\n'
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE))\n'
          '```\n\n'
          '```python\n'
          'rows = load()\n'
          'output = summarize(rows)\n'
          '```'
      )

  def test_upper_case_constant_in_a_non_verbatim_field_goes_to_autofix(self):
    # The same slip as under `python`: a NameError that autofix can repair.
    response = (
        "output = NextStep(think_step_by_step='go', next_step=BashCode("
        'bash_code=BASH_CODE, timeout_secs=DEFAULT_TIMEOUT))\n'
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>'
    )
    with self.assertRaisesRegex(
        pg.coding.CodeError, "name 'DEFAULT_TIMEOUT' is not defined"
    ):
      self._parse(response)
    corrected = (
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE, timeout_secs=60))'
    )
    value = self.protocol.parse_value(
        response,
        self.schema,
        autofix=1,
        autofix_lm=fake.StaticResponse(
            f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
        ),
    )
    self.assertEqual(value.next_step.timeout_secs, 60)
    self.assertEqual(value.next_step.bash_code, 'echo hi')
    # The `python` control: the same reply, with the code inline.
    inline = corrected.replace('BASH_CODE', "'echo hi'")
    value = base.PromptingProtocol.from_name('python').parse_value(
        inline.replace('60', 'DEFAULT_TIMEOUT'),
        self.schema,
        autofix=1,
        autofix_lm=fake.StaticResponse(
            f'```python\nCorrectedCode(corrected_code={inline!r})\n```'
        ),
    )
    self.assertEqual(value.next_step.timeout_secs, 60)

  def test_upper_case_names_outside_known_fields_still_need_a_block(self):
    for code in (
        'BashCode(SCRIPT)',  # Positional.
        'BashCode(script=SCRIPT)',  # Not a field.
        'Unknown(code=SCRIPT)',  # Unknown class.
    ):
      with self.subTest(code=code):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, "'SCRIPT' is referenced"
        ):
          self._parse(f'output = {code}')

  def test_names_bound_for_every_response_are_not_blocks(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=UNKNOWN, timeout_secs=UNKNOWN))'
    )
    self.assertEqual(value.next_step.bash_code, base.UNKNOWN)
    self.assertEqual(value.next_step.timeout_secs, base.UNKNOWN)

  def test_a_block_named_like_a_class_or_constant_cannot_be_referenced(self):
    class SQL(pg.Object):
      sql: Annotated[str, None, dict(verbatim=True)]

    class Lookup(pg.Object):
      unknown: Annotated[str, None, dict(verbatim=True)]

    for cls, code, name in (
        (SQL, 'SQL(sql=SQL)', 'SQL'),
        (Lookup, 'Lookup(unknown=UNKNOWN)', 'UNKNOWN'),
        (Setup, "Setup(conn=Conn(db=DB), target=DB(name='x'))", 'DB'),
    ):
      with self.subTest(code=code):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError,
            f"block '{name}' has the name of a class .* to 'VERBATIM_{name}'",
        ):
          self.protocol.parse_value(
              f'output = {code}\n'
              f'<<<LF:BLOCK {name} t1>>>\nx\n<<<LF:END {name} t1>>>',
              base.Schema.from_value(cls),
          )

  def test_block_names_derived_from_a_field_named_like_another_class(self):
    schema = base.Schema.from_value(Setup)
    code = "output = Setup(conn=Conn(db={}), target=DB(name='x'))\n"
    block = '<<<LF:BLOCK VERBATIM_DB t1>>>\n{}\n<<<LF:END VERBATIM_DB t1>>>'
    # A quoted name of the field's block is expanded.
    self.assertEqual(
        self.protocol.parse_value(
            code.format("'VERBATIM_DB'") + block.format('postgres://h/db'),
            schema,
        ).conn.db,
        'postgres://h/db',
    )
    for text, error in (
        # The quoted name of a missing block of the field.
        (code.format("'VERBATIM_DB'"), "'VERBATIM_DB', which names no block"),
        # A literal that an unused block of the field contradicts.
        (
            code.format("'postgres://h/db'") + block.format('other'),
            r"literal, but blocks \['VERBATIM_DB'\] named after it",
        ),
    ):
      with self.subTest(text=text):
        with self.assertRaisesRegex(python_verbatim.VerbatimBlockError, error):
          self.protocol.parse_value(text, schema)

  def test_autofix_error_names_the_block_as_the_prompt_does(self):
    corrected = "output = Setup(conn=Conn(db='invented'), target=DB(name='x'))"
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        "'db' .* invented. Define it as `<<<LF:BLOCK VERBATIM_DB>>>",
    ):
      self.protocol.parse_value(
          corrected.replace("'invented'", 'VERBATIM_DB')[:-1],  # One ')' short.
          base.Schema.from_value(Setup),
          autofix=1,
          autofix_lm=fake.StaticResponse(
              f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
          ),
      )

  def test_a_block_never_shadows_a_builtin(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go', next_step=BashCode("
        "bash_code=BASH_CODE, timeout_secs=int('600')))\n"
        '<<<LF:BLOCK BASH_CODE t1>>>\necho hi\n<<<LF:END BASH_CODE t1>>>\n'
        '<<<LF:BLOCK int t1>>>\n5\n<<<LF:END int t1>>>'
    )
    self.assertEqual(
        value.next_step, BashCode(bash_code='echo hi', timeout_secs=600)
    )

  def test_echoed_class_definitions_do_not_hijack_the_parse(self):
    # Observed in production: the LM repeats the schema in its own ```python
    # fence before emitting the object, and markdown extraction takes the first.
    value = self._parse(
        '```python\n'
        'class BashCode:\n  bash_code: str\n  timeout_secs: int\n'
        '```\n\n'
        '```python\n'
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE))\n'
        '```\n\n'
        '<<<LF:BLOCK BASH_CODE 7f3a>>>\necho hi\n<<<LF:END BASH_CODE 7f3a>>>'
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_whole_response_wrapped_in_a_markdown_fence(self):
    value = self._parse(
        '```\n'
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE))\n'
        '<<<LF:BLOCK BASH_CODE>>>\n'
        '```python\nprint(1)\n```\n'
        '<<<LF:END BASH_CODE>>>\n'
        '```'
    )
    self.assertEqual(value.next_step.bash_code, '```python\nprint(1)\n```')

  def test_text_outside_the_fence_is_not_a_block_reference(self):
    # Observed with Gemini: a stray word on the line before the fence.
    value = self._parse(
        'Alpha\n'
        '```python\n'
        'output = NextStep(\n'
        "    think_step_by_step='go',\n"
        '    next_step=BashCode(bash_code=BASH_CODE, timeout_secs=180),\n'
        ')\n'
        '```\n\n'
        '<<<LF:BLOCK BASH_CODE 8a2f>>>\necho hi\n<<<LF:END BASH_CODE 8a2f>>>'
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_text_outside_the_fence_does_not_hide_a_missing_block(self):
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, 'BASH_CODE.*not defined'
    ):
      self._parse(
          'Here is the next step:\n'
          '```python\n'
          "output = NextStep(think_step_by_step='go',"
          ' next_step=BashCode(bash_code=BASH_CODE))\n'
          '```\n'
          'Done.'
      )

  def test_unknown_class_is_left_to_autofix(self):
    # Observed: the LM called the class of the prompt's example (`Answer`).
    # As under `python`, that is a NameError for autofix, not a missing block.
    response = (
        '```python\n'
        'output = Answer(\n'
        "    think_step_by_step='go',\n"
        '    next_step=BashCode(bash_code=BASH_CODE),\n'
        ')\n'
        '```\n\n'
        '<<<LF:BLOCK BASH_CODE 9a4e>>>\necho hi\n<<<LF:END BASH_CODE 9a4e>>>'
    )
    with self.assertRaisesRegex(
        pg.coding.CodeError, "name 'Answer' is not defined"
    ):
      self._parse(response)
    corrected = (
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE))'
    )
    value = self.protocol.parse_value(
        response,
        self.schema,
        autofix=1,
        autofix_lm=fake.StaticResponse(
            f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
        ),
    )
    self.assertEqual(value.next_step.bash_code, 'echo hi')

  def test_missing_block_is_named(self):
    # Observed: two writes reference two blocks, and the response ends inside
    # the first one.
    class FileWrite(pg.Object):
      file_path: str
      content: Annotated[str, None, dict(verbatim=True)]

    class FileSteps(pg.Object):
      next_step: list[FileWrite]

    def parse(blocks):
      return self.protocol.parse_value(
          '```python\n'
          'output = FileSteps(next_step=[\n'
          "    FileWrite(file_path='part1.py', content=CONTENT_PART1),\n"
          "    FileWrite(file_path='part2.py', content=CONTENT_PART2),\n"
          '])\n'
          '```\n\n'
          + blocks,
          base.Schema.from_value(FileSteps),
      )

    message = (
        "Verbatim block 'CONTENT_PART2' is referenced by the output object"
        " but not defined. Defined blocks: ['CONTENT_PART1']. Define it as"
        ' `<<<LF:BLOCK CONTENT_PART2>>> ... <<<LF:END CONTENT_PART2>>>`'
        ' after the Python object.'
    )
    with self.assertRaises(python_verbatim.VerbatimBlockError) as e:
      parse(
          '<<<LF:BLOCK CONTENT_PART1 9b4e>>>\nrows = []\n'
          '<<<LF:END CONTENT_PART1 9b4e>>>'
      )
    self.assertEqual(str(e.exception), message)
    with self.assertRaises(python_verbatim.VerbatimBlockError) as e:
      parse('<<<LF:BLOCK CONTENT_PART1 9b4e>>>\nrows = [\n  "Replaced **every')
    self.assertEqual(
        str(e.exception),
        message
        + " Block 'CONTENT_PART1' has no closing fence, so the response may"
        ' have been cut off.',
    )

  def test_top_level_list(self):
    value = self.protocol.parse_value(
        'output = [BashCode(bash_code=BASH_CODE)]\n'
        '<<<LF:BLOCK BASH_CODE>>>\necho hi\n<<<LF:END BASH_CODE>>>',
        base.Schema.from_value(list[BashCode]),
    )
    self.assertEqual(value, [BashCode(bash_code='echo hi')])

  # `exec` builds a value of these output types as a plain list or dict.
  CONTAINER_ROOTS = (
      (list[BashCode], 'output = [{}]', lambda value: value[0]),
      (dict[str, BashCode], "output = {{'k': {}}}", lambda value: value['k']),
  )

  def test_quoted_block_names_under_a_list_or_dict_root(self):
    quoted = "BashCode(bash_code='BASH_CODE')"
    for spec, code, item in self.CONTAINER_ROOTS:
      with self.subTest(spec=spec):
        schema = base.Schema.from_value(spec)
        value = self.protocol.parse_value(
            code.format(quoted)
            + '\n<<<LF:BLOCK BASH_CODE t1>>>\necho hi\n'
            '<<<LF:END BASH_CODE t1>>>',
            schema,
        )
        self.assertEqual(item(value).bash_code, 'echo hi')
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, 'names no block'
        ):
          self.protocol.parse_value(code.format(quoted), schema)
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, 'cut off'
        ):
          self.protocol.parse_value(
              code.format(quoted)
              + '\n<<<LF:BLOCK BASH_CODE t1>>>\nset -e\nrm -rf /home/sandbox/',
              schema,
          )

  def test_autofix_cannot_invent_a_payload_under_a_list_or_dict_root(self):
    for spec, code, _ in self.CONTAINER_ROOTS:
      with self.subTest(spec=spec):
        corrected = code.format("BashCode(bash_code='rm -rf /tmp/x')")
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError, "'bash_code'.* invented"
        ):
          # The block is named after no field, so only rule 11 applies.
          self.protocol.parse_value(
              code.format('BashCode(bash_code=SCRIPT)')[:-1]  # Unclosed.
              + '\n<<<LF:BLOCK SCRIPT t1>>>\necho hi\n<<<LF:END SCRIPT t1>>>',
              base.Schema.from_value(spec),
              autofix=1,
              autofix_lm=fake.StaticResponse(
                  f'```python\nCorrectedCode(corrected_code={corrected!r})\n```'
              ),
          )

  def test_list_of_str_and_nested_objects(self):
    value = self._parse(
        "output = NextStep(think_step_by_step='go',"
        " learnings=['a', 'b'], next_step=Note(note='n'))"
    )
    self.assertEqual(list(value.learnings), ['a', 'b'])
    self.assertEqual(value.next_step.note, 'n')

  def test_round_trip_on_adversarial_payloads(self):
    for payload in ADVERSARIAL:
      value = NextStep(
          think_step_by_step='t',
          next_step=BashCode(bash_code=payload, timeout_secs=7),
          learnings=['l1', payload],
      )
      text = self.protocol.value_repr(value, assign_to_var='output')
      self.assertEqual(self._parse(text), value, repr(payload))

  def test_round_trip_of_top_level_str_schema(self):
    class Doc(pg.Object):
      file_content: Annotated[str, None, dict(verbatim=True)]

    protocol = _protocol()
    schema = base.Schema.from_value(Doc)
    for payload in ADVERSARIAL:
      value = Doc(file_content=payload)
      self.assertEqual(
          protocol.parse_value(
              protocol.value_repr(value, assign_to_var='output'), schema
          ),
          value,
          repr(payload),
      )

  def test_payload_equal_to_a_block_name_round_trips(self):
    # Only a string literal of the code can be a quoted block name.
    for value in (
        Job(setup=BashCode(bash_code='MAIN_CODE'), main_code='echo main'),
        Edits(
            reasoning='r',
            edits=[
                FileEdit(file_path='a', new_text='NEW_TEXT_2'),
                FileEdit(file_path='b', new_text='echo b'),
            ],
        ),
    ):
      with self.subTest(value=value):
        self.assertEqual(
            self.protocol.parse_value(
                self.protocol.value_repr(value, assign_to_var='output'),
                base.Schema.from_value(value.__class__),
            ),
            value,
        )


class QuotedBlockNameTest(unittest.TestCase):
  """Rule 13: a quoted value that names no block."""

  SCHEMA = base.Schema.from_value(BashCode)

  def _parse(self, text):
    return _protocol().parse_value(text, self.SCHEMA).bash_code

  def test_quoted_name_of_the_field_needs_its_block(self):
    for text in (
        'output = BashCode(bash_code="BASH_CODE")',
        'output = BashCode(bash_code="BASH_CODE_2")',
        # A block under another name is not renamed for a quoted name.
        (
            'output = BashCode(bash_code="BASH_CODE")\n'
            '<<<LF:BLOCK OTHER t1>>>\necho hi\n<<<LF:END OTHER t1>>>'
        ),
    ):
      with self.subTest(text=text):
        with self.assertRaisesRegex(
            python_verbatim.VerbatimBlockError,
            "'bash_code' is set to 'BASH_CODE(_2)?', which names no block",
        ):
          self._parse(text)

  def test_other_literals_are_kept(self):
    # E.g. a short final answer: `KMT2D`, `E0277`, `C99`.
    for literal in ('PWD', 'KMT2D', 'CODE', 'BASH_CODE_X', 'bash_code'):
      with self.subTest(literal=literal):
        self.assertEqual(
            self._parse(f'output = BashCode(bash_code={literal!r})'), literal
        )

  def test_block_text_equal_to_a_block_name_is_kept(self):
    self.assertEqual(
        self._parse(
            'output = BashCode(bash_code=OTHER)\n'
            '<<<LF:BLOCK OTHER t1>>>\nBASH_CODE\n<<<LF:END OTHER t1>>>'
        ),
        'BASH_CODE',
    )

  def test_quoted_name_of_another_fields_block_is_kept(self):
    class FileWrite(pg.Object):
      content: Annotated[str, None, dict(verbatim=True)]
      old_str: Annotated[str | None, None, dict(verbatim=True)] = None

    schema = base.Schema.from_value(FileWrite)
    block = '<<<LF:BLOCK OLD_STR t1>>>\nold\n<<<LF:END OLD_STR t1>>>'
    for code in (
        "FileWrite(content='OLD_STR', old_str=OLD_STR)",
        "FileWrite(content='OLD_STR', old_str='OLD_STR')",
    ):
      with self.subTest(code=code):
        self.assertEqual(
            _protocol().parse_value(f'output = {code}\n{block}', schema),
            FileWrite(content='OLD_STR', old_str='old'),
        )
    # A block that no other field claims may still be quoted.
    self.assertEqual(
        _protocol()
        .parse_value(
            "output = FileWrite(content='BODY')\n"
            '<<<LF:BLOCK BODY t1>>>\nnew\n<<<LF:END BODY t1>>>',
            schema,
        )
        .content,
        'new',
    )

  def test_block_names_with_the_reserved_prefix(self):
    class Doc(pg.Object):
      block_content: Annotated[str, None, dict(verbatim=True)]

    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError, 'VERBATIM_BLOCK_CONTENT'
    ):
      _protocol().parse_value(
          "output = Doc(block_content='VERBATIM_BLOCK_CONTENT')",
          base.Schema.from_value(Doc),
      )


class ApplicabilityTest(unittest.TestCase):
  """`verbatim` changes nothing for a schema without verbatim fields."""

  ECHOED = (
      '```python\nclass Plain:\n  answer: int\n  note: str\n```\n\n'
      "```python\noutput = Plain(answer=1, note='NOTE')\n```"
  )

  def test_prompt_equals_python(self):
    for schema in (Plain, int):
      with self.subTest(schema=schema):
        self.assertEqual(
            querying.query_prompt('1 + 1 =', schema, protocol=PROTOCOL).text,
            querying.query_prompt('1 + 1 =', schema, protocol='python').text,
        )

  def test_parse_uses_select_but_not_blocks(self):
    schema = base.Schema.from_value(Plain)
    self.assertEqual(
        _protocol().parse_value(self.ECHOED, schema),
        Plain(answer=1, note='NOTE'),
    )
    with self.assertRaisesRegex(pg.coding.CodeError, 'Class definition'):
      base.PromptingProtocol.from_name('python').parse_value(
          self.ECHOED, schema
      )
    # Blocks are not cut: the block name reaches `exec` as an unknown name.
    with self.assertRaisesRegex(pg.coding.CodeError, 'NOTE'):
      _protocol().parse_value(
          'output = Plain(answer=1, note=NOTE)\n'
          '<<<LF:BLOCK NOTE>>>\nhi\n<<<LF:END NOTE>>>',
          schema,
      )
    with self.assertRaisesRegex(pg.coding.CodeError, 'TEXT'):
      _protocol().parse_value(
          'output = TEXT\n<<<LF:BLOCK TEXT>>>\nhi\n<<<LF:END TEXT>>>'
      )

  def test_only_the_marks_of_the_schema_classes_count(self):
    # The prompt does not render base classes, e.g. one whose mark the schema
    # class drops.
    class Marked(pg.Object):
      code: Annotated[str, None, dict(verbatim=True)]

    class Unmarked(Marked):
      code: Annotated[str, None, dict(verbatim=False)]

    feature = python_verbatim._Verbatim()  # pylint: disable=protected-access
    self.assertTrue(feature.applies_to(base.Schema.from_value(Marked)))
    self.assertFalse(feature.applies_to(base.Schema.from_value(Unmarked)))

  def test_verbatim_does_not_imply_select(self):
    verbatim_only = base.PromptingProtocol.from_name('python+verbatim')
    self.assertEqual(verbatim_only.name, 'python+verbatim:1.0')
    with self.assertRaisesRegex(pg.coding.CodeError, 'Class definition'):
      verbatim_only.parse_value(self.ECHOED, base.Schema.from_value(Plain))

  def test_the_query_runs_without_the_feature(self):
    with querying.track_queries() as queries:
      value = querying.query(
          '1 + 1 =', Plain, lm=fake.StaticResponse(self.ECHOED),
          protocol=PROTOCOL,
      )
    self.assertEqual(value, Plain(answer=1, note='NOTE'))
    self.assertEqual(queries[0].protocol, RECORDED)

  def test_fences_in_string_literals_do_not_close_the_fence(self):
    schema = base.Schema.from_value(Plain)
    expected = Plain(answer=1, note='Run:\n```bash\nls\n```\ndone')
    response = (
        '```python\noutput = Plain(answer=1, note="""Run:\n```bash\nls\n```\n'
        'done""")\n```'
    )
    for protocol in ('python', PROTOCOL):
      with self.subTest(protocol=protocol):
        self.assertEqual(
            base.PromptingProtocol.from_name(protocol).parse_value(
                response, schema
            ),
            expected,
        )
    echoed = '```python\nclass Plain:\n  answer: int\n  note: str\n```\n\n'
    self.assertEqual(
        _protocol().parse_value(echoed + response, schema), expected
    )

  def test_correction_with_an_indented_fence_in_a_string_literal(self):
    # As in a recorded reply: `select` also reads the correction, here one in
    # the format of a plain `python` correction, whose code holds a markdown
    # answer with an indented fence.
    code = "_ANS = '''Steps:\n   ```go\n   for {}\n   ```\n'''"
    correction = (
        '```python\noutput = CorrectedCode(\n'
        f'    corrected_code="""{code}"""\n)\n```'
    )
    with querying.track_queries() as queries:
      querying.query(
          '1 + 1 =',
          Plain,
          default=None,
          lm=fake.StaticResponse("output = Plain(answer=1, note='x'"),
          autofix=1,
          autofix_lm=fake.StaticResponse(correction),
          protocol=PROTOCOL,
      )
    self.assertEqual(
        [q.protocol for q in queries], [RECORDED, 'python:2.0+select:1.0']
    )
    self.assertEqual(queries[1].output.corrected_code, code)


# The `verbatim:1.0` preamble, as cl/1005423462 snapshot 2 renders it.
_GOLDEN_PREAMBLE = '''\
Please respond to the last REQUEST with OUTPUT PYTHON OBJECT only according to OUTPUT PYTHON TYPE.

Emit exactly ONE ```python fence, containing only the `output = ...` assignment. Never repeat the class definitions: they are given to you above.

A field marked `(verbatim field)` MUST NOT be given a Python string literal. Assign it a block name and put its text in a block placed AFTER the ```python fence. A block is copied byte for byte: escape nothing, indent nothing, and do not wrap it in a fence. Name the block after its field, in upper case. Every block in a response carries the same short random token that you pick, in both of its fences:

  <<<LF:BLOCK THE_BLOCK_NAME token>>>
  the exact text
  <<<LF:END THE_BLOCK_NAME token>>>

REQUEST:
  Write a script that generates a report.

OUTPUT PYTHON TYPE:
  Answer

  ```python
  class Script:
    # (verbatim field: do not write a string literal; assign a block name, e.g. `code=CODE`, and define that block below)
    code: str

  class Answer:
    note: str
    script: Script
  ```

OUTPUT PYTHON OBJECT:
  ```python
  output = Answer(
    note='A heredoc keeps the Python source unescaped.',
    script=Script(code=CODE)
  )
  ```

  <<<LF:BLOCK CODE 7f3a>>>
  cat > /tmp/report.py <<'EOF'
  with open('/tmp/report.md', 'w') as f:
      f.write("# Title\\n\\nDone.\\n")
  EOF
  python3 /tmp/report.py
  <<<LF:END CODE 7f3a>>>'''

_GOLDEN_NEXT_STEP = _GOLDEN_PREAMBLE + '\n\n' + '''\
REQUEST:
  Write a script that writes report.md and run it.

OUTPUT PYTHON TYPE:
  NextStep

  ```python
  class BashCode:
    """Bash code to run."""
    # Bash code to execute.
    # (verbatim field: do not write a string literal; assign a block name, e.g. `bash_code=BASH_CODE`, and define that block below)
    bash_code: str
    timeout_secs: int

  class Note:
    note: str

  class NextStep:
    """The next step."""
    think_step_by_step: str
    next_step: Union[BashCode, Note, None]
    learnings: list[str] | None
  ```

OUTPUT PYTHON OBJECT:'''

_GOLDEN_FILE_EDITS = _GOLDEN_PREAMBLE + '\n\n' + '''\
REQUEST:
  Rename foo to bar in /tmp/a.py.

OUTPUT PYTHON TYPE:
  Edits

  ```python
  class FileEdit:
    """Replaces text in a file."""
    # ABSOLUTE path of the file.
    file_path: str
    # (verbatim field: do not write a string literal; assign a block name, e.g. `new_text=NEW_TEXT`, and define that block below)
    new_text: str
    # Exact text to replace.
    # (verbatim field: do not write a string literal; assign a block name, e.g. `old_text=OLD_TEXT`, and define that block below)
    old_text: str | None
    # Appended as is.
    # (verbatim field: do not write a string literal; assign a block name, e.g. `block_content=VERBATIM_BLOCK_CONTENT`, and define that block below)
    block_content: str | None

  class Edits:
    reasoning: str
    edits: list[FileEdit]
  ```

OUTPUT PYTHON OBJECT:'''

_GOLDEN_FEWSHOT_AND_INPUT = _GOLDEN_PREAMBLE + '\n\n' + '''\
REQUEST:
  Say hi twice.

OUTPUT PYTHON TYPE:
  NextStep

  ```python
  class BashCode:
    """Bash code to run."""
    # Bash code to execute.
    # (verbatim field: do not write a string literal; assign a block name, e.g. `bash_code=BASH_CODE`, and define that block below)
    bash_code: str
    timeout_secs: int

  class Note:
    note: str

  class NextStep:
    """The next step."""
    think_step_by_step: str
    next_step: Union[BashCode, Note, None]
    learnings: list[str] | None
  ```

OUTPUT PYTHON OBJECT:
  ```python
  output = NextStep(
    think_step_by_step='Echo it.',
    next_step=BashCode(
      # Bash code to execute.
      bash_code=BASH_CODE,
      timeout_secs=5
    ),
    learnings=None
  )
  ```

  <<<LF:BLOCK BASH_CODE 9132>>>
  echo "hi\\nhi"
  <<<LF:END BASH_CODE 9132>>>


REQUEST:
  ```python
  Task(
    title='Count lines.',
    # Text to process.
    payload=PAYLOAD
  )
  ```

  <<<LF:BLOCK PAYLOAD d9cc>>>
  a\tb
  "c"

  <<<LF:END PAYLOAD d9cc>>>

OUTPUT PYTHON TYPE:
  NextStep

  ```python
  class BashCode:
    """Bash code to run."""
    # Bash code to execute.
    # (verbatim field: do not write a string literal; assign a block name, e.g. `bash_code=BASH_CODE`, and define that block below)
    bash_code: str
    timeout_secs: int

  class Note:
    note: str

  class NextStep:
    """The next step."""
    think_step_by_step: str
    next_step: Union[BashCode, Note, None]
    learnings: list[str] | None
  ```

OUTPUT PYTHON OBJECT:'''

# No verbatim field: the `python:2.0` prompt.
_GOLDEN_PLAIN = '''\
Please respond to the last REQUEST with OUTPUT PYTHON OBJECT only according to OUTPUT PYTHON TYPE.

REQUEST:
  1 + 1 =

OUTPUT PYTHON TYPE:
  Answer

  ```python
  class Answer:
    final_answer: int
  ```

OUTPUT PYTHON OBJECT:
  ```python
  output = Answer(
    final_answer=2
  )
  ```

REQUEST:
  1 + 1 =

OUTPUT PYTHON TYPE:
  Plain

  ```python
  class Plain:
    answer: int
    note: str
  ```

OUTPUT PYTHON OBJECT:'''

# No verbatim field: the `python:2.0` prompt.
_GOLDEN_INT = '''\
Please respond to the last REQUEST with OUTPUT PYTHON OBJECT only according to OUTPUT PYTHON TYPE.

REQUEST:
  1 + 1 =

OUTPUT PYTHON TYPE:
  Answer

  ```python
  class Answer:
    final_answer: int
  ```

OUTPUT PYTHON OBJECT:
  ```python
  output = Answer(
    final_answer=2
  )
  ```

REQUEST:
  1 + 1 =

OUTPUT PYTHON TYPE:
  int

OUTPUT PYTHON OBJECT:'''


def golden_prompts(protocol):
  """Returns {case: prompt text} rendered under `protocol`."""
  example = mapping.MappingExample(
      input='Say hi twice.',
      schema=NextStep,
      output=NextStep(
          think_step_by_step='Echo it.',
          next_step=BashCode(bash_code='echo "hi\\nhi"', timeout_secs=5),
      ),
  )
  cases = dict(
      next_step=('Write a script that writes report.md and run it.', NextStep, None),
      file_edits=('Rename foo to bar in /tmp/a.py.', Edits, None),
      fewshot_and_input=(
          Task(title='Count lines.', payload='a\tb\n"c"\n'), NextStep, [example]
      ),
      plain=('1 + 1 =', Plain, None),
      int=('1 + 1 =', int, None),
  )
  return {
      name: querying.query_prompt(
          prompt, schema, examples=examples, protocol=protocol
      ).text
      for name, (prompt, schema, examples) in cases.items()
  }


class GoldenPromptTest(unittest.TestCase):
  """`python+verbatim` prompts equal the `python-verbatim:1.0` ones of cl/1005423462."""

  def test_golden_prompts(self):
    prompts = golden_prompts(PROTOCOL)
    for name, expected in [
        ('next_step', _GOLDEN_NEXT_STEP),
        ('file_edits', _GOLDEN_FILE_EDITS),
        ('fewshot_and_input', _GOLDEN_FEWSHOT_AND_INPUT),
        # No verbatim field: `python-verbatim:1.0` added its preamble anyway.
        ('plain', _GOLDEN_PLAIN),
        ('int', _GOLDEN_INT),
    ]:
      with self.subTest(name=name):
        self.assertEqual(prompts[name], expected)

  def test_pinned_name_renders_the_same_prompts(self):
    self.assertEqual(golden_prompts(RECORDED), golden_prompts(PROTOCOL))

  def test_parse_instructions_are_the_block_instructions(self):
    instructions = _protocol().parse_instructions()
    self.assertIn('\n\n' + instructions + '\n\n', _GOLDEN_PREAMBLE)
    self.assertTrue(instructions.startswith('A field marked'))

  def test_query_preamble_keeps_the_first_line_of_the_base_one(self):
    preamble = _protocol().query_preamble('Respond.\n\nAn example.')
    self.assertTrue(preamble.startswith('Respond.\n\nEmit exactly ONE'))
    self.assertNotIn('An example.', preamble)


class QueryTest(unittest.TestCase):
  """End-to-end `lf.query` with `protocol='python+verbatim+select'`."""

  # A payload the `python` protocol reliably mangles: a raw-string regex with an
  # escape, plus escapes inside an ordinary inner literal.
  PAYLOAD = inspect.cleandoc(r"""
      cat > /tmp/r.py <<'EOF'
      import re
      parts = re.split(r'[.!?\n]+', text)
      f.write("# Title\n\nDone.\n")
      EOF
      """)

  def test_query(self):
    response = (
        '```python\n'
        'output = NextStep(\n'
        "    think_step_by_step='write the report',\n"
        '    next_step=BashCode(bash_code=BASH_CODE, timeout_secs=600),\n'
        ')\n'
        '```\n\n'
        + python_verbatim.block_repr('BASH_CODE', self.PAYLOAD, '7f3a')
    )
    value = querying.query(
        'write a report',
        NextStep,
        lm=fake.StaticResponse(response),
        protocol=PROTOCOL,
    )
    self.assertEqual(value.next_step.bash_code, self.PAYLOAD)
    self.assertEqual(value.next_step.timeout_secs, 600)

  def test_query_with_an_indented_response(self):
    response = textwrap.indent(
        'output = NextStep(\n'
        "    think_step_by_step='t',\n"
        '    next_step=BashCode(bash_code=BASH_CODE),\n'
        ')\n'
        + python_verbatim.block_repr('BASH_CODE', self.PAYLOAD, '7f3a'),
        '    ',
    )
    value = querying.query(
        'write a report',
        NextStep,
        lm=fake.StaticResponse(response),
        protocol=PROTOCOL,
    )
    self.assertEqual(value.next_step.bash_code, self.PAYLOAD)

  def test_prompt_contains_block_instructions(self):
    prompt = querying.query_prompt(
        'write a report', NextStep, protocol=PROTOCOL
    )
    self.assertIn('<<<LF:BLOCK', prompt.text)
    self.assertIn('verbatim field', prompt.text)
    # The worked example must show an unescaped payload.
    self.assertIn(r'f.write("# Title\n\nDone.\n")', prompt.text)

  def test_fewshot_example_round_trips(self):
    example = mapping.MappingExample(
        input='write a report',
        output=NextStep(
            think_step_by_step='t',
            next_step=BashCode(bash_code=self.PAYLOAD),
        ),
        schema=NextStep,
    )
    prompt = querying.query_prompt(
        'another report',
        NextStep,
        examples=[example],
        protocol=PROTOCOL,
    )
    # The mapping template indents each example by 2; parsing dedents it back.
    self.assertIn(textwrap.indent(self.PAYLOAD, '  '), prompt.text)
    blocks, _ = python_verbatim.extract_blocks(prompt.text)
    self.assertEqual(blocks['BASH_CODE'], self.PAYLOAD)

  def test_recorded_protocol_replays_the_query(self):
    def in_context(context, fn, *args, **kwargs):
      if context is None:
        return fn(*args, **kwargs)
      with querying.query_protocol(context):
        return fn(*args, **kwargs)

    def check(
        prompt,
        schema,
        response,
        live_context=None,
        replay_context=None,
        **kwargs,
    ):
      if live_context is None:
        kwargs['protocol'] = PROTOCOL
      with querying.track_queries() as queries:
        live = in_context(
            live_context,
            querying.query,
            prompt,
            schema,
            lm=fake.StaticResponse(response),
            returns_message=True,
            **kwargs,
        )
      invocation = queries[0]
      self.assertEqual(invocation.protocol, RECORDED)
      self.assertEqual(invocation.output, live.result)
      self.assertEqual(
          in_context(
              replay_context,
              querying.query_output,
              invocation.lm_response,
              schema,
              protocol=invocation.protocol,
          ),
          live.result,
      )
      self.assertEqual(
          in_context(replay_context, lambda: invocation.lm_request.text),
          live.lm_input.text,
      )

    marked = dict(
        prompt='write a report',
        schema=NextStep,
        response=(
            '```python\n'
            "output = NextStep(think_step_by_step='t',"
            ' next_step=BashCode(bash_code=BASH_CODE))\n'
            '```\n\n'
            + python_verbatim.block_repr('BASH_CODE', self.PAYLOAD, '7f3a')
        ),
    )
    example = mapping.MappingExample(
        input=Task(title='Count lines.', payload='a\tb\n"c"\n'),
        schema=NextStep,
        output=NextStep(
            think_step_by_step='t', next_step=BashCode(bash_code=self.PAYLOAD)
        ),
    )
    for name, kwargs in dict(
        marked_schema=marked,
        unmarked_schema=dict(
            prompt='1 + 1 =',
            schema=Plain,
            response="output = Plain(answer=2, note='two')",
        ),
        examples_with_a_marked_input=dict(
            marked,
            prompt=Task(title='Count words.', payload='x\ty\n'),
            examples=[example],
        ),
        # Replays the same prompt, whatever a caller preamble renders as.
        caller_preamble=dict(marked, preamble='Be brief.'),
        query_protocol_context=dict(
            marked, live_context=PROTOCOL, replay_context=PROTOCOL
        ),
        replay_outside_the_context=dict(marked, live_context=PROTOCOL),
    ).items():
      with self.subTest(name):
        check(**kwargs)

  def test_the_prompts_own_hint_parses_for_fields_named_like_a_class(self):
    class SQL(pg.Object):
      sql: Annotated[str, None, dict(verbatim=True)]

    class Lookup(pg.Object):
      unknown: Annotated[str, None, dict(verbatim=True)]

    for cls, key, text in ((SQL, 'sql', 'SELECT 1'), (Lookup, 'unknown', '?')):
      with self.subTest(cls=cls):
        prompt = querying.query_prompt('Fill it in.', cls, protocol=PROTOCOL)
        # The block name of the hint, e.g. NAME in `sql=NAME`.
        name = prompt.text.split(f'`{key}=')[1].split('`')[0]
        response = (
            f'```python\noutput = {cls.__name__}({key}={name})\n```\n\n'
            + python_verbatim.block_repr(name, text, 't1')
        )
        self.assertEqual(
            querying.query(
                'Fill it in.',
                cls,
                lm=fake.StaticResponse(response),
                protocol=PROTOCOL,
            ),
            cls(**{key: text}),
        )

  def test_the_prompts_hint_parses_for_a_field_named_like_another_class(self):
    value = Setup(conn=Conn(db='postgres://h/db'), target=DB(name='x'))
    example = mapping.MappingExample(
        input='Fill it in.', schema=Setup, output=value
    )
    prompt = querying.query_prompt(
        'Fill it in.', Setup, examples=[example], protocol=PROTOCOL
    ).text
    # The block name of the hint, e.g. NAME in `db=NAME`.
    name = prompt.split('`db=')[1].split('`')[0]
    self.assertEqual(name, 'VERBATIM_DB')
    # The example names its block alike.
    self.assertIn(f'<<<LF:BLOCK {name} ', prompt)
    code = f"output = Setup(conn=Conn(db={name}), target=DB(name='x'))"
    response = f'```python\n{code}\n```\n\n' + python_verbatim.block_repr(
        name, 'postgres://h/db', 't1'
    )
    self.assertEqual(
        querying.query(
            'Fill it in.',
            Setup,
            lm=fake.StaticResponse(response),
            protocol=PROTOCOL,
        ),
        value,
    )

  def test_feature_order_and_versions_are_normalized(self):
    for protocol, recorded in [
        ('python+select+verbatim', RECORDED),
        ('python:2.0+verbatim:1.0+select', RECORDED),
        (RECORDED, RECORDED),
        # `verbatim` does not imply `select`.
        ('python+verbatim', 'python:2.0+verbatim:1.0'),
    ]:
      with self.subTest(protocol=protocol):
        with querying.track_queries() as queries:
          querying.query(
              'write a report',
              NextStep,
              lm=fake.StaticResponse("NextStep(think_step_by_step='t')"),
              protocol=protocol,
          )
        self.assertEqual(queries[0].protocol, recorded)

  def test_unsupported_protocols(self):
    for protocol, error in [
        ('python:1.0+verbatim', 'does not support python:1.0'),
        ('json+verbatim', 'does not support json'),
        ('python+verbatim+verbatim', 'appears twice'),
        # Snapshot-2 benchmark logs replay with the tooling of that snapshot.
        ('python-verbatim', "Protocol 'python-verbatim' is not supported"),
    ]:
      with self.subTest(protocol=protocol):
        with self.assertRaisesRegex(ValueError, error):
          querying.query_prompt('write a report', NextStep, protocol=protocol)

  def test_new_feature_versions_leave_unpinned_names_alone(self):
    prompt = querying.query_prompt(
        'write a report', NextStep, protocol=PROTOCOL
    )
    registry = base._ProtocolFeature._REGISTRY  # pylint: disable=protected-access
    with mock.patch.dict(registry, {n: dict(v) for n, v in registry.items()}):

      class _SelectV11(python_verbatim._Select):  # pylint: disable=protected-access,unused-variable
        VERSION = '1.1'
        DEFAULT = False

      class _VerbatimV11(python_verbatim._Verbatim):  # pylint: disable=protected-access,unused-variable
        VERSION = '1.1'
        DEFAULT = False

      for protocol, recorded in [
          (PROTOCOL, RECORDED),
          ('python:2.0+verbatim:1.0+select', RECORDED),
          ('python+select', 'python:2.0+select:1.0'),
          ('python+verbatim+select:1.1', 'python:2.0+verbatim:1.0+select:1.1'),
      ]:
        with self.subTest(protocol=protocol):
          with querying.track_queries() as queries:
            querying.query(
                'write a report',
                NextStep,
                lm=fake.StaticResponse("NextStep(think_step_by_step='t')"),
                protocol=protocol,
            )
          self.assertEqual(queries[0].protocol, recorded)
      self.assertEqual(
          querying.query_prompt(
              'write a report', NextStep, protocol=PROTOCOL
          ).text,
          prompt.text,
      )

  def test_a_preamble_argument_wins(self):
    prompt = querying.query_prompt(
        'write a report', NextStep, protocol=PROTOCOL, preamble='Be brief.'
    ).text
    # Followed by the block instructions, which the field hints refer to.
    instructions = _protocol().parse_instructions()
    self.assertTrue(
        prompt.startswith(f'Be brief.\n\n{instructions}\n\nREQUEST:'), prompt
    )
    self.assertIn('(verbatim field:', prompt)
    prompt = querying.query_prompt(
        '1 + 1 =', Plain, protocol=PROTOCOL, preamble='Be brief.'
    ).text
    self.assertTrue(prompt.startswith('Be brief.\n\nREQUEST:'), prompt)

  def test_autofix_keeps_the_protocol(self):
    response = (
        "output = NextStep(think_step_by_step='go',"
        ' next_step=BashCode(bash_code=BASH_CODE)\n'  # Missing paren.
        '<<<LF:BLOCK BASH_CODE t1>>>\necho "a\\nb"\n<<<LF:END BASH_CODE t1>>>'
    )
    corrected = (
        "output = NextStep(think_step_by_step='go\\n',"
        ' next_step=BashCode(bash_code=BASH_CODE))'
    )
    # The correction carries the corrected code in a block, unescaped.
    correction = (
        '```python\noutput = CorrectedCode(corrected_code=CORRECTED_CODE)\n'
        '```\n\n'
        + python_verbatim.block_repr('CORRECTED_CODE', corrected, 'c0de')
    )
    with querying.track_queries() as queries:
      value = querying.query(
          'say hi',
          NextStep,
          lm=fake.StaticResponse(response),
          autofix=1,
          autofix_lm=fake.StaticResponse(correction),
          protocol=PROTOCOL,
      )
    self.assertEqual(value.think_step_by_step, 'go\n')
    self.assertEqual(value.next_step.bash_code, 'echo "a\\nb"')
    self.assertEqual(
        [q.schema.spec.cls.__name__ for q in queries],
        ['NextStep', 'CorrectedCode'],
    )
    self.assertEqual([q.protocol for q in queries], [RECORDED, RECORDED])
    self.assertIn(
        '`corrected_code=CORRECTED_CODE`', queries[1].lm_request.text
    )


class ParseTest(unittest.TestCase):
  """`lf.parse` with `protocol='python+verbatim+select'`."""

  RESPONSE = (
      "output = BashCode(bash_code=BASH_CODE, timeout_secs=5)\n"
      '<<<LF:BLOCK BASH_CODE t1>>>\nprintf "%s\\n" hi\n<<<LF:END BASH_CODE t1>>>'
  )

  def test_parse(self):
    message = parsing.parse(
        'Run printf.',
        BashCode,
        lm=fake.StaticResponse(self.RESPONSE),
        protocol=PROTOCOL,
        returns_message=True,
    )
    self.assertEqual(
        message.result, BashCode(bash_code='printf "%s\\n" hi', timeout_secs=5)
    )
    prompt = message.lm_input.text
    self.assertIn(
        '2. RESULT_OBJECT must be created solely based on the information'
        ' provided in LM_RESPONSE.\n\n'
        + _protocol().parse_instructions()
        + '\n\nLM_RESPONSE:',
        prompt,
    )
    self.assertIn('`bash_code=BASH_CODE`', prompt)

  def test_a_preamble_argument_gets_the_block_instructions(self):
    message = parsing.parse(
        'Run printf.',
        BashCode,
        lm=fake.StaticResponse(self.RESPONSE),
        protocol=PROTOCOL,
        returns_message=True,
        preamble='Be brief.',
    )
    self.assertEqual(
        message.result, BashCode(bash_code='printf "%s\\n" hi', timeout_secs=5)
    )
    self.assertTrue(
        message.lm_input.text.startswith(
            f'Be brief.\n\n{_protocol().parse_instructions()}\n\nLM_RESPONSE:'
        ),
        message.lm_input.text,
    )

  def test_schema_without_verbatim_fields(self):
    def prompt(protocol):
      return parsing.parse(
          'one',
          Note,
          lm=fake.StaticResponse("Note(note='one')"),
          protocol=protocol,
          returns_message=True,
      ).lm_input.text

    self.assertEqual(prompt(PROTOCOL), prompt('python'))


class NameWordsTest(unittest.TestCase):

  def test_name_words(self):
    words = python_verbatim._name_words  # pylint: disable=protected-access
    self.assertEqual(words('BASH_CODE_2'), ['bash', 'code', '2'])
    self.assertEqual(words('BLOCK2'), ['block', '2'])
    self.assertEqual(words('bash_code'), ['bash', 'code'])
    # Block names are snake_case, so CamelCase is one word.
    self.assertEqual(words('BashCode'), ['bashcode'])


if __name__ == '__main__':
  unittest.main()
