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
"""Tests for the `python-verbatim` prompting protocol."""

import ast
import inspect
import textwrap
from typing import Annotated
import unittest
from unittest import mock

from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import querying
from langfun.core.structured.schema import base
from langfun.core.structured.schema import python_verbatim
import pyglove as pg


class BashCode(pg.Object):
  """Bash code to run."""

  bash_code: Annotated[str, 'Bash code to execute.']
  timeout_secs: int = 60


class Note(pg.Object):
  note: str


class NextStep(pg.Object):
  """The next step."""

  think_step_by_step: str
  next_step: BashCode | Note | None = None
  learnings: list[str] | None = None


class Job(pg.Object):
  setup: BashCode
  main_code: str


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

  def test_auto_detect_by_name(self):
    class A(pg.Object):
      bash_code: str
      code: str
      python_script: str
      file_content: str
      note: str
      count: int

    for name in ['bash_code', 'code', 'python_script', 'file_content']:
      self.assertTrue(
          python_verbatim.is_verbatim_field(self._field(A, name)), name
      )
    for name in ['note', 'count']:
      self.assertFalse(
          python_verbatim.is_verbatim_field(self._field(A, name)), name
      )

  def test_explicit_opt_in_and_out(self):
    class B(pg.Object):
      body: Annotated[str, 'Body.', dict(verbatim=True)]
      code: Annotated[str, 'Not verbatim.', dict(verbatim=False)]

    self.assertTrue(python_verbatim.is_verbatim_field(self._field(B, 'body')))
    self.assertFalse(python_verbatim.is_verbatim_field(self._field(B, 'code')))

  def test_optional_str_field(self):
    class C(pg.Object):
      code: str | None = None

    self.assertTrue(python_verbatim.is_verbatim_field(self._field(C, 'code')))


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
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A 7f3a>>>\nsee <<<LF:END A 7f3a>>>\nreal content\n'
        '<<<LF:END A 7f3a>>>'
    )
    self.assertEqual(blocks, {'A': 'see <<<LF:END A 7f3a>>>\nreal content'})

  def test_payload_containing_opening_fence(self):
    blocks, _ = python_verbatim.extract_blocks(
        '<<<LF:BLOCK A>>>\n<<<LF:BLOCK B>>>\nx\n<<<LF:END A>>>'
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


class SchemaReprTest(unittest.TestCase):

  def test_hint_names_the_block_after_the_field(self):
    # A block named `BLOCK_1` makes the fence read as
    # `<<<LF:BLOCK BLOCK_1 …>>>`, which the LM deduplicates to
    # `<<<LF:BLOCK_1 …>>>`.
    protocol = python_verbatim.PythonVerbatimPromptingProtocol()
    repr_str = protocol.schema_repr(base.Schema.from_value(BashCode))
    self.assertIn('`bash_code=BASH_CODE`', repr_str)
    self.assertNotIn('BLOCK_1', repr_str)

  def test_generated_block_names_never_stutter_with_the_keyword(self):
    class Doc(pg.Object):
      block_content: str
      end_content: str

    text = python_verbatim.PythonVerbatimPromptingProtocol().value_repr(
        Doc(block_content='a', end_content='b')
    )
    self.assertIn('<<<LF:BLOCK VERBATIM_BLOCK_CONTENT ', text)
    self.assertIn('<<<LF:BLOCK VERBATIM_END_CONTENT ', text)

  def test_verbatim_hint_in_class_definition(self):
    protocol = python_verbatim.PythonVerbatimPromptingProtocol()
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
    protocol = python_verbatim.PythonVerbatimPromptingProtocol()
    self.assertNotIn(
        'verbatim field',
        protocol.schema_repr(base.Schema.from_value(Note)),
    )


class ValueReprTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.protocol = python_verbatim.PythonVerbatimPromptingProtocol()

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
    self.protocol = python_verbatim.PythonVerbatimPromptingProtocol()
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

  def test_parse_without_schema(self):
    self.assertEqual(
        self.protocol.parse_value(
            'output = TEXT\n<<<LF:BLOCK TEXT>>>\nhi\n<<<LF:END TEXT>>>'
        ),
        'hi',
    )

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
      python_verbatim.PythonVerbatimPromptingProtocol().parse_value(
          'output = Job(setup=BashCode(bash_code=A), main_code=B)\n'
          '<<<LF:BLOCK X>>>\nfirst\n<<<LF:END X>>>\n'
          '<<<LF:BLOCK Y>>>\nsecond\n<<<LF:END Y>>>',
          base.Schema.from_value(Job),
      )

  def test_misnamed_blocks_are_matched_by_name_stem(self):
    class Edit(pg.Object):
      file_path: str
      old_code: str
      new_code: str

    value = python_verbatim.PythonVerbatimPromptingProtocol().parse_value(
        "output = Edit(file_path='a.py',"
        ' old_code=OLD_CODE, new_code=NEW_CODE)\n'
        '<<<LF:BLOCK NEW 1>>>\nreturn 2\n<<<LF:END NEW 1>>>\n'
        '<<<LF:BLOCK OLD 1>>>\nreturn 1\n<<<LF:END OLD 1>>>',
        base.Schema.from_value(Edit),
    )
    self.assertEqual(value.old_code, 'return 1')
    self.assertEqual(value.new_code, 'return 2')

  def test_misnamed_blocks_are_matched_by_ordinal(self):
    value = python_verbatim.PythonVerbatimPromptingProtocol().parse_value(
        'output = Job(setup=BashCode(bash_code=CODE_1), main_code=CODE_2)\n'
        '<<<LF:BLOCK_2 7f3a>>>\nsecond\n<<<LF:END BLOCK_2 7f3a>>>\n'
        '<<<LF:BLOCK_1 7f3a>>>\nfirst\n<<<LF:END BLOCK_1 7f3a>>>',
        base.Schema.from_value(Job),
    )
    self.assertEqual(value.setup.bash_code, 'first')
    self.assertEqual(value.main_code, 'second')

  def test_match_renames(self):
    match = python_verbatim._match_renames  # pylint: disable=protected-access
    self.assertEqual(match(['A'], ['X']), {'A': 'X'})
    self.assertEqual(match([], ['X', 'Y']), {})
    self.assertEqual(match(['NEW_CODE'], ['CODE', 'OLD']), {'NEW_CODE': 'CODE'})
    # No signal; two blocks agree with one name; one block with two names.
    self.assertIsNone(match(['A', 'B'], ['X', 'Y']))
    self.assertIsNone(match(['CODE_1'], ['BASH_CODE', 'MAIN_CODE']))
    self.assertIsNone(match(['OLD_CODE', 'OLD_TEXT'], ['OLD', 'NEW']))

  def test_error_lists_the_blocks_as_defined(self):
    # Blocks are renamed only if that resolves every reference.
    with self.assertRaisesRegex(
        python_verbatim.VerbatimBlockError,
        r"'B' is referenced .* Defined blocks: \['X'\]",
    ):
      python_verbatim.PythonVerbatimPromptingProtocol().parse_value(
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
    value = python_verbatim.PythonVerbatimPromptingProtocol().parse_value(
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

  def test_string_literals(self):
    literals = python_verbatim._string_literals  # pylint: disable=protected-access
    self.assertEqual(
        literals("x = ('a'  # Note.\n     'b', b'\u00e9', f'c', 'd'"),
        {'a', 'b', 'ab', 'd'},
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
      content: str

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
      file_content: str

    protocol = python_verbatim.PythonVerbatimPromptingProtocol()
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


class DiagnoseEscapeDamageTest(unittest.TestCase):

  def test_detects_eaten_escape(self):
    damaged = (
        "cat <<'EOF' > /tmp/a.py\n"
        'import re\n'
        "parts = re.split(r'[.!?\n]+', text)\n"
        'EOF\n'
    )
    diagnosis = python_verbatim.diagnose_escape_damage(damaged)
    self.assertIsNotNone(diagnosis)
    self.assertIn('unterminated', diagnosis)

  def test_clean_payload(self):
    clean = (
        "cat <<'EOF' > /tmp/a.py\n"
        'import re\n'
        "parts = re.split(r'[.!?\\n]+', text)\n"
        'EOF\n'
    )
    self.assertIsNone(python_verbatim.diagnose_escape_damage(clean))

  def test_non_python_heredoc_is_ignored(self):
    self.assertIsNone(
        python_verbatim.diagnose_escape_damage(
            "cat <<'EOF' > /tmp/a.txt\nsome 'unbalanced text\nEOF\n"
        )
    )

  def test_other_syntax_errors_are_not_escape_damage(self):
    self.assertIsNone(
        python_verbatim.diagnose_escape_damage(
            "cat <<'EOF' > /tmp/a.py\nimport os\ndef f(:\n  pass\nEOF\n"
        )
    )

  def test_python_too_complex_to_parse_is_skipped(self):
    body = 'import os\nx = ' + 'not ' * 100_000 + 'y\n'
    with self.assertRaises((MemoryError, RecursionError)):
      ast.parse(body)
    self.assertIsNone(
        python_verbatim.diagnose_escape_damage(
            f"cat <<'EOF' > /tmp/a.py\n{body}EOF\n"
        )
    )


class QueryTest(unittest.TestCase):
  """End-to-end `lf.query` through the registered protocol."""

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
        protocol='python-verbatim',
    )
    self.assertEqual(value.next_step.bash_code, self.PAYLOAD)
    self.assertEqual(value.next_step.timeout_secs, 600)
    self.assertIsNone(
        python_verbatim.diagnose_escape_damage(value.next_step.bash_code)
    )

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
        protocol='python-verbatim',
    )
    self.assertEqual(value.next_step.bash_code, self.PAYLOAD)

  def test_prompt_contains_block_instructions(self):
    prompt = querying.query_prompt(
        'write a report', NextStep, protocol='python-verbatim'
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
        protocol='python-verbatim',
    )
    # The mapping template indents each example by 2; parsing dedents it back.
    self.assertIn(textwrap.indent(self.PAYLOAD, '  '), prompt.text)
    blocks, _ = python_verbatim.extract_blocks(prompt.text)
    self.assertEqual(blocks['BASH_CODE'], self.PAYLOAD)


class CamelCaseFieldDetectionTest(unittest.TestCase):
  """Auto-detection must work for CamelCase schemas mirroring external tools.

  Agents that name their action fields after a tool's parameters (e.g.
  `CodeContent`, `ReplacementContent`, `CommandLine`) would otherwise have their
  code payloads silently left inline by a snake_case-only matcher.
  """

  def _field(self, cls, name):
    return cls.__schema__[name]

  def test_camel_case_code_fields_are_verbatim(self):

    class Edit(pg.Object):
      # pylint: disable=invalid-name
      TargetFile: str
      TargetContent: str
      ReplacementContent: str
      CodeContent: str
      CommandLine: str
      Description: str
      StartLine: int

    for name in (
        'TargetContent',
        'ReplacementContent',
        'CodeContent',
        'CommandLine',
    ):
      self.assertTrue(
          python_verbatim.is_verbatim_field(self._field(Edit, name)), name
      )
    for name in ('TargetFile', 'Description', 'StartLine'):
      self.assertFalse(
          python_verbatim.is_verbatim_field(self._field(Edit, name)), name
      )

  def test_snake_case_still_works(self):

    class Bash(pg.Object):
      bash_code: str
      timeout_secs: int
      question: str

    self.assertTrue(
        python_verbatim.is_verbatim_field(self._field(Bash, 'bash_code'))
    )
    self.assertFalse(
        python_verbatim.is_verbatim_field(self._field(Bash, 'question'))
    )

  def test_field_name_words(self):
    self.assertEqual(
        python_verbatim.field_name_words('ReplacementContent'),
        ['replacement', 'content'],
    )
    self.assertEqual(
        python_verbatim.field_name_words('bash_code'), ['bash', 'code']
    )
    self.assertEqual(
        python_verbatim.field_name_words('HTTPResponseBody'),
        ['http', 'response', 'body'],
    )


if __name__ == '__main__':
  unittest.main()
