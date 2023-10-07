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
"""Handling Python code."""

import ast
import contextlib
import enum
import inspect
import io
import re
import textwrap
from typing import Annotated, Any

import langfun.core as lf
import pyglove as pg


class CodePermission(enum.Flag):
  """Permissions for code execution."""

  # Allows basic Python code: Creating objects, assignment, operations.
  BASIC = enum.auto()

  # Allows conditions.
  CONDITION = enum.auto()

  # Allows loops.
  LOOP = enum.auto()

  # Allows exception.
  EXCEPTION = enum.auto()

  # Allows class definitions.
  CLASS_DEFINITION = enum.auto()

  # Allows function definitions.
  FUNCTION_DEFINITION = enum.auto()

  # Allows import.
  IMPORT = enum.auto()

  @classmethod
  @property
  def ALL(cls) -> 'CodePermission':    # pylint: disable=invalid-name
    """Returns all permissions."""
    return (
        CodePermission.BASIC | CodePermission.CONDITION | CodePermission.LOOP |
        CodePermission.EXCEPTION | CodePermission.CLASS_DEFINITION |
        CodePermission.FUNCTION_DEFINITION | CodePermission.IMPORT)


class CodeError(RuntimeError):
  """Python code error."""

  def __init__(self, code: str, cause: Exception):
    self.code = code
    self.cause = cause

  def __str__(self):
    r = io.StringIO()
    r.write(
        lf.colored(f'{self.cause.__class__.__name__}: {self.cause}', 'magenta'))

    if isinstance(self.cause, SyntaxError):
      r.write('\n\n')
      r.write(textwrap.indent(
          lf.colored(self.cause.text, 'magenta'),
          ' ' * 2
      ))
      if not self.cause.text.endswith('\n'):
        r.write('\n')

    r.write('\n')
    r.write(lf.colored('Generated Code:', 'red'))
    r.write('\n\n')
    r.write(lf.colored('  ```python\n', 'magenta'))
    r.write(textwrap.indent(
        lf.colored(self.code, 'magenta'),
        ' ' * 2
    ))
    r.write(lf.colored('\n  ```\n', 'magenta'))
    return r.getvalue()


class PythonCodeParser(lf.Component):
  """Python code parser with permission control."""

  _ID_REGEX = re.compile('^[a-zA-Z_\\-]*$')

  class _CodeValidator(ast.NodeVisitor):
    """Python AST node visitor for ensuring code are permitted."""

    def __init__(self, code: str, perm: CodePermission):
      super().__init__()
      self.code = code
      self.perm = perm

    def verify(self, node, flag: CodePermission, node_type, error_message: str):
      if isinstance(node, node_type) and not (self.perm & flag):
        raise SyntaxError(
            error_message, (
                '<generated-code>',
                node.lineno,
                node.col_offset,
                self._code_line(node.lineno),
                node.end_lineno,
                node.end_col_offset,
            ))

    def _code_line(self, lineno):
      return self.code.split('\n')[lineno - 1]

    def generic_visit(self, node):
      self.verify(
          node, CodePermission.CONDITION, (ast.If, ast.Match),
          'Condition is not allowed.')

      self.verify(
          node,
          CodePermission.LOOP,
          (
              ast.For,
              ast.While,
              ast.AsyncFor,
              ast.AsyncWith
          ),
          'Loop is not allowed.')

      self.verify(
          node,
          CodePermission.EXCEPTION,
          (
              ast.Try,
              ast.Raise,
              ast.Assert
          ),
          'Exception is not allowed.')

      self.verify(
          node,
          CodePermission.CLASS_DEFINITION,
          ast.ClassDef,
          'Class definition is not allowed.')

      self.verify(
          node,
          CodePermission.FUNCTION_DEFINITION,
          (
              ast.FunctionDef,
              ast.AsyncFunctionDef,
              ast.Return,
              ast.Yield,
              ast.YieldFrom
          ),
          'Function definition is not allowed.')

      self.verify(
          node,
          CodePermission.IMPORT,
          (
              ast.Import,
              ast.ImportFrom
          ),
          '`import` is not allowed.')

      super().generic_visit(node)

  def parse(self, code: str, perm: CodePermission) -> tuple[str, ast.AST]:
    code = self.clean(code)
    try:
      parsed_code = ast.parse(code, mode='exec')
      PythonCodeParser._CodeValidator(code, perm).visit(parsed_code)
    except SyntaxError as e:
      raise CodeError(code, e) from e
    return code, parsed_code

  def clean(self, code_text: str) -> str:
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
      if (not in_comment
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
          # case.
          if i < len(code_text) - 1 and code_text[i + 1] not in '.,]}) \t\n+*':
            c = f'\\{c}'
          else:
            quote_char = None
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
      if pos > 0 and self._ID_REGEX.match(code[:pos]):
        code = code[pos:]
    else:
      # Maybe-code that resides not within a code markdown block.
      code = code_text
    return inspect.cleandoc(code).strip()


# Key in the returned dict that represents the final result.
_FINAL_RESULT_KEY = '__result__'


_TLS_CODE_RUN_PERMISSION = '__code_run_permission__'
_TLS_CODE_RUN_CONTEXT = '__code_run_context__'


@contextlib.contextmanager
def permission(perm: CodePermission):
  """Context manager for controling the permission for code execution.

  When the `permission` context manager is nested, the outtermost permission
  will be used. This design allows users to control permission at the top level.

  Args:
    perm: Code execution permission.

  Yields:
    Actual permission applied.
  """

  outter_perm = pg.object_utils.thread_local_get(_TLS_CODE_RUN_PERMISSION, None)

  # Use the top-level permission as the actual permission
  if outter_perm is not None:
    perm = outter_perm

  pg.object_utils.thread_local_set(_TLS_CODE_RUN_PERMISSION, perm)

  try:
    yield perm
  finally:
    if outter_perm is None:
      pg.object_utils.thread_local_del(_TLS_CODE_RUN_PERMISSION)


def get_permission() -> CodePermission:
  """Gets the current permission for code execution."""
  return pg.object_utils.thread_local_get(
      _TLS_CODE_RUN_PERMISSION, CodePermission.ALL)


@contextlib.contextmanager
def context(**kwargs):
  """Context manager to inject symbols for code execution."""
  ctx = get_context()
  ctx.update(kwargs)
  pg.object_utils.thread_local_push(_TLS_CODE_RUN_CONTEXT, ctx)

  try:
    yield ctx
  finally:
    pg.object_utils.thread_local_pop(_TLS_CODE_RUN_CONTEXT)


def get_context() -> dict[str, Any]:
  """Gets the current context for code execution."""
  context_stack = pg.object_utils.thread_local_get(_TLS_CODE_RUN_CONTEXT, None)
  return dict(context_stack[-1]) if context_stack else {}


def run(
    code: str,
    perm: CodePermission | None = None,
    **kwargs
    ) -> dict[str, Any]:
  """Executes Python code.

  Features:
    * Fine-grained execution policy for limiting what APIs could be executed.
      This eliminates the need for sandboxing.
    * It exposes both the final results and intermediate results (variables).

  Args:
    code: Python code to run.
    perm: Permission for the Python code to run.
    **kwargs: The override to the key value pairs provided in `run_context`,
      which will be exposed as symbols to be referenced by the code.

  Returns:
    A dict of variable names to their evaluated values as the output of the
    code to run. The value for the last line can be accessed by key
    '__result__'.
  """
  # Set up the permission and context.
  perm = perm or get_permission()
  ctx = dict(get_context())
  ctx.update(kwargs)

  # Parse the code str.
  code, code_block = PythonCodeParser().parse(code, perm)
  global_vars, local_vars = ctx, {}

  if hasattr(code_block.body[-1], 'value'):
    last_expr = code_block.body.pop()  # pytype: disable=attribute-error
    result_vars = [_FINAL_RESULT_KEY]

    if isinstance(last_expr, ast.Assign):
      for name_node in last_expr.targets:
        result_vars.append(name_node.id)

    last_expr = ast.Expression(last_expr.value)  # pytype: disable=attribute-error

    try:
      # Execute the lines before the last expression.
      exec(compile(code_block, '', mode='exec'), global_vars, local_vars)  # pylint: disable=exec-used

      # Evaluate the last expression.
      result = eval(  # pylint: disable=eval-used
          compile(last_expr, '', mode='eval'), global_vars, local_vars)
    except Exception as e:
      raise CodeError(code, e) from e

    for result_var in result_vars:
      local_vars[result_var] = result
  else:
    try:
      exec(compile(code_block, '', mode='exec'), global_vars, local_vars)  # pylint: disable=exec-used
    except Exception as e:
      raise CodeError(code, e) from e
    local_vars[_FINAL_RESULT_KEY] = list(local_vars.values())[-1]
  return local_vars


class PythonCode(pg.Functor):
  """Symbolic class for Python code."""

  source: Annotated[
      str,
      'Source code.'
  ]

  def _call(self) -> Any:
    """Returns the final result."""
    return self.eval()[_FINAL_RESULT_KEY]

  def eval(self) -> dict[str, Any]:
    """Evaluates the code and return a dict of variables to their values."""
    return run(self.source)
