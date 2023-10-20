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
import functools
import inspect
import io
import multiprocessing
import re
import textwrap
from typing import Annotated, Any, Callable, ContextManager

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
          if nt in (',', '[', ']', '}', ')', '+', '*', '%', '\n', ':'):
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
      if pos > 0 and self._ID_REGEX.match(code[:pos]):
        code = code[pos:]
    else:
      # Maybe-code that resides not within a code markdown block.
      code = code_text
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
    **kwargs: The override to the key value pairs provided in `context`,
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


def sandbox_call(
    func: Callable[..., Any],
    *args,
    timeout: int | None = None,
    **kwargs) -> Any:
  """Calls a function with sandboxing.

  Args:
    func: Function to call.
    *args: Positional arguments for `func`
    timeout: Execution timeout in seconds. If None, wait `func` to complete.
    **kwargs: Keyword arguments for `func`.

  Returns:
    Return value from `func`.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception raised from `func`.
  """
  def _call(q, *args, **kwargs):
    try:
      q.put(pg.to_json_str(func(*args, **kwargs)))
    except Exception as e:  # pylint: disable=broad-exception-caught
      q.put(e)

  q = multiprocessing.Queue()
  p = multiprocessing.Process(
      target=_call, args=tuple([q] + list(args)), kwargs=kwargs)
  p.start()
  p.join(timeout=timeout)
  if p.is_alive():
    p.terminate()
    raise TimeoutError(f'Execution time exceed {timeout} seconds.')
  x = q.get()
  if isinstance(x, Exception):
    raise x
  return pg.from_json_str(x)


def sandbox_run(
    code: str,
    perm: CodePermission | None = None,
    timeout: int | None = None,
    **kwargs) -> dict[str, Any]:
  """Run Python code with sandboxing.

  Args:
    code: Code to run.
    perm: Permissiong to run.
    timeout: Execution timeout in seconds. If None, wait the code the complete.
    **kwargs: Globals that could be accessed within the code.

  Returns:
    A dict of local variables, with the value of the last expression kept
      in key `__result__`.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception  that are raised from the code.
  """
  return sandbox_call(run, code, perm, time=timeout, **kwargs)


class PythonCode(pg.Object):
  """Symbolic class for Python code."""

  source: Annotated[
      str,
      'Source code.'
  ]

  _TLS_AUTO_RUN = '__auto_run__'

  @classmethod
  def auto_run(
      cls,
      enabled: bool = True,
      sandbox: bool = True,
      timeout: int | None = 5) -> ContextManager[None]:
    """Returns a context manager to enable/disable auto run of PythonCode.

    `auto_run` is thread-safe and can be nested. For example::

      with lf.coding.PythonCode.auto_run(True, sandbox=True, timeout=1):
        a = PythonCode('1 + 2')
        assert a == 1
        with lf.coding.PythonCode.auto_run(False):
          b = PythonCode('2 + 3')
          assert isinstance(b, lf.PythonCode)

    Args:
      enabled: If True, enable auto call for functors.
        Otherwise, auto call will be disabled.
      sandbox: If True, execute the python code in a sandbox. Applicable when
        `enabled` is set to True.
      timeout: Timeout in seconds. Applicable when both `enabled` and `sandbox`
        are set to True.

    Returns:
      A context manager for enabling/disabling auto call for functors.
    """
    return pg.object_utils.thread_local_value_scope(
        cls._TLS_AUTO_RUN,
        pg.Dict(enabled=enabled, sandbox=sandbox, timeout=timeout),
        pg.Dict(enabled=False, sandbox=False, timeout=None),
    )

  def __new__(cls, *args, **kwargs):
    instance = object.__new__(cls)
    auto_run = pg.object_utils.thread_local_get(
        cls._TLS_AUTO_RUN,
        pg.Dict(enabled=False, sandbox=False, timeout=None)
    )
    if auto_run.enabled:
      instance.__init__(*args, **kwargs)
      return instance(
          sandbox=auto_run.sandbox, timeout=auto_run.timeout)
    return instance

  def __call__(
      self,
      *,
      sandbox: bool = True,
      timeout: int | None = 5,
      global_vars: dict[str, Any] | None = None
      ) -> Any:
    """Returns the value of the last expression from the source.

    Args:
      sandbox: If True, evaluate the code in a sand box.
      timeout: Timeout in seconds. If None, there is no timeout.
        Applicable when sandbox is set to True.
      global_vars: Global variables that could be accessed from the source code.

    Returns:
      The value of the last expression in the source code.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    return self.eval(
        sandbox=sandbox,
        timeout=timeout,
        global_vars=global_vars)[_FINAL_RESULT_KEY]

  def eval(
      self,
      *,
      sandbox: bool = True, timeout: int | None = 5,
      global_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluates the code and return a dict of local variable names to values.

    Args:
      sandbox: If True, evaluate the code in a sand box.
      timeout: Timeout in seconds. If None, there is no timeout.
        Applicable when sandbox is set to True.
      global_vars: Global variables that could be accessed from the source code.

    Returns:
      A dict of local variable names defined in the source code to their values.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    global_vars = global_vars or {}
    if sandbox:
      return sandbox_run(self.source, timeout=timeout, **global_vars)
    return run(self.source, **global_vars)


class PythonFunction(pg.Object):
  """Generated Python function via source code."""
  name: str
  args: dict[str, str]
  returns: str
  source: Annotated[
      str,
      'Source code for the Python function. '
  ]

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('implementation', None)

  @functools.cached_property
  def implementation(self) -> Callable[..., Any]:
    """Returns the function implementation based on source code."""
    return run(self.source)[_FINAL_RESULT_KEY]

  def __call__(
      self,
      *args,
      sandbox: bool = True,
      timeout: int | None = 5,
      **kwargs) -> Any:
    """Calls the function.

    Args:
      *args: Positional arguments that will be passed to the implementation.
      sandbox: If True, call the implementation in sandbox.
      timeout: Timeout in seconds. If None, there is no timeout. Applicable when
        sandbox is set to True.
      **kwargs: Keyword arguments that will be passed to the implementation.

    Returns:
      The return value of the implementation.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    if sandbox:
      return sandbox_call(self.implementation, *args, timeout=timeout, **kwargs)
    return self.implementation(*args, **kwargs)
