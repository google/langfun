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
"""Python code execution."""

import ast
import contextlib
import io
import multiprocessing
from typing import Any, Callable

from langfun.core.coding.python import errors
from langfun.core.coding.python import parsing
from langfun.core.coding.python import permissions
import pyglove as pg


# Key in returned dict that captures stdout.
STDOUT_KEY = '__stdout__'

# Key in the returned dict that represents the final result.
RESULT_KEY = '__result__'
_TLS_CODE_RUN_CONTEXT = '__code_run_context__'


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


def evaluate(
    code: str,
    *,
    global_vars: dict[str, Any] | None = None,
    permission: permissions.CodePermission | None = None,
    outputs_intermediate: bool = False,
) -> Any | dict[str, Any]:
  """Executes Python code.

  Features:
    * Fine-grained execution policy for limiting what APIs could be executed.
      This eliminates the need for sandboxing.
    * It exposes both the final results and intermediate results (variables).

  Args:
    code: Python code to run.
    global_vars: An optional dict as the globals that could be referenced by the
      code.
    permission: Permission for the Python code to run.
    outputs_intermediate: If True, intermediate output will be outputted as a
      dict, with the last line's value accessible by key '__result__'. Otherwise
      the value of the last line will be returned.

  Returns:
    The value of the last line of the code. Or a dict of variable name to
    their values if `outputs_intermediate` is set to True, with the final result
    accessible by key '__result__'.
  """
  # Set up the permission and context.
  permission = permission or permissions.get_permission()
  ctx = dict(get_context())
  if global_vars:
    ctx.update(global_vars)

  # Parse the code str.
  code, code_block = parsing.PythonCodeParser().parse(code, permission)
  global_vars, orig_global_vars = ctx, ctx.copy()

  # No code.
  if not code_block.body:
    return {} if outputs_intermediate else None

  stdout = io.StringIO()
  with contextlib.redirect_stdout(stdout):
    if hasattr(code_block.body[-1], 'value'):
      last_expr = code_block.body.pop()  # pytype: disable=attribute-error
      result_vars = [RESULT_KEY]

      if isinstance(last_expr, ast.Assign):
        for name_node in last_expr.targets:
          result_vars.append(name_node.id)

      last_expr = ast.Expression(last_expr.value)  # pytype: disable=attribute-error

      try:
        # Execute the lines before the last expression.
        # NOTE(daiyip): Only a `globals` dict is specified here, which will also
        # be used to output intermediate values by `exec`. We do not specify a
        # separate `locals` dict here, for - "If exec gets two separate objects
        # as globals and locals, the code will be executed as if it were
        # embedded in a class definition." - as the Python document explains.
        # The outcome is that new functions defined in the code block could not
        # be called by other newly defined functions.
        # Refer to https://stackoverflow.com/questions/
        # 73940751/why-cant-i-call-a-function-from-another-function-using-exec
        # for more details.
        exec(compile(code_block, '', mode='exec'), global_vars)  # pylint: disable=exec-used

        # Evaluate the last expression.
        result = eval(  # pylint: disable=eval-used
            compile(last_expr, '', mode='eval'), global_vars
        )
      except Exception as e:
        raise errors.CodeError(code, e) from e

      for result_var in result_vars:
        global_vars[result_var] = result
    else:
      try:
        exec(compile(code_block, '', mode='exec'), global_vars)  # pylint: disable=exec-used
      except Exception as e:
        raise errors.CodeError(code, e) from e
      global_vars[RESULT_KEY] = list(global_vars.values())[-1]

  if outputs_intermediate:
    outputs = {}
    for k, v in global_vars.items():
      if k == '__builtins__':
        continue
      if k not in orig_global_vars or v is not orig_global_vars[k]:
        outputs[k] = v
    # Add stdout to outputs.
    outputs[STDOUT_KEY] = stdout.getvalue()
    return outputs
  return global_vars[RESULT_KEY]


def sandbox_call(
    func: Callable[..., Any],
    *args,
    timeout: float | None = None,
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
    # NOTE(daiyip): if `q` is closed by the main process when `q.put` is called
    # on a subprocess, ValueError will be raised. This is okay since the main
    # process is no longer waiting for the result, and the subprocess could
    # recycled with non-zero error code, which does not affect the main
    # process.
    def _run():
      r = func(*args, **kwargs)
      try:
        return pg.to_json_str(r)
      except Exception as e:
        raise errors.SerializationError(
            f'Cannot serialize sandbox result: {r}', e
        ) from e

    try:
      q.put(_run())
    except Exception as e:  # pylint: disable=broad-exception-caught
      q.put(e)

  q = multiprocessing.Queue()
  try:
    p = multiprocessing.Process(
        target=_call, args=tuple([q] + list(args)), kwargs=kwargs)
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
      # We use `kill` instead of `terminate` to release process resources
      # right away.
      p.kill()
      raise TimeoutError(f'Execution time exceed {timeout} seconds.')
    x = q.get()
    if isinstance(x, Exception):
      raise x
    try:
      return pg.from_json_str(x)
    except Exception as e:
      raise errors.SerializationError(
          'Cannot deserialize the output from sandbox.', e
      ) from e
  finally:
    q.close()


def call(
    func: Callable[..., Any],
    *args,
    sandbox: bool | None = None,
    timeout: float | None = None,
    **kwargs
) -> Any:
  """Calls a function with sandbox support.

  Args:
    func: Function to call.
    *args: Postional args that will be passed to `func`.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: Execution timeout in seconds. If None, wait the code the complete.
    **kwargs: Keyword args that will be passed to `func`.

  Returns:
    The return value of `func`.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception  that are raised from `func`.
  """
  if sandbox is None:
    try:
      return sandbox_call(func, *args, timeout=timeout, **kwargs)
    # NOTE(daiyip): output could be serialized across processes, giving it
    # already finishes on sandbox, so it should be much safer to run under
    # current process.
    except errors.SerializationError:
      return func(*args, **kwargs)
  elif sandbox:
    return sandbox_call(func, *args, timeout=timeout, **kwargs)
  else:
    return func(*args, **kwargs)


def run(
    code: str,
    *,
    global_vars: dict[str, Any] | None = None,
    permission: permissions.CodePermission | None = None,
    outputs_intermediate: bool = False,
    sandbox: bool | None = None,
    timeout: float | None = None,
) -> Any | dict[str, Any]:
  """Executes Python code.

  Features:
    * Fine-grained execution policy for limiting what APIs could be executed.
      This eliminates the need for sandboxing.
    * It exposes both the final results and intermediate results (variables).

  Args:
    code: Python code to run.
    global_vars: An optional dict of
    permission: Permission for the Python code to run.
    outputs_intermediate: If True, all variables created as locals will be
      returned, with the final result accessible by key '__result__'. Otherwise
      only the final result will be returned.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: Execution timeout in seconds. If None, wait the code the complete.

  Returns:
    The value of the last line of the code block. Or a dict of variable
    names of all locals to their evaluated values as the output of the code to
    run. The value for the last line can be accessed by key '__result__'.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception  that are raised from the code.
  """
  return call(
      evaluate, code=code, global_vars=global_vars, permission=permission,
      outputs_intermediate=outputs_intermediate,
      sandbox=sandbox, timeout=timeout)
