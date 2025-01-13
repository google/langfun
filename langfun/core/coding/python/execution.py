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

from typing import Any

from langfun.core.coding.python import parsing
import pyglove as pg


context = pg.coding.context
CodeError = pg.coding.CodeError
CodePermission = pg.coding.CodePermission
permission = pg.coding.permission


def evaluate(
    code: str,
    *,
    global_vars: dict[str, Any] | None = None,
    permission: CodePermission | None = None,   # pylint: disable=redefined-outer-name
    returns_stdout: bool = False,
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
    returns_stdout: If True, the stdout (a str) will be returned.
    outputs_intermediate: Applicable when returns_stdout is False. If True,
      intermediate output will be outputted as a dict, with the last line's
      value accessible by key '__result__' and the std output accessible by
      key '__stdout__'. Otherwise the value of the last line will be returned.

  Returns:
    The value of the last line of the code block. Or a dict of variable
    names of all locals to their evaluated values as the output of the code to
    run. The value for the last line can be accessed by key '__result__'. Or the
    stdout as a str.
  """
  return pg.coding.evaluate(
      parsing.clean(code),
      global_vars=global_vars,
      permission=permission,
      returns_stdout=returns_stdout,
      outputs_intermediate=outputs_intermediate,
  )


def run(
    code: str,
    *,
    global_vars: dict[str, Any] | None = None,
    permission: CodePermission | None = None,  # pylint: disable=redefined-outer-name
    returns_stdout: bool = False,
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
    returns_stdout: If True, the stdout (a str) will be returned.
    outputs_intermediate: Applicable when returns_stdout is False. If True,
      intermediate output will be outputted as a dict, with the last line's
      value accessible by key '__result__' and the std output accessible by
      key '__stdout__'. Otherwise the value of the last line will be returned.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: Execution timeout in seconds. If None, wait the code the complete.

  Returns:
    The value of the last line of the code block. Or a dict of variable
    names of all locals to their evaluated values as the output of the code to
    run. The value for the last line can be accessed by key '__result__'. Or the
    stdout as a str.

  Raises:
    TimeoutError: If the execution time exceeds the timeout.
    Exception: Exception  that are raised from the code.
  """
  return pg.coding.run(
      parsing.clean(code),
      global_vars=global_vars,
      permission=permission,
      returns_stdout=returns_stdout,
      outputs_intermediate=outputs_intermediate,
      sandbox=sandbox,
      timeout=timeout,
  )
