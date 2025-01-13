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
"""Structures for Python code generation."""

import functools
from typing import Annotated, Any, Callable, ContextManager
import langfun.core as lf
from langfun.core.coding.python import correction
from langfun.core.coding.python import execution
import pyglove as pg


class PythonCode(pg.Object):
  """Symbolic class for Python code.

  The value of the last expression of the source will be the returned value.
  """

  source: Annotated[
      str,
      'Source code. Avoid using docstring for defined functions and classes.',
  ]

  _TLS_AUTO_RUN = '__auto_run__'

  @classmethod
  def auto_run(
      cls,
      enabled: bool = True,
      sandbox: bool | None = None,
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
      sandbox: If True, run code in sandbox; If False, run code in current
        process. If None, run in sandbox first, if the output could not be
        serialized and pass to current process, run the code again in current
        process. Applicable when `enabled` is set to True.
      timeout: Timeout in seconds. Applicable when both `enabled` and `sandbox`
        are set to True.

    Returns:
      A context manager for enabling/disabling auto call for functors.
    """
    return pg.object_utils.thread_local_value_scope(
        cls._TLS_AUTO_RUN,
        pg.Dict(enabled=enabled, sandbox=sandbox, timeout=timeout),
        pg.Dict(enabled=False, sandbox=None, timeout=None),
    )

  def __new__(cls, *args, **kwargs):
    instance = object.__new__(cls)
    auto_run = pg.object_utils.thread_local_get(
        cls._TLS_AUTO_RUN,
        pg.Dict(enabled=False, sandbox=None, timeout=None)
    )
    if auto_run.enabled:
      instance.__init__(*args, **kwargs)
      return instance(
          sandbox=auto_run.sandbox, timeout=auto_run.timeout)
    return instance

  def __call__(
      self,
      *,
      sandbox: bool | None = None,
      timeout: int | None = 5,
      global_vars: dict[str, Any] | None = None,
      returns_stdout: bool = False,
      outputs_intermediate: bool = False,
      autofix: int = 3,
      autofix_lm: lf.LanguageModel | None = None,
  ) -> Any:
    """Returns the value of the last expression from the source.

    Args:
      sandbox: If True, run code in sandbox; If False, run code in current
        process. If None, run in sandbox first, if the output could not be
        serialized and pass to current process, run the code again in current
        process.
      timeout: Timeout in seconds. If None, there is no timeout. Applicable when
        sandbox is set to True.
      global_vars: Global variables that could be accessed from the source code.
      returns_stdout: If True, the stdout (a str) will be returned.
      outputs_intermediate: Applicable when returns_stdout is False. If True,
        intermediate output will be outputted as a dict, with the last line's
        value accessible by key '__result__' and the std output accessible by 
        key '__stdout__'. Otherwise the value of the last line will be returned.
      autofix: Number of attempts to auto fix the generated code. If 0, autofix
        is disabled.
      autofix_lm: Language model to be used. If not specified, it will try to
        use the `lm` under `lf.context`.

    Returns:
      The value of the last expression in the source code. Or a dict of local
      variable names defined in the source code to their values if
      `outputs_intermediate` is set to True. The value for the last line can be
      accessed by key '__result__'. Or the stdout as a str if `returns_stdout`
      is set to True.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    result, updated_code = correction.run_with_correction(
        self.source,
        global_vars=global_vars,
        sandbox=sandbox,
        timeout=timeout,
        max_attempts=autofix,
        lm=autofix_lm,
        returns_code=True,
        returns_stdout=returns_stdout,
        outputs_intermediate=outputs_intermediate,
    )
    self.rebind(source=updated_code)
    return result

  def eval(
      self,
      *,
      sandbox: bool | None = None,
      timeout: int | None = 5,
      global_vars: dict[str, Any] | None = None,
      autofix: int = 3,
      autofix_lm: lf.LanguageModel = lf.contextual(),
  ) -> Any | tuple[Any, str]:
    """Evaluates the code and return a dict of local variable names to values.

    Args:
      sandbox: If True, run code in sandbox; If False, run code in current
        process. If None, run in sandbox first, if the output could not be
        serialized and pass to current process, run the code again in current
        process.
      timeout: Timeout in seconds. If None, there is no timeout. Applicable when
        sandbox is set to True.
      global_vars: Global variables that could be accessed from the source code.
      autofix: Number of attempts to auto fix the generated code. If 0, autofix
        is disabled. Auto-fix is not supported for 'json' protocol.
      autofix_lm: Language model to be used. If not specified, it will try to
        use the `lm` under `lf.context`.

    Returns:
      A dict of local variable names defined in the source code to their values
      if `returns_code` is set to False (default), otherwise a tuple
      of (dict, final code str).

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    return self(
        sandbox=sandbox,
        timeout=timeout,
        global_vars=global_vars,
        autofix=autofix,
        autofix_lm=autofix_lm,
        outputs_intermediate=True,
    )


class PythonFunction(pg.Object):
  """Generated Python function via source code.

  The source code will be directly passed into eval() for execution and the
  output of the function will be returned.
  """

  name: str
  args: dict[str, str]
  returns: str
  source: Annotated[str, 'Source code for the Python function. ']

  def _on_bound(self):
    super()._on_bound()
    self.__dict__.pop('implementation', None)

  @functools.cached_property
  def implementation(self) -> Callable[..., Any]:
    """Returns the function implementation based on source code."""
    return execution.run(self.source)

  def __call__(
      self,
      *args,
      sandbox: bool | None = None,
      timeout: int | None = 5,
      **kwargs) -> Any:
    """Calls the function.

    Args:
      *args: Positional arguments that will be passed to the implementation.
      sandbox: If True, run code in sandbox; If False, run code in current
        process. If None, run in sandbox first, if the output could not be
        serialized and pass to current process, run the code again in current
        process.
      timeout: Timeout in seconds. If None, there is no timeout. Applicable when
        sandbox is set to True.
      **kwargs: Keyword arguments that will be passed to the implementation.

    Returns:
      The return value of the implementation.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    return pg.coding.maybe_sandbox_call(
        self.implementation, *args, sandbox=sandbox, timeout=timeout, **kwargs
    )
