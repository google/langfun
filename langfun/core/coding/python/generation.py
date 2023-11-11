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

from langfun.core.coding.python import execution
import pyglove as pg


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
      global_vars: dict[str, Any] | None = None
      ) -> Any:
    """Returns the value of the last expression from the source.

    Args:
      sandbox: If True, run code in sandbox; If False, run code in current
        process. If None, run in sandbox first, if the output could not be
        serialized and pass to current process, run the code again in current
        process.
      timeout: Timeout in seconds. If None, there is no timeout.
        Applicable when sandbox is set to True.
      global_vars: Global variables that could be accessed from the source code.

    Returns:
      The value of the last expression in the source code.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    return execution.run(
        self.source, global_vars=global_vars, sandbox=sandbox, timeout=timeout)

  def eval(
      self,
      *,
      sandbox: bool | None = None,
      timeout: int | None = 5,
      global_vars: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluates the code and return a dict of local variable names to values.

    Args:
      sandbox: If True, run code in sandbox; If False, run code in current
        process. If None, run in sandbox first, if the output could not be
        serialized and pass to current process, run the code again in current
        process.
      timeout: Timeout in seconds. If None, there is no timeout.
        Applicable when sandbox is set to True.
      global_vars: Global variables that could be accessed from the source code.

    Returns:
      A dict of local variable names defined in the source code to their values.

    Raises:
      TimeoutError: If `sandbox` is True and timeout has reached.
      Exception: Any errors that the source code has raised.
    """
    return execution.run(
        self.source,
        global_vars=global_vars,
        sandbox=sandbox,
        timeout=timeout,
        outputs_intermediate=True)


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
    return execution.call(
        self.implementation, *args, sandbox=sandbox, timeout=timeout, **kwargs)
