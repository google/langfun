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
"""Python code error correction."""
from typing import Any
import langfun.core as lf
from langfun.core.coding.python import errors
from langfun.core.coding.python import execution
import pyglove as pg


class CodeWithError(pg.Object):
  """Python code with error."""

  code: str
  error: str


class CorrectedCode(pg.Object):
  corrected_code: str


def run_with_correction(
    code: str,
    error: str | None = None,
    *,
    global_vars: dict[str, Any] | None = None,
    lm: lf.LanguageModel | None = None,
    max_attempts: int = 5,
    sandbox: bool | None = None,
    timeout: int | None = 5,
    returns_code: bool = False,
    outputs_intermediate: bool = False,
) -> Any | tuple[Any, str]:
  """Correct code with a language model via self-play.

  Args:
    code: The source code that may or may not be problematic.
    error: An optional initial error for `code` when it's problematic, usually
      caught from elsewhere when it ran. If None, code will be executed once to
      verify if its good and obtain a feedback error message.
    global_vars: A dict of str to value as the global variables that could be
      accessed within the corrected code.
    lm: Language model to be used. If not specified, it will try to use the `lm`
      under `lf.context`.
    max_attempts: Max number of attempts for the correction.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: The timeout for running the corrected code. If None, there is no
      timeout. Applicable only when sandbox is set to True.
    returns_code: If True, the return value is a tuple of (result, final code).
      Otherwise the return value is the result only.
    outputs_intermediate: If True, intermediate output will be outputted as a
      dict, with the last line's value accessible by key '__result__'. Otherwise
      the value of the last line will be returned.

  Returns:
    Run result if `returns_code` is set to False (default), otherwise a tuple
      of (result, final code str).

  Raises:
    `lf.CodeError`: If code cannot be corrected after `max_attempts`.
  """
  # Delay import at runtime to avoid circular depenency.
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from langfun.core.structured import querying
  # pytype: enable=import-error
  # pylint: enable=g-import-not-at-top

  if max_attempts == 0:
    result = _maybe_custom_validate(
        execution.run(
            code,
            global_vars=global_vars,
            sandbox=sandbox,
            timeout=timeout,
            outputs_intermediate=outputs_intermediate,
        )
    )
    return (result, code) if returns_code else result

  def result_and_error(code: str) -> tuple[Any, str | None]:
    try:
      result = _maybe_custom_validate(
          execution.run(
              code,
              global_vars=global_vars,
              sandbox=sandbox,
              timeout=timeout,
              outputs_intermediate=outputs_intermediate,
          )
      )
      return (result, None)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return (None, _error_feedback_str(e))

  if error is None:
    result, error = result_and_error(code)
    if error is None:
      return (result, code) if returns_code else result

  num_attempts = 0
  for _ in range(max_attempts):
    num_attempts += 1
    # Catch query() error when the LM could not formulate a completion
    # structure.
    try:
      # Disable autofix for code correction to avoid recursion.
      correction = querying.query(
          CodeWithError(code=code, error=error), CorrectedCode, lm=lm, autofix=0
      )
    except errors.CodeError:
      break

    code = correction.corrected_code
    result, error = result_and_error(code)
    if error is None:
      return (result, code) if returns_code else result

  raise errors.CodeError(
      code,
      RuntimeError(
          f"Cannot correct code after {num_attempts} attempts. "
          f"Last error: {error}."
      ),
  )


def correct(
    code: str,
    error: str | None = None,
    *,
    global_vars: dict[str, Any] | None = None,
    lm: lf.LanguageModel = lf.contextual(),
    max_attempts: int = 5,
    sandbox: bool | None = None,
    timeout: int | None = 5,
) -> str:
  """Correct code with a language model via self-play.

  Args:
    code: The source code that may or may not be problematic.
    error: An optional initial error for `code` when it's problematic, usually
      caught from elsewhere when it ran. If None, code will be executed once to
      verify if its good and obtain a feedback error message.
    global_vars: A dict of str to value as the global variables that could be
      accessed within the corrected code.
    lm: Language model to be used. If not specified, it will try to use the `lm`
      under `lf.context`.
    max_attempts: Max number of attempts for the correction.
    sandbox: If True, run code in sandbox; If False, run code in current
      process. If None, run in sandbox first, if the output could not be
      serialized and pass to current process, run the code again in current
      process.
    timeout: The timeout for running the corrected code. If None, there is no
      timeout. Applicable only when sandbox is set to True.

  Returns:
    The final correct code if corrections are successful.

  Raises:
    `lf.CodeError`: If code cannot be corrected after `max_attempts`.
  """
  return run_with_correction(
      code,
      error=error,
      global_vars=global_vars,
      lm=lm,
      max_attempts=max_attempts,
      sandbox=sandbox,
      timeout=timeout,
      returns_code=True,
  )[1]


def _error_feedback_str(error: Exception) -> str:
  """Returns the error str for feedback."""
  if isinstance(error, errors.CodeError):
    return lf.text_formatting.decolored(
        error.format(include_complete_code=False)
    )
  else:
    return f"Encountered {error.__class__.__name__}: {error}"


def _maybe_custom_validate(result: Any) -> Any:
  """Apply custom validation through __validate_generation__ method."""
  if isinstance(result, dict) and "__result__" in result:
    r = result["__result__"]
  else:
    r = result

  if hasattr(r, "__validate__"):
    r.__validate__()
  return result
