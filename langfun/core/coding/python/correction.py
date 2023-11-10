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

import inspect
from typing import Any
import langfun.core as lf
from langfun.core.coding.python import errors
from langfun.core.coding.python import generation
import pyglove as pg


class CodeWithError(pg.Object):
  """Python code with error."""

  code: str
  error: str


class CodeCorrection(pg.Object):
  """Structure used for code correction."""

  latest_code: CodeWithError
  correction_history: list[CodeWithError]
  corrected_code: str


def correct(
    code: str,
    error: str | None = None,
    *,
    lm: lf.LanguageModel = lf.contextual(),
    examples: list[CodeCorrection] | None = None,
    sandbox: bool = True,
    timeout: int | None = 5,
    global_vars: dict[str, Any] | None = None,
    max_attempts: int = 5,
) -> str:
  """Correct code with a language model via self-play.

  Args:
    code: Problematic source code to correct.
    error: Error message for source code. If None, code will be executed once to
      get the error message.
    lm: Language model to be used. If not specified, it will try to use the `lm`
      under `lf.context`.
    examples: Code correction examples to use.
    sandbox: If True, run the corrected code within a sandbox.
    timeout: The timeout for running the corrected code. If None, there is no
      timeout. Applicable only when sandbox is set to True.
    global_vars: A dict of str to value as the global variables that could be
      accessed within the corrected code.
    max_attempts: Max number of attempts for the correction.

  Returns:
    The corrected source code.

  Raises:
    `lf.CodeError`: If code cannot be corrected after `max_attempts`.
  """
  # Delay import at runtime to avoid circular depenency.
  from langfun.core.structured import completion  # pylint: disable=g-import-not-at-top

  def error_from_code(code: str) -> str | None:
    try:
      _ = generation.PythonCode(code)(
          sandbox=sandbox, timeout=timeout, global_vars=global_vars
      )
      return None
    except errors.CodeError as e:
      error = lf.text_formatting.decolored(
          e.format(include_complete_code=False)
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      error = f"Encountered {e.__class__.__name__}: {e}"
    return error

  if error is None:
    error = error_from_code(code)

  examples = examples or DEFAULT_CODE_CORRECTION_EXAMPLES
  examples = [  # pylint: disable=g-complex-comprehension
      completion.completion_example(
          CodeCorrection.partial(ex.latest_code, ex.correction_history),
          ex,
      )
      for ex in examples
  ]

  history = []
  for _ in range(max_attempts):
    correction = CodeCorrection.partial(
        CodeWithError(code=code, error=error), history
    )
    correction = completion.complete(correction, lm=lm, examples=examples)
    history.append(CodeWithError(code=code, error=error))

    code = correction.corrected_code
    error = error_from_code(code)
    if error is None:
      return code
  raise errors.CodeError(
      code, RuntimeError(f"Cannot correct code after {max_attempts} attempts.")
  )


DEFAULT_CODE_CORRECTION_EXAMPLES = [
    CodeCorrection(
        CodeWithError(
            inspect.cleandoc("""
                class A(pg.Object):
                  x: str

                a = A('1')
                b = a + 'foo'
                """),
            (
                "TypeError: unsupported operand type(s) for +: 'A' and "
                "'str' (<unknown>, line 5)\n"
                "  b = a + 'foo'"
            ),
        ),
        correction_history=[
            CodeWithError(
                inspect.cleandoc("""
                    class A(pg.Object):
                      x: str

                    a = A(1
                    b = a + 'foo'
                    """),
                (
                    "SyntaxError: '(' was never closed (<unknown>, line 4)\n"
                    "  a = A(1"
                ),
            ),
            CodeWithError(
                inspect.cleandoc("""
                    class A(pg.Object):
                      x: str

                    a = A(1)
                    b = a + 'foo'
                    """),
                (
                    "TypeError: Expect <class 'str'> but encountered "
                    "<class 'int'>: 1. (path=x) (<unknown>, line 4)\n"
                    "  a = A(1)"
                ),
            ),
        ],
        corrected_code=inspect.cleandoc("""
            class A(pg.Object):
              x: str

            a = A('1')
            b = a.x + 'foo'
            """),
    ),
]
