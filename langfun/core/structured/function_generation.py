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
"""LLM-based function generation."""

import functools
import inspect
import re
from typing import Any, Callable, Optional, Tuple

from langfun.core import language_model
from langfun.core import template
from langfun.core.coding import python
from langfun.core.structured import prompting
import pyglove as pg


def unittest_gen(signature, lm, num_retries=10):
  """Generates unit tests for a python function signature."""

  class UnitTest(pg.Object):
    """A valid unit test for a python function."""

    input: dict[str, Any]
    expected_output: Any

  class PythonFunctionSignature(pg.Object):
    signature: str

  unittest_examples = None
  for _ in range(num_retries):
    r = prompting.query(
        PythonFunctionSignature(signature=signature),
        list[UnitTest],
        lm=lm,
        default=None,
    )
    if isinstance(r, list) and r:
      unittest_examples = []
      for unit_test in r:
        unittest_examples.append((unit_test.input, unit_test.expected_output))
      break

  return unittest_examples


def unittest_with_test_cases(f, unittests):
  """Applies unit tests to a python function to be tested."""
  if not unittests:
    raise ValueError(f"No unit tests provided: {unittests}")

  for unit_test in unittests:
    inputs = unit_test[0]
    if isinstance(inputs, dict):
      actual = f(**inputs)
    elif isinstance(inputs, tuple):
      actual = f(*inputs)
    else:
      actual = f(inputs)

    expected = unit_test[1]
    assert (
        actual == expected
    ), f"Test FAILED: Inputs: {inputs}, Expected: {expected}, Actual: {actual}"


def _function_gen(
    func: Callable[..., Any],
    signature: str,
    lm: language_model.LanguageModel,
    num_retries: int = 10,
    unittest: Optional[
        Callable[[Callable[..., Any]], None] | list[Tuple[Any, Any]]
    ] = None,
):
  """Generates a python function with LLM and verify its quality with unit testing."""

  class PythonFunctionPrompt(template.Template):
    r"""A template for a python function generation.

    Please reply to the last PYTHON_FUNCTION_SIGNATURE with a self-sufficient,
    error-free, and efficiently coded PYTHON_FUNCTION, crafted to the standards
    of a world-class programmer.

      PYTHON_FUNCTION_SIGNATURE:
        ```python
        def calculate_area_circle(radius: float) -> float:
        \"\"\"Calculates the area of a circle given its radius.

        Args:
            radius: The radius of the circle.

        Returns:
            The area of the circle.
        \"\"\"
        ```

      PYTHON_FUNCTION:
        ```python
        def calculate_area_circle(radius: float) -> float:
        \"\"\"Calculates the area of a circle given its radius.

        Args:
            radius: The radius of the circle.

        Returns:
            The area of the circle.
        \"\"\"
        import math

        area = math.pi * radius**2
        return area
        ```

      PYTHON_FUNCTION_SIGNATURE:
        ```python
        {{signature}}
        ```

      PYTHON_FUNCTION:
    """

  unittest_examples = None
  if unittest is None:
    unittest_examples = unittest_gen(signature, lm=lm)
  elif not callable(unittest):
    unittest_examples = unittest

  for _ in range(num_retries):
    try:
      source_code = prompting.query(
          PythonFunctionPrompt(signature=signature), lm=lm
      )
      f = python.evaluate(source_code)

      # Check whether the sigantures are the same.
      if inspect.signature(f) != inspect.signature(func):
        continue

      if callable(unittest):
        unittest(f)
      else:
        unittest_with_test_cases(f, unittest_examples)

      return f, source_code
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  return None, None


def _process_signature(signature):
  # Remove the decorator.
  pattern = r"^\@.*function_gen.*$"
  signature = re.sub(pattern, "", signature, flags=re.MULTILINE)
  # Remove the possible 'pass' in an empty function.
  pattern = r"^\s*pass\s*$"
  signature = re.sub(pattern, "", signature, flags=re.MULTILINE)
  return signature.strip()


def function_gen(
    lm: language_model.LanguageModel,
    cache_filename: str | None = None,
    num_retries: int = 10,
    unittest: Optional[
        Callable[[Callable[..., Any]], None] | list[Tuple[Any, Any]]
    ] = None,
):
  """A decorator for automating function generation using a language model.

  This decorator should be applied to functions that are not yet implemented. It
  facilitates the implementation via the specified LLM, ensuring
  quality through unit tests.

  Args:
      lm (lf.LanguageModel): The language model used for generating function
        implementations.
      cache_filename (str | None): Optional. The path of the file where
        generated function implementations are loaded from or saved to.
      num_retries (int): Maximum number of attempts the language model should
        make to generate a suitable function implementation.
      unittest: This optional parameter enables the definition of custom unit
        tests. You can either provide a list of test cases as tuples of inputs
        and outputs, or a function that throws an error if a test fails. If left
        as None (the default setting), the LLM will automatically create the
        unit test cases.

  Returns:
      The implemented function object.
  """

  def _decorate(func):
    setattr(func, "__function__", None)
    setattr(func, "__source_code__", None)

    @functools.wraps(func)
    def lm_generated_func(*args, **kwargs):
      if func.__function__ is not None:
        return func.__function__(*args, **kwargs)

      signature = _process_signature(inspect.getsource(func))
      cache = pg.Dict()
      if cache_filename is not None:
        try:
          cache = pg.load(cache_filename)
        except FileNotFoundError:
          pg.logging.warning(
              "Creating a new cache as cache file '%s' does not exist.",
              cache_filename,
          )

        if signature in cache:
          func.__source_code__ = cache[signature]
          func.__function__ = python.evaluate(func.__source_code__)
          return func.__function__(*args, **kwargs)

      func.__function__, func.__source_code__ = _function_gen(
          func, signature, lm, num_retries=num_retries, unittest=unittest
      )
      if func.__function__ is None:
        raise ValueError(f"Function generation failed. Signature:\n{signature}")

      if cache_filename is not None:
        cache[signature] = func.__source_code__
        cache.save(cache_filename)
      return func.__function__(*args, **kwargs)

    lm_generated_func.__name__ = func.__name__
    lm_generated_func.__qualname__ = func.__qualname__
    lm_generated_func.__module__ = func.__module__
    lm_generated_func.source = lambda: func.__source_code__
    return lm_generated_func

  return _decorate
