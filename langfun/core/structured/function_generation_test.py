# Copyright 2024 The Langfun Authors
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
import inspect
import os
import tempfile
import unittest
from langfun.core.llms import fake
from langfun.core.structured import function_generation
import pyglove as pg


class FunctionGenerationTest(unittest.TestCase):

  def test_generate_function(self):
    function_gen_lm_response = inspect.cleandoc("""
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        """)
    unittest_lm_response = inspect.cleandoc("""
        ```python
          [
            UnitTest(
              input={
                'items': [1, 2, 3, 4, 5],
                'target': 3
              },
              expected_output=2
            ),
            UnitTest(
              input={
                'items': [1, 2, 3, 4, 5],
                'target': 6
              },
              expected_output=-1
            )
          ]
        ```
        """)

    lm = fake.StaticSequence([unittest_lm_response, function_gen_lm_response])

    @function_generation.function_gen(lm=lm, unittest='auto')
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)
    self.assertEqual(linear_search.source(), function_gen_lm_response)

  def test_generate_function_without_unittest(self):
    function_gen_lm_response = inspect.cleandoc("""
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        """)

    lm = fake.StaticSequence([function_gen_lm_response])

    @function_generation.function_gen(lm=lm)
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)
    self.assertEqual(linear_search.source(), function_gen_lm_response)

  def test_custom_unittest_examples(self):
    function_gen_lm_response = inspect.cleandoc("""
        ```python
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        ```
        """)

    lm = fake.StaticSequence([function_gen_lm_response])

    custom_unittest = [(([1, 2, 3, 4, 5], 3), 2)]

    @function_generation.function_gen(lm=lm, unittest=custom_unittest)
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)

  def test_custom_unittest_fn(self):
    function_gen_lm_response = inspect.cleandoc("""
        ```python
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        ```
        """)

    lm = fake.StaticSequence([function_gen_lm_response])

    def _unittest_fn(func):
      assert func([1, 2, 3, 4, 5], 3) == 2
      assert func([1, 2, 3, 4, 5], 6) == -1

    custom_unittest = _unittest_fn

    @function_generation.function_gen(lm=lm, unittest=custom_unittest)
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)

  def test_load_function_from_cache_file(self):
    lm = fake.StaticSequence([])

    def _unittest_fn(func):
      assert func([1, 2, 3, 4, 5], 3) == 2
      assert func([1, 2, 3, 4, 5], 6) == -1

    cache_file_dir = tempfile.gettempdir()
    cache_file = os.path.join(cache_file_dir, 'cache_file.json')

    cache_key = """@function_generation.function_gen(
        lm=lm,
        unittest=_unittest_fn,
        cache_filename=cache_file,
    )
    def linear_search(items, target):  # pylint: disable=unused-argument
      \"\"\"Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      \"\"\""""
    cache_value = """
        ```python
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        ```
    """
    cache = pg.Dict()
    cache[cache_key] = cache_value
    cache.save(cache_file)

    @function_generation.function_gen(
        lm=lm,
        unittest=_unittest_fn,
        cache_filename=cache_file,
    )
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)
    self.assertEqual(linear_search(['a', 'b', 'c'], 'd'), -1)

  def test_empty_cache_file(self):
    function_gen_lm_response = inspect.cleandoc("""
        ```python
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        ```
        """)

    lm = fake.StaticSequence([function_gen_lm_response])

    def _unittest_fn(func):
      assert func([1, 2, 3, 4, 5], 3) == 2
      assert func([1, 2, 3, 4, 5], 6) == -1

    cache_file_dir = tempfile.gettempdir()
    cache_file = os.path.join(cache_file_dir, 'cache_file.json')

    @function_generation.function_gen(
        lm=lm,
        unittest=_unittest_fn,
        cache_filename=cache_file,
    )
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)

  def test_context_passthrough(self):

    class Number(pg.Object):
      value: int

    function_gen_lm_response = inspect.cleandoc("""
        ```python
        def add(a: Number, b: Number) -> Number:
            \"\"\"Adds two numbers together.\"\"\"
            return Number(a.value + b.value)
        ```
        """)

    lm = fake.StaticSequence(
        [function_gen_lm_response]
    )

    def _unittest_fn(func):
      assert func(Number(1), Number(2)) == Number(3)

    custom_unittest = _unittest_fn

    @function_generation.function_gen(
        lm=lm, unittest=custom_unittest, num_retries=1
    )
    def add(a: Number, b: Number) -> Number:  # pylint: disable=unused-argument
      """Adds two numbers together."""

    self.assertEqual(add(Number(2), Number(3)), Number(5))

  def test_siganture_check(self):
    incorrect_signature_lm_response = inspect.cleandoc("""
        ```python
        def dummy():
          pass
        ```
        """)
    function_gen_lm_response = inspect.cleandoc("""
        ```python
        def linear_search(items, target):
            \"\"\"
            Performs a linear search on a list to find a target value.

            Args:
                items (list): The list to search within.
                target: The value to search for.

            Returns:
                int: The index of the target value if found, otherwise -1.
            \"\"\"
            for i, item in enumerate(items):
                if item == target:
                    return i
            return -1
        ```
        """)

    lm = fake.StaticSequence(
        [incorrect_signature_lm_response, function_gen_lm_response]
    )

    def _unittest_fn(func):
      assert func([1, 2, 3, 4, 5], 3) == 2
      assert func([1, 2, 3, 4, 5], 6) == -1

    custom_unittest = _unittest_fn

    @function_generation.function_gen(
        lm=lm, unittest=custom_unittest, num_retries=2
    )
    def linear_search(items, target):  # pylint: disable=unused-argument
      """Performs a linear search on a list to find a target value.

      Args:
          items (list): The list to search within.
          target: The value to search for.

      Returns:
          int: The index of the target value if found, otherwise -1.
      """

    self.assertEqual(linear_search(['a', 'b', 'c'], 'c'), 2)


if __name__ == '__main__':
  unittest.main()
