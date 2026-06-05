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
import unittest

from langfun.core.llms import fake
from langfun.core.structured import mapping
from langfun.core.structured import schema_generation


class GenerateClassTest(unittest.TestCase):

  def test_generate_class_prompt(self):
    input_message = schema_generation.generate_class(
        'Trip',
        'Generate a trip class',
        skip_lm=True,
        returns_message=True,
    )
    self.maxDiff = None
    self.assertEqual(
        input_message.text,
        inspect.cleandoc("""
            Help generate a class based on the last CLASS_NAME and GENERATION_CONTEXT.

            Instructions:
            - Use `Object` as the base class for all generated classes
            - Create auxillary classes for composition if needed.
            - Use Python type annotation for declaraing fields:
              (e.g. bool, str, int, float, Optional[str], List[int], Union[str, int])
            - Do not use types that need import.
            - Avoid self-referential types. e.g:
              ```
              class Node(Object):
                children: list[Node]
              ```
            - Do not generate methods.

            CLASS_NAME:
              Solution

            GENERATION_CONTEXT:
              How to evaluate an arithmetic expression?

            OUTPUT_CLASS:
              ```python
              class Step(Object):
                description: str
                output: float

              class Solution(Object):
                steps: list[Step]
                result: float
              ```


            CLASS_NAME:
              Trip

            GENERATION_CONTEXT:
              Generate a trip class

            OUTPUT_CLASS:
            """),
    )

  def test_generate_class(self):
    lm = fake.StaticResponse("""
        ```python
        class A(Object):
          x: int

        class B(Object):
          a: A
        ```
        """)
    cls = schema_generation.generate_class(
        'B',
        'Generate a B class with a field pointing to another class A',
        lm=lm,
    )
    self.assertIs(cls.__name__, 'B')

    with self.assertRaises(mapping.MappingError):
      schema_generation.generate_class(
          'Foo',
          'Generate a Foo class with a field pointing to another class A',
          lm=lm,
      )


if __name__ == '__main__':
  unittest.main()
