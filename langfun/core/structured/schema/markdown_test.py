# Copyright 2025 The Langfun Authors
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
"""Tests for markdown prompting protocol."""

import unittest
from langfun.core.llms import fake
from langfun.core.structured.schema import base
from langfun.core.structured.schema import markdown
import pyglove as pg


class MarkdownPromptingProtocolSchemaReprTest(unittest.TestCase):
  """Tests for schema representation in markdown."""

  def test_repr_simple(self):
    """Test markdown schema with simple fields."""

    class Solution(pg.Object):
      reasoning: str
      answer: int

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    expected = """## reasoning
<str>

...

## answer
<int>

...

---

**Type Definitions:**

```python
class Solution:
  reasoning: str
  answer: int
```"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_with_code_field(self):
    """Test automatic code block suggestion for 'code' field."""

    class Solution(pg.Object):
      reasoning: str
      code: str

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    expected = """## reasoning
<str>

...

## code
<str>

```python
...
```

---

**Type Definitions:**

```python
class Solution:
  reasoning: str
  code: str
```"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_with_code_fields(self):
    """Test automatic code block detection for *_code fields."""

    class SolutionWithTests(pg.Object):
      cpp_code: str
      terminal_code: str
      bash_code: str
      python_code: str  # Added to test the default python branch

    schema = base.Schema(SolutionWithTests)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    expected = """## cpp_code
<str>

```cpp
...
```

## terminal_code
<str>

```bash
...
```

## bash_code
<str>

```bash
...
```

## python_code
<str>

```python
...
```

---

**Type Definitions:**

```python
class SolutionWithTests:
  cpp_code: str
  terminal_code: str
  bash_code: str
  python_code: str
```"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_nested_object(self):
    """Test markdown schema with nested objects."""

    class Inner(pg.Object):
      value: int

    class Outer(pg.Object):
      inner: Inner
      name: str

    schema = base.Schema(Outer)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    expected = """## inner
...

## name
<str>

...

---

**Type Definitions:**

```python
class Inner:
  value: int

class Outer:
  inner: Inner
  name: str
```"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_union_with_pyobject(self):
    """Test that Union types use 'pyobject' language marker in schema."""

    class FileRead(pg.Object):
      file_path: str
      mode: str

    class FileWrite(pg.Object):
      file_path: str
      content: str

    class FinalizeAnswer(pg.Object):
      pass

    class NextStep(pg.Object):
      next_step: FileRead | FileWrite | FinalizeAnswer | None

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    expected_parts = [
        '## next_step',
        '```pyobject',
        'Union[FileRead, FileWrite, FinalizeAnswer, None]',
        '```',
        '**Type Definitions:**',
        'class FileRead:',
        'class FileWrite:',
        'class FinalizeAnswer:',
        'class NextStep:',
    ]

    # Verify all expected parts are present
    for part in expected_parts:
      self.assertIn(part, markdown_repr)

    # Verify python Union is NOT used
    self.assertNotIn('```python\nUnion[', markdown_repr)

  def test_repr_union_with_code_classes_and_objects(self):
    """Test Union with both code classes and other objects uses correct markers."""

    class BashCode(pg.Object):
      bash_code: str

    class PythonCode(pg.Object):
      python_code: str

    class FileRead(pg.Object):
      file_path: str

    class NextStep(pg.Object):
      next_step: BashCode | PythonCode | FileRead | None

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    # Should have bash and python code block examples
    self.assertIn('```bash', markdown_repr)
    self.assertIn('```python', markdown_repr)
    # Should use pyobject for non-code objects
    self.assertIn('```pyobject', markdown_repr)
    self.assertIn('FileRead(...)', markdown_repr)

  def test_repr_list_of_objects_with_pyobject(self):
    """Test that List[Object] uses 'pyobject' language marker in schema."""

    class Item(pg.Object):
      name: str
      value: int

    class Container(pg.Object):
      items: list[Item]

    schema = base.Schema(Container)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    # Should use pyobject for List[Object] type annotation
    self.assertIn('```pyobject', markdown_repr)
    self.assertIn('list[Item]', markdown_repr)
    # Should NOT use python for type annotation
    self.assertNotIn('```python\nlist[', markdown_repr)

  def test_print_schema_example(self):
    """Print example schema output to demonstrate the pyobject format."""

    class BashCode(pg.Object):
      """Execute bash commands."""

      bash_code: str

    class FileRead(pg.Object):
      """Read a file from the filesystem."""

      file_path: str
      mode: str

    class FinalizeAnswer(pg.Object):
      """Finalize the answer to the question."""

      pass

    class NextStep(pg.Object):
      """Next step in the research process."""

      think_step_by_step: str
      next_step: BashCode | FileRead | FinalizeAnswer | None

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    # Print the actual output for demonstration
    print('\n' + '=' * 80)
    print('EXAMPLE SCHEMA OUTPUT WITH PYOBJECT FORMAT:')
    print('=' * 80)
    print(markdown_repr)
    print('=' * 80 + '\n')

    # Verify it has the expected markers
    self.assertIn('```bash', markdown_repr)
    self.assertIn('```pyobject', markdown_repr)
    self.assertIn('FileRead(...)', markdown_repr)
    self.assertIn('FinalizeAnswer(...)', markdown_repr)


class MarkdownPromptingProtocolValueReprTest(unittest.TestCase):
  """Tests for value representation in markdown."""

  def test_repr(self):
    """Test markdown value representation."""

    class Solution(pg.Object):
      reasoning: str
      cpp_code: str

    solution = Solution(
        reasoning='Use dynamic programming',
        cpp_code='#include <iostream>\nint main() { return 0; }',
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(solution)

    expected = """## reasoning
Use dynamic programming

## cpp_code
```cpp
#include <iostream>
int main() { return 0; }
```
"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_bash_detection(self):
    """Test bash code detection in markdown repr."""

    class ScriptSolution(pg.Object):
      bash_script_code: str

    solution = ScriptSolution(
        bash_script_code='#!/bin/bash\necho "Hello"',
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(solution)

    expected = """## bash_script_code
```bash
#!/bin/bash
echo "Hello"
```
"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_bash_detection_cat(self):
    """Test bash code detection with cat > pattern."""

    class TestCode(pg.Object):
      test_code: str

    solution = TestCode(
        test_code='cat > test.txt\nSome content',
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(solution)

    expected = """## test_code
```bash
cat > test.txt
Some content
```
"""

    self.assertEqual(markdown_repr, expected)

  def test_repr_python_detection(self):
    """Test python code detection (default fallback)."""

    class PythonSolution(pg.Object):
      python_code: str

    solution = PythonSolution(
        python_code='def foo():\n  return 42',
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(solution)

    expected = """## python_code
```python
def foo():
  return 42
```
"""

    self.assertEqual(markdown_repr, expected)


class MarkdownPromptingProtocolParseValueTest(unittest.TestCase):
  """Tests for parsing markdown into structured values."""

  def test_parse_simple(self):
    """Test parsing markdown text into structured object."""

    class Solution(pg.Object):
      reasoning: str
      code: str

    markdown_text = """
## reasoning
This is my reasoning

## code
```python
def foo():
  pass
```
"""

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertEqual(result.reasoning, 'This is my reasoning')
    self.assertEqual(result.code, 'def foo():\n  pass')

  def test_parse_with_code_blocks(self):
    """Test parsing code blocks from markdown."""

    class SolutionWithTests(pg.Object):
      cpp_code: str
      terminal_code: str

    markdown_text = """
## cpp_code
```cpp
#include <iostream>
int main() { return 0; }
```

## terminal_code
```bash
g++ -o solution solution.cpp
./solution
```
"""

    schema = base.Schema(SolutionWithTests)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertIn('#include <iostream>', result.cpp_code)
    self.assertIn('g++ -o solution', result.terminal_code)

  def test_parse_missing_required_field(self):
    """Test that missing required fields raise ValueError."""

    class Solution(pg.Object):
      reasoning: str
      code: str

    markdown_text = """
## reasoning
This is my reasoning
"""

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()
    with self.assertRaisesRegex(ValueError, 'Required field "code" not found'):
      protocol.parse_value(markdown_text, schema)

  def test_parse_optional_field(self):
    """Test parsing with optional fields."""

    class Solution(pg.Object):
      reasoning: str
      code: str | None

    markdown_text = """
## reasoning
This is my reasoning
"""

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertEqual(result.reasoning, 'This is my reasoning')
    self.assertIsNone(result.code)

  def test_extract_section(self):
    """Test section extraction helper method."""
    markdown_text = """
## section1
Content 1

## section2
Content 2
"""

    protocol = markdown.MarkdownPromptingProtocol()
    section1 = protocol._extract_section(markdown_text, 'section1')
    section2 = protocol._extract_section(markdown_text, 'section2')

    self.assertEqual(section1, 'Content 1')
    self.assertEqual(section2, 'Content 2')

  def test_extract_code_block(self):
    """Test code block extraction helper method."""
    section_content = """
Some text
```python
def foo():
  pass
```
More text
"""

    protocol = markdown.MarkdownPromptingProtocol()
    code_info = protocol._extract_code_block(section_content)

    self.assertIsNotNone(code_info)
    code, language = code_info
    self.assertEqual(code, 'def foo():\n  pass')
    self.assertEqual(language, 'python')

  def test_autofix_with_missing_field(self):
    """Test that autofix is triggered when a required field is missing."""

    class Solution(pg.Object):
      reasoning: str
      code: str

    # Markdown missing the 'code' field
    markdown_text = """
## reasoning
This is my reasoning
"""

    # Create a fake LLM that will provide the missing field
    corrected_markdown = """
## reasoning
This is my reasoning

## code
```python
def solution():
  pass
```
"""

    fix_lm = fake.StaticResponse(corrected_markdown)

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()

    # With autofix=1, should call fix_lm and succeed
    result = protocol.parse_value(
        markdown_text,
        schema,
        autofix=1,
        autofix_lm=fix_lm,
    )

    self.assertEqual(result.reasoning, 'This is my reasoning')
    self.assertEqual(result.code, 'def solution():\n  pass')

  def test_autofix_not_triggered_when_disabled(self):
    """Test that autofix is not triggered when autofix=0."""

    class Solution(pg.Object):
      reasoning: str
      code: str

    # Markdown missing the 'code' field
    markdown_text = """
## reasoning
This is my reasoning
"""

    schema = base.Schema(Solution)
    protocol = markdown.MarkdownPromptingProtocol()

    # With autofix=0, should raise ValueError
    with self.assertRaisesRegex(ValueError, 'Required field "code" not found'):
      protocol.parse_value(
          markdown_text,
          schema,
          autofix=0,
      )


class MarkdownPromptingProtocolListTest(unittest.TestCase):
  """Tests for List type support in markdown protocol."""

  def test_simple_list_schema_repr(self):
    """Test schema representation for simple list."""

    class SimpleList(pg.Object):
      title: str
      items: list[str]

    schema = base.Schema(SimpleList)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    self.assertIn('## title', markdown_repr)
    self.assertIn('## items', markdown_repr)
    self.assertIn('- item 1', markdown_repr)

  def test_object_list_schema_repr(self):
    """Test schema representation for list of objects."""

    class TestCase(pg.Object):
      description: str
      code: str

    class TestSuite(pg.Object):
      name: str
      test_cases: list[TestCase]

    schema = base.Schema(TestSuite)
    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.schema_repr(schema)

    self.assertIn('## test_cases', markdown_repr)
    self.assertIn('### TestCase 1', markdown_repr)
    self.assertIn('#### description', markdown_repr)
    self.assertIn('#### code', markdown_repr)

  def test_simple_list_value_repr(self):
    """Test value representation for simple list."""

    class SimpleList(pg.Object):
      title: str
      items: list[str]

    value = SimpleList(title='Shopping List', items=['Milk', 'Eggs', 'Bread'])

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(value)

    self.assertIn('## title', markdown_repr)
    self.assertIn('Shopping List', markdown_repr)
    self.assertIn('## items', markdown_repr)
    self.assertIn('- Milk', markdown_repr)
    self.assertIn('- Eggs', markdown_repr)
    self.assertIn('- Bread', markdown_repr)

  def test_object_list_value_repr(self):
    """Test value representation for list of objects."""

    class TestCase(pg.Object):
      description: str
      input_code: str

    class TestSuite(pg.Object):
      name: str
      test_cases: list[TestCase]

    value = TestSuite(
        name='Math Tests',
        test_cases=[
            TestCase(
                description='Test addition',
                input_code='assert add(1, 2) == 3',
            ),
            TestCase(
                description='Test subtraction',
                input_code='assert subtract(5, 3) == 2',
            ),
        ],
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(value)

    self.assertIn('## name', markdown_repr)
    self.assertIn('Math Tests', markdown_repr)
    self.assertIn('## test_cases', markdown_repr)
    self.assertIn('### TestCase 1', markdown_repr)
    self.assertIn('#### description', markdown_repr)
    self.assertIn('Test addition', markdown_repr)
    self.assertIn('#### input_code', markdown_repr)
    self.assertIn('assert add(1, 2) == 3', markdown_repr)
    self.assertIn('### TestCase 2', markdown_repr)
    self.assertIn('Test subtraction', markdown_repr)

  def test_parse_simple_list(self):
    """Test parsing simple list from markdown."""

    class SimpleList(pg.Object):
      title: str
      items: list[str]

    markdown_text = """
## title
Shopping List

## items
- Milk
- Eggs
- Bread
"""

    schema = base.Schema(SimpleList)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertEqual(result.title, 'Shopping List')
    self.assertEqual(result.items, ['Milk', 'Eggs', 'Bread'])

  def test_parse_object_list(self):
    """Test parsing list of objects from markdown."""

    class TestCase(pg.Object):
      description: str
      input_code: str
      expected_output: str

    class TestSuite(pg.Object):
      name: str
      test_cases: list[TestCase]

    markdown_text = """
## name
Math Tests

## test_cases

### TestCase 1

#### description
Test addition

#### input_code
```python
assert add(1, 2) == 3
```

#### expected_output
Pass

### TestCase 2

#### description
Test subtraction

#### input_code
```python
assert subtract(5, 3) == 2
```

#### expected_output
Pass
"""

    schema = base.Schema(TestSuite)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertEqual(result.name, 'Math Tests')
    self.assertEqual(len(result.test_cases), 2)
    self.assertEqual(result.test_cases[0].description, 'Test addition')
    self.assertEqual(result.test_cases[0].input_code, 'assert add(1, 2) == 3')
    self.assertEqual(result.test_cases[0].expected_output, 'Pass')
    self.assertEqual(result.test_cases[1].description, 'Test subtraction')
    self.assertEqual(
        result.test_cases[1].input_code, 'assert subtract(5, 3) == 2'
    )
    self.assertEqual(result.test_cases[1].expected_output, 'Pass')

  def test_parse_int_list(self):
    """Test parsing list of integers."""

    class NumberList(pg.Object):
      numbers: list[int]

    markdown_text = """
## numbers
- 1
- 2
- 3
"""

    schema = base.Schema(NumberList)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertEqual(result.numbers, [1, 2, 3])

  def test_extract_list_items(self):
    """Test list item extraction helper method."""
    section_content = """
- Item 1
- Item 2
- Item 3
"""

    protocol = markdown.MarkdownPromptingProtocol()
    items = protocol._extract_list_items(section_content)

    self.assertEqual(items, ['Item 1', 'Item 2', 'Item 3'])


class MarkdownPromptingProtocolUnionTest(unittest.TestCase):
  """Tests for Union type support in markdown protocol."""

  def test_parse_union_object(self):
    """Test parsing Union of objects."""

    class Action1(pg.Object):
      name: str
      value: int

    class Action2(pg.Object):
      title: str

    class TestUnion(pg.Object):
      action: Action1 | Action2

    # Test first candidate
    markdown_text = """## action
```pyobject
Action1(name='test', value=42)
```
"""

    schema = base.Schema(TestUnion)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertIsInstance(result.action, Action1)
    self.assertEqual(result.action.name, 'test')
    self.assertEqual(result.action.value, 42)

  def test_parse_union_with_nested_objects(self):
    """Test parsing Union with nested objects (e.g., BrowseWeb with Question)."""

    class Question(pg.Object):
      question: str
      context: dict[str, str]

    class BrowseWeb(pg.Object):
      question: Question

    class FileRead(pg.Object):
      file_path: str

    class NextStep(pg.Object):
      next_step: BrowseWeb | FileRead | None

    # Test BrowseWeb with nested Question object
    markdown_text = """## next_step
```pyobject
BrowseWeb(question=Question(question='What is the answer?', context={}))
```
"""

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertIsInstance(result.next_step, BrowseWeb)
    self.assertIsInstance(result.next_step.question, Question)
    self.assertEqual(result.next_step.question.question, 'What is the answer?')
    self.assertEqual(result.next_step.question.context, {})

  def test_parse_union_list_vs_single(self):
    """Test parsing Union of list vs single object."""

    class Item(pg.Object):
      name: str

    class TestUnion(pg.Object):
      items: list[Item] | Item | None

    # Test single object
    markdown_text1 = """## items
```pyobject
Item(name='single')
```
"""

    schema = base.Schema(TestUnion)
    protocol = markdown.MarkdownPromptingProtocol()
    result1 = protocol.parse_value(markdown_text1, schema)

    self.assertIsInstance(result1.items, Item)
    self.assertEqual(result1.items.name, 'single')

  def test_parse_union_code_class(self):
    """Test parsing Union with code class (BashCode)."""

    class BashCode(pg.Object):
      bash_code: str

    class FileRead(pg.Object):
      file_path: str

    class TerminalNextStep(pg.Object):
      think_step_by_step: str
      next_step: BashCode | FileRead | None

    # Test BashCode with comment (like LLM generates from schema example)
    markdown_text = """## think_step_by_step
Calculate 1002 * 0.04 and round up.

## next_step
```bash
# BashCode
python -c "import math; print(math.ceil(1002 * 0.04))"
```
"""

    schema = base.Schema(TerminalNextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    self.assertIsInstance(result.next_step, BashCode)
    # The bash_code should include the comment (it's valid bash)
    self.assertIn('# BashCode', result.next_step.bash_code)
    self.assertIn('python -c', result.next_step.bash_code)

  def test_parse_union_python_vs_bash(self):
    """Test that Python code blocks are not misidentified as BashCode."""

    class BashCode(pg.Object):
      bash_code: str

    class PythonCode(pg.Object):
      python_code: str

    class NextStep(pg.Object):
      reasoning: str
      action: BashCode | PythonCode | None

    # Test Python code block - should be parsed as PythonCode
    markdown_text = """## reasoning
Use Python to finalize the answer.

## action
```python
FinalizeAnswer()
```
"""

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    # Should be PythonCode, not BashCode
    self.assertIsInstance(result.action, PythonCode)
    self.assertEqual(result.action.python_code, 'FinalizeAnswer()')

    # Test Bash code block - should be parsed as BashCode
    markdown_text2 = """## reasoning
Use bash to execute a command.

## action
```bash
echo "Hello World"
```
"""

    result2 = protocol.parse_value(markdown_text2, schema)

    # Should be BashCode
    self.assertIsInstance(result2.action, BashCode)
    self.assertEqual(result2.action.bash_code, 'echo "Hello World"')

  def test_finalize_answer_not_parsed_as_bash(self):
    """Test the original user scenario: FinalizeAnswer() should not be BashCode.

    This reproduces the exact issue reported by the user where a Python code
    block containing FinalizeAnswer() was being incorrectly parsed as BashCode.
    """

    class BashCode(pg.Object):
      bash_code: str

    class FinalizeAnswer(pg.Object):
      """Dummy FinalizeAnswer for testing."""

      pass

    class TerminalNextStep(pg.Object):
      think_step_by_step: str
      next_step: BashCode | FinalizeAnswer | None

    # This is the user's exact scenario
    markdown_text = """## think_step_by_step
The problem has been successfully solved in the previous steps.

1.  **Initial Attempt (Step 1):** The first script failed with a `KeyError`.
2.  **Investigation (Step 2):** A second script was used to inspect the structure.
3.  **Successful Execution (Step 3):** A third script was created and succeeded.
4.  **Result:** The script executed successfully and printed the final answer.

The goal has been achieved. The correct action is to finalize the answer.

## next_step
```pyobject
FinalizeAnswer()
```
"""

    schema = base.Schema(TerminalNextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    # Critical assertion: Python code should be parsed as FinalizeAnswer
    self.assertIsInstance(
        result.next_step,
        FinalizeAnswer,
        f'Expected FinalizeAnswer but got {type(result.next_step).__name__}. '
        'Python code blocks should not be misidentified as BashCode!',
    )

  def test_parse_list_of_union_with_nested_objects(self):
    """Test parsing list of Union types with nested objects in pyobject format.

    This tests the scenario where an LLM generates a Python list literal
    containing multiple objects with nested dependencies (e.g., Terminal and
    BrowseWeb, each containing a Question object).
    """

    class Question(pg.Object):
      question: str
      context: dict[str, str] | None

    class Terminal(pg.Object):
      question: Question

    class BrowseWeb(pg.Object):
      question: Question

    class FileRead(pg.Object):
      file_path: str

    class NextStep(pg.Object):
      next_step: list[Terminal | BrowseWeb | FileRead] | None

    # This is the exact format an LLM might generate
    markdown_text = """## next_step
```pyobject
[
    Terminal(
        question=Question(
            question="Please check the transcript of the audio file to identify the recommended reading page numbers for the Calculus mid-term. You can use available command-line tools or install Python libraries like SpeechRecognition to process the file. If the file is MP3, you might need to convert it to WAV first.",
            context={
                'file_path': './question/attachments/1f975693-876d-457b-a649-393859e79bf3.mp3'
            }
        )
    ),
    BrowseWeb(
        question=Question(
            question="What are the recommended reading page numbers for Professor Willowbrook's Calculus mid-term?",
            context=None
        )
    )
]
```
"""

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    # Verify it's a list
    self.assertIsInstance(result.next_step, list)
    self.assertEqual(len(result.next_step), 2)

    # Verify first item is Terminal with nested Question
    self.assertIsInstance(result.next_step[0], Terminal)
    self.assertIsInstance(result.next_step[0].question, Question)
    self.assertIn('Calculus mid-term', result.next_step[0].question.question)
    self.assertIsInstance(result.next_step[0].question.context, dict)
    self.assertIn('file_path', result.next_step[0].question.context)

    # Verify second item is BrowseWeb with nested Question
    self.assertIsInstance(result.next_step[1], BrowseWeb)
    self.assertIsInstance(result.next_step[1].question, Question)
    self.assertIn(
        'Professor Willowbrook', result.next_step[1].question.question
    )
    self.assertIsNone(result.next_step[1].question.context)

  def test_value_repr_with_object_field(self):
    """Test that value_repr generates pyobject code blocks for Object fields.

    This ensures that few-shot examples are formatted correctly with pyobject
    blocks instead of inline code or plain str(). Uses the actual example from
    browse.NEXT_STEP_EXAMPLES.
    """

    class NavigateTo(pg.Object):
      url: str

    class NextStep(pg.Object):
      think_step_by_step: str
      next_step: NavigateTo

    # This is the actual example from browse.NEXT_STEP_EXAMPLES
    value = NextStep(
        think_step_by_step=(
            'I should use the NavigateTo action to navigate to the Google'
            ' homepage.'
        ),
        next_step=NavigateTo('https://www.google.com/'),
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(value)

    # Should use pyobject code block for the NavigateTo object
    self.assertIn('```pyobject', markdown_repr)
    self.assertIn('NavigateTo', markdown_repr)
    self.assertIn('https://www.google.com/', markdown_repr)
    # Should NOT use inline code (single backticks) around the object
    # The entire object should be in a code block, not inline
    lines = markdown_repr.split('\n')
    # Find the next_step section
    next_step_idx = None
    for i, line in enumerate(lines):
      if line.strip() == '## next_step':
        next_step_idx = i
        break
    self.assertIsNotNone(next_step_idx)
    # The line after ## next_step should be ```pyobject
    self.assertEqual(lines[next_step_idx + 1], '```pyobject')

  def test_value_repr_no_backticks(self):
    """Test that value_repr does NOT generate backticks around objects.

    This verifies the fix for the issue where LLMs were seeing backticks
    in few-shot examples and copying them, causing parsing errors.
    """

    class NavigateTo(pg.Object):
      url: str

    class NextStep(pg.Object):
      think_step_by_step: str
      next_step: NavigateTo

    value = NextStep(
        think_step_by_step=(
            'I should use the NavigateTo action to navigate to the Google'
            ' homepage.'
        ),
        next_step=NavigateTo(url='https://www.google.com/'),
    )

    protocol = markdown.MarkdownPromptingProtocol()
    markdown_repr = protocol.value_repr(value)

    # Expected output should NOT have backticks around NavigateTo(...)
    expected = """## think_step_by_step
I should use the NavigateTo action to navigate to the Google homepage.

## next_step
```pyobject
NavigateTo(url='https://www.google.com/')
```
"""

    self.assertEqual(markdown_repr, expected)

  def test_parse_with_backticks_defensive(self):
    """Test that parser can handle backticks defensively if LLM adds them."""

    class Question(pg.Object):
      question: str
      context: dict[str, str] | None

    class BrowseWeb(pg.Object):
      question: Question

    class NextStep(pg.Object):
      next_step: BrowseWeb | None

    # This is what caused the original error - backticks around the object
    markdown_text = """## next_step
```pyobject
`BrowseWeb(question=Question(question='Find the photograph with accession number 2022.128 in the Whitney Museum of American Art collection.', context=None))`
```
"""

    schema = base.Schema(NextStep)
    protocol = markdown.MarkdownPromptingProtocol()
    result = protocol.parse_value(markdown_text, schema)

    # Should successfully parse despite the backticks
    self.assertIsInstance(result.next_step, BrowseWeb)
    self.assertIsInstance(result.next_step.question, Question)
    self.assertEqual(
        result.next_step.question.question,
        'Find the photograph with accession number 2022.128 in the Whitney'
        ' Museum of American Art collection.',
    )
    self.assertIsNone(result.next_step.question.context)


if __name__ == '__main__':
  unittest.main()
