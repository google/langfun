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
"""Markdown-based prompting protocol."""

import re
from typing import Any
import langfun.core as lf
from langfun.core.structured.schema import base
import pyglove as pg


class MarkdownPromptingProtocol(base.PromptingProtocol):
  """Markdown-based prompting protocol."""

  NAME = 'markdown'

  def schema_repr(self, schema: base.Schema, **kwargs) -> str:
    """Returns markdown representation of the schema."""
    del kwargs
    lines = []

    # Only process Object schemas
    if not isinstance(schema.spec, pg.typing.Object):
      raise ValueError(
          f'Markdown protocol only supports Object schemas, got: {schema.spec}'
      )

    cls = schema.spec.cls
    for key, field in cls.__schema__.items():
      if not isinstance(key, pg.typing.ConstStrKey):
        continue

      field_name = str(key)
      lines.append(f'## {field_name}')

      # Handle Union type
      if isinstance(field.value, pg.typing.Union):
        # Separate code classes from other candidates
        code_classes = []  # [(cls, field_name, language)]
        other_candidates = []

        for candidate in field.value.candidates:
          if isinstance(candidate, pg.typing.Object):
            # Check if this is a code class (single *_code field)
            cls_fields = list(candidate.cls.__schema__.items())
            if len(cls_fields) == 1:
              cls_key, cls_field = cls_fields[0]
              if isinstance(cls_key, pg.typing.ConstStrKey):
                cls_field_name = str(cls_key)
                if cls_field_name.endswith('_code') and isinstance(
                    cls_field.value, pg.typing.Str
                ):
                  # This is a code class
                  language = self._detect_code_language(cls_field_name)
                  code_classes.append((candidate.cls, cls_field_name, language))
                  continue
            # Not a code class
            other_candidates.append(candidate)
          else:
            other_candidates.append(candidate)

        if code_classes:
          # Special format: code blocks OR python objects
          lines.append('Choose ONE of:')
          lines.append('')

          # Add code block options
          for i, (cls, _, language) in enumerate(code_classes):
            lines.append(f'```{language}')
            lines.append('...')
            lines.append('```')
            if i < len(code_classes) - 1 or other_candidates:
              lines.append('')
              lines.append('OR')
              lines.append('')

          # Add other object options
          if other_candidates:
            lines.append('```pyobject')
            for i, candidate in enumerate(other_candidates):
              if isinstance(candidate, pg.typing.Object):
                lines.append(f'{candidate.cls.__name__}(...)')
              elif candidate == pg.typing.MISSING_VALUE:
                lines.append('None')
              if i < len(other_candidates) - 1:
                lines.append('# OR')
            lines.append('```')
        else:
          # Normal Union format (no code classes)
          union_annotation = base.annotation(field.value)
          lines.append('```pyobject')
          lines.append(union_annotation)
          lines.append('```')
      # Handle List type
      elif isinstance(field.value, pg.typing.List):
        element_spec = field.value.element.value
        if isinstance(element_spec, pg.typing.Object):
          # List of Objects - use pyobject code block for type
          list_annotation = base.annotation(field.value)
          lines.append('```pyobject')
          lines.append(list_annotation)
          lines.append('```')
          lines.append('')
          # Show nested structure
          lines.append(f'### {element_spec.cls.__name__} 1')
          lines.append('')
          for obj_key, _ in element_spec.cls.__schema__.items():
            if isinstance(obj_key, pg.typing.ConstStrKey):
              obj_field_name = str(obj_key)
              lines.append(f'#### {obj_field_name}')
              lines.append('...')
              lines.append('')
        else:
          # List of primitives - use angle brackets
          list_annotation = base.annotation(field.value)
          lines.append(f'<{list_annotation}>')
          lines.append('')
          lines.append('- item 1')
          lines.append('- item 2')
          lines.append('- ...')
      # Handle string type
      elif isinstance(field.value, pg.typing.Str):
        lines.append('<str>')
        lines.append('')
        # Detect if this is a code field and suggest code block
        if field_name.endswith('_code') or field_name == 'code':
          language = self._detect_code_language(field_name)
          lines.append(f'```{language}')
          lines.append('...')
          lines.append('```')
        else:
          lines.append('...')
      # Handle other primitive types
      elif isinstance(field.value, pg.typing.Int):
        lines.append('<int>')
        lines.append('')
        lines.append('...')
      elif isinstance(field.value, pg.typing.Float):
        lines.append('<float>')
        lines.append('')
        lines.append('...')
      elif isinstance(field.value, pg.typing.Bool):
        lines.append('bool')
        lines.append('')
        lines.append('...')
      else:
        # Unknown type, just show placeholder
        lines.append('...')
      lines.append('')

    # Add Python class definitions for all dependent types
    # This helps LLM understand the structure of Object types used in
    # Union fields
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    from langfun.core.structured.schema import python
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top

    py_protocol = python.PythonPromptingProtocol()
    class_defs = py_protocol.class_definitions(schema, markdown=True)

    if class_defs:
      lines.append('---')
      lines.append('')
      lines.append('**Type Definitions:**')
      lines.append('')
      lines.append(class_defs)

    return '\n'.join(lines)

  def _detect_code_language(self, field_name: str) -> str:
    """Detects programming language from field name."""
    # Check for language-specific prefixes
    if field_name.startswith('cpp_') or field_name.startswith('c++_'):
      return 'cpp'
    elif field_name.startswith('bash_') or field_name.startswith('shell_'):
      return 'bash'
    elif field_name.startswith('terminal_'):
      return 'bash'
    elif field_name.startswith('python_'):
      return 'python'
    elif field_name.startswith('java_'):
      return 'java'
    elif field_name.startswith('javascript_') or field_name.startswith('js_'):
      return 'javascript'
    # Default to python for generic *_code fields
    return 'python'

  def value_repr(
      self, value: Any, schema: base.Schema | None = None, **kwargs
  ) -> str:
    """Returns markdown representation of a value."""
    del schema, kwargs
    if not isinstance(value, pg.Object):
      return str(value)

    lines = []
    for key, val in value.sym_items():
      field_name = str(key)
      lines.append(f'## {field_name}')

      # Handle List type
      if isinstance(val, list):
        for idx, item in enumerate(val, 1):
          if isinstance(item, pg.Object):
            # Nested Object - use ### for item header
            item_type = item.__class__.__name__
            lines.append(f'### {item_type} {idx}')
            lines.append('')
            # Recursively render object fields with ####
            for item_key, item_val in item.sym_items():
              item_field_name = str(item_key)
              lines.append(f'#### {item_field_name}')
              # Handle code in nested objects
              if isinstance(item_val, str):
                language = self._detect_code_language_from_content(
                    item_field_name, item_val
                )
                if language:
                  lines.append(f'```{language}')
                  lines.append(item_val)
                  lines.append('```')
                else:
                  lines.append(item_val)
              else:
                lines.append(str(item_val))
              lines.append('')
          else:
            # Simple type - use list item
            lines.append(f'- {item}')
      # Check if value looks like code
      elif isinstance(val, str):
        language = self._detect_code_language_from_content(field_name, val)
        if language:
          lines.append(f'```{language}')
          lines.append(val)
          lines.append('```')
        else:
          lines.append(val)
      # Handle pg.Object values - use pyobject code block
      elif isinstance(val, pg.Object):
        lines.append('```pyobject')
        # Use pg.format to get proper Python representation
        lines.append(
            pg.format(val, compact=True, verbose=False, python_format=True)
        )
        lines.append('```')
      else:
        lines.append(str(val))
      lines.append('')

    return '\n'.join(lines)

  def _detect_code_language_from_content(
      self, field_name: str, content: str
  ) -> str | None:
    """Detects if content is code and returns language."""
    # First check field name
    if field_name.endswith('_code') or field_name == 'code':
      # Check content for language hints
      if content.strip().startswith('#include'):
        return 'cpp'
      elif content.strip().startswith('#!/bin/bash') or 'cat >' in content:
        return 'bash'
      elif 'def ' in content or 'class ' in content:
        return 'python'
      # Use field name detection as fallback
      return self._detect_code_language(field_name)
    return None

  def parse_value(
      self,
      text: str,
      schema: base.Schema | None = None,
      *,
      autofix=0,
      autofix_lm: lf.LanguageModel = lf.contextual(),
      **kwargs,
  ) -> Any:
    """Parses markdown text into a structured object."""
    del kwargs
    if schema is None:
      raise ValueError('Schema is required for markdown parsing')

    # Without autofix: parse directly
    if autofix == 0:
      return self._parse_markdown(text, schema)

    # With autofix: use correction mechanism
    error = None
    for attempt in range(autofix + 1):
      try:
        return self._parse_markdown(text, schema)
      except Exception as e:  # pylint: disable=broad-exception-caught
        error = e
        if attempt < autofix:
          # Try to fix the markdown using LLM
          text = self._fix_markdown(text, schema, error, autofix_lm)
        else:
          raise

    # Should not reach here, but just in case
    raise error  # type: ignore

  def _parse_markdown(self, text: str, schema: base.Schema) -> Any:
    """Internal method to parse markdown text."""
    if not isinstance(schema.spec, pg.typing.Object):
      raise ValueError(
          f'Markdown protocol only supports Object schemas, got: {schema.spec}'
      )

    cls = schema.spec.cls
    result = {}

    # Get all class dependencies from schema (like Python protocol does)
    dependencies = schema.class_dependencies(
        include_base_classes=False, include_subclasses=False
    )
    all_dependencies = {d.__name__: d for d in dependencies}

    # Extract sections for each field
    for key, field in cls.__schema__.items():
      if not isinstance(key, pg.typing.ConstStrKey):
        continue

      field_name = str(key)
      section_content = self._extract_section(text, field_name)

      if section_content is None:
        # Field not found - check if it's required
        if not field.value.is_noneable:
          raise ValueError(
              f'Required field "{field_name}" not found in markdown'
          )
        result[field_name] = None
        continue

      # Parse based on field type
      if isinstance(field.value, pg.typing.Union):
        # Handle Union type - try each candidate in order
        result[field_name] = self._parse_union_field(
            section_content, field.value, field_name, all_dependencies
        )
      elif isinstance(field.value, pg.typing.List):
        # Parse List type
        element_spec = field.value.element.value

        # Check if this is a pyobject block with Python list literal
        code_info = self._extract_code_block(section_content)
        if code_info and code_info[1] == 'pyobject':
          # Parse as Python list literal
          # Delay import at runtime to avoid circular dependency.
          # pylint: disable=g-import-not-at-top
          # pytype: disable=import-error
          from langfun.core.structured.schema import python
          # pytype: enable=import-error
          # pylint: enable=g-import-not-at-top

          # Build global_vars with all classes
          global_vars = all_dependencies.copy()

          try:
            parsed_list = python.structure_from_python(
                code_info[0],
                global_vars=global_vars,
                permission=pg.coding.CodePermission.CALL,
            )
            # Verify it's a list
            if isinstance(parsed_list, list):
              result[field_name] = parsed_list
            else:
              raise TypeError(
                  f'Expected list, got {type(parsed_list).__name__}'
              )
          except Exception as e:
            raise ValueError(
                f'Failed to parse list for field "{field_name}": {e}'
            ) from e
        elif isinstance(element_spec, pg.typing.Object):
          # List of Objects - extract items with ### headers
          items = self._extract_list_objects(section_content, element_spec.cls)
          result[field_name] = items
        else:
          # List of primitives - extract list items
          items = self._extract_list_items(section_content)
          # Convert to appropriate type
          if isinstance(element_spec, pg.typing.Int):
            result[field_name] = [int(item) for item in items]
          elif isinstance(element_spec, pg.typing.Float):
            result[field_name] = [float(item) for item in items]
          elif isinstance(element_spec, pg.typing.Bool):
            result[field_name] = [
                item.lower() in ('true', 'yes', '1') for item in items
            ]
          else:
            result[field_name] = items
      elif isinstance(field.value, pg.typing.Str):
        # Try to extract code block first
        code_info = self._extract_code_block(section_content)
        result[field_name] = (
            code_info[0] if code_info else section_content.strip()
        )
      elif isinstance(field.value, (pg.typing.Int, pg.typing.Float)):
        result[field_name] = field.value.value_type(section_content.strip())
      elif isinstance(field.value, pg.typing.Bool):
        result[field_name] = section_content.strip().lower() in (
            'true',
            'yes',
            '1',
        )
      elif isinstance(field.value, pg.typing.Object):
        # Handle Object type - check if it's a pyobject block
        code_info = self._extract_code_block(section_content)
        if code_info and code_info[1] == 'pyobject':
          # Delegate to Python protocol for parsing
          # Delay import at runtime to avoid circular dependency.
          # pylint: disable=g-import-not-at-top
          # pytype: disable=import-error
          from langfun.core.structured.schema import python
          # pytype: enable=import-error
          # pylint: enable=g-import-not-at-top

          code = code_info[0]
          # Strip any backticks that LLMs might add
          code = code.strip('`')

          # Build global_vars with all class dependencies
          global_vars = all_dependencies.copy()

          try:
            result[field_name] = python.structure_from_python(
                code,
                global_vars=global_vars,
                permission=pg.coding.CodePermission.CALL,
            )
          except Exception as e:
            raise ValueError(
                f'Failed to parse pyobject for field "{field_name}": {e}'
            ) from e
        else:
          # Not a pyobject block - treat as error
          raise ValueError(
              f'Object field "{field_name}" must use pyobject code block'
          )
      else:
        result[field_name] = section_content.strip()

    # Create object instance
    return cls(**result)

  def _fix_markdown(
      self,
      text: str,
      schema: base.Schema,
      error: Exception,
      lm: lf.LanguageModel,
  ) -> str:
    """Fix malformed markdown using LLM."""
    # Delay import at runtime to avoid circular dependency.
    # This follows the same pattern as python/correction.py
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    from langfun.core.structured import querying
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top

    # Build schema description
    schema_desc = self.schema_repr(schema)

    # Build correction prompt
    correction_prompt = f"""The following markdown output has an error:

```markdown
{text}
```

Error: {error}

Expected schema:
{schema_desc}


Please provide the corrected markdown output that matches the expected schema."""

    # Query LLM for correction (disable autofix to avoid recursion)
    corrected = querying.query(
        correction_prompt,
        str,
        lm=lm,
        autofix=0,
    )

    return corrected

  def _extract_section(self, text: str, section_name: str) -> str | None:
    """Extract content from a markdown section."""
    # Match: ## section_name\n<content>
    # (until next ## that's not ### or ####, or end)
    # Use negative lookahead to ensure ## is not followed by another #
    pattern = rf'##\s+{re.escape(section_name)}\s*\n(.*?)(?=\n##(?!#)|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
      return match.group(1).strip()
    return None

  def _extract_code_block(self, section_content: str) -> tuple[str, str] | None:
    """Extract code and language from markdown code block.

    Args:
      section_content: The markdown section content to extract from.

    Returns:
      Tuple of (code_content, language) if found, None otherwise.
      Language defaults to 'python' if not specified.
    """
    # Match: ```language\n<code>\n```
    pattern = r'```([\w]*)\s*\n(.*?)\n```'
    match = re.search(pattern, section_content, re.DOTALL)
    if match:
      language = match.group(1) or 'python'
      code = match.group(2)
      return (code, language)
    return None

  def _extract_list_items(self, section_content: str) -> list[str]:
    """Extract items from a markdown list."""
    if not section_content:
      return []

    items = []
    for line in section_content.split('\n'):
      line = line.strip()
      if line.startswith('- ') or line.startswith('* '):
        items.append(line[2:].strip())

    return items

  def _extract_list_objects(
      self, section_content: str, obj_cls: type[pg.Object]
  ) -> list[pg.Object]:
    """Extract list of objects from markdown with ### headers."""
    if not section_content:
      return []

    # Split by ### headers to get individual items
    # Pattern: ### ClassName N or ### Item N
    pattern = r'###\s+(?:\w+\s+)?\d+\s*\n'
    parts = re.split(pattern, section_content)

    # First part before any ### is usually empty or description
    items = []
    for part in parts[1:]:  # Skip first empty part
      if not part.strip():
        continue

      # Parse object fields from this part
      obj_data = {}
      for key, field in obj_cls.__schema__.items():
        if not isinstance(key, pg.typing.ConstStrKey):
          continue

        field_name = str(key)
        # Extract #### field_name content
        field_pattern = (
            rf'####\s+{re.escape(field_name)}\s*\n(.*?)(?=\n####|\Z)'
        )
        field_match = re.search(field_pattern, part, re.DOTALL | re.IGNORECASE)

        if field_match:
          field_content = field_match.group(1).strip()

          # Parse based on field type
          if isinstance(field.value, pg.typing.Str):
            code_info = self._extract_code_block(field_content)
            obj_data[field_name] = code_info[0] if code_info else field_content
          elif isinstance(field.value, pg.typing.Int):
            obj_data[field_name] = int(field_content)
          elif isinstance(field.value, pg.typing.Float):
            obj_data[field_name] = float(field_content)
          elif isinstance(field.value, pg.typing.Bool):
            obj_data[field_name] = field_content.lower() in (
                'true',
                'yes',
                '1',
            )
          else:
            obj_data[field_name] = field_content
        elif not field.value.is_noneable:
          raise ValueError(
              f'Required field "{field_name}" not found in list item'
          )

      items.append(obj_cls(**obj_data))

    return items

  def _parse_union_field(
      self,
      section_content: str,
      union_spec: pg.typing.Union,
      field_name: str,
      all_dependencies: dict[str, type[pg.Object]],
  ) -> Any:
    """Parse a Union type field by trying each candidate in order."""
    if not section_content:
      # Empty content - check if None is allowed
      if union_spec.is_noneable:
        return None
      raise ValueError(f'Field "{field_name}" is empty but not optional')

    # Check if this is a pyobject code block
    code_info = self._extract_code_block(section_content)
    if code_info and code_info[1] == 'pyobject':
      # Delegate to Python protocol for parsing
      # This handles Union matching automatically through PyGlove's type system
      # Delay import at runtime to avoid circular dependency.
      # pylint: disable=g-import-not-at-top
      # pytype: disable=import-error
      from langfun.core.structured.schema import python
      # pytype: enable=import-error
      # pylint: enable=g-import-not-at-top

      code = code_info[0]
      # Strip any backticks that LLMs might add
      code = code.strip('`')

      # Build global_vars with all class dependencies
      global_vars = all_dependencies.copy()

      try:
        result = python.structure_from_python(
            code,
            global_vars=global_vars,
            permission=pg.coding.CodePermission.CALL,
        )
        return result
      except Exception as e:
        raise ValueError(
            f'Failed to parse pyobject for field "{field_name}": {e}'
        ) from e

    # Use the dependencies passed from _parse_markdown (like Python protocol)
    all_classes = all_dependencies.copy()
    code_classes = {}

    for candidate in union_spec.candidates:
      if isinstance(candidate, pg.typing.Object):
        # Add the candidate class itself
        all_classes[candidate.cls.__name__] = candidate.cls

        # Check if this is a code class
        cls_fields = list(candidate.cls.__schema__.items())
        if len(cls_fields) == 1:
          cls_key, cls_field = cls_fields[0]
          if isinstance(cls_key, pg.typing.ConstStrKey):
            cls_field_name = str(cls_key)
            if cls_field_name.endswith('_code') and isinstance(
                cls_field.value, pg.typing.Str
            ):
              code_classes[candidate.cls.__name__] = (
                  candidate.cls,
                  cls_field_name,
              )

    # Sort candidates: Objects first, then primitives, then List last
    # This prevents List from matching too eagerly
    def candidate_priority(candidate):
      if isinstance(candidate, pg.typing.Object):
        return 0  # Highest priority
      elif isinstance(candidate, pg.typing.List):
        return 2  # Lowest priority
      else:
        return 1  # Medium priority

    sorted_candidates = sorted(union_spec.candidates, key=candidate_priority)

    # Try each candidate type in priority order
    errors = []
    for candidate in sorted_candidates:
      try:
        # Try to parse as this candidate type
        if isinstance(candidate, pg.typing.Object):
          if candidate.cls.__name__ in code_classes:
            # This is a code class - check language marker
            code_info = self._extract_code_block(section_content)
            if code_info:
              code_content, actual_lang = code_info
              # Check if this code class matches the language
              current_expected_lang = self._detect_code_language(
                  code_classes[candidate.cls.__name__][1]
              )

              # If language matches this code class, use it
              if actual_lang == current_expected_lang:
                code_cls, code_field_name = code_classes[candidate.cls.__name__]
                return code_cls(**{code_field_name: code_content})

              # If language doesn't match, skip this candidate
              raise ValueError(
                  f'Code block language "{actual_lang}" does not match expected'
                  f' language "{current_expected_lang}" for'
                  f' {candidate.cls.__name__}'
              )
          else:
            # This is a complex object - should have been handled by
            # pyobject above
            raise ValueError(
                f'Complex object {candidate.cls.__name__} must use'
                ' pyobject marker'
            )
        elif isinstance(candidate, pg.typing.List):
          # Check if this is a pyobject block with Python list literal
          code_info = self._extract_code_block(section_content)
          if code_info:
            _, language = code_info
            if language == 'pyobject':
              # Should have been handled above
              raise ValueError('pyobject blocks handled earlier')

          # Try markdown-style list parsing
          # (has ### headers or - list items)
          if '###' in section_content or section_content.strip().startswith(
              '-'
          ):
            element_spec = candidate.element.value
            if isinstance(element_spec, pg.typing.Object):
              items = self._extract_list_objects(
                  section_content, element_spec.cls
              )
              if items:  # Only return if we actually found items
                return items
            else:
              items = self._extract_list_items(section_content)
              if items:  # Only return if we actually found items
                if isinstance(element_spec, pg.typing.Int):
                  return [int(item) for item in items]
                elif isinstance(element_spec, pg.typing.Float):
                  return [float(item) for item in items]
                else:
                  return items
          # If doesn't look like a list, skip this candidate
          raise ValueError('Content does not look like a list')
        elif isinstance(candidate, pg.typing.Str):
          code_info = self._extract_code_block(section_content)
          if code_info:
            # If this is a pyobject block, skip Str candidate
            # Let Object candidates handle it
            if code_info[1] == 'pyobject':
              raise ValueError(
                  'pyobject code blocks should be parsed as Objects, not Str'
              )
            return code_info[0]
          return section_content.strip()
        elif isinstance(candidate, pg.typing.Int):
          return int(section_content.strip())
        elif isinstance(candidate, pg.typing.Float):
          return float(section_content.strip())
        elif isinstance(candidate, pg.typing.Bool):
          return section_content.strip().lower() in ('true', 'yes', '1')
        else:
          # Unknown type, skip
          continue
      except Exception as e:  # pylint: disable=broad-exception-caught
        errors.append((candidate, e))
        continue

    # If we get here, all candidates failed
    error_msg = (
        f'Failed to parse field "{field_name}" as any Union candidate:\\n'
    )
    for candidate, error in errors:
      error_msg += f'  - {candidate}: {error}\\n'
    raise ValueError(error_msg)

  def _parse_as_object(
      self,
      section_content: str,
      obj_cls: type[pg.Object],
      all_classes: dict[str, type[pg.Object]] | None = None,
  ) -> pg.Object:
    """Parse content as a PyGlove Object using Python eval."""
    # Delay import at runtime to avoid circular dependency.
    # pylint: disable=g-import-not-at-top
    # pytype: disable=import-error
    from langfun.core.structured.schema import python
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top

    # Extract code if wrapped in triple backticks (```...```)
    code_info = self._extract_code_block(section_content)
    if code_info:
      code = code_info[0]
    else:
      code = section_content.strip()

    # Strip any leading/trailing backticks that LLMs might add
    # (e.g., `BrowseWeb(...)` instead of BrowseWeb(...))
    code = code.strip('`')

    # Build global_vars with all classes
    global_vars = all_classes.copy() if all_classes else {}
    # Ensure the target class is included
    global_vars[obj_cls.__name__] = obj_cls

    # Use Python protocol to parse the object
    try:
      result = python.structure_from_python(
          code,
          global_vars=global_vars,
          permission=pg.coding.CodePermission.CALL,
      )
      # Verify it's the right type
      if isinstance(result, obj_cls):
        return result
      raise TypeError(
          f'Expected {obj_cls.__name__}, got {type(result).__name__}'
      )
    except Exception as e:
      raise ValueError(f'Failed to parse as {obj_cls.__name__}: {e}') from e
