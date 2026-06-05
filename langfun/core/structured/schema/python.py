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
"""Python-based prompting protocol."""

import inspect
import io
import re
import sys
import textwrap
import typing
from typing import Any, Sequence, Type
import langfun.core as lf
from langfun.core.coding.python import correction
from langfun.core.structured.schema import base
import pyglove as pg


class PythonPromptingProtocol(base.PromptingProtocol):
  """Python-based prompting protocol."""

  NAME = 'python'

  def schema_repr(
      self,
      schema: base.Schema,
      *,
      include_result_definition: bool = True,
      markdown: bool = True,
      **kwargs,
  ) -> str:
    ret = ''
    if include_result_definition:
      ret += self.result_definition(schema)
    class_definition_str = self.class_definitions(
        schema, markdown=markdown, **kwargs
    )
    if class_definition_str:
      ret += f'\n\n{class_definition_str}'
    return ret.strip()

  def class_definitions(
      self,
      schema: base.Schema,
      additional_dependencies: list[Type[Any]] | None = None,
      **kwargs
  ) -> str | None:
    """Returns a string containing of class definitions from a schema."""
    deps = schema.class_dependencies(
        include_base_classes=False, include_subclasses=True
    )
    allowed_dependencies = set(deps)
    if additional_dependencies:
      allowed_dependencies.update(additional_dependencies)
    return class_definitions(
        deps, allowed_dependencies=allowed_dependencies, **kwargs
    )

  def result_definition(self, schema: base.Schema) -> str:
    return base.annotation(schema.spec)

  def value_repr(
      self,
      value: Any,
      schema: base.Schema | None = None,
      *,
      compact: bool = True,
      verbose: bool = False,
      markdown: bool = True,
      assign_to_var: str | None = None,
      **kwargs) -> str:
    del schema
    if inspect.isclass(value):
      cls_schema = base.Schema.from_value(value)
      if isinstance(cls_schema.spec, pg.typing.Object):
        object_code = self.class_definitions(
            cls_schema,
            markdown=markdown,
            # We add `pg.Object` as additional dependencies to the class
            # definition so exemplars for class generation could show
            # pg.Object as their bases.
            additional_dependencies=[pg.Object]
        )
        assert object_code is not None
        return object_code
      else:
        object_code = self.result_definition(cls_schema)
    elif isinstance(value, lf.Template):
      return str(value)
    else:
      object_code = pg.format(
          value, compact=compact, verbose=verbose, python_format=True
      )
      if assign_to_var is not None:
        object_code = f'{assign_to_var} = {object_code}'
    if markdown:
      return f'```python\n{object_code}\n```'
    return object_code

  def parse_value(
      self,
      text: str,
      schema: base.Schema | None = None,
      *,
      additional_context: dict[str, Type[Any]] | None = None,
      permission: pg.coding.CodePermission = (
          pg.coding.CodePermission.ASSIGN | pg.coding.CodePermission.CALL
      ),
      autofix=0,
      autofix_lm: lf.LanguageModel = lf.contextual(),
      **kwargs,
  ) -> Any:
    """Parses a Python string into a structured object."""
    del kwargs
    global_vars = additional_context or {}
    if schema is not None:
      dependencies = schema.class_dependencies()
      global_vars.update({d.__name__: d for d in dependencies})
    return structure_from_python(
        text,
        global_vars=global_vars,
        autofix=autofix,
        autofix_lm=autofix_lm,
        permission=permission,
    )


def structure_from_python(
    code: str,
    *,
    global_vars: dict[str, Any] | None = None,
    permission: pg.coding.CodePermission = (
        pg.coding.CodePermission.ASSIGN | pg.coding.CodePermission.CALL
    ),
    autofix=0,
    autofix_lm: lf.LanguageModel = lf.contextual(),
) -> Any:
  """Evaluates structure from Python code with access to symbols."""
  global_vars = global_vars or {}
  global_vars.update({
      'pg': pg,
      'Object': pg.Object,
      'Any': typing.Any,
      'List': typing.List,
      'Tuple': typing.Tuple,
      'Dict': typing.Dict,
      'Sequence': typing.Sequence,
      'Optional': typing.Optional,
      'Union': typing.Union,
      # Special value markers.
      'UNKNOWN': base.UNKNOWN,
  })
  # We are creating objects here, so we execute the code without a sandbox.
  return correction.run_with_correction(
      code,
      global_vars=global_vars,
      sandbox=False,
      max_attempts=autofix,
      lm=autofix_lm,
      permission=permission,
  )


def source_form(value, compact: bool = True, markdown: bool = False) -> str:
  """Returns the source code form of an object."""
  return PythonPromptingProtocol().value_repr(
      value, compact=compact, markdown=markdown
  )


def include_method_in_prompt(method):
  """Decorator to include a method in the class definition of the prompt."""
  setattr(method, '__show_in_prompt__', True)
  return method


def should_include_method_in_prompt(method):
  """Returns True if the method should be shown in the prompt."""
  return getattr(method, '__show_in_prompt__', False)


def class_definition(
    cls,
    strict: bool = False,
    allowed_dependencies: set[Type[Any]] | None = None,
) -> str:
  """Returns the Python class definition."""
  out = io.StringIO()
  schema = pg.schema(cls)
  eligible_bases = []
  for base_cls in cls.__bases__:
    if base_cls is not object:
      if allowed_dependencies is None or base_cls in allowed_dependencies:
        eligible_bases.append(base_cls.__name__)

  if eligible_bases:
    base_cls_str = ', '.join(eligible_bases)
    out.write(f'class {cls.__name__}({base_cls_str}):\n')
  else:
    out.write(f'class {cls.__name__}:\n')

  if cls.__doc__:
    doc_lines = cls.__doc__.strip().split('\n')
    if len(doc_lines) == 1:
      out.write(f'  """{cls.__doc__}"""\n')
    else:
      out.write('  """')

      # Since Python 3.13, the indentation of docstring lines is removed.
      # Therefore, we add two spaces to each non-empty line to keep the
      # indentation consistent with the class definition.
      if sys.version_info >= (3, 13):
        for i in range(1, len(doc_lines)):
          if doc_lines[i]:
            doc_lines[i] = ' ' * 2 + doc_lines[i]

      for line in doc_lines:
        out.write(line)
        out.write('\n')
      out.write('  """\n')

  empty_class = True
  if schema.fields:
    for key, field in schema.items():
      if not isinstance(key, pg.typing.ConstStrKey):
        pg.logging.warning(
            'Variable-length keyword arguments is not supported in '
            f'structured parsing or query. Encountered: {cls}, Schema: {schema}'
        )
        continue

      # Skip fields that are marked as excluded from the prompt sent to LLM
      # for OOP.
      if field.metadata.get('exclude_from_prompt', False):
        continue

      # Write field doc string as comments before the field definition.
      if field.description:
        for line in field.description.split('\n'):
          if line:
            out.write('  # ')
            out.write(line)
            out.write('\n')

      annotation_str = base.annotation(
          field.value, strict=strict, allowed_dependencies=allowed_dependencies
      )
      out.write(f'  {field.key}: {annotation_str}')
      out.write('\n')
      empty_class = False

  for method in _iter_newly_defined_methods(cls, allowed_dependencies):
    source = inspect.getsource(method)
    # Remove decorators from the method definition.
    source = re.sub(r'\s*@.*\.include_method_in_prompt.*\n', '', source)
    out.write('\n')
    out.write(
        textwrap.indent(
            inspect.cleandoc('\n' + source), ' ' * 2)
    )
    out.write('\n')
    empty_class = False

  if empty_class:
    out.write('  pass\n')
  return out.getvalue()


def _iter_newly_defined_methods(
    cls, allowed_dependencies: set[Type[Any]] | None):
  names = {attr_name: True for attr_name in dir(cls)}
  for base_cls in cls.__bases__:
    if allowed_dependencies is None or base_cls in allowed_dependencies:
      for name in dir(base_cls):
        names.pop(name, None)
  for name in names.keys():
    attr = getattr(cls, name)
    if callable(attr) and should_include_method_in_prompt(attr):
      yield attr


def class_definitions(
    classes: Sequence[Type[Any]],
    *,
    allowed_dependencies: set[Type[Any]] | None = None,
    strict: bool = False,
    markdown: bool = False,
) -> str | None:
  """Returns a string for class definitions."""
  if not classes:
    return None
  def_str = io.StringIO()
  for i, cls in enumerate(classes):
    if i > 0:
      def_str.write('\n')
    def_str.write(
        class_definition(
            cls,
            strict=strict,
            allowed_dependencies=allowed_dependencies,
        )
    )
  ret = def_str.getvalue()
  if markdown and ret:
    ret = f'```python\n{ret}```'
  return ret
