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

import ast
import inspect
import io
import re
import sys
import textwrap
import typing
from typing import Any, Callable, Sequence, Type
import langfun.core as lf
from langfun.core.coding.python import correction
from langfun.core.structured.schema import base
import pyglove as pg


class PythonPromptingProtocol(base.PromptingProtocol):
  """Python-based prompting protocol."""

  NAME = 'python'
  SUPPORTS_AUTOFIX = True

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
      returns_code: bool = False,
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
        returns_code=returns_code,
    )


# Names that every `structure_from_python()` call binds, besides `global_vars`.
STRUCTURE_GLOBALS: dict[str, Any] = {
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
}


def structure_from_python(
    code: str,
    *,
    global_vars: dict[str, Any] | None = None,
    permission: pg.coding.CodePermission = (
        pg.coding.CodePermission.ASSIGN | pg.coding.CodePermission.CALL
    ),
    autofix=0,
    autofix_lm: lf.LanguageModel = lf.contextual(),
    returns_code: bool = False,
) -> Any:
  """Evaluates structure from Python code with access to symbols.

  Args:
    code: The Python code to evaluate.
    global_vars: Names the code may reference, besides `STRUCTURE_GLOBALS`.
    permission: What the code is allowed to do.
    autofix: Max number of attempts to repair failing code with `autofix_lm`.
    autofix_lm: The LM that repairs failing code.
    returns_code: If True, returns a tuple of (value, the code that produced
      it), which differs from `code` if autofix rewrote it.

  Returns:
    The value of the code, or a tuple of (value, code) if `returns_code`.
  """
  global_vars = global_vars or {}
  global_vars.update(STRUCTURE_GLOBALS)
  # We are creating objects here, so we execute the code without a sandbox.
  return correction.run_with_correction(
      code,
      global_vars=global_vars,
      sandbox=False,
      max_attempts=autofix,
      lm=autofix_lm,
      permission=permission,
      returns_code=returns_code,
  )


_CODE_FENCE_RE = re.compile(
    r'^[ \t]*```(?:python|py)?[ \t]*\n(.*?)^[ \t]*```', re.DOTALL | re.MULTILINE
)
_DEFINITION_RE = re.compile(r'^\s*(class|import|from|def)\b')


def select_object_code(
    text: str,
    assign_to_var: str = 'output',
    class_names: Sequence[str] = (),
) -> str:
  """Returns the part of an LM response that holds the output object.

  LMs sometimes echo the schema's class definitions before emitting the object,
  either in a separate ```python fence (markdown extraction then takes the
  first one) or above the object inside the same fence. Both fail with
  `Class definition is not allowed`, losing a response whose object is present
  and correct.

  Args:
    text: An LM response, possibly containing several fenced code blocks.
    assign_to_var: The variable the output object is expected to be assigned to.
    class_names: Classes of the output object. A block that assigns an instance
      of one of them to `assign_to_var` wins over any other code block, e.g. a
      script the LM fenced as Python that assigns its own `output`.

  Returns:
    Code containing just the output object, or `text` unchanged if there is
    nothing to repair.
  """
  blocks = _CODE_FENCE_RE.findall(text)
  if not blocks:
    return text
  if len(blocks) > 1:
    assignment = re.compile(
        rf'^\s*{re.escape(assign_to_var)}\s*=', re.MULTILINE
    )
    construction = re.compile(
        rf'^\s*{re.escape(assign_to_var)}\s*=\s*'
        rf'(?:{"|".join(map(re.escape, class_names))})\s*\(',
        re.MULTILINE,
    )

    def _score(block: str) -> int:
      return (
          8 * bool(class_names and construction.search(block))
          + 4 * bool(assignment.search(block))
          + 2 * (not _DEFINITION_RE.match(block))
          + bool('(' in block)
      )

    # `max` keeps the first maximum; reverse so ties go to the last block,
    # which is where the object goes when the LM prefixes it with something.
    chosen = max(reversed(blocks), key=_score)
    return _trim_to_object(chosen) or chosen
  return _trim_to_object(blocks[0]) or text


def _trim_to_object(code: str) -> str | None:
  """Returns `code` reduced to the output object, or None if unchanged."""
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return None
  call = _as_constructor_call(tree)
  if call is not None:
    return ast.unparse(call)
  definitions = (ast.ClassDef, ast.FunctionDef, ast.Import, ast.ImportFrom)
  kept = [n for n in tree.body if not isinstance(n, definitions)]
  if not kept or len(kept) == len(tree.body):
    return None
  return '\n'.join(ast.unparse(n) for n in kept)


def _as_constructor_call(tree: ast.Module) -> ast.Expr | None:
  r"""Rewrites `class X:\n  a = 1` into `X(a=1)`, which is what the LM meant.

  A schema class definition (`class X:\n  a: int`) has no assigned values and
  is left alone, so class-generating queries are unaffected.

  Args:
    tree: The parsed response.

  Returns:
    The equivalent constructor call, or None if the response is not a class
    body of assignments.
  """
  if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
    return None
  cls = tree.body[0]
  keywords = []
  for stmt in cls.body:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
      target, value = stmt.targets[0], stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
      target, value = stmt.target, stmt.value
    else:
      return None
    if not isinstance(target, ast.Name):
      return None
    keywords.append(ast.keyword(arg=target.id, value=value))
  return ast.fix_missing_locations(
      ast.Expr(
          value=ast.Call(
              func=ast.Name(id=cls.name, ctx=ast.Load()),
              args=[],
              keywords=keywords,
          )
      )
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
    field_hint: Callable[[pg.typing.Field], str | None] | None = None,
) -> str:
  """Returns the Python class definition.

  Args:
    cls: The class to render.
    strict: If True, render value specs in `pg.typing` form.
    allowed_dependencies: Classes that may be referenced by name.
    field_hint: An optional callable that returns an extra comment line to emit
      right above a field, e.g. to tell the LM how to provide its value.

  Returns:
    The Python source of the class definition.
  """
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

      hint = field_hint(field) if field_hint is not None else None
      if hint:
        out.write(f'  # {hint}\n')

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
    field_hint: Callable[[pg.typing.Field], str | None] | None = None,
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
            field_hint=field_hint,
        )
    )
  ret = def_str.getvalue()
  if markdown and ret:
    ret = f'```python\n{ret}```'
  return ret
