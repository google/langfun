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
import functools
import inspect
import io
import re
import sys
import textwrap
import tokenize
import typing
from typing import Any, Callable, Iterator, Sequence, Type
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
    kwargs.setdefault(
        'field_hint', functools.partial(self._field_hint, schema=schema)
    )
    return class_definitions(
        deps, allowed_dependencies=allowed_dependencies, **kwargs
    )

  def _field_hint(
      self, field: pg.typing.Field, schema: base.Schema
  ) -> str | None:
    hints = [h for f in self.features if (h := f.field_hint(field, schema))]
    return ' '.join(hints) or None

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
    if self.features and not (
        inspect.isclass(value) or isinstance(value, lf.Template)
    ):
      # The first feature renders the value around the protocol without it.
      rest = self._with_features(self.features[1:])
      return self.features[0].render_value(
          value,
          schema,
          functools.partial(
              rest.value_repr,
              schema=schema,
              compact=compact,
              verbose=verbose,
              markdown=markdown,
              assign_to_var=assign_to_var,
              **kwargs,
          ),
      )
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
    """Parses a Python string into a structured object.

    Features that apply to `schema` run their hooks by phase: each rewrites
    the response in turn, then each checks the code that will run, and after
    execution each checks the value.

    Args:
      text: The LM response.
      schema: The schema of the value.
      additional_context: Names the code may reference, besides the schema
        classes and `STRUCTURE_GLOBALS`.
      permission: What the code is allowed to do.
      autofix: Max number of attempts to repair failing code with an LM.
      autofix_lm: The LM that repairs failing code. Its query uses this
        protocol if the protocol has features.
      **kwargs: Unused.

    Returns:
      The parsed value.
    """
    del kwargs
    protocol = self.for_schema(schema)
    hooks = [f.parse_hooks(schema) for f in protocol.features]
    for h in hooks:
      text = h.before_exec(text)
    global_vars = additional_context or {}
    classes = {}
    if schema is not None:
      classes = {d.__name__: d for d in schema.class_dependencies()}
    if hooks:
      # The checks see every name the code may use. Names they define rank
      # below the schema classes and `STRUCTURE_GLOBALS` at execution.
      global_vars = {**global_vars, **classes, **STRUCTURE_GLOBALS}
      for h in hooks:
        h.check_code(text, global_vars)
    global_vars.update(classes)
    value, code = structure_from_python(
        text,
        global_vars=global_vars,
        autofix=autofix,
        autofix_lm=autofix_lm,
        permission=permission,
        returns_code=True,
        # Features carry over to the autofix query, e.g. so that corrected code
        # travels in a block; a plain protocol leaves it to the default.
        protocol=protocol.name if protocol.features else None,
    )
    for h in hooks:
      value = h.after_exec(value, code)
    return value


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
    protocol: str | None = None,
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
    protocol: The prompting protocol of the autofix query. If None, the
      protocol from `lf.query_protocol` is used.

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
      protocol=protocol,
  )


_OPENING_FENCE_RE = re.compile(r'[ \t]*```(?:python|py)?[ \t]*')
_CLOSING_FENCE_RE = re.compile(r'[ \t]*```')
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
    Code containing just the output object, fenced if it contains ```, or
    `text` unchanged if there is nothing to repair or the code that builds the
    object does not parse.
  """
  blocks = _code_blocks(text)
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
    if not _parses(chosen):
      return text
    return _fenced(_trim_to_object(chosen) or chosen)
  trimmed = _trim_to_object(blocks[0])
  return text if trimmed is None else _fenced(trimmed)


def _code_blocks(text: str) -> list[str]:
  """Returns the code in the ```python fences of `text`."""
  lines = text.split('\n')
  blocks = []
  i = 0
  # The last line cannot open a fence: an opening fence ends with a newline.
  while i < len(lines) - 1:
    if _OPENING_FENCE_RE.fullmatch(lines[i]):
      end = _closing_fence(lines, i + 1)
      if end is not None:
        blocks.append(''.join(line + '\n' for line in lines[i + 1 : end]))
        i = end
    i += 1
  return blocks


def _closing_fence(lines: list[str], start: int) -> int | None:
  """Returns the index of the line that closes a fence opened above `start`.

  That is the first ``` line that does not begin inside a string literal of
  the code, e.g. of a docstring holding a markdown code block.

  Args:
    lines: The lines of the response.
    start: The index of the first line inside the fence.

  Returns:
    The index of the closing line, or None if the fence is not closed.
  """
  spans = _string_spans(
      functools.partial(next, (line + '\n' for line in lines[start:]), '')
  )
  span = (0, 0)  # Before the first string literal.
  for i in range(start, len(lines)):
    if not _CLOSING_FENCE_RE.match(lines[i]):
      continue
    row = i - start + 1
    while span is not None and span[1] < row:
      span = next(spans, None)
    if span is None or span[0] >= row:
      return i
  return None


def _string_spans(readline: Callable[[], str]) -> Iterator[tuple[int, int]]:
  """Yields the first and last rows of each string literal of some code.

  Rows are 1-based, and a string literal inside an f-string is part of it, so
  the spans are disjoint and in order. Tokenizing stops at the first error:
  e.g. the prose after a fence, or a string literal that is not closed.

  Args:
    readline: Returns the next line of the code, or '' at its end.
  """
  # Start rows of the open f-strings (and t-strings), which are tokenized in
  # parts since Python 3.12.
  open_rows = []
  try:
    for token in tokenize.generate_tokens(readline):
      kind = tokenize.tok_name[token.type]
      if kind.endswith('STRING_START'):
        open_rows.append(token.start[0])
      elif kind.endswith('STRING_END'):
        first = open_rows.pop()
        if not open_rows:
          yield first, token.end[0]
      elif kind == 'STRING' and not open_rows:
        yield token.start[0], token.end[0]
  except (tokenize.TokenError, SyntaxError):
    return


def _fenced(code: str) -> str:
  # Outside a fence, `parsing.clean` reads ``` as a fence even in a string.
  if '```' not in code:
    return code
  return '```python\n' + code.rstrip('\n') + '\n```'


def _parses(code: str) -> bool:
  try:
    ast.parse(code)
  except SyntaxError:
    return False
  return True


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
