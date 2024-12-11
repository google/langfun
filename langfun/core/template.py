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
"""langfun text template with compositionality."""

import contextlib
import dataclasses
import functools
import inspect
from typing import Annotated, Any, Callable, Iterator, Set, Tuple, Type, Union

import jinja2
from jinja2 import meta as jinja2_meta
from langfun.core import component
from langfun.core import message as message_lib
from langfun.core import modality
from langfun.core import natural_language
from langfun.core import subscription
import pyglove as pg


# Include this string anywhere in the class docstr in order not to treat
# the docstring as a template.
NO_TEMPLATE_DOCSTR_SIGN = 'THIS IS NOT A TEMPLATE'

# Keys for storing render stack in the
# thread-local storage.
_TLS_RENDER_STACK = '_template_render_stack'
_TLS_RENDER_RESULT_CACHE = '_template_render_result_cache'

# The prefix for fields or contextual attributes to be treated as additional
# metadata for rendered message.
_ADDITIONAL_METADATA_PREFIX = 'metadata_'


class Template(
    natural_language.NaturalLanguageFormattable,
    component.Component,
    pg.typing.CustomTyping,
    pg.views.HtmlTreeView.Extension
):
  """Langfun string template.

  Langfun uses jinja2 as its template engine. Pleaes check out
  https://jinja.palletsprojects.com/en/3.1.x/templates/ for detailed
  explanation on the template language.
  """

  template_str: Annotated[
      str,
      (
          'A template string in jinja2 syntax. During `render`, the variables '
          'will be resolved from 1) the `kwargs` dict passed to the `render` '
          'method; 2) the attributes of this object; and 3) the attributes of '
          'its containing objects.'
      ),
  ]

  clean: Annotated[
      bool,
      (
          'If True, `inspect.cleandoc` will be applied on `template_str` to '
          'additional indention, etc. Otherwise, the original form of the '
          'template string will be used.'
      ),
  ] = True

  __kwargs__: Annotated[
      Any,
      (
          'Wildcard keyword arguments for `__init__` that can be referred in '
          'the template string. This allows modularization of prompt with '
          'fully or partially bound variables.'
      ),
  ]

  def __init_subclass__(cls):
    # NOTE(daiyip): class attribute `template_str` may exists in 2 ways:
    # A string: specified by the user as the new default value for template_str.
    # A property: inherited from the base class. In this case, we only update
    #   its default value when template_str can be extracted from the docstr.
    # Here we try to detect `template_str` from docstr when it's not specified
    # by the user.
    template_str = getattr(cls, 'template_str', None)

    if not isinstance(template_str, str):
      # Here we try to update the default value of `template_str` based on
      # the class docstring. Unless it's not provided or
      # 'THIS IS NOT A TEMPLATE' appears in the docstring.
      template_str = cls._template_str_from_docstr()
      if template_str:
        # Set the `template_str` attribute will change the default value
        # of the `template_str` symbolic field.
        setattr(cls, 'template_str', template_str)

    if template_str:
      # Declare template variables as symbolic attributes.
      template_vars = Template.resolve_vars(template_str)
      for var_name in template_vars:
        if 'DEFAULT' == var_name:
          raise ValueError(
              '`{{ DEFAULT }}` cannot be used in pre-configured templates. '
              f'Encountered: {template_str!r}'
          )
        # NOTE(daiyip): This is to avoid warning from accessing
        # `pg.Object.schema`, which was replaced by `pg.Object.__schema__`.
        if var_name == 'schema' or not hasattr(cls, var_name):
          setattr(cls, var_name, component.contextual())

    super().__init_subclass__()

  @classmethod
  def _template_str_from_docstr(cls) -> str:
    """Extract template string from docstring."""
    docstr = cls.__doc__
    if (
        not docstr
        or docstr is cls.__base__.__doc__
        or NO_TEMPLATE_DOCSTR_SIGN in docstr
    ):
      return ''

    docstr = inspect.cleandoc(docstr)
    doc_start = docstr.find('\n')
    if doc_start == -1:
      return ''

    return docstr[doc_start + 1 :].strip()

  @classmethod
  def resolve_vars(cls, template_str: str) -> Set[str]:
    try:
      env = jinja2.Environment()
      ast = env.parse(template_str)
      return jinja2_meta.find_undeclared_variables(ast)
    except jinja2.TemplateSyntaxError as e:
      raise ValueError(f'Bad template string:\n\n{template_str}') from e

  def _on_bound(self) -> None:
    super()._on_bound()

    # Invalidate cached properties.
    if self.clean:
      self.rebind(
          template_str=inspect.cleandoc(self.template_str),
          skip_notification=True,
      )

    # Invalidate cached variables and template cache.
    self.__dict__.pop('_variables', None)
    self.__dict__.pop('_template', None)

    # Use contextual value for unassigned attributes.
    # TODO(daiyip): Consider to delay template parsing upon usage.
    unassigned_vars = {}
    for k in self._variables:
      if k not in ('DEFAULT',) and not hasattr(self, k):
        unassigned_vars[k] = component.contextual()
    if unassigned_vars:
      self.rebind(unassigned_vars, skip_notification=True)

    # Last render output.
    self._cached_render_output = None

  @property
  def render_output(self) -> message_lib.Message | None:
    """Returns last render output."""
    return self._cached_render_output

  @functools.cached_property
  def _variables(self) -> Set[str]:
    """Returns all declared variables from the template."""
    return Template.resolve_vars(self.template_str)

  @functools.cached_property
  def _template(self) -> jinja2.Template:
    return jinja2.Template(self.template_str)

  def vars(
      self,
      specified: bool | None = None,
      closure: bool = False,
      leaf: bool | None = None,
  ) -> Set[str]:
    """Returns referred variables.

    Args:
      specified: If True, include only variables that are specified. If False,
        include only variables that are not specified. If None, include both.
      closure: If True, include variables from referred LangFuncs recursively.
        Otherwise, include the immediate used variables.
      leaf: If True, include only the non-LangFunc variables. If False, include
        LangFunc variables. If None, include both.

    Returns:
      A list of variable names that match the criteria.
    """
    variables = set()
    for k in self._variables:
      v = getattr(self, k, pg.MISSING_VALUE)

      match specified:
        case None:
          include = True
        case True:
          include = v != pg.MISSING_VALUE
        case _:
          include = v == pg.MISSING_VALUE

      if include and leaf is not None:
        include = leaf != isinstance(v, Template)

      if include:
        variables.add(k)

      if closure and isinstance(v, Template):
        variables.update(v.vars(closure=True, specified=specified, leaf=leaf))
    return variables

  @property
  def missing_vars(self) -> Set[str]:
    """Returns the missing variable names."""
    return self.vars(closure=True, specified=False)

  @classmethod
  def raw_str(cls, text: str) -> str:
    """Returns a template string that preserve the text as original."""
    return '{% raw %}' + text + '{% endraw %}'

  @classmethod
  def from_raw_str(cls, text: str) -> 'Template':
    """Returns a template that preserve the text as original."""
    return cls(cls.raw_str(text), clean=False)

  def render(
      self,
      *,
      allow_partial: bool = False,
      implicit: bool = False,
      message_cls: Type[message_lib.Message] = message_lib.UserMessage,
      **kwargs,
  ) -> message_lib.Message:
    """Renders the template with variables from the context.

    Args:
      allow_partial: Allow partial rendering, this means that unresolved
        variables are allowed and remain in the output text.
      implicit: If True, reuse the rendering output if a parent LangFunc
        is rendering current LangFunc multiple times. This is important
        for making sure all references to the same LangFunc within a single
        top-level rendering would return the same result. If False, every call
        to `render` will trigger the actual rendering process.
      message_cls: The message class used for creating the return value.
      **kwargs: Values for template variables.

    Returns:
      An Message object as the rendered result.
    """
    try:
      pg.object_utils.thread_local_push(_TLS_RENDER_STACK, self)

      # NOTE(daiyip): Rendering could be triggered explicitly (by the user code)
      # or implicitly (when `str` gets called during template rendering). Since
      # a LangFunc object could be referenced in a template multiple times, we
      # need to have all the references to the same LangFunc object produce the
      # same output. Rendering each sub-LangFunc object once and only once would
      # also greatly improve the rendering performance.
      #
      # To achieve this, we install a result cache for every explicit rendering.
      # Thus, when sub-LangFunc are rendered, they could retrieve the output
      # from the cache.

      cache = {}
      if implicit:
        cache = pg.object_utils.thread_local_get(_TLS_RENDER_RESULT_CACHE, {})

        if id(self) in cache:
          return cache[id(self)]

        caching_context = contextlib.nullcontext()
      else:
        caching_context = pg.object_utils.thread_local_value_scope(
            _TLS_RENDER_RESULT_CACHE, cache, {}
        )

      with caching_context:
        # We use **kwargs for both the missing parts of the child components
        # and as the overriden attributes of current component.
        with component.context(**kwargs):
          with self.override(**kwargs):
            inputs = dict()
            for var_name in self._variables:
              var_value = getattr(self, var_name, pg.MISSING_VALUE)
              if var_value == pg.MISSING_VALUE:
                if allow_partial:
                  var_value = _UnresolvedExpression(var_name)
                else:
                  raise ValueError(
                      f'The value for template variable {var_name!r} is not '
                      'provided.'
                  )
              inputs[var_name] = var_value

            # Enable Python format for builtin types during template rendering,
            # versus the default PyGlove format (e.g. [0: 'abc'] for list).
            # User-defined classes may have their own format.
            with pg.object_utils.str_format(
                # Use compact (single-line) Python format (vs. PyGlove format)
                # quoted with markdown notions (e.g. `Foo(1)`)
                # for non-natural-language-formattable symbolic objects.
                markdown=True,
                compact=True,
                python_format=True,
            ):
              # Natural language formattable objects will be returned in natural
              # language when they are directly returned as rendering elements
              # in the template.
              with modality.format_modality_as_ref():
                rendered_text = self._template.render(**inputs)

            # Carry additional metadata.
            metadata = self.additional_metadata()

        if self.clean:
          rendered_text = rendered_text.strip()

        metadata.update(
            {k: pg.Ref(v) for k, v in inputs.items() if not inspect.ismethod(v)}
        )

        # Fill the variables for rendering the template as metadata.
        message = message_cls(text=rendered_text, metadata=metadata)

        # Tag input as rendered message.
        message.tag(message_lib.Message.TAG_RENDERED)

        # Adding result to cache.
        cache[id(self)] = message

        # Set last render output to message.
        self._cached_render_output = message

        # Emit TemplateRenderEvent.
        # Event handlers could access the cached result from nested templates
        # using `str()`.
        render_stack = list(pg.object_utils.thread_local_get(_TLS_RENDER_STACK))
        render_stack.pop()
        subscription.emit(
            TemplateRenderEvent(
                sender=self, output=message, render_stack=render_stack
            )
        )
      return message
    finally:
      top = pg.object_utils.thread_local_pop(_TLS_RENDER_STACK)
      assert top is self, (top, self)

  def additional_metadata(self) -> dict[str, Any]:
    """Returns additional metadta to be carried in the rendered message."""
    metadata = {}
    # Carry metadata from `lf.context`.
    for k, v in component.all_contextual_values().items():
      if k.startswith(_ADDITIONAL_METADATA_PREFIX):
        metadata[k.removeprefix(_ADDITIONAL_METADATA_PREFIX)] = v

    # Carry metadata from fields.
    for k, v in self.sym_init_args.items():
      if k.startswith(_ADDITIONAL_METADATA_PREFIX):
        metadata[k.removeprefix(_ADDITIONAL_METADATA_PREFIX)] = v
    return metadata

  #
  # Implements `pg.typing.CustomTyping`.
  #

  def custom_apply(
      self,
      path: pg.KeyPath,
      value_spec: pg.typing.ValueSpec,
      allow_partial: bool,
      child_transform: Callable[[pg.KeyPath, pg.typing.Field, Any], Any]
      | None = None,
  ) -> Tuple[bool, Any]:
    """Makes it applicable to pg.typing.Str()."""
    del allow_partial
    del child_transform

    # Check if value_spec directly accepts `self`.
    if value_spec.value_type and isinstance(self, value_spec.value_type):
      return (False, self)

    pg.typing.ensure_value_spec(value_spec, pg.typing.Str().noneable(), path)
    return (False, self)

  #
  # Implements `lf.NaturalLanguageFormattable`.
  #

  def natural_language_format(self) -> str:
    """Returns the natural language format representation."""
    return self.render(allow_partial=True, implicit=True).text

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, str):
      return not self.missing_vars and str(self) == other
    return super().__eq__(other)

  def __hash__(self) -> int:
    # Override __hash__ since __eq__ has changed.
    return object.__hash__(self)

  #
  # Special methods.
  #

  @property
  def DEFAULT(self) -> 'Template':
    """Referring to the default value used for this template.

    This method is intended to be used in template for referring to the default
    value of current template. There are two scenarios:

    Scenario 1: Use instance-level template_str to override the class default.

    ```
    class Foo(lf.Template):
       '''Foo template.

       This is {{x}}.
       '''

    f = Foo(template_str='<h1>{{DEFAULT}}</h1>', x=1)
    f.render()

    >> <h1>This is 1.</h1>
    ```

    Scenario 2: Use an ad-hoc template to override a predefined field.

    ```
    class Bar(lf.Template):
      '''Bar template.

      {{preamble}}
      {{prompt}}
      '''
      preamble: lf.Template = lf.Template('You are a chat bot.')
      prompt: lf.Template = lf.Template('User: hi')

    b = Bar(preamble=lf.Template('<h1>{{DEFAULT}}<h1>'),
            prompt=lf.Template('<h2>{{DEFAULT}}</h2>')
    b.render()

    >> <h1>You are a chat bot.<h1>
    >> <h2>User: hi</h2>
    ```

    Returns:
      The default (pre-configured) value used for this template.
    """
    base_template = self.__class__.__schema__['template_str'].default_value
    if base_template == pg.MISSING_VALUE:
      if not self.sym_path:
        raise ValueError(
            f'No DEFAULT template found for {self!r}: '
            'The template neither has a default `template_str` nor is '
            'contained under another object.'
        )
      key = self.sym_path.key
      assert self.sym_parent is not None
      assigned_field = self.sym_parent.sym_attr_field(key)
      container_cls = self.sym_parent.__class__

      if (
          assigned_field is None
          or assigned_field.default_value == pg.MISSING_VALUE
      ):
        raise ValueError(
            f'No DEFAULT template found for {self!r}: '
            f'`{container_cls.__name__}.{key}` '
            'does not have a default value. '
        )
      base_template = assigned_field.default_value
      if isinstance(base_template, Template):
        base_template = base_template.template_str
      if not isinstance(base_template, str):
        raise ValueError(
            f'No DEFAULT template found for {self!r}: The default '
            f'value {base_template!r} of '
            f'`{container_cls.__name__}.{key}` is not a '
            '`lf.Template` object or str.'
        )
    t = Template(base_template)
    # NOTE(daiyip): Set the parent of the newly created template to self so
    # it could access all the contextual variables.
    t.sym_setparent(self)
    return t

  @classmethod
  def from_value(
      cls,
      value: Union[str, message_lib.Message, 'Template'],
      **kwargs
  ) -> 'Template':
    """Create a template object from a string or template."""
    if isinstance(value, cls):
      return value.clone(override=kwargs) if kwargs else value  # pylint: disable=no-value-for-parameter
    if isinstance(value, str):
      return cls(template_str=value, **kwargs)
    if isinstance(value, message_lib.Message):
      kwargs.update(value.metadata)
      return cls(template_str=value.text, **kwargs)
    if isinstance(value, Template):
      lfun = cls(template_str=value.template_str, **kwargs)
      # So lfun could acccess all attributes from value.
      lfun.sym_setparent(value)
      return lfun
    return cls(template_str='{{input}}', input=value, **kwargs)

  def _html_tree_view_content(
      self,
      *,
      view: pg.views.HtmlTreeView,
      root_path: pg.KeyPath | None = None,
      collapse_level: int | None = None,
      extra_flags: dict[str, Any] | None = None,
      debug: bool = False,
      **kwargs,
  ):
    extra_flags = extra_flags if extra_flags is not None else {}
    collapse_template_vars_level: int | None = extra_flags.get(
        'collapse_template_vars_level', 1
    )

    def render_template_str():
      return pg.Html.element(
          'div',
          [
              pg.Html.element('span', [self.template_str])
          ],
          css_classes=['template-str'],
      )

    def render_fields():
      return view.complex_value(
          {k: v for k, v in self.sym_items()},
          name='fields',
          root_path=root_path,
          parent=self,
          exclude_keys=['template_str', 'clean'],
          collapse_level=max(
              collapse_template_vars_level, collapse_level
          ) if collapse_level is not None else None,
          extra_flags=extra_flags,
          debug=debug,
          **view.get_passthrough_kwargs(
              remove=['exclude_keys'],
              **kwargs,
          )
      )

    return pg.views.html.controls.TabControl([
        pg.views.html.controls.Tab(
            'template_str',
            render_template_str(),
        ),
        pg.views.html.controls.Tab(
            'variables',
            render_fields(),
        ),
    ], selected=1)

  @classmethod
  def _html_tree_view_css_styles(cls) -> list[str]:
    return super()._html_tree_view_css_styles() + [
        """
        /* Langfun Template styles. */
        .template-str {
            padding: 10px;
            margin: 10px 5px 10px 5px;
            font-style: italic;
            font-size: 1.1em;
            white-space: pre-wrap;
            border: 1px solid #EEE;
            border-radius: 5px;
            background-color: #EEE;
            color: #cc2986;
        }
        """
    ]

  @classmethod
  @functools.cache
  def _html_tree_view_config(cls) -> dict[str, Any]:
    return pg.views.HtmlTreeView.get_kwargs(
        super()._html_tree_view_config(),
        dict(
            css_classes=['lf-template'],
        )
    )


# Register converter from str to LangFunc, therefore we can always
# pass strs to attributes that accept LangFunc.
pg.typing.register_converter(str, Template, Template)


@dataclasses.dataclass
class TemplateRenderEvent(subscription.Event[Template]):
  """Template rendering event."""

  output: message_lib.Message
  render_stack: list[Template]


class _UnresolvedExpression(pg.Object):
  """Unresolved expression in a Jinja2 template for partial rendering."""

  expression: str

  def __repr__(self) -> str:
    return self.expression

  def __str__(self) -> str:
    return '{{' + self.expression + '}}'

  def __call__(self, *args, **kwargs) -> '_UnresolvedExpression':
    items = [repr(x) for x in args] + [f'{k}={v!r}' for k, v in kwargs.items()]
    arg_str = ', '.join(items)
    return _UnresolvedExpression(f'{self.expression}({arg_str})')

  def __getitem__(self, key: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression}[{key!r}]')

  def __getattr__(self, key: str) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression}.{key}')

  def __len__(self) -> '_UnresolvedExpression':     # pylint: disable=invalid-length-returned
    return _UnresolvedExpression(f'len({self.expression})')

  def __neg__(self) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'-{self.expression}')

  def __add__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} + {other!r}')

  def __radd__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} + {self.expression}')

  def __sub__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} - {other!r}')

  def __rsub__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} - {self.expression}')

  def __mul__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} * {other!r}')

  def __rmul__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} * {self.expression}')

  def __pow__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} ** {other!r}')

  def __rpow__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} ** {self.expression}')

  def __floordiv__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} // {other!r}')

  def __rfloordiv__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} // {self.expression}')

  def __truediv__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} / {other!r}')

  def __rtruediv__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} / {self.expression}')

  def __mod__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} % {other!r}')

  def __rmod__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{other!r} % {self.expression}')

  def __eq__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} == {other!r}')

  def __ne__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} != {other!r}')

  def __lt__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} < {other!r}')

  def __le__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} <= {other!r}')

  def __gt__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} > {other!r}')

  def __ge__(self, other: Any) -> '_UnresolvedExpression':
    return _UnresolvedExpression(f'{self.expression} >= {other!r}')

  def __iter__(self) -> Iterator['_UnresolvedExpression']:
    for i in range(3):
      yield getattr(self, f'item{i}')

  def keys(self) -> Iterator[Any]:
    for i in range(3):
      yield getattr(self, f'key{i}')

  def values(self) -> Iterator[Any]:
    for i in range(3):
      yield getattr(self, f'value{i}')

  def items(self) -> Iterator[tuple[Any, Any]]:
    for i in range(3):
      yield getattr(self, f'key{i}'), getattr(self, f'value{i}')
