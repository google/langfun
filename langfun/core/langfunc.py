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
"""LangFunc: Natural-language driven component."""

import contextlib
import dataclasses
import inspect
from typing import Annotated, Any, Callable, Iterator, Set, Tuple, Type

import jinja2
from jinja2 import meta as jinja2_meta
from langfun.core import component
from langfun.core import language_model
from langfun.core import message as message_lib
from langfun.core import message_transform
from langfun.core import natural_language
from langfun.core import subscription
import pyglove as pg


# Include this string anywhere in the class docstr in order not to treat
# the docstring as a template.
NO_TEMPLATE_DOCSTR_SIGN = 'THIS IS NOT A TEMPLATE'

# Keys for storing render/call stack (invoked LangFunc objects) in the
# thread-local storage.
_TLS_LFUN_CALL_STACK = '_langfunc_callstack'
_TLS_RENDER_STACK = '_langfunc_render_stack'
_TLS_RENDER_RESULT_CACHE = '_langfunc_render_result_cache'


# NOTE(daiyip): Only the template string belongs to the positional arguments,
# all others are keyword-only for clarity.
@pg.use_init_args(['template_str'])
class LangFunc(
    natural_language.NaturalLanguageFormattable,
    # LangFunc is also a component that transforms an message to another
    # message, so we can chain it.
    message_transform.MessageTransform,
    pg.typing.CustomTyping,
):
  r"""Base class for natural-language driven component.

  ``LangFunc`` is a language-driven component that enables users to
  seamlessly interact with Language Models (LLMs) using a blend of natural
  language and code. It empowers users to easily modularize prompt/execution
  logics, compose them, and simplify the creation of Language Model (LLM)-based
  components and applications.

  LangFunc can be conceptualized as a string template with embeddable code,
  but it distinguishes itself from traditional template systems in four key
  ways.

  Firstly, it enables easy modularization of templates along with the required
  values with OO principles, providing a reusable way for LLM-based content
  generation. For example:

    ```
    class FewshotExamples(lf.LangFunc):
      '''Base for fewshot prompt.

      {% for example in examples %}
      {{ example }}
      {% endfor %}
      '''

    # Usage 1: __init__ time binding.
    assert FewshotPrompt(examples=['foo', 'bar'])() == 'foo\nbar'

    # Usage 2: __call__ time binding.
    assert FewshotPrompt()(examples=['foo', 'bar']) == 'foo\nbar'

    class ToolDescription(lf.LangFunc):
      '''Tool descriptions.

      {% for tool in tools %}
      {{ tool.description }}
      {% endfor %}
      '''
      # We want to constrain tools to be a list of `Tool` objects.
      tools: list[Tool]

    # Raises: runtime type checking will fail on [1, 2, 3].
    ToolDescription(tools=[1, 2, 3])
    ```

  Secondly, it has the capability to compose multiple LangFuncs together,
  enabling the accomplishment of complex language tasks with maximum reuse.
  It allows users to provide program inputs to all the LangFuncs within a
  composition at the top level, significantly simplifying the process of
  providing context for users. For example:

    ```
    class ReAct(lf.LangFunc):
      '''ReAct prompt for tool-use.

      {{ preamble }}
      {{ tool_description }}
      {{ tool_examples }}
      {{ user_input }}
      '''
      # Default preamble, which could be overriden from subclass
      # or parsed from the `__init__` argument.
      preamble = 'Please help me on my task based on the following tools.',

    react = ReAct(
        tool_description=ToolDescription()
        tool_examples=FewshotExamples(),
        # Partially bind `tools` and `examples`.
        tools=my_tools,
        examples=[t.examples for t in my_tools]
        )

    # Late bind `user_input` at __call__ time.
    react(user_input='Help me get a lunch to go, veggie please.' )
    ```

  Thirdly, it allows the flexibility to encapsulate complex compositions to
  reusable classes and modify them. For example:

    ```
    # The compound decorator converts a function into a LangFunc.
    @lf.compound
    def react_with_tools(preamble, tools: list[Tool]):
      return ReAct(
          preamble=preamble,
          tool_description=ToolDescription()
          tool_examples=FewshotExamples(),
          # Partially bind `tools` and `examples`.
          tools=my_tools,
          examples=[t.examples for t in my_tools]
      )

    # Actually, the entire chat application is a LangFunc.
    class Chat(lt.LangFunc):
      '''LLM-based Chat application.

      llm({{ prompt }})
      '''

    chat = Chat(
        llm=Bard24B(),
        prompt=react_with_tools(
            preamble=(
                f'Please help me solve my problem using tools. '
                f'Current time is {{datetime.datetime.now()}}'),
            tools=my_tools))

    chat(user_input='Help me get a lunch to go, veggie please.')
    ```

  Fourthly, LangFunc is built on top of PyGlove symbolic programming power,
  it could be manipulated programmatically, turned into a space for data
  sampling, or even tuned by AutoML. For example:

    ```
    import pyglove as pg

    prompt_space = react_with_tools(
        preamble=pg.oneof([
            'Help me solve my problem using the following tools:',
            'Help me with the tools below:',
            ...
        ])
        # Choose any two of the tools for generating data.
        tools=pg.manyof(2, [
            google_search(...),
            doordash(...),
            ...
        ])

    for prompt in pg.random_sample(prompt_space):
      print(prompt(user_input='Help me book a conf room please.'))

    ```

  For more capabilities on symbolic programming with PyGlove, please checkout
  https://pyglove.readthedocs.io/en/latest/.

  Final note: always include these capitalized words if you don't want to treat
  the docstr as the template str: THIS IS NOT A TEMPLATE. So as a result, this
  docstr is not treated as a template str :).
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

  returns: Annotated[
      Any,
      (
          'PyGlove extended annotation for the structured return value of this '
          'LangFunc, which can be accessed via the `result` property of the '
          'returned message upon invocation. If None, no structured output '
          'will be returned. Here are a few applicable values: '
          'int, str, str | None, list[str], {"x": int, "y": str}, Foo (a '
          '`pg.Object` subclass). '
          '(This is an experimental feature and is not stable) '
      )
  ] = None

  lm: Annotated[
      language_model.LanguageModel,
      (
          'The language model used to sample the responses based on the '
          'rendered template. It could either be provided at `__init__` time '
          'or accessed at `__call__` time from its context.'
      ),
  ] = component.contextual()

  input_transform: Annotated[
      message_transform.MessageTransform | None,
      (
          'External input transform, which intercepts LM input before calling '
          'the internal `transform_input` method. It is designed to apply '
          'extra structures to the LM input (e.g. COT).'
          'We set the default value to None as we do not want the child '
          "LangFun to use the parent's transform accidentally."
      ),
  ] = None

  output_transform: Annotated[
      message_transform.MessageTransform | None,
      (
          'Extenral output transform, which intercepts LM response before '
          'calling the internal `transform_output` method. It is designed to '
          'clean up LM response before structured parsing. We set the default '
          'value to None as we do not want the child LangFun to use the '
          "parent's transform accidentally."
      ),
  ] = None

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
      template_str = cls._extract_template_from_docstr()
      if template_str:
        # Set the `template_str` attribute will change the default value
        # of the `template_str` symbolic field.
        setattr(cls, 'template_str', template_str)

    if template_str:
      # Declare template variables as symbolic attributes.
      template_vars = _Template.get_template_variables(template_str)
      for var_name in template_vars:
        var_attr = getattr(cls, var_name, pg.MISSING_VALUE)
        if var_attr == pg.MISSING_VALUE:
          setattr(cls, var_name, component.contextual())

    super().__init_subclass__()

  @classmethod
  def _extract_template_from_docstr(cls) -> str:
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

  def _on_bound(self):
    super()._on_bound()

    # Set internal output transform based on return schema.
    internal_output_transform = None
    if self.returns is not None:
      internal_output_transform = message_transform.Identity().as_structured(
          self.returns
      )
    self._internal_output_transform = internal_output_transform

    if self.clean:
      self.rebind(
          template_str=inspect.cleandoc(self.template_str),
          skip_notification=True,
      )
    self._template = _Template(self.template_str)

    # Use contextual value for unassigned attributes.
    unassigned_vars = {}
    for k in self._template.variables:
      if not hasattr(self, k):
        unassigned_vars[k] = component.contextual()
    if unassigned_vars:
      self.rebind(unassigned_vars, skip_notification=True)

    # Cached results.
    self._cached_lm_input = None
    self._cached_lm_output = None

  @property
  def lm_input(self) -> message_lib.Message | None:
    """Returns the cached LM input from last __call__."""
    return self._cached_lm_input

  @property
  def lm_output(self) -> message_lib.Message | None:
    """Returns the cached LM output from last __call__."""
    return self._cached_lm_output

  @property
  def missing_vars(self) -> Set[str]:
    """Returns the missing variable names."""
    return self.vars(closure=True, specified=False)

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
    for k in self._template.variables:
      v = self.get(k, pg.MISSING_VALUE)

      match specified:
        case None:
          include = True
        case True:
          include = v != pg.MISSING_VALUE
        case _:
          include = v == pg.MISSING_VALUE

      if include and leaf is not None:
        include = leaf != isinstance(v, LangFunc)

      if include:
        variables.add(k)

      if closure and isinstance(v, LangFunc):
        variables.update(v.vars(closure=True, specified=specified, leaf=leaf))
    return variables

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

  def get(self, var_name: str, default: Any = pg.MISSING_VALUE) -> Any:
    return getattr(self, var_name, default)

  def __call__(
      self,
      *,
      lm: language_model.LanguageModel | None = None,
      lm_input: message_lib.Message | None = None,
      skip_input_transform: bool = False,
      skip_lm: bool = False,
      skip_output_transform: bool = False,
      **variables) -> message_lib.Message:
    """Calls language model with `lm_input` or rendered text.

    Args:
      lm: Language model to use. When present, it takes precedence over 
        the value from the `lm` attribute as well as from the containing chain.
      lm_input: An optional `lf.Message` object as the `lm_input`. If present,
        it will be directly used to call the LM. Otherwise, `render()` will be
        called to produce an LM input first.
      skip_input_transform: If True, the input transform will be skipped.
      skip_lm: If True, skipping LM. In such case, the input message will be
        returned.
      skip_output_transform: If True, the output transform will be skipped.
      **variables: Template variables applicable to this or child LangFunc.

    Returns:
      An output message from LM.
    """
    return self._call_once(
        lm=lm,
        lm_input=lm_input,
        skip_input_transform=skip_input_transform,
        skip_lm=skip_lm,
        skip_output_transform=skip_output_transform,
        **variables,
    )

  def _call_once(
      self,
      *,
      lm: language_model.LanguageModel | None = None,
      lm_input: message_lib.Message | None = None,
      skip_input_transform: bool = False,
      skip_lm: bool = False,
      skip_output_transform: bool = False,
      **variables,
  ) -> message_lib.Message:
    """Call the language model once, with invoking the output transform."""
    try:
      pg.object_utils.thread_local_push(_TLS_LFUN_CALL_STACK, self)

      kwargs = dict(variables)
      if lm is not None:
        kwargs['lm'] = lm

      with self.override(**kwargs):
        # Render the LM input text and creates a user message.
        if lm_input is None:
          lm_input = self.render(
              skip_input_transform=skip_input_transform, **kwargs)
        else:
          self._cached_lm_input = lm_input

        if not skip_lm:
          # Send rendered text to LM.
          lm_input.tag(message_lib.Message.TAG_LM_INPUT)
          lm_output = self.lm(lm_input)

          # Track the input as the source of the output.
          lm_output.source = lm_input
          lm_output.tag(message_lib.Message.TAG_LM_RESPONSE)

          # Transform the output message if applicable.
          if not skip_output_transform:

            # Call the external output transform first to clean up LM response.
            if self.output_transform is not None:
              lm_output = self.output_transform.transform(lm_output)

            lm_output = self.transform_output(lm_output)

          lm_output.tag(message_lib.Message.TAG_LM_OUTPUT)

          # We cache the transformed output instead of the original one
          # since the old one is tracked with `sym_origin`.
          self._cached_lm_output = lm_output
        else:
          lm_output = lm_input
          self._cached_lm_output = None

      # Emit LangFuncCallEvent.
      lm_callstack = list(
          pg.object_utils.thread_local_get(_TLS_LFUN_CALL_STACK))
      lm_callstack.pop()
      subscription.emit(
          LangFuncCallEvent(
              sender=self,
              lm_input=lm_input,
              lm_output=lm_output,
              lm_callstack=lm_callstack,
          )
      )
      return lm_output
    finally:
      top = pg.object_utils.thread_local_pop(_TLS_LFUN_CALL_STACK, self)
      assert top is self, (top, self)

  def render(
      self,
      *,
      allow_partial: bool = False,
      implicit: bool = False,
      skip_input_transform: bool = False,
      message_cls: Type[message_lib.Message] = message_lib.UserMessage,
      **kwargs
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
      skip_input_transform: If True, the input transform will be skipped.
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
            rendered_text, rendered_vars = self._template.render(
                self.get, allow_partial=allow_partial)

        if self.clean:
          rendered_text = rendered_text.strip()

        lm_input = message_cls(text=rendered_text, metadata={
            # NOTE(daiyip): Make a reference to the rendered variables
            # so the nested LangFunc could still access their contextual
            # variables. This also minimize copying so it's fast.
            k: pg.Ref(v) for k, v in rendered_vars.items()
        })

        # Tag input as rendered message.
        lm_input.tag(message_lib.Message.TAG_RENDERED)

        # Transform the input message if applicable.
        if not skip_input_transform:
          # Call the external input transform first.
          lm_input_transformed = lm_input
          if self.input_transform is not None:
            lm_input_transformed = self.input_transform.transform(
                lm_input_transformed)
          lm_input_transformed = self.transform_input(lm_input_transformed)

          if (lm_input_transformed is lm_input
              and isinstance(lm_input_transformed.result, str)):
            lm_input_transformed = lm_input.clone(
                override={
                    'text': lm_input.result,
                    'tags': [],
                    'metadata.result': pg.MISSING_VALUE
                })
            lm_input_transformed.source = lm_input
            lm_input_transformed.tag(message_lib.Message.TAG_TRANSFORMED)

          lm_input = lm_input_transformed

        # Adding result to cache
        cache[id(self)] = lm_input

        # We cache the transformed input instead of the original one
        # since the old one is tracked with `sym_origin`.
        self._cached_lm_input = lm_input

        # Emit LangFuncRenderEvent.
        # Event handlers could access the cached result from nested templates
        # using `str()`.
        render_stack = list(pg.object_utils.thread_local_get(_TLS_RENDER_STACK))
        render_stack.pop()
        subscription.emit(
            LangFuncRenderEvent(
                sender=self, lm_input=lm_input, render_stack=render_stack
            )
        )

      return lm_input
    finally:
      top = pg.object_utils.thread_local_pop(_TLS_RENDER_STACK)
      assert top is self, (top, self)

  #
  # Internal input and output transforms.
  # Subclasses can override.
  #

  def transform_input(
      self, lm_input: message_lib.Message) -> message_lib.Message:
    """Transforms the input message before sending to LM."""
    return lm_input

  def transform_output(
      self, lm_output: message_lib.Message) -> message_lib.Message:
    """Transforms the output message before returning from __call__."""
    if self._internal_output_transform is not None:
      transform_output = self._internal_output_transform.transform(lm_output)
      lm_output.result = transform_output.result
    return lm_output

  #
  # Override MessageTransform methods.
  #

  def _transform_path(
      self,
      message: message_lib.Message,
      input_path: str,
      value: Any
      ) -> message_lib.Message:
    """Implements MessageTransform._transform_path."""
    if input_path in (
        message_lib.Message.PATH_TEXT, message_lib.Message.PATH_ROOT):
      input_message = message
    else:
      if isinstance(value, message_lib.Message):
        message.set(input_path, pg.MISSING_VALUE)
        input_message = value
      elif isinstance(value, str):
        input_message = message.clone(override={
            'text': value,
            'tags': [message_lib.Message.TAG_TRANSFORMED],
            f'metadata.{input_path}': pg.MISSING_VALUE
        })
      else:
        raise TypeError(
            f'Metadata {repr(input_path)} should be a string or '
            f'a `lf.Message`. Encountered: {value!r}'
        )
      input_message.source = message

    # For LangFunc that are used as transforms, its template could access the
    # input via 'message'.
    output_message = self(message=input_message)

    # Trace back the source for the root.
    output_message.root.source = input_message
    return output_message

  def __rshift__(self, x):
    """Override >> to chain output transform and return self."""
    self.rebind(
        output_transform=(
            self.output_transform >> message_transform.make_transform(x)),
        skip_notification=True
    )
    return self

  #
  # Implements NaturalLanguageFormattable
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

  def __repr__(self) -> str:
    exclude_keys = []
    if self.input_path is None:
      exclude_keys.append('input_path')
    if self.output_path is None:
      exclude_keys.append('output_path')
    return self.format(compact=True, exclude_keys=exclude_keys)


# Register converter from str to LangFunc, therefore we can always
# pass strs to attributes that accept LangFunc.
pg.typing.register_converter(str, LangFunc, LangFunc)


#
# LangFunc related events.
#


class LangFuncEvent(subscription.Event[LangFunc]):
  """Base class for LangFunc events."""


@dataclasses.dataclass
class LangFuncRenderEvent(LangFuncEvent):
  """LangFunc rendering event."""

  lm_input: message_lib.Message
  render_stack: list[LangFunc]


@dataclasses.dataclass
class LangFuncCallEvent(LangFuncEvent):
  """LangFunc call event."""

  lm_input: message_lib.Message
  lm_output: message_lib.Message
  lm_callstack: list[LangFunc]


class _Template(pg.Object):
  """(Mutable) String template."""

  template_str: Annotated[str, 'Jinja2 template string.']

  def _on_bound(self) -> None:
    super()._on_bound()

    # Invalidate cached properties.
    self._variables = None
    self._template = None

  @property
  def variables(self) -> Set[str]:
    """Returns all declared variables from the template."""
    if self._variables is None:
      self._variables = self.get_template_variables(self.template_str)
    return self._variables

  @classmethod
  def get_template_variables(cls, template_str: str) -> Set[str]:
    env = jinja2.Environment()
    ast = env.parse(template_str)
    return jinja2_meta.find_undeclared_variables(ast)

  @property
  def template(self) -> jinja2.Template:
    if self._template is None:
      self._template = jinja2.Template(self.template_str)
    return self._template

  def render(
      self,
      variables: dict[str, Any] or Callable[[str], Any],
      allow_partial: bool = False) -> tuple[str, dict[str, Any]]:
    """Renders the template with variable getter function."""
    if callable(variables):
      inputs = dict()
      for var_name in self.variables:
        var_value = variables(var_name)
        if var_value == pg.MISSING_VALUE:
          if allow_partial:
            var_value = _UnresolvedExpression(var_name)
          else:
            raise ValueError(
                f'The value for template variable {var_name!r} is not provided.'
            )
        inputs[var_name] = var_value
    elif isinstance(variables, dict):
      inputs = variables
      for var_name in self.variables:
        if var_name not in inputs:
          if allow_partial:
            inputs[var_name] = _UnresolvedExpression(var_name)
          else:
            raise ValueError(
                f'The value for template variable {var_name!r} is not provided.'
            )
    else:
      raise ValueError(
          '`variables` should be either a dict or a callable object. '
          f'Encountered: {variables!r}'
      )
    # Natural language formattable objects will be returned in natural language
    # when they are directly returned as rendering elements in the template.
    return self.template.render(**inputs), inputs


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
