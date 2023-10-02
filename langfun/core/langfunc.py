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
"""LangFunc: Language-based functions."""

import dataclasses
from typing import Annotated, Any, Type

from langfun.core import component
from langfun.core import language_model
from langfun.core import message as message_lib
from langfun.core import message_transform
from langfun.core import subscription
from langfun.core import template as template_lib
import pyglove as pg


# Keys for storing render/call stack (invoked LangFunc objects) in the
# thread-local storage.
_TLS_LFUN_CALL_STACK = '_langfunc_callstack'


# NOTE(daiyip): Only the template string belongs to the positional arguments,
# all others are keyword-only for clarity.
@pg.use_init_args(['template_str'])
class LangFunc(
    template_lib.Template,
    # LangFunc is also a component that transforms an message to another
    # message, so we can chain it.
    message_transform.MessageTransform,
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

  def _on_bound(self):
    super()._on_bound()

    # Set internal output transform based on return schema.
    internal_output_transform = None
    if self.returns is not None:
      internal_output_transform = message_transform.Identity().as_structured(
          self.returns
      )
    self._internal_output_transform = internal_output_transform

    # Last LM input and output.
    self._cached_lm_input = None
    self._cached_lm_output = None

  @property
  def lm_input(self) -> message_lib.Message | None:
    """Returns the cached LM input from the last invocation to `__call__`."""
    return self._cached_lm_input

  @property
  def lm_output(self) -> message_lib.Message | None:
    """Returns the cached LM output from the last invocation to `__call__`."""
    return self._cached_lm_output

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
    render_output = super().render(
        allow_partial=allow_partial,
        implicit=implicit,
        message_cls=message_cls,
        **kwargs,
    )

    # Transform the input message if applicable.
    if not skip_input_transform:
      # Call the external input transform first.
      render_transformed = render_output
      if self.input_transform is not None:
        render_transformed = self.input_transform.transform(render_transformed)
      render_transformed = self.transform_input(render_transformed)

      if render_transformed is render_output and isinstance(
          render_transformed.result, str
      ):
        render_transformed = render_output.clone(
            override={
                'text': render_output.result,
                'tags': [],
                'metadata.result': pg.MISSING_VALUE,
            }
        )
        render_transformed.source = render_output
        render_transformed.tag(message_lib.Message.TAG_TRANSFORMED)

      render_output = render_transformed
    return render_output

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

  def __repr__(self) -> str:
    exclude_keys = []
    if self.input_path is None:
      exclude_keys.append('input_path')
    if self.output_path is None:
      exclude_keys.append('output_path')
    return self.format(
        compact=True, use_inferred=True, exclude_keys=exclude_keys)


# Register converter from str to LangFunc, therefore we can always
# pass strs to attributes that accept LangFunc.
pg.typing.register_converter(str, LangFunc, LangFunc)


#
# LangFunc events.
#


@dataclasses.dataclass
class LangFuncCallEvent(subscription.Event[LangFunc]):
  """LangFunc call event."""

  lm_input: message_lib.Message
  lm_output: message_lib.Message
  lm_callstack: list[LangFunc]


def call(
    prompt: str | template_lib.Template,
    returns: Any = None, **kwargs
    ) -> Any:
  """Call a language model with prompt and formulate response in return type.

  Examples::

    # Call with constant string-type prompt.
    lf.call('Compute one plus one', lm=lf.llms.Gpt35())
    >> "two"

    # Call with returning a structured (int) type.
    lf.call('Compute one plus one', int, lm=lf.llms.Gpt35())
    >> 2

    # Call with a template string with variables.
    lf.call('Compute {{x}} plus {{y}}', int,
            x='one', y='one', lm=lf.llms.Gpt35())
    >> 2

    # Call with an `lf.Template` object with variables.
    lf.call(lf.Template('Compute {{x}} plus {{y}}', x=1), int,
            y=1, lm=lf.llms.Gpt35())
    >> 2

  Args:
    prompt: User prompt that will be sent to LM, which could be a string or a
      string template whose variables are provided from **kwargs.
    returns: Type annotations for return type. If None, the raw LM response will
      be returned (str). Otherwise, the response will be parsed based on the
      return type.
    **kwargs: Keyword arguments. Including options that control the calling
      behavior, such as `lm`, `temperature`, etc. As well as variables that will
      be fed to the prompt if it's a string template.

  Returns:
    A string if `returns` is None or an instance of the return type.
  """
  if isinstance(prompt, LangFunc):
    lfun = prompt.as_structured(returns)
  elif isinstance(prompt, template_lib.Template):
    lfun = LangFunc(prompt.render(**kwargs).text, returns=returns)
  elif isinstance(prompt, str):
    lfun = LangFunc(prompt, returns=returns)
  else:
    raise TypeError(
        '`prompt` should be a string or an `lf.Template` object. '
        f'Encountered {prompt!r}.')

  message = lfun(**kwargs)
  if returns is None:
    return message.text
  return message.result
