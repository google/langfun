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
from typing import Annotated, Type

from langfun.core import component
from langfun.core import language_model
from langfun.core import message as message_lib
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

  lm: Annotated[
      language_model.LanguageModel,
      (
          'The language model used to sample the responses based on the '
          'rendered template. It could either be provided at `__init__` time '
          'or accessed at `__call__` time from its context.'
      ),
  ] = component.contextual()

  def _on_bound(self):
    super()._on_bound()

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
      cache_seed: int | None = 0,
      skip_lm: bool = False,
      **variables,
  ) -> message_lib.Message:
    """Calls language model with `lm_input` or rendered text.

    Args:
      lm: Language model to use. When present, it takes precedence over the
        value from the `lm` attribute as well as from the containing chain.
      lm_input: An optional `lf.Message` object as the `lm_input`. If present,
        it will be directly used to call the LM. Otherwise, `render()` will be
        called to produce an LM input first.
      cache_seed: Seed for computing cache key. The cache key is determined by a
        tuple of (lm, prompt, cache seed). If None, cache will be disabled for
        the query even cache is configured by the LM.
      skip_lm: If True, returns the rendered prompt as a UserMessage object.
        otherwise return the LLM response based on the rendered prompt.
      **variables: Template variables applicable to this or child LangFunc.

    Returns:
      An output message from LM.
    """
    return self._call_once(
        lm=lm,
        lm_input=lm_input,
        cache_seed=cache_seed,
        skip_lm=skip_lm,
        **variables,
    )

  def _call_once(
      self,
      *,
      lm: language_model.LanguageModel | None = None,
      lm_input: message_lib.Message | None = None,
      cache_seed: int | None = 0,
      skip_lm: bool = False,
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
          lm_input = self.render(**kwargs)

        if skip_lm:
          return lm_input

        self._cached_lm_input = lm_input

        # Send rendered text to LM.
        lm_output = self.lm(lm_input, cache_seed=cache_seed)

        # Attach cache seed.
        lm_input.metadata.cache_seed = cache_seed

        # Transform the output message.
        lm_output = self.transform_output(lm_output)
        lm_output.tag(message_lib.Message.TAG_LM_OUTPUT)

        # We cache the transformed output instead of the original one
        # since the old one is tracked with `sym_origin`.
        self._cached_lm_output = lm_output

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
      message_cls: Type[message_lib.Message] = message_lib.UserMessage,
      **kwargs,
  ) -> message_lib.Message:
    lm_input = super().render(
        allow_partial=allow_partial,
        implicit=implicit,
        message_cls=message_cls,
        **kwargs,
    )

    with component.context(**kwargs):
      with self.override(**kwargs):
        return self.transform_input(lm_input)

  #
  # Input and output transforms.
  # Subclasses can override.
  #

  def transform_input(
      self, lm_input: message_lib.Message) -> message_lib.Message:
    """Transforms the input message before sending to LM."""
    return lm_input

  def transform_output(
      self, lm_output: message_lib.Message) -> message_lib.Message:
    """Transforms the output message before returning from __call__."""
    return lm_output


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
