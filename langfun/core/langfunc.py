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
class LangFunc(template_lib.Template):
  r"""Base class for Language-based functions.

  LangFunc represents a function powered by a language model. It is a subclass
  of `lf.Template` and can be thought of as a `lf.Template` augmented with an LM
  and an output transformation. Calling a `lf.LangFunc` is equivalent to calling
  the LM with the rendered prompt and transforming the output.

  LangFunc can be directly constructed and used.

  ```python
  import langfun as lf

  func = lf.LangFunc("Hello, {{name}}!")
  print(func(name="Gemini", lm=lf.llms.Gemini25Flash()))
  # Output: Hello, how are you today?
  ```

  Or it can be subclassed:

  ```python
  import langfun as lf

  class Compute(lf.LangFunc):
    '''Compute a simple arithmetic expression.

    {{expression}} = ?
    '''
    expression: str

    def transform_output(self, lm_output: lf.Message) -> lf.Message:
      lm_output.metadata.result = float(lm_output.text)
      return lm_output

  r = Compute(expression="1 + 1")(lm=lf.llms.Gemini25Flash())
  print(r.result)
  # Output: 2.0

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
    """Renders the template and transforms it as LM input message.

    Args:
      allow_partial: If True, allows partial rendering, which leaves unresolved
        variables in place in the output text. Otherwise, raises error when
        there are unresolved variables.
      implicit: If True, reuse the rendering output if a parent `lf.Template`
        is rendering current `lf.Template` multiple times. This is important
        for making sure all references to the same `lf.Template` within a single
        top-level rendering would return the same result. If False, every call
        to `render` will trigger the actual rendering process.
      message_cls: The message class used for creating the return value.
      **kwargs: Values for template variables, which override values from
        member attributes or context.

    Returns:
      A Message object containing the rendered result.
    """
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
