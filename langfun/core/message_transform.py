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
"""Langfun transform base."""

import abc
import json
import re
from typing import Annotated, Any, Callable, Type, Union
from langfun.core import component
from langfun.core import message as message_lib
import pyglove as pg


RAISE_IF_HAS_ERROR = (pg.MISSING_VALUE,)


class MessageTransform(component.Component):
  """Base class for message transform."""

  input_path: Annotated[
      str | None,
      (
          'The path for the input value which will be passed to '
          '`lf.Message.get`, e.g.: `text` or `result.a[0]`. '
          'If None, the input path will be set automatically based on the '
          'prior or parent transforms.'
      ),
  ] = None

  output_path: Annotated[
      str | None,
      (
          'The path for writing the output value to the message. '
          'If empty and the output value is a message, the output message will '
          'replace the input message. By default it writes to `result`.'
      ),
  ] = None

  def _on_bound(self):
    super()._on_bound()

    # Set the input path.
    self._input_search_paths = (
        (message_lib.Message.PATH_RESULT, message_lib.Message.PATH_ROOT)
        if self.input_path is None else (self.input_path,)
    )
    self._output_path = (
        message_lib.Message.PATH_RESULT
        if self.output_path is None else self.output_path
    )

  def _input_path_and_value(self, message: message_lib.Message) -> Any:
    for input_path in self._input_search_paths:
      v = message.get(input_path, pg.MISSING_VALUE)
      if v != pg.MISSING_VALUE:
        return input_path, v
    raise KeyError(
        f'Input path {repr(self._input_search_paths[0])} does not exist in the '
        f'message. Message: {repr(message)}, Transform: {repr(self)}.'
    )

  def transform(self, message: message_lib.Message) -> message_lib.Message:
    output_message = self._transform(message)
    output_message.tag(message_lib.Message.TAG_TRANSFORMED)

    if output_message.root is not message.root:
      output_message.root.source = message
    return output_message

  def _transform(self, message: message_lib.Message) -> message_lib.Message:
    """Transforms an input message to an output message."""
    input_path, input_value = self._input_path_and_value(message)
    output = self._transform_path(message, input_path, input_value)

    if isinstance(output, message_lib.Message):
      return output
    else:
      message.set(self._output_path, output)
    return message

  @abc.abstractmethod
  def _transform_path(self,
                      message: message_lib.Message,
                      input_path: str,
                      value: Any
                      ) -> Any:
    """Transforms the value at the location specified by `input_path`."""

  def _chain(self, x, cls, **kwargs):
    """Chains an operation with self."""
    if x is None:
      return self

    x = make_transform(x)

    # Chain the transforms.
    if x.input_path is None and self.output_path:
      x.rebind(input_path=self.output_path)

    def child_transforms(t):
      return t.transforms if isinstance(t, cls) else [t]

    return cls(child_transforms(self) + child_transforms(x), **kwargs)

  def __rshift__(self, x):
    """The sequential operator (>>)."""
    return self._chain(x, Sequential)

  def __rrshift__(self, x):
    """The right-hand sequential operator (>>)."""
    if x is None:
      return self
    return make_transform(x) >> self

  def __or__(self, x):
    """The logical or operator (|)."""
    return self._chain(x, LogicalOr)

  def __ror__(self, x):
    if x is None:
      return self
    return make_transform(x) | self

  def as_structured(
      self,
      annotation: Any,
      default: Any = RAISE_IF_HAS_ERROR,
      examples: list[Any] | None = None,
      **kwargs,
  ) -> Any:
    """Returns a structured representation of message text."""
    del annotation, default, examples, kwargs
    assert (
        False
    ), 'This method will be overridden in transforms/parse_structured.py'

  def as_text(self) -> 'MessageTransform':
    return self >> SaveAs('text')

  def as_metadata(self, key: str) -> 'MessageTransform':
    return self >> SaveAs(key)

  def retry(
      self,
      max_attempts: int,
      errors_to_retry: (
          Type[Exception] | tuple[Type[Exception], ...]
      ) = Exception,
  ):
    """Retry current transform multiple times when it fails."""
    return Retry(
        self, max_attempts=max_attempts, errors_to_retry=errors_to_retry
    )

  def to_bool(self, default: Any = RAISE_IF_HAS_ERROR):
    """Converts the output to int."""
    return self >> ParseBool(default)

  def to_int(self, default: Any = RAISE_IF_HAS_ERROR):
    """Converts the output to int."""
    return self >> ParseInt(default)

  def to_float(self, default: Any = RAISE_IF_HAS_ERROR):
    """Converts the output to int."""
    return self >> ParseFloat(default)

  def to_dict(self, default: Any = RAISE_IF_HAS_ERROR):
    """Parses the output of current transform as a dict in JSON format."""
    return self >> ParseJson(default)

  def match(self,
            pattern: str,
            multiple: bool = False,
            default: Any = RAISE_IF_HAS_ERROR):
    """Extract matched sub-strings using regular expression."""
    return self >> Match(pattern, multiple=multiple, default=default)

  def match_block(
      self,
      prefix: str,
      suffix: str,
      inclusive: bool = False,
      multiple: bool = False,
      default: Any = RAISE_IF_HAS_ERROR,
  ):
    """Extract text blocks based on prefix and suffix."""
    return self >> MatchBlock(
        prefix, suffix, inclusive=inclusive, multiple=multiple, default=default
    )


#
# Basic transforms.
#


@pg.use_init_args(['value_transform', 'input_path', 'output_path'])
class Lambda(MessageTransform):
  """Message transform from an lambda function."""

  # Set input_path and output_path to empty strings to transform the
  # entire message by default.

  value_transform: Annotated[
      Callable[[Any], Any],
      (
          'A callable object that transforms the input value (identified by '
          '`input_path`) from the message. If the transform returns an '
          '`lf.Message` object and the `output_path` is empty, the entire '
          'message will be replaced.'
      ),
  ]

  def _transform_path(
      self, message: message_lib.Message, input_path: str, value: Any) -> Any:
    del message, input_path
    return self.value_transform(value)


# Register automatic conversion from function to MessageTransform for
# symbolic attribute assignment.
pg.typing.register_converter(type(lambda: 0), MessageTransform, Lambda)


_TYPE_BASED_TRANSFORMS = {
    int: lambda x: ParseInt(),
    float: lambda x: ParseFloat(),
    bool: lambda x: ParseBool(),
}


def make_transform(exp: Any) -> MessageTransform:
  """Maps an expression to a transform.

  Args:
    exp: The expression used for creating a transform. Currently the following
      expressions are supported:
        A callable object - based on which a `Lambda` transform will be created.
        A type (int, float, bool) - which maps to ParseInt/ParseFloat/ParseBool.
        A `re.Pattern` object - which maps to Match.

  Returns:
    A transform object that corresponds to the expression.

  Raises:
    TypeError: if the expression as a transform is not supported.
  """
  if isinstance(exp, MessageTransform):
    return exp

  # Map supported types to conversions.
  if exp in _TYPE_BASED_TRANSFORMS:
    return _TYPE_BASED_TRANSFORMS[exp](exp)

  # Create match from regular expression pattern.
  if isinstance(exp, re.Pattern):
    return Match(exp.pattern)

  # Make lambda transform from callable object.
  if callable(exp):
    return Lambda(exp)
  raise ValueError(f'Unsupported expression for `lf.MessageTransform`: {exp}')


@pg.use_init_args(['copy', 'deep', 'input_path', 'output_path'])
class Identity(MessageTransform):
  """Identity transform."""

  copy: Annotated[bool, 'If True, copy the input message.'] = False

  deep: Annotated[
      bool,
      (
          'If True, deep copy the input message. Applicable when `copy` is set '
          'to True.'
      ),
  ] = False

  def _transform_path(
      self, message: message_lib.Message, input_path: str, value: Any) -> Any:
    del message, input_path
    if self.copy:
      value = pg.clone(value, deep=self.deep)
    return value


@pg.use_init_args(['output_path', 'input_path', 'remove_input'])
class SaveAs(MessageTransform):
  """Save the input value as a message field."""

  output_path = pg.MISSING_VALUE  # Make output_path required again.
  remove_input: bool = True

  def transform(self, message: message_lib.Message) -> message_lib.Message:
    if self.input_path == self.output_path:
      return message

    k, v = self._input_path_and_value(message)

    output_path = self.output_path
    if output_path and output_path != 'text':
      output_path = f'metadata.{output_path}'

    updates = {output_path: v}
    if self.remove_input and k and k != 'text':
      updates[f'metadata.{k}'] = pg.MISSING_VALUE

    message.rebind(updates, raise_on_no_change=False)
    return message

  def _transform_path(
      self, message: message_lib.Message, input_path: str, value: Any) -> Any:
    assert False, 'Not needed.'


#
# Compositional transforms.
#


class Compositional(MessageTransform):
  """Base class for compositional transforms."""

  def _on_bound(self):
    super()._on_bound()
    self._update_input_output_paths()

  def _transform_path(self, input_path: str, value: Any) -> Any:
    assert False, 'Not called.'

  @abc.abstractmethod
  def _update_input_output_paths(self):
    """update input and output paths for the composition."""


@pg.use_init_args(['transforms'])
class MultiTransformComposition(Compositional):
  """Base class for compositions that have multiple child transforms."""

  transforms: Annotated[list[MessageTransform], 'Child transforms.'] = []

  def __getitem__(self, index: int | slice):
    return self.transforms[index]


#
# Multi-transform compositions.
#


class Sequential(MultiTransformComposition):
  """Sequentially performing multiple transforms."""

  def _update_input_output_paths(self):
    # NOTE(daiyip): Assume a transform is defined as `t(input_key, output_key)`,
    # A sequence is defined as `s(input_key, output_key)[child_transforms]`
    #
    # Case 1:
    # INPUT:  s(None, None)[x(None, None), y(None, None), z(None, None)]
    # OUTPUT: s(None, None)[x(None, None), y(None, None), z(None, None)]
    #
    # Case 2:
    # INPUT:  s(i1, None)[x(None, None), y(None, None), z(None, o1)]
    # OUTPUT: s(i1, o1)[x(i1, o1), y(o1, o1), z(o1, o1)]
    #
    # Case 3:
    # INPUT:  s(None, o1)[x(i1, None),  y(None, None),  z(None, None)]
    # OUTPUT: s(i1, o1)[x(i1, o1), y(o1, o1), z(o1, o1)]
    #
    # Case 4:
    # INPUT:  s(None, None)[x(i1, o1) >> y(None, None) >> z(None, o2)
    # OUTPUT: s(i1, o2)[x(i1, o1) >> y(o1, o2) >> z(o2, o2)]
    # Meaning that y's output will be overridden by z.

    # Deduce the output paths for child transforms in a reversed order.
    succeeding_output_path = self.output_path

    for t in reversed(self.transforms):
      if t.output_path is None and succeeding_output_path is not None:
        t.rebind(output_path=succeeding_output_path, notify_parents=False)
      succeeding_output_path = t.output_path

    # Use the last transform's output path if the sequence output path is None.
    if (
        self.output_path is None
        and self.transforms
        and self.transforms[-1].output_path is not None
    ):
      self.rebind(
          output_path=self.transforms[-1].output_path, skip_notification=True
      )

    # Deduece the input paths for child transforms.
    prior_output_path = self.input_path
    for t in self.transforms:
      if t.input_path is None and prior_output_path is not None:
        t.rebind(
            input_path=prior_output_path,
            raise_on_no_change=False,
            notify_parents=False,
        )
      prior_output_path = t.output_path

    if (
        self.transforms
        and self.transforms[0].input_path is not None
        and self.input_path != self.transforms[0].input_path
    ):
      self.rebind(
          input_path=self.transforms[0].input_path, skip_notification=True
      )

  def transform(self, message: message_lib.Message) -> message_lib.Message:
    for t in self.transforms:
      message = t.transform(message)
    return message


class LogicalOr(MultiTransformComposition):
  """Try transforms one by one until one succeeds."""

  def _update_input_output_paths(self):
    # NOTE(daiyip): Assume a transform is defined as `t(input_key, output_key)`,
    # A logical-or op is defined as `or(input_key, output_key)[transforms]`
    #
    # Case 1:
    # INPUT:  or(None, None)[x(None, None), y(None, None), z(None, None)]
    # OUTPUT: or(None, None)[x(None, None), y(None, None), z(None, None)]
    #
    # Case 2:
    # INPUT:  or(i1, None)[x(None, None), y(None, None), z(None, None)]
    # OUTPUT: or(i1, None)[x(i1, None), y(i1, None), z(i1, None)]
    #
    # Case 3:
    # INPUT:  or(None, o1)[x(None, None), y(None, None), z(None, None)]
    # OUTPUT: or(None, o1)[x(None, o1), y(None, o1), z(None, o1)]
    #
    # Case 4:
    # Input:  or(i1, None)[x(None, None), y(i2, None), z(i3, None)]
    # Input:  or(i1, None)[x(None, None), y(i2, None), z(i3, None)]
    #
    # Case 5: Invalid - child transforms should have no output path.
    # Input:  or(i1, None)[x(None, None), y(i2, None), z(i3, None)]

    input_path = self.input_path
    output_path = self.output_path
    child_output_paths = set()

    for t in self.transforms:
      if input_path is not None and t.input_path is None:
        t.rebind(input_path=input_path, notify_parents=False)
      if output_path is not None and t.output_path is None:
        t.rebind(output_path=output_path, notify_parents=False)
      child_output_paths.add(t.output_path)
    if len(child_output_paths) > 1:
      raise ValueError(
          'The branches of `LogicalOr` should have the same output path. '
          f'Encountered: {repr(self)}.'
      )

  def transform(self, message: message_lib.Message) -> message_lib.Message:
    for t in self.transforms:
      try:
        return t.transform(message)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    raise ValueError('None of the child transforms has run successfully.')


#
# Single-transform compositions.
#


@pg.use_init_args(['child_transform'])
class SingleTransformComposition(Compositional):
  """Base class for composition that takes a single child transform."""

  child_transform: Annotated[MessageTransform, 'Transform to repeat.']


@pg.use_init_args(['child_transform', 'max_attempts', 'errors_to_retry'])
class Retry(SingleTransformComposition):
  """Retry a transform multiple times when it fails."""

  max_attempts: Annotated[int, 'Max attempts if transform fails.'] = 5

  errors_to_retry: Annotated[
      Type[Exception] | tuple[Type[Exception], ...], 'Errors to retry.'
  ] = Exception

  def _update_input_output_paths(self):
    # Exchange the input_path between `self` and child transform.
    if self.child_transform.input_path is not None:
      self.rebind(
          input_path=self.child_transform.input_path, skip_notification=True
      )
    elif self.input_path is not None:
      self.child_transform.rebind(
          input_path=self.input_path, notify_parents=False
      )

    # Exchange the output_path between `self` and child transform.
    if self.child_transform.output_path is not None:
      self.rebind(
          output_path=self.child_transform.output_path, skip_notification=True
      )
    elif self.output_path is not None:
      self.child_transform.rebind(
          output_path=self.output_path, notify_parents=False
      )

  def transform(self, message: message_lib.Message) -> message_lib.Message:
    attempts = 0
    while True:
      try:
        return self.child_transform.transform(message)
      except self.errors_to_retry as e:  # pylint: disable=broad-exception-caught
        attempts += 1
        if attempts < self.max_attempts:
          continue

        raise ValueError(
            f'{self.child_transform!r} failed after '
            f'{self.max_attempts} attempts. '
        ) from e


#
# Common parsers.
#


@pg.use_init_args(['default'])
class Parser(MessageTransform):
  """Base clasas for conversion-type transform."""

  default: Annotated[
      Any,
      (
          'The default value to use if parsing failed. '
          'If unspecified, error will be raisen.'
      )
  ] = RAISE_IF_HAS_ERROR

  def _transform_path(
      self, message: message_lib.Message, input_path: str, value: Any) -> Any:
    del message
    if isinstance(value, message_lib.Message):
      value = value.text

    if not isinstance(value, str):
      raise TypeError(
          f'Metadata {input_path!r} must be a string. Encountered: {value}.')
    try:
      return self.parse(value)
    except Exception as e:  # pylint: disable=broad-exception-caught
      if self.default == RAISE_IF_HAS_ERROR:
        raise e
      return self.default

  @abc.abstractmethod
  def parse(self, text: str) -> Any:
    """Parse a string value."""


class ParseInt(Parser):
  """Int parser."""

  def parse(self, text: str) -> Any:
    return int(text)


class ParseFloat(Parser):
  """Float parser."""

  def parse(self, text: str) -> Any:
    return float(text)


class ParseBool(Parser):
  """Boolean parser."""

  def parse(self, text: str) -> Any:
    lc = text.lower()
    if lc in ['true', 'yes', '1']:
      return True
    elif lc in ['false', 'no', '0']:
      return False
    else:
      raise ValueError(f'Cannot convert {text!r} to bool.')


class ParseJson(Parser):
  """Json parser."""

  def parse(self, text: str) -> Any:
    return json.loads(text)


@pg.use_init_args(['pattern', 'multiple', 'default'])
class Match(Parser):
  r"""Regular expression based information parseor.

  Examples:

    ```python
    t = lf.transforms.Match('\((\d+),\s*(\d+)\)')
    t('(1, 2)')
    >> ('1, '2')

    t = lf.transforms.Match('\((?P<x>\d+),\s*(?P<y>\d+)\)')
    t('(1, 2)')
    >> {'x': '1', 'y': '2'}

    t = lf.transforms.Match('\((\d+),\s*(\d+)\)', multiple=True)
    t('(1, 2) (3, 4)')
    >> [('1', '2'), ('3', '4')]

    t = lf.transforms.Match('\((?P<x>\d+),\s*(?P<y>\d+)\)', multiple=True)
    t('(1, 2) (3, 4)')
    >> [{'x': '1', 'y': '2'}, {'x': '3', 'y': '4'}]
    ```
  """

  pattern: Annotated[
      str, 'Regular expression for the pattern to match and parse.'
  ]

  multiple: Annotated[
      bool,
      (
          'If True, it will return all matched non-overlapping instances as a '
          'list. If False, it will return the first matched instance.'
      ),
  ] = False

  def _on_bound(self):
    super()._on_bound()
    # Allow dot (.) to match line break (\n).
    self._pattern = re.compile(self.pattern, re.DOTALL)

  def parse(
      self, text: str
  ) -> Union[
      None,
      list[Union[tuple[str, ...], dict[str, str]]],
      tuple[str, ...],
      dict[str, str],
  ]:
    def value_from_match(m) -> dict[str, str] | tuple[str, ...]:
      result_dict = m.groupdict()
      if result_dict:
        return result_dict
      groups = m.groups()
      if not groups:
        start, end = m.span()
        return m.string[start:end]
      elif len(groups) == 1:
        return groups[0]
      return groups

    if self.multiple:
      instances = []
      for m in self._pattern.finditer(text):
        instances.append(value_from_match(m))
      if not instances:
        raise ValueError('No match found.')
      return instances
    else:
      m = self._pattern.search(text)
      if m:
        return value_from_match(m)
      else:
        raise ValueError('No match found.')


@pg.compound(Match)
def MatchBlock(  # pylint: disable=invalid-name
    prefix: str, suffix: str, inclusive: bool = False,
    multiple: bool = False, default: Any = RAISE_IF_HAS_ERROR,
):
  """Match a text block with a prefix and a suffix."""
  prefix = re.escape(prefix)
  suffix = re.escape(suffix)
  greedy_option = '?' if multiple else ''
  if inclusive:
    pattern = f'({prefix}.*{greedy_option}{suffix})'
  else:
    pattern = f'{prefix}(.*{greedy_option}){suffix}'
  return Match(pattern, multiple=multiple, default=default)
