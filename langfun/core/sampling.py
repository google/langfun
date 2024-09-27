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
"""Sampling LM input and responses."""

import functools
from typing import Any, Callable, Generator, Iterator, Tuple, Type, Union

from langfun.core import concurrent
from langfun.core import message as message_lib
from langfun.core.langfunc import LangFunc
import pyglove as pg


def sweep(
    lfun: LangFunc,
    num_examples: int | None = None,
    *,
    max_workers: int = 32,
    silence_on_errors: Union[
        Type[BaseException], Tuple[Type[BaseException], ...], None
    ] = None,
    ignore_examples_with_errors: bool = True,
    **kwargs,
) -> Iterator[
    Tuple[
        message_lib.Message | BaseException,              # LM input.
        Union[message_lib.Message, BaseException, None],  # LM output.
    ],
]:
  """Sweeps the input/output of this LangFunc concurrently.

  Args:
    lfun: An LangFunc object that contains `pg.oneof` as the search space 
      for sampling.
    num_examples: Number of examples to sample.
    max_workers: Max number of concurrent workers to do sampling.
    silence_on_errors: Return None for `lm_input` and `lm_output` when errors
      in this category happen. Otherwise error will be raised during sampling.
    ignore_examples_with_errors: If True, the examples with erratic lm_input
      or lm_output will not be included.
    **kwargs: Keyword arguments as rendering variables.

  Returns:
    An iterator of (lm_input, lm_output).
      `lm_input`/`lm_output` will be None when error happened the error
      matches the type of `silence_on_errors`.
  """
  return _concurrent_sample(
      lfun,
      pg.iter,
      num_examples=num_examples,
      max_workers=max_workers,
      silence_on_errors=silence_on_errors,
      ignore_examples_with_errors=ignore_examples_with_errors,
      **kwargs,
  )


def random_sample(
    lfun: LangFunc,
    num_examples: int | None = None,
    *,
    max_workers: int = 32,
    silence_on_errors: Union[
        Type[BaseException], Tuple[Type[BaseException], ...], None
    ] = None,
    ignore_examples_with_errors: bool = True,
    seed: int | None = None,
    **kwargs,
) -> Iterator[
    Tuple[
        message_lib.Message | BaseException,              # LM input.
        Union[message_lib.Message, BaseException, None],  # LM output.
    ],
]:
  """Random samples the input/output of this LangFunc concurrently.

  Args:
    lfun: An LangFunc object that contains `pg.oneof` as the search space 
      for sampling.
    num_examples: Number of examples to sample.
    max_workers: Max number of concurrent workers to do sampling.
    silence_on_errors: Return None for `lm_input` and `lm_output` when errors
      in this category happen. Otherwise error will be raised during sampling.
    ignore_examples_with_errors: If True, the examples with erratic lm_input
      or lm_output will not be included.
    seed: Random seed.
    **kwargs: Keyword arguments as rendering variables.

  Returns:
    An iterator of (lm_input, lm_output).
      `lm_input`/`lm_output` will be None when error happened the error
      matches the type of `silence_on_errors`.
  """
  return _concurrent_sample(
      lfun,
      functools.partial(pg.random_sample, seed=seed),
      num_examples=num_examples,
      max_workers=max_workers,
      silence_on_errors=silence_on_errors,
      ignore_examples_with_errors=ignore_examples_with_errors,
      **kwargs,
  )


def _concurrent_sample(
    lfun: LangFunc,
    pg_sample_fn: Callable[..., Iterator[Any]],
    num_examples: int | None = None,
    *,
    max_workers: int = 32,
    silence_on_errors: Union[
        Type[BaseException], Tuple[Type[BaseException], ...], None
    ] = None,
    ignore_examples_with_errors: bool = True,
    **kwargs,
) -> Generator[
    Tuple[
        message_lib.Message | BaseException,              # LM input.
        Union[message_lib.Message, BaseException, None],  # LM output.
    ],
    None,
    None,  # Sender type and return type.
]:
  """Concurrently sample the input/output of an LangFunc search space."""

  sampling_space = pg.Dict(
      lfun=lfun,
      kwargs=kwargs,
  )

  if pg.is_deterministic(sampling_space):
    num_examples = num_examples or None
    def repeat_example(example, num_examples=num_examples):
      for _ in range(num_examples):
        yield example.clone()
    pg_sample_fn = repeat_example

  def _error_of(func, *args, **kwargs):
    if silence_on_errors:
      try:
        func(*args, **kwargs)
      except silence_on_errors as e:
        return e
    else:
      func(*args, **kwargs)
    return None

  def _call_fn(example):
    lfun, kwargs = example.lfun, example.kwargs

    error = _error_of(lfun, **kwargs)
    lm_input = lfun.lm_input or error
    lm_output = lfun.lm_output or error
    return lm_input, lm_output

  # Concurrently sample.
  for _, result, error in concurrent.concurrent_map(
      _call_fn,
      pg_sample_fn(sampling_space, num_examples=num_examples),
      silence_on_errors=concurrent.RetryError,
      max_workers=max_workers
  ):
    if error is None:
      lm_input, lm_output = result
    else:
      lm_input, lm_output = error, error
    if (not ignore_examples_with_errors
        or not (isinstance(lm_input, BaseException)
                or isinstance(lm_output, BaseException))):
      yield lm_input, lm_output
