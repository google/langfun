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
"""Utility library for handling concurrency in langfun."""

import concurrent.futures
import dataclasses
import time
from typing import Any, Callable, Iterable, Iterator, Tuple, Type, Union

from langfun.core import component
import pyglove as pg


def with_context_access(func: Callable[..., Any]) -> Any:
  """Derives a user function with the access to the current context."""
  with component.context() as current_context:
    pass

  def _func(*args, **kwargs) -> Any:
    with component.context(**current_context):
      return func(*args, **kwargs)

  return _func


def concurrent_execute(
    func: Callable[[Any], Any],
    parallel_inputs: list[Any],
    max_workers: int = 32,
) -> list[Any]:
  """Executes a function concurrently under current component context."""
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    return list(executor.map(with_context_access(func), parallel_inputs))


@dataclasses.dataclass
class Job:
  """Thread pool job."""

  func: Callable[[Any], Any]
  arg: Any
  result: Any = pg.MISSING_VALUE
  error: Exception | None = None

  def __call__(self, max_attempts=1, retry_interval: float = 0.1) -> Any:
    attempts = 0
    while True:
      try:
        result = self.func(self.arg)
        self.result = result
        return result
      except Exception as e:  # pylint: disable=broad-exception-caught
        attempts += 1
        if attempts < max_attempts:
          time.sleep(retry_interval)
        else:
          self.error = e
          return e


def concurrent_map(
    func: Callable[[Any], Any],
    parallel_inputs: Iterable[Any],
    *,
    max_workers: int = 32,
    ordered: bool = False,
    timeout: int | None = None,
    max_attempts: int = 1,
    retry_interval: float = 0.1,
    silence_on_errors: Union[
        Type[Exception], Tuple[Type[Exception]], None
    ] = None,
) -> Iterator[tuple[Any, Any, Exception | None]]:
  """Maps inputs to outptus via func concurrently under current context."""
  start_time = time.time()

  def remaining_time():
    if timeout is None:
      return None
    return time.time() - start_time

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    future_to_job = {}
    pending_futures = []
    for inputs in parallel_inputs:
      job = Job(func, inputs)
      future = executor.submit(
          with_context_access(job),
          max_attempts=max_attempts,
          retry_interval=retry_interval,
      )
      pending_futures.append(future)
      future_to_job[future] = job

    remaining_futures = []
    if ordered:
      for i, future in enumerate(pending_futures):
        try:
          _ = future.result(timeout=remaining_time())
          job = future_to_job[future]
          if job.error is not None and not (
              silence_on_errors and isinstance(job.error, silence_on_errors)
          ):
            raise job.error
          del future_to_job[future]
          yield job.arg, job.result, job.error
        except concurrent.futures.TimeoutError:
          remaining_futures = pending_futures[i:]
          break
    else:
      for future in concurrent.futures.as_completed(
          pending_futures, timeout=remaining_time()
      ):
        job = future_to_job[future]
        del future_to_job[future]
        if job.error is not None and not (
            silence_on_errors and isinstance(job.error, silence_on_errors)
        ):
          raise job.error
        yield job.arg, job.result, job.error
      remaining_futures = future_to_job

    # Flush pending requests.
    for future in remaining_futures:
      job = future_to_job[future]
      if not future.done():
        future.cancel()
        job.error = TimeoutError(f'Execution time exceeds {timeout} seconds.')
      yield job.arg, job.result, job.error
