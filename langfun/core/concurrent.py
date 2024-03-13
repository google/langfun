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

import collections
import concurrent.futures
import dataclasses
import random
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Literal, Sequence, Tuple, Type, Union

from langfun.core import component
import pyglove as pg
from tqdm import auto as tqdm


def with_context_access(func: Callable[..., Any]) -> Callable[..., Any]:
  """Derives a user function with the access to the current context."""
  with component.context() as current_context:
    pass

  def _func(*args, **kwargs) -> Any:
    with component.context(**current_context):
      return func(*args, **kwargs)

  return _func


class RetryError(RuntimeError):
  """Retry error."""

  def __init__(
      self,
      func: Callable[..., Any],
      errors: list[Exception],
      wait_intervals: list[int],
  ):
    assert len(errors) == len(wait_intervals) + 1

    super().__init__()
    self.func = func
    self.errors = errors
    self.wait_intervals = wait_intervals

  @property
  def attempts(self) -> int:
    """Returns the number of attempts made."""
    return len(self.errors)

  def __repr__(self) -> str:
    return (
        'RetryError('
        + pg.object_utils.kvlist_str(
            [
                ('func', self.func, None),
                ('errors', self.errors, None),
                ('wait_intervals', self.wait_intervals, None),
            ],
            compact=True,
        )
        + ')'
    )

  def __str__(self) -> str:
    wait_interval_str = ', '.join([str(x) for x in self.wait_intervals])
    return (
        f'Calling {self.func!r} failed after {self.attempts} attempts '
        f'(wait time: {wait_interval_str} seconds). '
        f'Last error: {self.errors[-1]}'
    )

  def __eq__(self, other: 'RetryError') -> bool:
    if not isinstance(other, RetryError):
      return False
    return (self.func is other.func
            and self.errors == other.errors
            and self.wait_intervals == other.wait_intervals)

  def __ne__(self, other: 'RetryError') -> bool:
    return not self.__eq__(other)

  def __hash__(self) -> int:
    return hash((
        RetryError, self.func, tuple(self.errors), tuple(self.wait_intervals)))


def with_retry(
    func: Callable[[Any], Any],
    retry_on_errors: Union[
        Union[Type[Exception], Tuple[Type[Exception], str]],
        Sequence[Union[Type[Exception], Tuple[Type[Exception], str]]],
    ],
    max_attempts: int,
    retry_interval: int | tuple[int, int] = (5, 60),
    exponential_backoff: bool = True,
    seed: int | None = None,
) -> Callable[..., Any]:
  """Derives a user function with retry on error.

  Args:
    func: A user function.
    retry_on_errors: A sequence of exception types or tuples of exception type
      and error messages (described in regular expression) as the desired
      exception types to retry.
    max_attempts: Max number of attempts if an error to retry is encountered.
    retry_interval: The (base) retry interval in seconds. If a tuple, the retry
      interval will be randomly chosen between the first and the second element
      of the tuple.
    exponential_backoff: If True, exponential wait time will be applied on top
      of the base retry interval.
    seed: Random seed to generate retry interval. If None, the seed will be
      determined based on current time.

  Returns:
    A function with the same signature of the input function, with the retry
    capability.
  """
  rand = random if seed is None else random.Random(seed)

  def _func(*args, **kwargs) -> Any:
    def base_interval() -> int:
      if isinstance(retry_interval, tuple):
        return rand.randint(retry_interval[0], retry_interval[1])
      else:
        assert isinstance(retry_interval, int)
        return retry_interval

    def next_wait_interval(attempt: int) -> float:
      if not exponential_backoff:
        attempt = 1
      return base_interval() * (2 ** (attempt - 1))

    wait_interval = None
    wait_intervals = []
    errors = []
    while True:
      with pg.catch_errors(retry_on_errors) as error_context:
        return func(*args, **kwargs)

      # Branch when errors are met for retry.
      errors.append(error_context.error)
      if len(errors) < max_attempts:
        wait_interval = next_wait_interval(len(errors))
        wait_intervals.append(wait_interval)

        pg.logging.warning(
            f'Calling {func!r} encountered {error_context.error!r} '
            f'(attempts={len(errors)}), retrying in {wait_interval} seconds...'
        )

        time.sleep(wait_interval)
      else:
        raise RetryError(func, errors, wait_intervals)

  return _func


def concurrent_execute(
    func: Callable[[Any], Any],
    parallel_inputs: Iterable[Any],
    *,
    executor: Union[concurrent.futures.ThreadPoolExecutor, str, None] = None,
    max_workers: int = 32,
    retry_on_errors: Union[
        Union[Type[Exception], Tuple[Type[Exception], str]],
        Sequence[Union[Type[Exception], Tuple[Type[Exception], str]]],
        None,
    ] = None,
    max_attempts: int = 5,
    retry_interval: int | tuple[int, int] = (5, 60),
    exponential_backoff: bool = True,
) -> list[Any]:
  """Executes a function concurrently under current component context.

  Args:
    func: A user function.
    parallel_inputs: The inputs for `func` which will be processed in parallel.
    executor: A thread pool executor or a resource ID string to pool work items.
      When resource ID is used, a thread pool will be created and cached in a
      executor pool for future reuse. If None, a new thread pool executor will
      be created for current execution.
    max_workers: The max number of workers.
    retry_on_errors: A sequence of exception types or tuples of exception type
      and error messages (described in regular expression) as the desired
      exception types to retry.
    max_attempts: Max number of attempts if an error to retry is encountered.
    retry_interval: The (base) retry interval in seconds. If a tuple, the retry
      interval will be randomly chosen between the first and the second element
      of the tuple.
    exponential_backoff: If True, exponential wait time will be applied on top
      of the base retry interval.

  Returns:
    A list of ouputs. Each is the return value of `func` based on the input
      value. Order is preserved.
  """
  if retry_on_errors is not None:
    func = with_retry(
        func,
        retry_on_errors,
        max_attempts=max_attempts,
        retry_interval=retry_interval,
        exponential_backoff=exponential_backoff,
    )

  # NOTE(daiyip): when executor is not specified and max_worker is 1,
  # we don't need to create a executor pool. Instead, the inputs will be
  # processed by the user function in sequence within the current thread.
  if executor is None and max_workers == 1:
    return [func(i) for i in parallel_inputs]

  shutdown_after_finish = executor is None
  executor = _executor_pool.executor_from(executor, max_workers=max_workers)

  try:
    return list(executor.map(with_context_access(func), parallel_inputs))
  finally:
    if shutdown_after_finish:
      # Do not wait threads to finish if they are timed out.
      executor.shutdown(wait=False, cancel_futures=True)


@dataclasses.dataclass
class Job:
  """Thread pool job."""

  func: Callable[[Any], Any]
  arg: Any
  result: Any = pg.MISSING_VALUE
  error: Exception | None = None
  start_time: float | None = None
  end_time: float | None = None

  def __call__(self) -> Any:
    self.start_time = time.time()
    try:
      self.result = self.func(self.arg)
      return self.result
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.error = e
      return e
    finally:
      self.end_time = time.time()

  def mark_canceled(self, error: Exception) -> None:
    """Marks the job as canceled."""
    self.error = error
    self.end_time = time.time()

  @property
  def elapse(self) -> float:
    """Returns the running time in seconds since the job get started."""
    if self.start_time is None:
      return 0.0
    if self.end_time is None:
      return time.time() - self.start_time
    return self.end_time - self.start_time


@dataclasses.dataclass
class Progress:
  """Concurrent processing progress."""
  total: int

  _succeeded: int = 0
  _failed: int = 0
  _last_error: Exception | None = None
  _total_duration: float = 0.0
  _job: Job | None = None

  @property
  def succeeded(self) -> int:
    """Returns number of succeeded jobs."""
    return self._succeeded

  @property
  def failed(self) -> int:
    """Returns number of failed jobs."""
    return self._failed

  @property
  def completed(self) -> int:
    """Returns number of completed jobs."""
    return self.succeeded + self.failed

  @property
  def last_error(self) -> Exception | None:
    """Returns last error."""
    return self._last_error

  @property
  def job(self) -> Job | None:
    """Returns current job."""
    return self._job

  @property
  def success_rate(self) -> float:
    """Returns success rate."""
    if self.completed == 0:
      return 0.0
    return self.succeeded / self.completed

  @property
  def failure_rate(self) -> float:
    """Returns failure rate."""
    if self.completed == 0:
      return 0.0
    return self.failed / self.completed

  @property
  def avg_duration(self) -> float:
    """Returns average duration each job worked."""
    if self.completed == 0:
      return 0.0
    return self._total_duration / self.completed

  def update(self, job: Job) -> None:
    """Mark a job as completed."""
    self._job = job
    if job.error is None:
      self._succeeded += 1
    else:
      self._failed += 1
      self._last_error = job.error
    self._total_duration += job.elapse


class ProgressBar:
  """Progress bars that can be updated in concurrent threads.

  When progresses are reported during `lf.concurrent_map`, thread functions may
  have child calls to `lf.concurrent_map`, which requires multiple progress bar
  created/updated in multi-threads to be supported.

  However, tqdm requires all progress bars to be created and updated in the
  main thread. This class uses a queue system to communicate among the threads,
  allowing bar installation/uninstallation/update to be called within non-main
  thread.
  """

  @dataclasses.dataclass
  class Settings:
    """Progress bar settings."""
    label: str | None
    total: int
    color: str | None = None
    postfix: dict[str, str] | None = None

  @dataclasses.dataclass
  class Update:
    """Progress bar update."""
    bar_id: int
    delta: int
    postfix: Union[dict[str, str], str, None] = None
    color: str | None = None

  _progress_bars: dict[int, tqdm.tqdm] = {}
  _install_requests: list[tuple[int, Settings]] = []
  _updates: collections.deque[Update] = collections.deque()
  _uninstall_requests: list[int] = []
  _lock = threading.Lock()

  @classmethod
  def install(
      cls,
      label: str | None,
      total: int,
      color: str | None = None,
      postfix: dict[str, str] | None = None,
      ) -> int:
    """Installs a progress bar and returns a reference id."""
    with cls._lock:
      settings = ProgressBar.Settings(label, total, color, postfix)
      bar_id = id(settings)
      cls._install_requests.append((bar_id, settings))
      return bar_id

  @classmethod
  def update(
      cls,
      bar_id: int,
      delta: int = 0,
      postfix: Union[dict[str, str], str, None] = None,
      color: str | None = None,
      refresh: bool = True,
      ) -> None:
    """Report the progress for a label."""
    with cls._lock:
      cls._updates.append(
          ProgressBar.Update(
              bar_id=bar_id, delta=delta, postfix=postfix, color=color,
          )
      )
    if refresh:
      cls.refresh()

  @classmethod
  def uninstall(cls, bar_id: int) -> None:
    """Uninstalls a progress bar with the reference id returned by install."""
    with cls._lock:
      cls._uninstall_requests.append(bar_id)

  @classmethod
  def refresh(cls) -> None:
    """Update all progress bars when called within the main thread."""
    if threading.current_thread() is not threading.main_thread():
      return

    with cls._lock:
      # Process install requests.
      if cls._install_requests:
        for bar_id, settings in cls._install_requests:
          cls._progress_bars[bar_id] = tqdm.tqdm(
              total=settings.total,
              desc=settings.label,
              colour=settings.color,
              postfix=settings.postfix)
        cls._install_requests.clear()

      # Process updates.
      updated_bars = set()
      while cls._updates:
        update = cls._updates.popleft()
        bar = cls._progress_bars.get(update.bar_id)
        # Processing of updates may be delayed, in such case the bar might
        # be already uninstalled from a different thread.
        if bar is None:
          continue
        if update.delta > 0:
          bar.update(update.delta)

        if isinstance(update.postfix, str):
          bar.set_postfix_str(update.postfix, refresh=False)
        elif isinstance(update.postfix, dict):
          bar.set_postfix(update.postfix, refresh=False)
        elif update.postfix is not None:
          raise ValueError(f'Unsupported postfix: {update.postfix}')

        if update.color is not None:
          bar.colour = update.color
        updated_bars.add(bar)

      # Refresh each updated bar just once.
      for bar in updated_bars:
        bar.refresh()

      # Process uninstall requests.
      if cls._uninstall_requests:
        for bar_id in cls._uninstall_requests:
          cls._progress_bars.pop(bar_id, None)
        cls._uninstall_requests.clear()


def concurrent_map(
    func: Callable[[Any], Any],
    parallel_inputs: Iterable[Any],
    *,
    executor: Union[concurrent.futures.ThreadPoolExecutor, str, None] = None,
    max_workers: int = 32,
    ordered: bool = False,
    show_progress: bool | int = False,
    label: str | None = None,
    color: Literal[
        'red',
        'blue',
        'green',
        'black',
        'yellow',
        'magenta',
        'cyan',
        'white',
        None,
    ] = None,
    status_fn: Callable[[Progress], dict[str, Any]] | None = None,
    timeout: int | None = None,
    silence_on_errors: Union[
        Type[Exception], Tuple[Type[Exception], ...], None
    ] = Exception,
    retry_on_errors: Union[
        Type[Exception],
        Tuple[Type[Exception], ...],
        None,
    ] = None,
    max_attempts: int = 5,
    retry_interval: int | tuple[int, int] = (5, 60),
    exponential_backoff: bool = True,
) -> Iterator[tuple[Any, Any, Exception | None]]:
  """Maps inputs to outptus via func concurrently under current context.

  Args:
    func: A user function.
    parallel_inputs: The inputs for `func` which will be processed in parallel.
    executor: A thread pool executor or a resource ID string to pool work items.
      When resource ID is used, a thread pool will be created and cached in a
      executor pool for future reuse. If None, a new thread pool executor will
      be created for current execution.
    max_workers: The max number of workers.
    ordered: If True, the returned iterator will emit (input, output, error) in
      the order of the elements in `parallel_inputs`. Otherwise, elements that
      are finished earlier will be delivered first.
    show_progress: A boolean or an integer as the bar id returned from
      `lf.concurrent.ProgressBar.install`. If True, a bar will be created on the
      fly showing the progress of the execution. If False, no progress bar will
      be shown. If a bar id, the progress update will be associated with the
      bar.
    label: An optional label for the progress bar. Applicable when
      `show_progress` is set to True.
    color: Color of the progress bar. Applicable when `show_progress` is set to
      True or a bar id.
    status_fn: An optional callable object that receives a
      `lf.concurrent.Progress` object and returns a dict of kv pairs as the
      status to include in the progress bar. Applicable only when
      `show_progress` is set to True or a bar id. If None, the default status_fn
      will be used, which outputs the success and failure rate.
    timeout: The timeout in seconds for processing each input. It is the total
      processing time for each input, even multiple retries take place. If None,
      there is no timeout.
    silence_on_errors: If None, any errors raised during processing any inputs
      will be raised. Otherwise, the matched errors will not raise, instead,
      they will be returned as the third_element of the tuple.
    retry_on_errors: A sequence of exception types or tuples of exception type
      and error messages (described in regular expression) as the desired
      exception types to retry. These errors are usually transient error.
    max_attempts: Max number of attempts if an error to retry is encountered.
    retry_interval: The (base) retry interval in seconds. If a tuple, the retry
      interval will be randomly chosen between the first and the second element
      of the tuple.
    exponential_backoff: If True, exponential wait time will be applied on top
      of the base retry interval.

  Yields:
    An iterator of (input, output, error).

  Raises:
    Exception: Errors that are not in `silence_on_errors` or `retry_on_errors`,
      or retry on such errors has reached max attempts.
    TimeoutError: Any item timed out while TimeoutError is not silenced via
      `silence_on_errors`.
  """
    # Internal usage logging.

  if retry_on_errors:
    func = with_retry(
        func,
        retry_on_errors,
        max_attempts=max_attempts,
        retry_interval=retry_interval,
        exponential_backoff=exponential_backoff,
    )

  status_fn = status_fn or (lambda p: {   # pylint: disable=g-long-lambda
      'Succeeded': '%.2f%% (%d/%d)' % (
          p.success_rate * 100, p.succeeded, p.completed),
      'Failed': '%.2f%% (%d/%d)' % (
          p.failure_rate * 100, p.failed, p.completed),
  })

  shutdown_after_finish = executor is None
  executor = _executor_pool.executor_from(executor, max_workers=max_workers)

  future_to_job = {}
  pending_futures = []
  total = 0
  for inputs in parallel_inputs:
    job = Job(func, inputs)
    future = executor.submit(
        with_context_access(job),
    )
    pending_futures.append(future)
    future_to_job[future] = job
    total += 1

  # Setup progress bar.
  progress = Progress(total=total)
  if isinstance(show_progress, bool):
    external_bar = False
    bar_id = ProgressBar.install(label, total, color) if show_progress else None
  else:
    bar_id = show_progress
    external_bar = True
    show_progress = True

  def update_progress_bar(progress: Progress) -> None:
    if show_progress:
      status = status_fn(progress)
      status.update({
          'AvgDuration': '%.2f seconds' % progress.avg_duration
      })
      if progress.last_error is not None:
        error_text = repr(progress.last_error)
        if len(error_text) >= 64:
          error_text = error_text[:64] + '...'
        status['LastError'] = error_text
      ProgressBar.update(bar_id, delta=1, postfix=status)

  try:
    if ordered:
      for future in pending_futures:
        job = future_to_job[future]
        completed = False
        while True:
          try:
            _ = future.result(timeout=1)
            completed = True
          except concurrent.futures.TimeoutError:
            if timeout and timeout < job.elapse:
              future.cancel()
              last_error = TimeoutError(
                  f'Execution time ({job.elapse}) exceeds {timeout} seconds.')
              job.mark_canceled(last_error)
              completed = True

          if completed:
            if job.error is not None and not (
                silence_on_errors and isinstance(job.error, silence_on_errors)):
              raise job.error   # pylint: disable=g-doc-exception

            yield job.arg, job.result, job.error
            progress.update(job)
            update_progress_bar(progress)
            ProgressBar.refresh()
            break
          else:
            # There might be updates from other concurrent_map. So even there
            # is no progress on current map, we still update the progress
            # manager.
            ProgressBar.refresh()
    else:
      while pending_futures:
        completed_batch = set()
        try:
          for future in concurrent.futures.as_completed(
              pending_futures, timeout=1):
            job = future_to_job[future]
            del future_to_job[future]
            if job.error is not None and not (
                silence_on_errors and isinstance(job.error, silence_on_errors)):
              raise job.error   # pylint: disable=g-doc-exception
            yield job.arg, job.result, job.error
            progress.update(job)
            update_progress_bar(progress)
            completed_batch.add(future)
            ProgressBar.refresh()
        except concurrent.futures.TimeoutError:
          pass

        remaining_futures = []
        for future in pending_futures:
          if future in completed_batch:
            continue
          job = future_to_job[future]
          if timeout and job.elapse > timeout:
            if not future.done():
              future.cancel()
              job.mark_canceled(
                  TimeoutError(f'Execution time ({job.elapse}) '
                               f'exceeds {timeout} seconds.'))
              if not (silence_on_errors
                      and isinstance(job.error, silence_on_errors)):
                raise job.error  # pylint: disable=g-doc-exception

            yield job.arg, job.result, job.error
            progress.update(job)
            update_progress_bar(progress)
          else:
            remaining_futures.append(future)
        pending_futures = remaining_futures
        ProgressBar.refresh()
  finally:
    if show_progress and not external_bar:
      ProgressBar.uninstall(bar_id)

    if shutdown_after_finish:
      # Do not wait threads to finish if they are timed out.
      executor.shutdown(wait=False, cancel_futures=True)


class ExecutorPool:
  """A pool of managed executors.

  Managed executors are used for controlling the parallelism of execution based
  on resource id. This design is to honor overall rate limit of LMs globally
  (with current process).
  """

  def __init__(self):
    self._executors: dict[str, concurrent.futures.ThreadPoolExecutor] = {}

  def get(
      self, resource_id: str, max_workers: int | None = None
  ) -> concurrent.futures.ThreadPoolExecutor:
    """Gets or creates a thread pool executor associated with a resource id."""
    executor = self._executors.get(resource_id)
    if executor is None:
      executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
      self._executors[resource_id] = executor
    return executor

  @property
  def resource_ids(self) -> list[str]:
    """Returns the resource ids for active executors."""
    return list(self._executors.keys())

  def executor_from(
      self,
      maybe_executor: Union[concurrent.futures.ThreadPoolExecutor, str, None],
      max_workers: int | None = None,
  ) -> concurrent.futures.ThreadPoolExecutor:
    """Creates a thread pool executor."""
    if isinstance(maybe_executor, concurrent.futures.ThreadPoolExecutor):
      return maybe_executor
    elif isinstance(maybe_executor, str):
      return self.get(maybe_executor, max_workers)
    elif maybe_executor is None:
      return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    else:
      raise ValueError(f'Unsupported value: {maybe_executor}.')


# The global executor pool based on resource IDs.
_executor_pool = ExecutorPool()
