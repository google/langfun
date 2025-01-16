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

import abc
import collections
from collections.abc import Mapping
import concurrent.futures
import dataclasses
import io
import random
import sys
import threading
import time
from typing import Annotated, Any, Callable, Iterable, Iterator, Literal, Sequence, Tuple, Type, Union

import pyglove as pg


progress_bar: Literal['tqdm', 'console', None] = None

try:
  from tqdm import auto as tqdm   # pylint: disable=g-import-not-at-top
  progress_bar = 'tqdm'
except ImportError:
  progress_bar = 'console'
  tqdm = None


class RetryError(RuntimeError):
  """Retry error."""

  def __init__(
      self,
      func: Callable[..., Any],
      errors: list[BaseException],
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
        Union[Type[BaseException], Tuple[Type[BaseException], str]],
        Sequence[Union[Type[BaseException], Tuple[Type[BaseException], str]]],
    ],
    max_attempts: int,
    retry_interval: int | tuple[int, int] = (5, 60),
    exponential_backoff: bool = True,
    max_retry_interval: int = 300,
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
    max_retry_interval: The max retry interval in seconds. This is useful when
      the retry interval is exponential, to avoid the wait time to grow
      exponentially.
    seed: Random seed to generate retry interval. If None, the seed will be
      determined based on current time.

  Returns:
    A function with the same signature of the input function, with the retry
    capability.
  """

  def _func(*args, **kwargs):
    job = Job(
        func,
        args,
        kwargs,
        retry_on_errors=retry_on_errors,
        max_attempts=max_attempts,
        retry_interval=retry_interval,
        exponential_backoff=exponential_backoff,
        max_retry_interval=max_retry_interval,
        seed=seed,
    )
    job()
    if job.error:
      raise job.error
    return job.result

  return _func


class RetryEntry(pg.Object):
  """Retry entry."""

  call_interval: float
  error: BaseException | None = None
  wait_interval: float = 0.


def concurrent_execute(
    func: Callable[[Any], Any],
    parallel_inputs: Iterable[Any],
    *,
    executor: Union[concurrent.futures.ThreadPoolExecutor, str, None] = None,
    max_workers: int = 32,
    retry_on_errors: Union[
        Union[Type[BaseException], Tuple[Type[BaseException], str]],
        Sequence[Union[Type[BaseException], Tuple[Type[BaseException], str]]],
        None,
    ] = None,
    max_attempts: int = 5,
    retry_interval: int | tuple[int, int] = (5, 60),
    exponential_backoff: bool = True,
    max_retry_interval: int = 300,
    return_jobs: bool = False,
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
    max_retry_interval: The max retry interval in seconds. This is useful when
      the retry interval is exponential, to avoid the wait time to grow
      exponentially.
    return_jobs: If True, return a list of `Job` objects. Otherwise, return a
      list of outputs.

  Returns:
    A list of ouputs. Each is the return value of `func` based on the input
      value. Order is preserved.
  """
  jobs = []
  for inputs in parallel_inputs:
    jobs.append(
        Job(
            func,
            (inputs,),
            retry_on_errors=retry_on_errors,
            max_attempts=max_attempts,
            retry_interval=retry_interval,
            exponential_backoff=exponential_backoff,
            max_retry_interval=max_retry_interval,
        )
    )

  # NOTE(daiyip): when executor is not specified and max_worker is 1,
  # we don't need to create a executor pool. Instead, the inputs will be
  # processed by the user function in sequence within the current thread.
  if executor is None and max_workers == 1:
    for job in jobs:
      job()
      if job.error:
        raise job.error
    return jobs if return_jobs else [job.result for job in jobs]

  shutdown_after_finish = executor is None
  executor = _executor_pool.executor_from(executor, max_workers=max_workers)

  try:
    executed_jobs = list(
        executor.map(
            lambda job: job(),
            [pg.with_contextual_override(job) for job in jobs]
        )
    )
    for job in executed_jobs:
      if job.error:
        raise job.error
    return (
        executed_jobs if return_jobs else [job.result for job in executed_jobs]
    )
  finally:
    if shutdown_after_finish:
      # Do not wait threads to finish if they are timed out.
      executor.shutdown(wait=False, cancel_futures=True)


@dataclasses.dataclass
class Job:
  """Thread pool job."""

  func: Callable[[Any], Any]
  args: Sequence[Any] = ()
  kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
  _: dataclasses.KW_ONLY

  result: Any = pg.MISSING_VALUE
  error: Annotated[
      BaseException | None,
      'The non-retryable error encountered during the job execution.',
  ] = None
  retry_entries: Annotated[
      Sequence[RetryEntry], 'Records of retry attempts.'
  ] = dataclasses.field(default_factory=list)

  retry_on_errors: Annotated[
      Sequence[Type[BaseException] | str],
      (
          'A sequence of exception types or tuples of exception type and error '
          'messages (described in regular expression) as the desired exception '
          'types to retry.'
      ),
  ] = ()
  max_attempts: Annotated[
      int, 'Max number of attempts if an error to retry is encountered.'
  ] = 5
  retry_interval: Annotated[
      int | tuple[int, int],
      (
          'The (base) retry interval in seconds. If a tuple, the retry '
          'interval will be randomly chosen between the first and the second '
          'element of the tuple.'
      ),
  ] = (5, 60)
  exponential_backoff: Annotated[
      bool,
      (
          'If True, exponential wait time will be applied on top of the base '
          'retry interval.'
      ),
  ] = True
  max_retry_interval: Annotated[
      int,
      (
          'The max retry interval in seconds. This is useful when the retry '
          'interval is exponential, to avoid the wait time to grow '
          'exponentially.'
      ),
  ] = 300
  seed: Annotated[
      int | None,
      (
          'Random seed to generate retry interval. If None, the seed will be'
          ' determined based on current time.'
      ),
  ] = None

  timeit: pg.object_utils.TimeIt = dataclasses.field(
      default_factory=lambda: pg.object_utils.TimeIt('job')
  )

  @property
  def elapse(self) -> float:
    """Returns the running time in seconds since the job get started."""
    return self.timeit.elapse

  def _retry_call(self) -> 'Job':
    """Retries func call on args."""
    rand = random if self.seed is None else random.Random(self.seed)

    def base_interval() -> int:
      if isinstance(self.retry_interval, tuple):
        return rand.randint(*self.retry_interval)
      else:
        assert isinstance(self.retry_interval, int)
        return self.retry_interval

    def next_wait_interval(attempt: int) -> float:
      if not self.exponential_backoff:
        attempt = 1
      return min(
          self.max_retry_interval, base_interval() * (2 ** (attempt - 1))
      )

    retry_entries = []
    wait_interval = 0
    while True:
      with pg.catch_errors(self.retry_on_errors) as error_context:
        begin_time = time.time()
        self.result = self.func(*self.args, **self.kwargs)

      end_time = time.time()
      retry_entries.append(RetryEntry(
          call_interval=end_time - begin_time,
          wait_interval=wait_interval,
          error=error_context.error,
      ))
      if error_context.error is None:
        self.retry_entries = retry_entries
        return self

      # Branch when errors are met for retry.
      if len(retry_entries) < self.max_attempts:
        wait_interval = next_wait_interval(len(retry_entries))

        pg.logging.warning(
            f'Calling {self.func!r} encountered {error_context.error!r} '
            f'(attempts={len(retry_entries)}), retrying in '
            f'{wait_interval} seconds...'
        )

        time.sleep(wait_interval)
      else:
        errors = [e.error for e in retry_entries]
        # First wait interval is 0.
        wait_intervals = [e.wait_interval for e in retry_entries[1:]]
        raise RetryError(self.func, errors, wait_intervals)

  def __call__(self) -> 'Job':
    if getattr(self, '_has_call', False):
      raise ValueError('Job can only be called once.')
    self._has_call = True
    try:
      with self.timeit:
        if self.retry_on_errors:
          return self._retry_call()
        self.result = self.func(*self.args, **self.kwargs)
    except BaseException as e:  # pylint: disable=broad-exception-caught
      self.error = e
    return self

  def mark_canceled(self, error: BaseException) -> None:
    """Marks the job as canceled."""
    self.timeit.end(error)
    self.error = error


@dataclasses.dataclass
class Progress:
  """Concurrent processing progress."""
  total: int

  _succeeded: int = 0
  _failed: int = 0
  _last_error: BaseException | None = None
  _total_duration: float = 0.0
  _job: Job | None = None
  _timeit_summary: pg.object_utils.TimeIt.StatusSummary = dataclasses.field(
      default_factory=pg.object_utils.TimeIt.StatusSummary
  )

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
  def last_error(self) -> BaseException | None:
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

  @property
  def timeit_summary(self) -> pg.object_utils.TimeIt.StatusSummary:
    """Returns the aggregated summary for each `pg.timeit`."""
    return self._timeit_summary

  def timeit_summary_str(self) -> str | None:
    if not self.timeit_summary:
      return None
    return ', '.join([
        '%s (%.2fs, %d/%d)' % (
            k.lstrip('job.'), v.avg_duration, v.num_ended, v.num_started
        ) for k, v in self.timeit_summary.breakdown.items() if k != 'job'
    ])

  def last_error_str(self) -> str | None:
    if self.last_error is None:
      return None
    error_text = repr(self.last_error)
    if len(error_text) >= 64:
      error_text = error_text[:64] + '...'
    return error_text

  def update(self, job: Job) -> None:
    """Mark a job as completed."""
    self._job = job
    if job.error is None:
      self._succeeded += 1
    else:
      self._failed += 1
      self._last_error = job.error
    self._total_duration += job.elapse
    self._timeit_summary.aggregate(job.timeit.status())


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
    status: dict[str, Any] | None = None

  @dataclasses.dataclass
  class Update:
    """Progress bar update."""
    bar_id: int
    delta: int
    status: Union[dict[str, Any], str, None] = None
    color: str | None = None

  _progress_bars: dict[int, '_ProgressControl'] = {}
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
      status: dict[str, Any] | None = None,
      ) -> int:
    """Installs a progress bar and returns a reference id."""
    with cls._lock:
      settings = ProgressBar.Settings(label, total, color, status)
      bar_id = id(settings)
      cls._install_requests.append((bar_id, settings))
      return bar_id

  @classmethod
  def update(
      cls,
      bar_id: int,
      delta: int = 0,
      status: Union[dict[str, Any], str, None] = None,
      color: str | None = None,
      refresh: bool = True,
      ) -> None:
    """Report the progress for a label."""
    if status is not None and not isinstance(status, (str, dict)):
      raise ValueError(f'Unsupported status: {status}')
    with cls._lock:
      cls._updates.append(
          ProgressBar.Update(
              bar_id=bar_id, delta=delta, status=status, color=color,
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
          cls._progress_bars[bar_id] = _progress_control(
              total=settings.total,
              label=settings.label,
              color=settings.color,
              status=settings.status)
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

        if update.status is not None:
          bar.set_status(update.status)

        if update.color is not None:
          bar.set_color(update.color)
        updated_bars.add(bar)

      # Refresh each updated bar just once.
      for bar in updated_bars:
        bar.refresh()

      # Process uninstall requests.
      if cls._uninstall_requests:
        for bar_id in cls._uninstall_requests:
          bar = cls._progress_bars.pop(bar_id, None)
          if bar is not None:
            bar.close()
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
        Type[BaseException], Tuple[Type[BaseException], ...], None
    ] = Exception,
    retry_on_errors: Union[
        Type[BaseException],
        Tuple[Type[BaseException], ...],
        None,
    ] = None,
    max_attempts: int = 5,
    retry_interval: int | tuple[int, int] = (5, 60),
    exponential_backoff: bool = True,
    return_jobs: bool = False,
) -> Iterator[Any]:
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
    return_jobs: If True, the returned iterator will emit `Job` objects.

  Yields:
    An iterator of (input, output, error) or Job object.

  Raises:
    Exception: Errors that are not in `silence_on_errors` or `retry_on_errors`,
      or retry on such errors has reached max attempts.
    TimeoutError: Any item timed out while TimeoutError is not silenced via
      `silence_on_errors`.
  """
    # Internal usage logging.

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
    job = Job(
        func,
        (inputs,),
        retry_on_errors=retry_on_errors,
        max_attempts=max_attempts,
        retry_interval=retry_interval,
        exponential_backoff=exponential_backoff,
    )
    future = executor.submit(pg.with_contextual_override(job))
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
          'AvgDuration': '%.2fs' % progress.avg_duration
      })
      if progress.last_error is not None:
        status['LastError'] = progress.last_error_str()

      if progress.timeit_summary:
        status['TimeIt'] = progress.timeit_summary_str()
      ProgressBar.update(bar_id, delta=1, status=status)

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

            yield job if return_jobs else job.args[0], job.result, job.error
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
            yield job if return_jobs else job.args[0], job.result, job.error
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

            yield job.args[0], job.result, job.error
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


class _ProgressControl(pg.Object):
  """Abstract progress control."""
  # Disable symbolic comparison so the hash is based on object address.
  use_symbolic_comparison = False

  total: int
  label: str | None
  color: str | None
  status: str | dict[str, Any] | None

  def set_color(self, color: str | None):
    with pg.notify_on_change(False):
      self.rebind(color=color)

  def set_status(self, status: str | dict[str, Any] | None):
    with pg.notify_on_change(False):
      self.rebind(status=status)

  @abc.abstractmethod
  def update(self, delta):
    """Update progress."""

  @abc.abstractmethod
  def refresh(self) -> None:
    """Refresh progress bar."""

  @abc.abstractmethod
  def close(self) -> None:
    """Close progress bar."""


class _TqdmProgressControl(_ProgressControl):
  """Tqdm-based progress control."""

  def _on_bound(self):
    super()._on_bound()
    assert tqdm is not None
    self._tqdm = tqdm.tqdm(
        total=self.total,
        desc=self.label,
        colour=self.color,
        postfix=self.status,
    )

  def update(self, delta: int) -> None:
    self._tqdm.update(delta)

  def refresh(self):
    self._tqdm.set_description(self.label, refresh=False)
    if isinstance(self.status, str):
      self._tqdm.set_postfix_str(self.status, refresh=False)
    else:
      self._tqdm.set_postfix(self.status, refresh=False)
    self._tqdm.colour = self.color
    self._tqdm.refresh()

  def close(self):
    self._tqdm.close()


class _ConsoleProgressControl(_ProgressControl):
  """Simple progress control by printing the status to the console."""

  def _on_bound(self):
    super()._on_bound()
    self._progress = 0

  def update(self, delta: int) -> None:
    self._progress += delta

  def refresh(self):
    s = io.StringIO()
    if self.label is not None:
      s.write(pg.colored(self.label, 'red', styles=['bold']))
      s.write(': ')
    s.write(
        pg.colored(
            '%d%% (%d/%d)' %
            (
                self._progress * 100 // self.total,
                self._progress,
                self.total,
            ),
            color=self.color or 'green'
        )
    )
    if self.status is not None:
      status = repr(self.status) if isinstance(
          self.status, dict) else self.status
      s.write(f' : {status}')
    sys.stderr.write(s.getvalue() + '\n')

  def close(self):
    sys.stderr.flush()


class _NoopProgressControl(_ProgressControl):
  """No-op progress control."""

  def update(self, delta: int) -> None:
    pass

  def refresh(self) -> None:
    pass

  def close(self) -> None:
    pass


def _progress_control(
    total: int,
    label: str | None,
    color: str | None,
    status: str | dict[str, Any] | None,
) -> _ProgressControl:
  """Creates a process control."""
  if progress_bar == 'tqdm':
    if not tqdm:
      raise RuntimeError(
          'Please install package "tqdm" to use `tqdm` progress bar.'
      )
    return _TqdmProgressControl(total, label, color, status)
  elif progress_bar == 'console':
    return _ConsoleProgressControl(total, label, color, status)
  elif progress_bar is None:
    return _NoopProgressControl(total, label, color, status)
  else:
    raise ValueError(f'Unsupported progress bar type: {progress_bar}')


def get_executor(
    resource_id: str,
    max_workers: int | None = None) -> concurrent.futures.ThreadPoolExecutor:
  """Gets a thread pool executor associated with a resource id."""
  return _executor_pool.get(resource_id, max_workers)

# The global executor pool based on resource IDs.
_executor_pool = ExecutorPool()
