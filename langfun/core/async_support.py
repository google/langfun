# Copyright 2025 The Langfun Authors
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
"""Utility for async IO in Langfun."""

import asyncio
import contextlib
from typing import Any, Awaitable, Callable, Iterator
import anyio
import pyglove as pg


async def invoke_async(
    sync_callable: Callable[..., Any], *args, **kwargs
) -> Any:
  """Invokes a callable asynchronously with `lf.context` manager enabled."""
  return await asyncio.to_thread(
      # Enable `lf.context` manager for async calls.
      pg.with_contextual_override(sync_callable), *args, **kwargs
  )


def invoke_sync(
    async_callable: Callable[..., Awaitable[Any]],
    *args,
    **kwargs
) -> Any:
  """Invokes a async callable synchronously."""
  async def _invoke():
    return await async_callable(*args, **kwargs)
  invoke_fn = pg.with_contextual_override(_invoke)
  blocking_portal = pg.utils.thread_local_get('__blocking_portal__', None)
  if blocking_portal is None:
    return anyio.run(invoke_fn)
  return blocking_portal.call(invoke_fn)


@contextlib.contextmanager
def sync_context_manager(
    async_context_manager: contextlib.AbstractAsyncContextManager[Any]
) -> Iterator[Any]:
  """Adapts an async context manager to a sync context manager.

  sync_context_manager installs a blocking portal in current thread to run the
  async context manager in a blocking way. It's useful for running async code in
  sync context managers, e.g. `sync_context_manager` can be nested and share the
  same event loop.

  Example:

    ```python
    @contextlib.asynccontextmanager
    async def foo(x):
      try:
        yield x
      finally:
        pass

    with lf.sync_context_manager(foo(x)) as x
      with lf.sync_context_manager(foo(y)) as y:
        ...
    ```

  Args:
    async_context_manager: The async context manager to adapt.

  Yields:
    The value yielded by the async context manager.
  """
  blocking_portal = pg.utils.thread_local_get('__blocking_portal__', None)
  portal_exit_stack = None

  try:
    if blocking_portal is None:
      portal_exit_stack = contextlib.ExitStack()
      blocking_portal = portal_exit_stack.enter_context(
          anyio.from_thread.start_blocking_portal()
      )
      pg.utils.thread_local_set('__blocking_portal__', blocking_portal)
    context_manager = blocking_portal.wrap_async_context_manager(
        async_context_manager
    )
    with context_manager as value:
      yield value
  finally:
    if portal_exit_stack is not None:
      portal_exit_stack.close()
      pg.utils.thread_local_del('__blocking_portal__')
