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
"""Load balancers for environments."""

import abc
import threading

from langfun.env import interface
import pyglove as pg


class LoadBalancer(pg.Object):
  """Base class for load balancers."""

  @abc.abstractmethod
  def acquire(self, sandbox_pool: list[interface.Sandbox]) -> interface.Sandbox:
    """Acquires a free sandbox from a pool of sandboxes.

    The load balancer will pick a sandbox from the pool and mark it as pending.

    Args:
      sandbox_pool: The pool of sandboxes to pick from.

    Raises:
      IndexError: If all sandboxes in the pool are either busy or dead.
    """


class RoundRobin(LoadBalancer):
  """Round robin load balancer."""

  def _on_bound(self):
    super()._on_bound()
    self._counter = 0
    self._acquire_lock = threading.Lock()

  def acquire(self, sandbox_pool: list[interface.Sandbox]) -> interface.Sandbox:
    """Returns a free sandbox from the pool."""
    with self._acquire_lock:
      for _ in range(len(sandbox_pool)):
        sandbox = sandbox_pool[self._counter % len(sandbox_pool)]
        self._counter = self._counter + 1
        if sandbox.status == interface.Sandbox.Status.READY:
          # Mark the sandbox as acquired to prevent it from being acquired by
          # other threads.
          sandbox.set_acquired()
          return sandbox
    raise IndexError('No free sandbox in the pool.')
