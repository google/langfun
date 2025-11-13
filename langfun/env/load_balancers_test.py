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
import concurrent.futures
import contextlib
import time
from typing import Any, Iterator
import unittest

from langfun.env import interface
from langfun.env import load_balancers


class TestingSandbox(interface.Sandbox):
  sandbox_id: str
  status: interface.Sandbox.Status = interface.Sandbox.Status.READY
  image_id: str = 'test_image'

  __test__ = False

  def _on_bound(self) -> None:
    super()._on_bound()
    self._session_id = None

  @property
  def id(self) -> interface.Sandbox.Id:
    return interface.Sandbox.Id(
        environment_id=interface.Environment.Id('testing-env'),
        image_id=self.image_id,
        sandbox_id=self.sandbox_id
    )

  @property
  def environment(self) -> interface.Environment:
    raise NotImplementedError()

  @property
  def features(self) -> dict[str, interface.Feature]:
    raise NotImplementedError()

  @property
  def state_errors(self) -> list[interface.SandboxStateError]:
    return []

  def report_state_error(self, error: interface.SandboxStateError) -> None:
    pass

  def set_status(self, status: interface.Sandbox.Status) -> None:
    self.rebind(status=status, skip_notification=True)

  def set_acquired(self) -> None:
    self.set_status(self.Status.ACQUIRED)

  def start(self) -> None:
    pass

  def shutdown(self) -> None:
    pass

  def start_session(self, session_id: str) -> None:
    self._session_id = session_id

  def end_session(self, session_id: str) -> None:
    self._session_id = None

  @property
  def session_id(self) -> str | None:
    return self._session_id

  @contextlib.contextmanager
  def track_activity(
      self,
      name: str,
      feature: interface.Feature | None = None,
      **kwargs: Any
  ) -> Iterator[None]:
    try:
      yield
    finally:
      pass


class RoundRobinTest(unittest.TestCase):

  def test_basic(self):
    sandbox_pool = [
        TestingSandbox('0', interface.Sandbox.Status.OFFLINE),
        TestingSandbox('1', interface.Sandbox.Status.SETTING_UP),
        TestingSandbox('2', interface.Sandbox.Status.IN_SESSION),
        TestingSandbox('3', status=interface.Sandbox.Status.READY),
        TestingSandbox('4', status=interface.Sandbox.Status.READY),
    ]
    lb = load_balancers.RoundRobin()
    sandbox = lb.acquire(sandbox_pool)
    self.assertIs(sandbox, sandbox_pool[3])
    self.assertEqual(sandbox.status, interface.Sandbox.Status.ACQUIRED)

    sandbox = lb.acquire(sandbox_pool)
    self.assertIs(sandbox, sandbox_pool[4])
    self.assertEqual(sandbox.status, interface.Sandbox.Status.ACQUIRED)

    sandbox_pool[0].set_status(interface.Sandbox.Status.READY)
    sandbox = lb.acquire(sandbox_pool)
    self.assertIs(sandbox, sandbox_pool[0])
    self.assertEqual(sandbox.status, interface.Sandbox.Status.ACQUIRED)

    with self.assertRaisesRegex(IndexError, 'No free sandbox in the pool.'):
      lb.acquire(sandbox_pool)

  def test_thread_safety(self):
    sandbox_pool = [TestingSandbox(str(i)) for i in range(64)]

    lb = load_balancers.RoundRobin()

    def _thread_func(i):
      sandbox = lb.acquire(sandbox_pool)
      time.sleep(0.1)
      sandbox.set_status(interface.Sandbox.Status.IN_SESSION)
      time.sleep(0.1)
      sandbox.set_status(interface.Sandbox.Status.OFFLINE)
      time.sleep(0.1)
      sandbox.set_status(interface.Sandbox.Status.READY)
      return i

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
      for i, o in enumerate(executor.map(_thread_func, range(1024))):
        self.assertEqual(o, i)


if __name__ == '__main__':
  unittest.main()
