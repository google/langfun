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
import time
import unittest

from langfun.env import interface
from langfun.env import load_balancers


class TestingSandbox(interface.Sandbox):
  sandbox_id: str
  is_alive: bool = True
  is_pending: bool = False
  is_busy: bool = False

  def _on_bound(self) -> None:
    super()._on_bound()
    self._session_id = None

  @property
  def id(self) -> interface.SandboxId:
    return interface.SandboxId(
        environment_id=interface.EnvironmentId('testing-env'),
        sandbox_id=self.sandbox_id
    )

  @property
  def environment(self) -> interface.Environment:
    raise NotImplementedError()

  @property
  def features(self) -> dict[str, interface.Feature]:
    raise NotImplementedError()

  def set_pending(self, pending: bool = True) -> None:
    self.rebind(
        is_pending=pending, skip_notification=True, raise_on_no_change=False
    )

  def set_busy(self, busy: bool = True) -> None:
    self.rebind(
        is_busy=busy, skip_notification=True, raise_on_no_change=False
    )

  def set_alive(self, alive: bool = True) -> None:
    self.rebind(
        is_alive=alive, skip_notification=True, raise_on_no_change=False
    )

  def start(self) -> None:
    self.set_alive()

  def shutdown(self) -> None:
    self.set_alive(False)

  def ping(self) -> None:
    pass

  def start_session(self, session_id: str) -> None:
    self._session_id = session_id

  def end_session(self, session_id: str) -> None:
    self._session_id = None

  @property
  def session_id(self) -> str | None:
    return self._session_id


class RoundRobinTest(unittest.TestCase):

  def test_basic(self):
    sandbox_pool = [
        TestingSandbox(
            '0',
            is_alive=False,
            is_pending=False,
            is_busy=False,
        ),
        TestingSandbox(
            '1',
            is_alive=True,
            is_pending=True,
            is_busy=False,
        ),
        TestingSandbox(
            '2',
            is_alive=True,
            is_pending=False,
            is_busy=True,
        ),
        TestingSandbox(
            '3',
            is_alive=True,
            is_pending=False,
            is_busy=False,
        ),
        TestingSandbox(
            '4',
            is_alive=True,
            is_pending=False,
            is_busy=False,
        ),
    ]
    lb = load_balancers.RoundRobin()
    sandbox = lb.acquire(sandbox_pool)
    self.assertIs(sandbox, sandbox_pool[3])
    self.assertTrue(sandbox.is_pending)

    sandbox = lb.acquire(sandbox_pool)
    self.assertIs(sandbox, sandbox_pool[4])
    self.assertTrue(sandbox.is_pending)

    sandbox_pool[0].set_alive()
    sandbox = lb.acquire(sandbox_pool)
    self.assertIs(sandbox, sandbox_pool[0])
    self.assertTrue(sandbox.is_pending)

    with self.assertRaisesRegex(IndexError, 'No free sandbox in the pool.'):
      lb.acquire(sandbox_pool)

  def test_thread_safety(self):
    sandbox_pool = [TestingSandbox(str(i)) for i in range(64)]

    lb = load_balancers.RoundRobin()

    def _thread_func(i):
      sandbox = lb.acquire(sandbox_pool)
      time.sleep(0.1)
      sandbox.set_busy()
      sandbox.set_pending(False)
      time.sleep(0.1)
      sandbox.set_busy(False)
      sandbox.set_alive(False)
      time.sleep(0.1)
      sandbox.set_alive()
      return i

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
      for i, o in enumerate(executor.map(_thread_func, range(1024))):
        self.assertEqual(o, i)


if __name__ == '__main__':
  unittest.main()
