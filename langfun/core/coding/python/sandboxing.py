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
"""Python sandboxing."""

import abc
import os
import tempfile
from typing import Annotated

from langfun.core.coding.python import parsing
import pyglove as pg


class SandboxOutput(pg.Object):
  """Sandbox output."""

  stdout: Annotated[
      str,
      'The stdout of the sandbox execution.'
  ] = ''

  stderr: Annotated[
      str,
      'The stderr of the sandbox execution.'
  ] = ''

  output_files: Annotated[
      dict[str, bytes],
      'The output files of the sandbox execution.'
  ] = {}


class BaseSandbox(pg.Object):
  """Interface and partial implementation for Python sandbox."""

  def _on_bound(self):
    super()._on_bound()
    self._uploaded_files: dict[str, str] = {}

  def run(
      self,
      code: str,
      *,
      timeout: int | None = 30,
      **kwargs
  ) -> SandboxOutput:
    """Runs code in the sandbox. Raises pg.coding.CodeError if failed."""
    return self._run(self.normalize_code(code), timeout=timeout, **kwargs)

  def normalize_code(self, code: str) -> str:
    """Returns normalized code runnable in the sandbox."""
    for original_path, uploaded_path in self._uploaded_files.items():
      code = code.replace(original_path, uploaded_path)
    return parsing.clean(code)

  @abc.abstractmethod
  def _run(
      self,
      code: str,
      *,
      timeout: int | None = 30,
      **kwargs
  ) -> SandboxOutput:
    """Runs code in the sandbox. Raises pg.coding.CodeError if failed."""

  def upload(self, path: str) -> str:
    """Uploads a file to the sandbox. Returns the uploaded path."""
    uploaded_path = self._upload(path)
    self._uploaded_files[path] = uploaded_path
    return uploaded_path

  @abc.abstractmethod
  def _upload(self, path: str) -> str:
    """Uploads a file to the sandbox."""

  def setup(self) -> None:
    """Sets up the sandbox."""
    self._uploaded_files = {}
    self._setup()

  def _setup(self) -> None:
    """Sets up the sandbox."""

  def cleanup(self) -> None:
    """Cleans up the sandbox."""
    self._uploaded_files = {}
    self._cleanup()

  def _cleanup(self) -> None:
    """Cleans up the sandbox."""

  def __enter__(self):
    """Enters the sandbox."""
    self.setup()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    """Exits the sandbox."""
    self.cleanup()


class MultiProcessingSandbox(BaseSandbox):
  """Sandbox using multiprocessing."""

  def _on_bound(self):
    super()._on_bound()
    self._working_dir = None

  @property
  def working_dir(self) -> str | None:
    """Returns the directory of the sandbox."""
    return self._working_dir

  def _setup(self) -> None:
    """Sets up the sandbox."""
    self._working_dir = tempfile.TemporaryDirectory()

  def _cleanup(self) -> None:
    """Cleans up the sandbox."""
    assert self._working_dir is not None
    self._working_dir.cleanup()

  def _run(
      self,
      code: str,
      *,
      timeout: int | None = 30,
      **kwargs
  ) -> SandboxOutput:
    """Runs code in the sandbox."""
    stdout = pg.coding.run(
        code, returns_stdout=True, sandbox=True, timeout=timeout
    )
    return SandboxOutput(stdout=stdout)

  def _upload(self, path: str) -> str:
    """Uploads a file to the sandbox."""
    if self._working_dir is None:
      raise ValueError('Sandbox is not set up.')

    # Upload the file to the sandbox directory.
    uploaded_path = os.path.join(self._working_dir.name, os.path.basename(path))
    with pg.io.open(path, 'r') as r:
      with pg.io.open(uploaded_path, 'w') as w:
        w.write(r.read())
    return uploaded_path
