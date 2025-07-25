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
from typing import Any, Callable
import pyglove as pg


async def invoke_async(
    callable_object: Callable[..., Any], *args, **kwargs
) -> Any:
  """Invokes a callable asynchronously with `lf.context` manager enabled."""
  return await asyncio.to_thread(
      # Enable `lf.context` manager for async calls.
      pg.with_contextual_override(callable_object), *args, **kwargs
  )
