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
"""In-memory LM cache."""

from typing import Any
from langfun.core.llms.cache import base


class InMemory(base.LMCacheBase):
  """In memory cache."""

  def _get(self, key: Any) -> base.LMCacheEntry | None:
    """Returns a LM cache entry associated with the key."""
    return _CACHE_MEMORY.get(key, None)

  def _put(self, key: Any, entry: base.LMCacheEntry) -> None:
    """Puts a LM cache entry associated with the key."""
    _CACHE_MEMORY[key] = entry

  def reset(self) -> None:
    """Resets the cache."""
    _CACHE_MEMORY.clear()


# NOTE(daiyip): We install a process-level cache store, so different InMemory()
# object could access the same memory. This is not a problem across different
# language models, since the `model_id` of the language model is included as a
# part of the cache key.
_CACHE_MEMORY = {}
