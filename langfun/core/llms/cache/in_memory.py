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

import collections
import contextlib
from typing import Annotated, Any, Iterator
import langfun.core as lf
from langfun.core.llms.cache import base
import pyglove as pg


@pg.use_init_args(['filename', 'ttl', 'key'])
class InMemory(base.LMCacheBase):
  """In memory cache."""

  filename: Annotated[
      str | None,
      (
          'File name to load and save in memory cache.'
      )
  ] = None

  def _on_bound(self) -> None:
    super()._on_bound()
    self._cache = collections.defaultdict(dict)

    if self.filename is not None:
      try:
        records = pg.load(self.filename)
        for record in records:
          model_cache = {}
          for entry in record.entries:
            model_cache[entry.k] = entry.v
          self._cache[record.model_id] = model_cache
      except FileNotFoundError:
        pg.logging.warning(
            "Creating a new cache as cache file '%s' does not exist.",
            self.filename,
        )

  def model_ids(self) -> list[str]:
    """Returns the model ids of cached queires."""
    return list(self._cache.keys())

  def __len__(self) -> int:
    """Returns the number of entries in the cache."""
    return sum(len(v) for v in self._cache.values())

  def keys(self, model_id: str | None = None) -> Iterator[str]:
    """Returns the cached keys for a model."""
    if model_id is None:
      for model_cache in self._cache.values():
        for k in model_cache.keys():
          yield k
    else:
      for k in self._cache[model_id].keys():
        yield k

  def values(self, model_id: str | None = None) -> Iterator[base.LMCacheEntry]:
    """Returns the cached entries for a model."""
    if model_id is None:
      for model_cache in self._cache.values():
        for v in model_cache.values():
          yield v
    else:
      for v in self._cache[model_id].values():
        yield v

  def items(
      self,
      model_id: str | None = None
      ) -> Iterator[tuple[str, base.LMCacheEntry]]:
    """Returns the cached items for a model."""
    if model_id is None:
      for model_cache in self._cache.values():
        for k, v in model_cache.items():
          yield k, v
    else:
      for k, v in self._cache[model_id].items():
        yield k, v

  def _get(self, model_id: str, key: Any) -> base.LMCacheEntry | None:
    """Returns a LM cache entry associated with the key."""
    return self._cache[model_id].get(key, None)

  def _put(self, model_id: str, key: Any, entry: base.LMCacheEntry) -> None:
    """Puts a LM cache entry associated with the key."""
    self._cache[model_id][key] = entry

  def reset(self, model_id: str | None = None) -> None:
    """Resets the cache."""
    if model_id is not None:
      self._cache[model_id].clear()
    else:
      self._cache.clear()

  def _sym_clone(self, deep: bool, memo: Any = None) -> 'InMemory':
    v = super()._sym_clone(deep, memo)
    v._cache = self._cache  # pylint: disable=protected-access
    return v

  def save(self, path: str | None = None) -> None:
    """Saves the in-memory cache."""
    if path is None:
      if self.filename is None:
        raise ValueError('`path` must be specified.')
      path = self.filename

    # Do nothing if there is no update, this avoids unnecessary rewrites.
    if self.stats.num_updates == 0 and path == self.filename:
      return

    records = []
    for model_id in self.model_ids():
      entries = [dict(k=k, v=v) for k, v in self.items(model_id)]
      records.append(dict(model_id=model_id, entries=entries))
    pg.save(records, path)


@contextlib.contextmanager
def lm_cache(filename: str | None = None) -> Iterator[InMemory]:
  """Context manager to enable cache for LMs under the context.

  If LMs under the context manager have explicitly specified cache, they will
  use their own cache. Otherwise they will use the cache created by the context
  manager.

  Args:
    filename: If not None, JSON file to load and save the cache.

  Yields:
    A cache object created.
  """
  cache = InMemory(filename)
  try:
    with lf.context(cache=cache):
      yield cache
  finally:
    if filename is not None:
      cache.save()
