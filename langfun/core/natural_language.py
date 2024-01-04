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
"""Natural language utilities."""

import abc
import pyglove as pg


class NaturalLanguageFormattable(pg.Formattable):
  """Base class for natural language formattable objects."""

  @abc.abstractmethod
  def natural_language_format(self) -> str:
    """Returns the natural language format of current object."""

  def format(
      self,
      *args,
      natural_language: bool = False,
      **kwargs
  ) -> str:
    if natural_language:
      return self.natural_language_format()
    return super().format(*args, **kwargs)

  def __str__(self):
    return self.natural_language_format()
