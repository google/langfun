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
"""Text completion task."""

from typing import Annotated, Union
import langfun.core as lf


@lf.use_init_args(['prompt', 'lm_response'])
class Completion(lf.LangFunc):
  """LM-based text completion, this is base for all LM-based tasks.

  {{ prompt }}
  {% if response -%}
  {{ response }}
  {%- endif -%}
  """

  #
  # Template variables.
  #

  prompt: Annotated[
      lf.LangFunc,
      (
          'Prompt for completition.'
      )
  ]

  lm_response: Annotated[
      Union[None, str, lf.Message, lf.LangFunc],
      (
          '(Optional) LM rsesponse. This is the cached LM response when '
          '`cache_response` is set to True, or user provided response for '
          'training data generation.'
      ),
  ] = None

  # Task options.
  cache_response: bool = False

  @property
  def response(self) -> Union[None, str, lf.Message, lf.LangFunc]:
    """Formalized respone. Subclass could override."""
    return self.lm_response

  def __call__(self, **kwargs) -> lf.Message:
    lm_response = self.lm_response
    if lm_response is None:
      lm_response = super().__call__(**kwargs)
      if self.cache_response:
        self.rebind(lm_response=lm_response, skip_notification=True)
    return lm_response

  def clear_lm_response(self):
    """Clear LM response."""
    self.rebind(lm_response=None, skip_notification=True)
