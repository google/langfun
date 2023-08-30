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
"""Console utilities."""

from typing import Any
from langfun.core.text_formatting import colored


def write(
    value: Any,
    *,
    title: str | None = None,
    color: str | None = None,
    background: str | None = None,
    styles: list[str] | None = None
) -> None:
  """Writes text to console.

  Args:
    value: A Python value to write to console. If not a string. `str(value)`
      will be called as the text to write.
    title: An optional title printed in bold.
    color: A string for text colors. Applicable values are: 'red', 'green',
      'yellow', 'blue', 'magenta', 'cyan', 'white'.
    background: A string for background colors. Applicable values are: 'red',
      'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    styles: A list of strings for applying styles on the text.
      Applicable values are: 'bold', 'dark', 'underline', 'blink', 'reverse',
      'concealed'.
  """
  # Print title if present.
  if title is not None:
    print(colored(title, styles=['bold']))

  # Print body.
  print(colored(str(value), color=color, background=background, styles=styles))
