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
"""Video modality."""

import functools
from langfun.core.modalities import mime


class Video(mime.Mime):
  """Represents a video for communicating with language models.

  `lf.Video` can be initialized from a URI (HTTP/HTTPS URL or local path)
  using `lf.Video.from_uri()` or from raw bytes using `lf.Video.from_bytes()`.

  **Example:**

  ```python
  import langfun as lf

  # Load video from path
  video = lf.Video.from_path('/path/to/video.mp4')

  # Use video in a prompt
  prompt = lf.Template(
      'What is happening in this video? {{video}}', video=video
  )
  response = lf.query(prompt, lm=lf.llms.Gemini25Flash())
  print(response)
  ```
  """

  MIME_PREFIX = 'video'

  @functools.cached_property
  def video_format(self) -> str:
    return self.mime_type.removeprefix(self.MIME_PREFIX + '/')

  def _mime_control_for(self, uri: str) -> str:
    return f'<video controls> <source src="{uri}"> </video>'
