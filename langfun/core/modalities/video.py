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

import base64
from typing import cast

from langfun.core.modalities import mime
import magic


class Video(mime.MimeType):
  """Base class for Video."""

  @property
  def video_format(self) -> str:
    return cast(str, self.mime_type.lstrip('video/'))

  @property
  def mime_type(self) -> str:
    video_mime_type = magic.from_buffer(self.to_bytes(), mime=True)
    if 'video/' not in video_mime_type:
      raise ValueError(f'Not a video: {video_mime_type!r}.')
    return video_mime_type

  def _repr_html_(self) -> str:
    if self.uri and self.uri.lower().startswith(('http:', 'https:', 'ftp:')):
      return f'<video controls> <source src="{self.uri}"> </video>'
    video_raw = base64.b64encode(self.to_bytes()).decode()
    return (
        '<video controls> <source'
        f' src="data:video/{self.video_format};base64,{video_raw}"> </video>'
    )
