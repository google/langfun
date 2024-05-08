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
"""Image modality."""

import imghdr
from typing import cast
from langfun.core.modalities import mime


class Image(mime.MimeType):
  """Base class for image."""

  @property
  def image_format(self) -> str:
    iformat = imghdr.what(None, self.to_bytes())
    if iformat not in ['png', 'jpeg']:
      raise ValueError(f'Unsupported image format: {iformat!r}.')
    return cast(str, iformat)

  @property
  def mime_type(self) -> str:
    return f'image/{self.image_format}'

  def _repr_html_(self) -> str:
    if self.uri and self.uri.lower().startswith(('http:', 'https:', 'ftp:')):
      return f'<img src="{self.uri}">'
    return f'<img src="{self.content_uri}">'
