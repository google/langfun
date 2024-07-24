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

import functools
import io

from langfun.core.modalities import mime
from PIL import Image as pil_image


class Image(mime.Mime):
  """Image."""

  MIME_PREFIX = 'image'

  @functools.cached_property
  def image_format(self) -> str:
    return self.mime_type.removeprefix(self.MIME_PREFIX + '/')

  def _html(self, uri: str) -> str:
    return f'<img src="{uri}">'

  @functools.cached_property
  def size(self) -> tuple[int, int]:
    img = pil_image.open(io.BytesIO(self.to_bytes()))
    return img.size

  def to_pil_image(self) -> pil_image.Image:
    return pil_image.open(io.BytesIO(self.to_bytes()))

  @classmethod
  def from_pil_image(cls, img: pil_image.Image) -> 'Image':
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return cls.from_bytes(buf.getvalue())
