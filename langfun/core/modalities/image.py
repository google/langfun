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
from typing import Any

from langfun.core.modalities import mime

try:
  from PIL import Image as pil_image    # pylint: disable=g-import-not-at-top
  PILImage = pil_image.Image
  pil_open = pil_image.open
except ImportError:
  PILImage = Any

  def pil_open(*unused_args, **unused_kwargs):
    raise RuntimeError(
        'Please install "langfun[mime-pil]" to enable PIL image support.'
    )


class Image(mime.Mime):
  """Image."""

  MIME_PREFIX = 'image'

  @functools.cached_property
  def image_format(self) -> str:
    return self.mime_type.removeprefix(self.MIME_PREFIX + '/')

  def _mime_control_for(self, uri: str) -> str:
    return f'<img src="{uri}">'

  @functools.cached_property
  def size(self) -> tuple[int, int]:
    img = pil_open(io.BytesIO(self.to_bytes()))
    return img.size

  def to_pil_image(self) -> PILImage:   # pytype: disable=invalid-annotation
    return pil_open(io.BytesIO(self.to_bytes()))

  @classmethod
  def from_pil_image(cls, img: PILImage) -> 'Image':  # pytype: disable=invalid-annotation
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return cls.from_bytes(buf.getvalue())
