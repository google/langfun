# Copyright 2025 The Langfun Authors
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
"""langfun embedding models."""

# pylint: disable=g-bad-import-order
# pylint: disable=unused-import

from langfun.core.ems.rest import REST

from langfun.core.ems.openai import OpenAI
from langfun.core.ems.openai import TextEmbedding3Large
from langfun.core.ems.openai import TextEmbedding3Small
from langfun.core.ems.openai import TextEmbeddingAda002

from langfun.core.ems.vertexai import VertexAI
from langfun.core.ems.vertexai import VertexAIPredictAPI
from langfun.core.ems.vertexai import VertexAIGeminiEmbedding2
from langfun.core.ems.vertexai import VertexAIGeminiEmbedding1
from langfun.core.ems.vertexai import VertexAITextEmbedding005
from langfun.core.ems.vertexai import VertexAITextMultilingualEmbedding002

# pylint: enable=g-bad-import-order
# pylint: enable=unused-import
