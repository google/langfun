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
"""langfun LLM implementations."""

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top

# LMs for testing.
from langfun.core.llms.fake import Fake
from langfun.core.llms.fake import Echo
from langfun.core.llms.fake import StaticMapping
from langfun.core.llms.fake import StaticResponse
from langfun.core.llms.fake import StaticSequence

# Gemini models.
from langfun.core.llms.google_genai import GenAI
from langfun.core.llms.google_genai import GeminiPro
from langfun.core.llms.google_genai import GeminiProVision
from langfun.core.llms.google_genai import Palm2
from langfun.core.llms.google_genai import Palm2_IT

# OpenAI models.
from langfun.core.llms.openai import OpenAI

from langfun.core.llms.openai import Gpt4Turbo
from langfun.core.llms.openai import Gpt4Turbo_20240409
from langfun.core.llms.openai import Gpt4TurboPreview
from langfun.core.llms.openai import Gpt4TurboPreview_0125
from langfun.core.llms.openai import Gpt4TurboPreview_1106
from langfun.core.llms.openai import Gpt4VisionPreview
from langfun.core.llms.openai import Gpt4VisionPreview_1106
from langfun.core.llms.openai import Gpt4
from langfun.core.llms.openai import Gpt4_0613
from langfun.core.llms.openai import Gpt4_32K
from langfun.core.llms.openai import Gpt4_32K_0613

from langfun.core.llms.openai import Gpt35Turbo
from langfun.core.llms.openai import Gpt35Turbo_0125
from langfun.core.llms.openai import Gpt35Turbo_1106
from langfun.core.llms.openai import Gpt35Turbo_0613
from langfun.core.llms.openai import Gpt35Turbo16K
from langfun.core.llms.openai import Gpt35Turbo16K_0613


from langfun.core.llms.openai import Gpt35

from langfun.core.llms.openai import Gpt3
from langfun.core.llms.openai import Gpt3Curie
from langfun.core.llms.openai import Gpt3Babbage
from langfun.core.llms.openai import Gpt3Ada

from langfun.core.llms.anthropic import Anthropic
from langfun.core.llms.anthropic import Claude3Opus
from langfun.core.llms.anthropic import Claude3Sonnet
from langfun.core.llms.anthropic import Claude3Haiku


# LLaMA C++ models.
from langfun.core.llms.llama_cpp import LlamaCppRemote

# Placeholder for Google-internal imports.

# Include cache as sub-module.
from langfun.core.llms import cache

# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order
# pylint: enable=g-importing-member
