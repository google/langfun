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

# REST-based models.
from langfun.core.llms.rest import REST

# Gemini models.
from langfun.core.llms.google_genai import GenAI
from langfun.core.llms.google_genai import GeminiFlash1_5
from langfun.core.llms.google_genai import GeminiPro
from langfun.core.llms.google_genai import GeminiPro1_5
from langfun.core.llms.google_genai import GeminiProVision
from langfun.core.llms.google_genai import Palm2
from langfun.core.llms.google_genai import Palm2_IT

# OpenAI models.
from langfun.core.llms.openai import OpenAI

from langfun.core.llms.openai import GptO1Preview
from langfun.core.llms.openai import GptO1Preview_20240912
from langfun.core.llms.openai import GptO1Mini
from langfun.core.llms.openai import GptO1Mini_20240912

from langfun.core.llms.openai import Gpt4oMini
from langfun.core.llms.openai import Gpt4oMini_20240718
from langfun.core.llms.openai import Gpt4o
from langfun.core.llms.openai import Gpt4o_20240806
from langfun.core.llms.openai import Gpt4o_20240513

from langfun.core.llms.openai import Gpt4Turbo
from langfun.core.llms.openai import Gpt4Turbo_20240409
from langfun.core.llms.openai import Gpt4TurboPreview
from langfun.core.llms.openai import Gpt4TurboPreview_20240125
from langfun.core.llms.openai import Gpt4TurboPreview_20231106
from langfun.core.llms.openai import Gpt4VisionPreview
from langfun.core.llms.openai import Gpt4VisionPreview_20231106
from langfun.core.llms.openai import Gpt4
from langfun.core.llms.openai import Gpt4_20230613

from langfun.core.llms.openai import Gpt4_32K
from langfun.core.llms.openai import Gpt4_32K_20230613

from langfun.core.llms.openai import Gpt35Turbo
from langfun.core.llms.openai import Gpt35Turbo_20240125
from langfun.core.llms.openai import Gpt35Turbo_20231106
from langfun.core.llms.openai import Gpt35Turbo_20230613
from langfun.core.llms.openai import Gpt35Turbo16K
from langfun.core.llms.openai import Gpt35Turbo16K_20230613

# For backward compatibility.
Gpt4TurboPreview_0125 = Gpt4TurboPreview_20240125
Gpt4TurboPreview_1106 = Gpt4TurboPreview_20231106
Gpt4VisionPreview_1106 = Gpt4VisionPreview_20231106
Gpt4_0613 = Gpt4_20230613
Gpt4_32K_0613 = Gpt4_32K_20230613
Gpt35Turbo_0125 = Gpt35Turbo_20240125
Gpt35Turbo_1106 = Gpt35Turbo_20231106
Gpt35Turbo_0613 = Gpt35Turbo_20230613
Gpt35Turbo16K_0613 = Gpt35Turbo16K_20230613

from langfun.core.llms.openai import Gpt35

from langfun.core.llms.openai import Gpt3
from langfun.core.llms.openai import Gpt3Curie
from langfun.core.llms.openai import Gpt3Babbage
from langfun.core.llms.openai import Gpt3Ada

from langfun.core.llms.anthropic import Anthropic
from langfun.core.llms.anthropic import Claude35Sonnet
from langfun.core.llms.anthropic import Claude3Opus
from langfun.core.llms.anthropic import Claude3Sonnet
from langfun.core.llms.anthropic import Claude3Haiku

from langfun.core.llms.groq import Groq
from langfun.core.llms.groq import GroqLlama3_70B
from langfun.core.llms.groq import GroqLlama3_8B
from langfun.core.llms.groq import GroqLlama2_70B
from langfun.core.llms.groq import GroqMistral_8x7B
from langfun.core.llms.groq import GroqGemma7B_IT

from langfun.core.llms.vertexai import VertexAI
from langfun.core.llms.vertexai import VertexAIGeminiPro1_5
from langfun.core.llms.vertexai import VertexAIGeminiPro1_5_0514
from langfun.core.llms.vertexai import VertexAIGeminiPro1_5_0409
from langfun.core.llms.vertexai import VertexAIGeminiFlash1_5
from langfun.core.llms.vertexai import VertexAIGeminiFlash1_5_0514
from langfun.core.llms.vertexai import VertexAIGeminiPro1
from langfun.core.llms.vertexai import VertexAIGeminiPro1Vision
from langfun.core.llms.vertexai import VertexAIPalm2
from langfun.core.llms.vertexai import VertexAIPalm2_32K
from langfun.core.llms.vertexai import VertexAICustom


# LLaMA C++ models.
from langfun.core.llms.llama_cpp import LlamaCppRemote

# Placeholder for Google-internal imports.

# Include cache as sub-module.
from langfun.core.llms import cache

# pylint: enable=g-import-not-at-top
# pylint: enable=g-bad-import-order
# pylint: enable=g-importing-member
