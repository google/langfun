# Langfun Repository Map

## Project Overview
- **Name**: Langfun (Language as Functions)
- **Description**: OO for LLMs - PyGlove-powered library making language models fun to work with
- **License**: Apache-2.0
- **Repository**: https://github.com/google/langfun
- **Total Python LOC**: ~69,750 lines

## Module Structure

### `/langfun` (root package)
- `__init__.py` - Public API exports, version 0.1.2
- Core components: structured, llms, eval, coding, templates, agentic, mcp, memories, modalities

### `/langfun/core/` - Core Framework
- **`language_model.py`** (1907 LOC) - Base LanguageModel class, ModelInfo, rate limits, pricing
- **`message.py`** (1190 LOC) - Message class for prompts/responses
- **`template.py`** (889 LOC) - Template system for prompt composition
- **`concurrent.py`** (1133 LOC) - Concurrent execution utilities
- **`component.py`** - Component base class
- **`embedding_model.py`** - Embedding model interface

#### `/langfun/core/llms/` - LLM Provider Integrations
| File | LOC | Description |
|------|-----|-------------|
| `openai.py` | 1365 | OpenAI models (GPT-4, GPT-5, o1, o3, o4, etc.) |
| `anthropic.py` | 1246 | Anthropic models (Claude 3.5, Opus 4, etc.) |
| `gemini.py` | 1045 | Google Gemini models |
| `vertexai.py` | 836 | Vertex AI integration |
| `openai_compatible.py` | ~450 | OpenAI-compatible API support |
| `google_genai.py` | ~330 | Google Generative AI |
| `groq.py` | 402 | Groq integration |
| `deepseek.py` | ~200 | DeepSeek models |
| `azure_openai.py` | ~200 | Azure OpenAI |
| `rest.py` | ~350 | REST API base class |
| `fake.py` | ~200 | Fake/mock LLM for testing |
| `veo.py` | ~500 | Veo video generation |
| `llama_cpp.py` | ~150 | Ollama/Llama.cpp |
| `compositional.py` | ~200 | Compositional model wrapper |

#### `/langfun/core/structured/` - Structured Prompting
- **`querying.py`** (1360 LOC) - Query interface and execution
- **`schema/`** - Schema definitions (JSON, Python, base)
- **`parsing.py`** (420 LOC) - Output parsing
- **`completion.py`** - Text completion
- **`mapping.py`** (514 LOC) - Schema mapping
- **`function_generation.py`** - Function generation from schemas
- **`description.py`** - Object description generation

#### `/langfun/core/eval/` - Evaluation Framework
- **`base.py`** (2402 LOC) - Core evaluation infrastructure
- **`v2/experiment.py`** (1167 LOC) - Experiment tracking
- **`v2/evaluation.py`** (965 LOC) - Evaluation execution
- **`v2/metrics.py`** - Metrics computation
- **`v2/checkpointing.py`** - Experiment checkpointing
- **`v2/progress.py`** - Progress tracking
- **`v2/runners/`** - Execution runners

#### `/langfun/core/agentic/` - Agent Framework
- **`action.py`** (2398 LOC) - Action base class, multi-sampling
- **`action_test.py`** - Action tests

#### `/langfun/core/coding/` - Code Execution
- `python/` - Python sandbox execution

#### `/langfun/core/templates/` - Template Utilities
- Various templating utilities

#### `/langfun/core/data/` - Data Handling
- **`conversion/`** - Format conversion (Gemini, etc.)

#### `/langfun/core/modalities/` - Multi-Modal Support
- **`mime.py`** (372 LOC) - MIME type handling
- **`image.py`** - Image modality
- **`video.py`** - Video modality
- **`pdf.py`** - PDF modality

#### `/langfun/core/mcp/` - MCP Protocol Support
- `testing/` - MCP testing utilities

#### `/langfun/core/memories/` - Memory/Context Management

### `/langfun/assistant/` - Assistant Module

### `/langfun/env/` - Environment/Sandbox
- **`base_environment.py`** (827 LOC) - Environment interface
- **`base_sandbox.py`** (842 LOC) - Sandbox execution
- **`interface.py`** (1657 LOC) - User interface
- **`event_handlers/`** - Event logging, metric writing

## Key Dependencies
- **pyglove>=0.5.0.dev202510170226** - Core dependency (note: requires dev version)
- **jinja2>=3.1.2** - Templating
- **requests>=2.31.0** - HTTP
- **anyio>=4.7.0** - Async I/O
- **puremagic>=1.20** - MIME type detection
- **mcp>=1.17.0** - MCP protocol

## Extras Dependencies
- `vertexai`: google-auth>=2.16.0
- `mime-pil`: pillow>=10.0.0
- `ui`: termcolor==1.1.0, tqdm>=4.64.1

## Supported LLM Providers
1. **OpenAI** - GPT-4, GPT-5, o1/o2/o3/o4 series
2. **Anthropic** - Claude 3.5, Opus 4, Sonnet 4
3. **Google Gemini** - Gemini 1.5, 2.0, 2.5
4. **Vertex AI** - Google cloud Vertex models
5. **Azure OpenAI** - Microsoft Azure OpenAI
6. **DeepSeek** - DeepSeek models
7. **Groq** - Groq Cloud
8. **OpenAI-Compatible** - Any REST API following OpenAI format
9. **Ollama/Llama.cpp** - Local models
10. **Veo** - Video generation

## CI/CD Pipeline
- GitHub Actions CI on push/PR to main
- Python versions: 3.11, 3.12, 3.13, 3.14
- Parallel testing with pytest-xdist
- Coverage reporting to Codecov

## Test Files
- ~97 test files across the project
- Major test suites in: language_model_test, message_test, concurrent_test, querying_test