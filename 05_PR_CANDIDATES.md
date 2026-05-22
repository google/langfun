# Langfun PR Candidates

## Overview
Analysis of open issues, pending PRs, and codebase quality signals to identify good PR candidates for contribution.

## Quality Audit Findings

### CI/CD Issues
- **Outdated GitHub Actions**: Uses `actions/checkout@v2` and `actions/setup-python@v1` (PRs #667, #668 to upgrade)
- **Outdated codecov**: Uses `codecov/codecov-action@v2` (PR #668 addresses this)
- Node 20 deprecated in favor of Node 24 (needs action updates)

### Code Quality Signals (TODO/FIXME)
27 TODO comments found across codebase. Key clusters:
- **Async implementations**: 8 TODOs for native async calling/sampling/scoring/tokenization/querying
- **Groq pricing**: 2 TODOs indicating pricing not properly computed from token counts
- **Audio support**: 2 TODOs for audio_input support in OpenAI conversion
- **Schema autofix**: 1 TODO for schema error autofix
- **Template parsing**: 2 TODOs about delaying template parsing

### Open Issues Needing Contributions

| # | Title | Priority | Type | Difficulty |
|---|-------|----------|------|------------|
| 693 | Security: Unsandboxed code execution in PythonFunction | **CRITICAL** | Security | Medium |
| 433 | Scoring implementation missing for specific LLMs | High | Feature | Medium |
| 380 | How does conversation history work? | Medium | Documentation | Easy |
| 25 | Required PG version too high | Medium | Dependency | Easy |

### Pending PRs (Good Reference/Contributions)

| # | Title | Status | Type |
|---|-------|--------|------|
| 712 | Enable code execution for Gemini models on Vertex AI | Open | Feature |
| 711 | Fix Gemini toolConfig.functionCallingConfig.mode='NONE' | Open | Bug |
| 702 | Fix Template.from_value to preserve _referred_modalities | Open | Bug |
| 690 | Add MiniMax as LLM provider | Open | Feature |
| 668 | Upgrade GitHub Actions to latest versions | Open | CI/CD |
| 667 | Upgrade GitHub Actions for Node 24 compatibility | Open | CI/CD |
| 627 | Add Markdown Prompting Protocol | Open | Feature |
| 621 | Fix ExecutionTrace.remove missing-item handling | Open | Bug |
| 614 | Adds markdown protocol for structured querying | Open | Feature |

### Strategic PR Candidates

1. **Security Fix**: Help review/merge PR #692 (security fix for unsandboxed code execution)
2. **Groq Pricing**: Implement proper token-based pricing (2 TODOs in groq.py)
3. **Audio Support**: Add audio_input support to OpenAI conversion (2 TODOs)
4. **Async Foundation**: Implement native async versions of core methods
5. **Scoring Fix**: Investigate and fix scoring for specific LLM providers

### Files to Examine for Contributions
- `langfun/core/llms/groq.py` - Pricing TODOs
- `langfun/core/data/conversion/openai.py` - Audio input TODOs
- `langfun/core/language_model.py` - 4 async TODOs
- `langfun/core/structured/*.py` - 4 async TODOs
- `langfun/core/agentic/action.py` - 1 async TODO
- `langfun/core/eval/v2/runners/base.py` - Background error rendering TODO