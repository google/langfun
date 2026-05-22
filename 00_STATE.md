# Langfun Repository Analysis State

## Repository Info
- **Upstream**: https://github.com/google/langfun
- **Fork**: https://github.com/okwn/langfun
- **License**: Apache-2.0
- **Status**: Active (not archived)
- **Topics**: framework, llms, nlp

## Fork Status
- Fork created: 2026-05-22
- Fork sync status: In sync with upstream main (no local changes)

## Local Clone
- **Path**: /root/oss-pr-campaign/repos/langfun
- **Current branch**: main
- **Upstream remote**: upstream -> google/langfun

## Python Environment Issue
- Python 3.12 system pip installs pyglove 0.4.5
- requirements.txt requires pyglove>=0.5.0.dev202510170226
- Incompatibility prevents full import due to Modality type annotation changes
- Tests cannot be run without compatible pyglove version

## CI/CD
- CI workflow: .github/workflows/ci.yaml
- Python versions tested: 3.11, 3.12, 3.13, 3.14
- Test command: pytest -n auto -p no:threadexception --cov=langfun --cov-report=xml -vv

## Quick Stats
- Total Python files: ~97 test files + source files
- Largest modules: eval/base (2402 lines), agentic/action (2398 lines), language_model (1907 lines)
- LLM providers: OpenAI, Anthropic, Google Gemini, VertexAI, Azure OpenAI, DeepSeek, Groq, Ollama/LlamaCPP, Veo, OpenAI-Compatible

## Issues & PRs from Upstream
- Open Issues: 30
- Open PRs: 27

## Next Steps
1. Create 01_REPO_MAP.md with detailed module structure
2. Run baseline tests when compatible pyglove is available
3. Query issues/PRs for PR candidates
4. Quality audit
5. Create 05_PR_CANDIDATES.md and 06_SELECTED_5_PR_PLAN.md
