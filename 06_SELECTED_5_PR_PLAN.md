# Langfun Top 5 PR Selection and Plan

## Selected PRs for Contribution

### 1. [CRITICAL] Security Fix: Unsandboxed Code Execution (PR #692)
**Issue**: #693  
**Link**: https://github.com/google/langfun/pull/692  
**Priority**: CRITICAL  
**Type**: Security Bug  

**Summary**: PythonFunction.implementation calls execution.run() without sandbox parameter, allowing arbitrary code execution.  
**Why**: High-security impact, directly addresses a reported vulnerability.  
**Difficulty**: Medium  
**Action**: Review, test, and help merge the existing PR #692.

---

### 2. Gemini toolConfig Function Calling Fix (PR #711)
**Issue**: Tracked in PR #711  
**Link**: https://github.com/google/langfun/pull/711  
**Priority**: HIGH  
**Type**: Bug Fix  

**Summary**: Gemini base class hardcodes `toolConfig.functionCallingConfig.mode='NONE'`, breaking function calling. PR #711 changes default to 'AUTO' when tools are present.  
**Why**: Improves Gemini functionality significantly, unlocks tool use by default.  
**Difficulty**: Medium  
**Action**: Review, test, and merge PR #711.

---

### 3. Groq Pricing Fix (Code Contribution)
**Link**: `langfun/core/llms/groq.py` lines 182, 201  
**Priority**: MEDIUM  
**Type**: Bug  

**Summary**: TODO comments indicate pricing is not properly computed based on token counts.  
**Why**: Correct pricing is essential for cost tracking and eval accuracy.  
**Difficulty**: Medium  
**Action**: Implement proper token-count-based pricing for Groq models.

---

### 4. Template.from_value Modalities Fix (PR #702)
**Issue**: Tracked in PR #702  
**Link**: https://github.com/google/langfun/pull/702  
**Priority**: MEDIUM  
**Type**: Bug  

**Summary**: Template.from_value doesn't preserve _referred_modalities in all branches.  
**Why**: Modality tracking is important for multi-modal workflows.  
**Difficulty**: Easy-Medium  
**Action**: Review and test PR #702.

---

### 5. GitHub Actions Upgrade (PRs #667, #668)
**Links**: 
- https://github.com/google/langfun/pull/667
- https://github.com/google/langfun/pull/668  
**Priority**: MEDIUM  
**Type**: CI/CD Maintenance  

**Summary**: Upgrade actions/checkout to v6, actions/setup-python to v6, codecov to v5 for Node 24 compatibility.  
**Why**: Outdated actions will break when Node 20 reaches EOL in April 2026.  
**Difficulty**: Easy (mostly version bumps)  
**Action**: Review and merge the existing PRs.

---

## PR Merge Priority Order

| Priority | PR # | Action |
|----------|------|--------|
| 1 | #692 | Review/security merge |
| 2 | #711 | Review and merge |
| 3 | #702 | Review and merge |
| 4 | #667, #668 | Review and merge |
| 5 | #690 (MiniMax) | Test and merge if good |

## Fork Sync Note
- Fork is currently in sync with upstream main
- Local branch `main` matches `upstream/main`
- All PRs target `upstream/main`

## Notes
- Python 3.12 environment has pyglove version incompatibility (0.4.5 installed vs 0.5.0.dev required)
- Tests cannot be fully run locally until pyglove issue is resolved
- CI runs on Python 3.11, 3.12, 3.13, 3.14 with pytest-xdist