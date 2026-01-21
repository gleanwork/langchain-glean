## v0.3.4 (2026-01-21)

### Fix

- Use vars() instead of model_dump() for Speakeasy SDK models
- Update SDK method calls for glean-api-client 0.11.x

### Refactor

- Remove unused stream parameter from GleanRunAgentTool

## v0.3.3 (2025-07-18)

### Fix

- Ensure `act_as` and `GLEAN_ACT_AS` are propagated (#20)

## v0.3.2 (2025-06-25)

### Fix

- Update tests with new SDK import path (#17)

## v0.3.1 (2025-05-17)

### Fix

- Upgrades api-client to 0.4.2 (#14)

## v0.3.0 (2025-05-14)

### Feat

- Adding support for Agent APIs via Tools and ChatModel (#12)
- Adding GleanToolkit

### Fix

- Fixes incorrect call pattern with api-client (#13)

### Refactor

- Extracts auth and client configuration to a mixin

## v0.2.1 (2025-03-31)

### Fix

- ensure version files are properly synchronized

## v0.2.0 (2025-03-25)

### Feat

- Adds better error handling for retriever and tool
- Implements GleanSearchRetriever and GleanSearchTool

### Fix

- Fixing publish workflow
- Updating default for act_as to be empty string vs. None
- Ensure uv is executing python via venv
- Fixing imports (they were failing the check imports task
- Updating README.md to correct examples
- Fixing retriever type errors
- Fixing mypy errors
- Fixing actions workflows to use correct uv action
- Updating README.md and adding CONTRIBUTING.md
