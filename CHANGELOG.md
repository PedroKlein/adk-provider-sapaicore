# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-07-08

### Added

- Seed support: extract `Seed` from `GenerateContentConfig` for deterministic outputs
- TopK sampling: extract `TopK` from config, forwarded as `top_k` to models that support it
- Logprobs in response: `ResponseLogprobs` + `Logprobs` config fields now sent in requests; response logprobs parsed and returned as `LLMResponse.LogprobsResult` (both streaming and non-streaming)
- Streaming logprobs aggregation: per-chunk logprobs accumulated and populated in the final streamed response
- Token ID mapping: `TokenID` from OpenAI logprobs responses mapped to `genai.LogprobsResultCandidate.TokenID`
- Tool choice: `ToolConfig.FunctionCallingConfig` mapped to OpenAI `tool_choice` format (auto/none/required/named function) in both modes
- `WithModelParams` precedence documented: extra params override first-class config fields when keys collide
- `mise run fix` task for auto-fixing lint issues locally
- Smoke tests: seed determinism (contrast), logprobs (contrast + streaming), tool choice (contrast), topK
- Foundation mode provider helper for smoke tests (`newFoundationProvider`)

## [0.1.0] - 2025-07-08

### Added

- `Provider` with functional options pattern (`WithEndpoint`, `WithAuth`, `WithDeploymentID`, `WithDeployments`, etc.)
- Orchestration mode: single deployment routes to all models via SAP AI Core harmonized API
- Orchestration auto-discovery via `WithOrchestration()` — queries the deployments API at init time
- Foundation-models mode: per-model deployment IDs with direct OpenAI-compatible API
- Streaming (SSE) support for both modes with chunk aggregation
- Tool/function calling support (request and response round-trips)
- `WithModelParams` for provider-specific features (extended thinking, reasoning effort, logprobs)
- `ExtraParams` forwarding in both modes (merged into top-level JSON for foundation, into model.params for orchestration)
- OAuth2 client credentials flow with thread-safe token caching and automatic refresh
- `WithResourceGroup`, `WithHTTPClient`, `WithHeaders`, `WithTimeout`, `WithMaxRetries` for advanced configuration
- `NewProvider` accepts `context.Context` for initialization-time HTTP calls (e.g. deployment discovery)
- JSON schema response format (`application/json` MIME type with structured output)
- Sentinel errors: `ErrMissingConfig`, `ErrDeploymentNotFound`, `ErrDiscovery`, `ErrInference`
- Comprehensive unit tests and integration smoke tests against live SAP AI Core
- golangci-lint v2 configuration with strict linting rules
- mise tasks for local development (`mise run check`, `mise run smoke`)
