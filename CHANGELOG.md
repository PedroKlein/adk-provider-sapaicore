# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-08

### Added

- Content filtering: `WithFiltering(cfg)` enables Azure Content Safety and/or Llama Guard 3 8B on input/output. Pass nil for strict defaults (ALLOW_SAFE on all categories + prompt_shield)
- Data masking: `WithMasking(cfg)` enables SAP DPI anonymization/pseudonymization. Supports 25 standard entity types, custom regex entities, and allowlists. `CommonPIIEntities` preset for quick setup
- Translation: `WithTranslation(cfg)` enables SAP Document Translation on input and/or output
- Module fallback: `WithFallback(models...)` sends a modules array so the service tries each model in order until one succeeds
- Prompt caching: `WithPromptCaching(ttl...)` adds Anthropic `cache_control` annotations to system messages and tool definitions. Default 5m TTL, optional `CacheTTL1h` for supported models
- Module composition: provider-level defaults + model-level overrides. Same module at both levels → model replaces provider. `WithoutFiltering()`, `WithoutMasking()`, `WithoutTranslation()` for explicit opt-out
- Global stream options: `WithStreamOptions(opts)` for chunk_size and delimiters (required for translation + streaming)
- Validation: foundation mode + orchestration modules returns `ErrMissingConfig` at `Model()` time. Empty translation config or empty masking entities also error
- Smoke tests: filtering contrast, masking PII redaction, translation output, fallback recovery, prompt caching acceptance

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
