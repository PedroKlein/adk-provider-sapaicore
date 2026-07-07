# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-07

### Added

- `Provider` with functional options pattern (`WithEndpoint`, `WithAuth`, `WithDeploymentID`, `WithDeployments`, etc.)
- Orchestration mode: single deployment routes to all models via SAP AI Core harmonized API
- Foundation-models mode: per-model deployment IDs with direct OpenAI-compatible API
- Streaming (SSE) support for both modes
- Tool/function calling support (request and response)
- `WithModelParams` for provider-specific features (extended thinking, reasoning effort, 1M context)
- OAuth2 client credentials flow with thread-safe token caching
- `WithResourceGroup`, `WithHTTPClient`, `WithHeaders` for advanced configuration
- Sentinel errors: `ErrMissingConfig`, `ErrDeploymentNotFound`, `ErrTokenRefresh`, `ErrInference`
