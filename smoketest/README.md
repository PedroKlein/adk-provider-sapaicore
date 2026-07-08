# Smoke Tests

Integration tests that run against a live SAP AI Core instance. They verify the full request path: OAuth token exchange, orchestration deployment discovery, request building, streaming, and response parsing.

## Running

```bash
# Required for all tests
export AI_CORE_ENDPOINT="https://api.ai.xxx.aicore.cfapps.xxx.hana.ondemand.com"
export AI_CORE_CLIENT_ID="..."
export AI_CORE_CLIENT_SECRET="..."
export AI_CORE_AUTH_URL="https://xxx.authentication.xxx.hana.ondemand.com/oauth/token"

# Required for foundation-mode tests
export AI_CORE_FOUNDATION_DEPLOYMENT_ID="dXXXXXXXX"
export AI_CORE_FOUNDATION_MODEL="gpt-4.1-mini"  # optional, defaults to gpt-4.1-mini

# Required for explicit deployment ID test
export AI_CORE_DEPLOYMENT_ID="dXXXXXXXX"

# Required for resource group test
export AI_CORE_RESOURCE_GROUP="my-group"

# Run all smoke tests
go test -tags=smoke ./smoketest/ -v -timeout=5m

# Run a specific test
go test -tags=smoke ./smoketest/ -run TestSmoke_ToolCalling -v

# Enable the expensive >200K token test
SMOKE_LARGE_CONTEXT=1 go test -tags=smoke ./smoketest/ -run TestSmoke_Anthropic1MContext -v -timeout=10m
```

Tests skip automatically when their required environment variables are missing.

## Test Files

### `smoke_basic_test.go`

Core functionality across all three providers (GPT-4.1-mini, Claude 4.5 Sonnet, Gemini 2.5 Flash):

| Test | What it verifies |
|------|-----------------|
| `NonStreaming` | Basic request/response cycle, usage metadata returned |
| `Streaming` | Partial chunks arrive, final response aggregates correctly |
| `ModelParams` | `WithModelParams` caps output (max_tokens=50 truncates) |
| `MultiTurn` | Conversation history preserved across messages |
| `StopSequences` | Model stops generating at the specified sequence |
| `ResponseFormat_JSON` | Structured output with JSON schema constraint |

### `smoke_tools_test.go`

Tool calling and ADK integration patterns:

| Test | What it verifies |
|------|-----------------|
| `ToolCalling` | Single function call with correct name and arguments |
| `ToolCalling_Streaming` | Multiple tool calls assembled from streamed deltas |
| `ToolRoundTrip` | Full cycle: user question, tool call, tool result, final answer |
| `ToolCalling_MultiModel` | Same tool works across GPT, Claude, and Gemini |
| `ADK_BeforeModelCallback` | `req.Model` override routes to a different model at runtime |
| `CustomHeaders` | Extra HTTP headers don't break the request |
| `ADK_AgentStyleUsage` | Two-step agent loop simulating real ADK agent behavior |

### `smoke_advanced_test.go`

Provider-specific features and edge cases:

| Test | What it verifies |
|------|-----------------|
| `ExtendedThinking` | Claude's thinking mode produces correct math results |
| `ExtendedThinking_Streaming` | Thinking + streaming combined |
| `Anthropic1MContext` | 1M context beta accepts >200K prompt tokens (opt-in, expensive) |
| `TimeoutAndRetries` | Server-side timeout/retry config accepted |
| `ErrorHandling_InvalidModel` | Invalid model name produces an error (or graceful fallback) |
| `ParamForwarding_Logprobs` | `WithModelParams` logprobs param accepted |
| `ParamForwarding_ReasoningEffort` | `WithModelParams` reasoning_effort accepted |
| `SeedDeterminism` | Contrast: same seed = same output, different seed = different output (orchestration) |
| `Logprobs_InResponse` | Contrast: without logprobs → nil, with → populated with top candidates (foundation) |
| `ToolChoice_Required` | Contrast: without tool_choice → text, with required → forced tool call (foundation) |
| `TopK` | `top_k` param accepted by Gemini via orchestration |
| `Logprobs_Orchestration` | Logprobs work via orchestration model.params |
| `Logprobs_Streaming` | Streaming aggregates per-chunk logprobs into final response (foundation) |
| `ToolChoice_Orchestration` | Contrast: same as foundation test, via orchestration model.params |
| `Seed_Foundation` | Same seed determinism contrast test in foundation mode |

### `smoke_providers_test.go`

Alternative provider configurations and remaining API surface:

| Test | What it verifies |
|------|-----------------|
| `FoundationMode_NonStreaming` | Foundation-models mode with per-model deployment IDs |
| `FoundationMode_Streaming` | Streaming in foundation mode |
| `FoundationMode_ToolCalling` | Tool calls work in foundation mode |
| `WithDeploymentID` | Explicit orchestration deployment ID (no auto-discovery) |
| `WithResourceGroup` | Custom resource group header |
| `WithHTTPClient` | Custom `*http.Client` injection |
| `Refusal` | Model refuses harmful content (refusal field or safety finish) |
| `StreamingUsageMetadata` | Streaming final response includes token counts |

### `smoke_modules_test.go`

Orchestration modules with contrast assertions proving each module is active:

| Test | What it verifies |
|------|------------------|
| `Filtering_BlocksHarmfulInput` | Harmful prompt blocked with filtering, would succeed without |
| `Masking_RedactsPII` | Contrast: email/phone absent from response with masking, present without |
| `Translation_TranslatesOutput` | Contrast: response in German with translation, English without |
| `Fallback_RecoverFromInvalidModel` | Invalid primary model succeeds with fallback, errors without |
| `PromptCaching_AnthropicSucceeds` | cache_control annotation accepted by Claude API |

### `smoke_multimodal_test.go`

Multi-modal input and advanced masking strategies:

| Test | What it verifies |
|------|------------------|
| `ImageInput_InlineData` | 50x50 red PNG recognized by GPT, Claude, and Gemini (answers "red") |
| `ImageInput_Streaming` | Image + streaming produces partial chunks and correct final answer |
| `FileInput_PDF` | Minimal PDF with "BANANA" read correctly by Claude and Gemini |
| `ImageInput_FileDataURL` | HTTPS URL passed to model (skips if deployment doesn't support external URLs) |
| `FabricatedMasking_RedactsPII` | `FabricatedEntity` accepted by API; anonymization mode hides PII, pseudonymization unmasks |
| `FileInput_WithMasking` | PDF + masking with `MaskFileSkip` — model still reads file content |

## Helpers (`helpers_test.go`)

Shared test utilities that keep each test body short:

- `newProvider(t)` - creates a provider with auto-discovery, skips if env vars missing
- `newFoundationProvider(t)` - creates a foundation-mode provider, skips if `AI_CORE_FOUNDATION_DEPLOYMENT_ID` missing
- `withTimeout(t, d)` - returns a context with deadline (prevents hanging on slow APIs)
- `generateOne(t, ctx, llm, req)` - non-streaming call, fails test on error
- `generateStream(t, ctx, llm, req)` - streaming call, returns partials + final
- `requireText(t, resp)` - extracts text or fails
- `requireFunctionCalls(t, resp)` - extracts function calls or fails
- `simpleReq(text)` - one-liner user message

## Adding Tests

1. Pick the file that matches the concern (basic, tools, advanced, providers, modules)
2. Use the helpers to keep tests under 30 lines
3. Set a timeout with `withTimeout(t, ...)` appropriate for the operation
4. Use `t.Logf` for useful debug output (shown with `-v`)
5. Prefix test names with `TestSmoke_`
6. For tests that need extra env vars, use `envOrSkip` so they skip gracefully
7. Module tests use the **contrast pattern**: test with module vs without, assert observable difference
