# adk-provider-sapaicore

ADK Go v2 model provider for [SAP AI Core](https://help.sap.com/docs/sap-ai-core/generative-ai/generative-ai-hub).

Implements [`model.LLM`](https://github.com/google/adk-go) so any ADK agent can use GPT, Claude, Gemini, Mistral, or any other model deployed on SAP AI Core without changing agent code.

## Install

```bash
go get github.com/PedroKlein/adk-provider-sapaicore
```

Requires Go 1.25+.

## Quick Start

The simplest setup uses orchestration mode with auto-discovery (the default):

```go
provider, err := sapaicore.NewProvider(
    sapaicore.WithEndpoint(os.Getenv("AI_CORE_ENDPOINT")),
    sapaicore.WithAuth(
        os.Getenv("AI_CORE_CLIENT_ID"),
        os.Getenv("AI_CORE_CLIENT_SECRET"),
        os.Getenv("AI_CORE_AUTH_URL"),
    ),
)
```

Then pass the model to any ADK agent:

```go
llm, _ := provider.Model("gpt-4.1")

agent := llmagent.New(llmagent.Config{
    Name:        "my-agent",
    Model:       llm,
    Instruction: "You are a helpful assistant.",
})

r := runner.New(runner.Config{AppName: "my-app", Agent: agent})

for event, err := range r.Run(ctx, userID, sessionID, userMsg) {
    // handle events
}
```

Any model available on your SAP AI Core instance works by name:

```go
provider.Model("gpt-4.1-mini")
provider.Model("anthropic--claude-4.5-sonnet")
provider.Model("gemini-2.5-flash")
provider.Model("o4-mini")
provider.Model("mistralai--mistral-large-instruct")
provider.Model("deepseek-r1-0528")
```

## Streaming

Streaming works through the standard ADK interface. The provider yields partial text chunks followed by a final aggregated response:

```go
for resp, err := range llm.GenerateContent(ctx, req, true) {
    if resp.Partial {
        fmt.Print(resp.Content.Parts[0].Text) // incremental text
        continue
    }
    // Final response with full text, usage metadata, and finish reason
    fmt.Println(resp.Content.Parts[0].Text)
    fmt.Println(resp.UsageMetadata.PromptTokenCount)
}
```

## Tool Calling

Define tools using standard ADK/genai types. The provider handles the full cycle: function call requests from the model, and function results sent back.

```go
llm, _ := provider.Model("gpt-4.1-mini")

req := &model.LLMRequest{
    Contents: []*genai.Content{
        {Parts: []*genai.Part{{Text: "What's the weather in Berlin?"}}, Role: "user"},
    },
    Config: &genai.GenerateContentConfig{
        Tools: []*genai.Tool{{
            FunctionDeclarations: []*genai.FunctionDeclaration{{
                Name:        "get_weather",
                Description: "Get current weather for a city",
                Parameters:  &genai.Schema{Type: "OBJECT", Properties: map[string]*genai.Schema{
                    "city": {Type: "STRING"},
                }, Required: []string{"city"}},
            }},
        }},
    },
}

// Model returns FunctionCall parts
for resp, _ := range llm.GenerateContent(ctx, req, false) {
    for _, part := range resp.Content.Parts {
        if part.FunctionCall != nil {
            fmt.Printf("Call: %s(%v)\n", part.FunctionCall.Name, part.FunctionCall.Args)
        }
    }
}
```

Sending tool results back works through the standard content history:

```go
// After executing the tool, send the result back:
contents := []*genai.Content{
    {Parts: []*genai.Part{{Text: "What's the weather?"}}, Role: "user"},
    {Parts: []*genai.Part{{FunctionCall: call}}, Role: "model"},
    {Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
        ID: call.ID, Name: call.Name,
        Response: map[string]any{"temp": "22°C", "condition": "sunny"},
    }}}, Role: "user"},
}
```

## Multi-modal Input (Images & Files)

Send images or documents alongside text using standard ADK `genai.Part` types:

```go
// Image from bytes (InlineData)
req := &model.LLMRequest{
    Contents: []*genai.Content{{
        Role: "user",
        Parts: []*genai.Part{
            {Text: "What's in this image?"},
            {InlineData: &genai.Blob{Data: pngBytes, MIMEType: "image/png"}},
        },
    }},
}

// Image from URL (FileData)
req := &model.LLMRequest{
    Contents: []*genai.Content{{
        Role: "user",
        Parts: []*genai.Part{
            {Text: "Describe this image."},
            {FileData: &genai.FileData{
                FileURI:  "https://example.com/photo.jpg",
                MIMEType: "image/jpeg",
            }},
        },
    }},
}

// PDF document (Claude and Gemini only)
req := &model.LLMRequest{
    Contents: []*genai.Content{{
        Role: "user",
        Parts: []*genai.Part{
            {Text: "Summarize this document."},
            {InlineData: &genai.Blob{Data: pdfBytes, MIMEType: "application/pdf"}},
        },
    }},
}
```

Supported image formats: PNG, JPEG, GIF, WebP. File support varies by model (Claude: PDF; Gemini: PDF, CSV, MP3).

## Content Filtering

Enable input/output safety filtering with zero configuration:

```go
// Strictest defaults: Azure Content Safety ALLOW_SAFE + prompt_shield on all categories.
provider, _ := sapaicore.NewProvider(ctx,
    sapaicore.WithEndpoint(endpoint),
    sapaicore.WithAuth(clientID, clientSecret, authURL),
    sapaicore.WithFiltering(nil), // nil = sensible defaults
)
```

Or customize thresholds and providers:

```go
provider, _ := sapaicore.NewProvider(ctx,
    sapaicore.WithEndpoint(endpoint),
    sapaicore.WithAuth(clientID, clientSecret, authURL),
    sapaicore.WithFiltering(&sapaicore.FilteringConfig{
        Input: &sapaicore.InputFilterConfig{
            AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
                Hate:         sapaicore.AzureThresholdAllowSafeLow,
                SelfHarm:     sapaicore.AzureThresholdAllowSafe,
                Sexual:       sapaicore.AzureThresholdAllowSafe,
                Violence:     sapaicore.AzureThresholdAllowSafeLow,
                PromptShield: true,
            },
            LlamaGuard: &sapaicore.LlamaGuardConfig{
                Hate: true,
                ChildExploitation: true,
            },
        },
        Output: &sapaicore.OutputFilterConfig{
            AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
                Hate: sapaicore.AzureThresholdAllowSafe,
            },
        },
    }),
)
```

## Data Masking

Redact PII before messages reach the LLM:

```go
provider, _ := sapaicore.NewProvider(ctx,
    sapaicore.WithEndpoint(endpoint),
    sapaicore.WithAuth(clientID, clientSecret, authURL),
    sapaicore.WithMasking(sapaicore.MaskingConfig{
        Method:   sapaicore.Anonymization,
        Entities: sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
        Allowlist: []string{"SAP", "Joule"},
    }),
)
```

Custom regex entities:

```go
sapaicore.WithMasking(sapaicore.MaskingConfig{
    Entities: []sapaicore.MaskingEntity{
        sapaicore.StandardEntity(sapaicore.EntityEmail),
        sapaicore.StandardEntity(sapaicore.EntityPhone),
        sapaicore.CustomMaskingEntity(`\b[0-9]{2}-SAP-[0-9]{3}\b`, "REDACTED_ID"),
    },
})
```

Replacement strategies for standard entities:

```go
sapaicore.WithMasking(sapaicore.MaskingConfig{
    Method: sapaicore.Pseudonymization,
    Entities: []sapaicore.MaskingEntity{
        sapaicore.StandardEntity(sapaicore.EntityEmail),           // DPI default
        sapaicore.FabricatedEntity(sapaicore.EntityPerson),       // realistic fake data
        sapaicore.ConstantEntity(sapaicore.EntityPhone, "PHONE"), // fixed value + incrementing number
    },
})
```

When using file input with masking, set `MaskFileInputMethod`:

```go
sapaicore.WithMasking(sapaicore.MaskingConfig{
    Method:              sapaicore.Anonymization,
    Entities:            sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
    MaskFileInputMethod: sapaicore.MaskFileSkip, // or sapaicore.MaskFileAnonymization
})
```

## Translation

Translate input before the LLM or output after generation:

```go
provider, _ := sapaicore.NewProvider(ctx,
    sapaicore.WithEndpoint(endpoint),
    sapaicore.WithAuth(clientID, clientSecret, authURL),
    sapaicore.WithTranslation(sapaicore.TranslationConfig{
        Output: &sapaicore.TranslationOutputConfig{
            TargetLanguage: "de-DE",
        },
    }),
)
```

## Module Fallback

Try the primary model first, fall back to alternatives on failure:

```go
llm, _ := provider.Model("gpt-4.1",
    sapaicore.WithModelFallback("gpt-4.1-mini", "gpt-4.1-nano"),
)
```

Fallback models inherit all module configurations (filtering, masking, translation).

## Prompt Caching

Enable Anthropic prompt caching (auto-annotates system message and tools with `cache_control`):

```go
llm, _ := provider.Model("anthropic--claude-4.5-sonnet",
    sapaicore.WithModelPromptCaching(),
)
```

For 1-hour TTL on supported models (Claude Opus 4.5, Haiku 4.5, Sonnet 4.5):

```go
llm, _ := provider.Model("anthropic--claude-4.5-sonnet",
    sapaicore.WithModelPromptCaching(sapaicore.CacheTTL1h),
)
```

## Module Composition

Modules can be set at provider level (defaults for all models) and overridden per-model:

```go
// Provider-level: all models get filtering + masking.
provider, _ := sapaicore.NewProvider(ctx,
    sapaicore.WithEndpoint(endpoint),
    sapaicore.WithAuth(clientID, clientSecret, authURL),
    sapaicore.WithFiltering(nil),
    sapaicore.WithMasking(sapaicore.MaskingConfig{
        Entities: sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
    }),
)

// This model inherits both filtering and masking.
llm1, _ := provider.Model("gpt-4.1")

// This model replaces filtering with a permissive config, keeps masking.
llm2, _ := provider.Model("gpt-4.1-mini",
    sapaicore.WithModelFiltering(&sapaicore.FilteringConfig{
        Input: &sapaicore.InputFilterConfig{
            AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
                Hate: sapaicore.AzureThresholdAllowAll,
            },
        },
    }),
)

// This model has no filtering at all, keeps masking.
llm3, _ := provider.Model("gpt-4.1-mini", sapaicore.WithoutFiltering())
```

Composition rules:
- Same module at both levels → model replaces provider
- Different modules → both apply
- `Without*()` → explicitly removes inherited module

Orchestration modules in foundation-models mode produce an error at `Model()` time:

```go
// This returns ErrMissingConfig — modules require orchestration mode.
provider, _ := sapaicore.NewProvider(ctx,
    sapaicore.WithEndpoint(endpoint),
    sapaicore.WithAuth(id, secret, url),
    sapaicore.WithDeployments(map[string]string{"gpt-4.1": "d1"}),
    sapaicore.WithFiltering(nil), // ← orchestration module
)
_, err := provider.Model("gpt-4.1") // err: orchestration modules require orchestration mode
```

Invalid configs also error at `Model()` time:
- `TranslationConfig` with neither Input nor Output set
- `MaskingConfig` with empty Entities

## Foundation-Models Mode

For per-model deployments (dedicated capacity, custom fine-tunes):

```go
provider, _ := sapaicore.NewProvider(
    sapaicore.WithEndpoint(os.Getenv("AI_CORE_ENDPOINT")),
    sapaicore.WithAuth(clientID, clientSecret, authURL),
    sapaicore.WithDeployments(map[string]string{
        "gpt-4.1":      "d1234abc",
        "gpt-4.1-mini": "d5678def",
    }),
)

llm, _ := provider.Model("gpt-4.1") // routes to deployment d1234abc
```

## Extra Model Parameters

For provider-specific features beyond what ADK exposes:

```go
// Claude extended thinking
llm, _ := provider.Model("anthropic--claude-4.5-sonnet",
    sapaicore.WithModelParams(map[string]any{
        "thinking":   map[string]any{"type": "enabled", "budget_tokens": 16384},
        "max_tokens": 64000,
    }),
)

// Claude 4.6+ native 1M context window
llm, _ := provider.Model("anthropic--claude-4.6-sonnet",
    sapaicore.WithModelParams(map[string]any{
        "max_tokens": 200000,
    }),
)

// OpenAI reasoning effort
llm, _ := provider.Model("o4-mini",
    sapaicore.WithModelParams(map[string]any{
        "reasoning_effort": "high",
    }),
)

// Logprobs
llm, _ := provider.Model("gpt-4.1-mini",
    sapaicore.WithModelParams(map[string]any{
        "logprobs":     true,
        "top_logprobs": 3,
    }),
)
```

## Features

### Supported

| Feature | Orchestration | Foundation | Notes |
|---------|:---:|:---:|-------|
| Non-streaming generation | ✅ | ✅ | |
| Streaming (SSE) | ✅ | ✅ | Partial chunks + final aggregated response |
| Tool calling | ✅ | ✅ | Function calls and function results |
| Tool calling (streaming) | ✅ | ✅ | Deltas assembled into complete calls |
| Tool choice (auto/none/required) | ✅ | ✅ | Via `ToolConfig.FunctionCallingConfig` |
| System instructions | ✅ | ✅ | |
| Multi-turn conversation | ✅ | ✅ | Full message history |
| Temperature / TopP / TopK | ✅ | ✅ | |
| Seed (deterministic outputs) | ✅ | ✅ | Best-effort determinism |
| MaxOutputTokens | ✅ | ✅ | |
| StopSequences | ✅ | ✅ | |
| FrequencyPenalty / PresencePenalty | ✅ | ✅ | |
| Logprobs in response | ✅ | ✅ | `ResponseLogprobs` + `Logprobs` → `LogprobsResult` |
| JSON response format | ✅ | ✅ | With schema validation |
| Extra model params (`WithModelParams`) | ✅ | ✅ | Thinking, reasoning_effort, etc. |
| Server-side timeout/retries | ✅ | - | Orchestration only |
| OAuth2 token caching | ✅ | ✅ | Auto-refresh before expiry |
| Auto-discovery | ✅ | - | Finds orchestration deployment automatically |
| Custom HTTP client | ✅ | ✅ | |
| Custom headers | ✅ | ✅ | |
| Resource groups | ✅ | ✅ | |
| Refusal handling | ✅ | ✅ | ErrorCode="refusal" |
| BeforeModelCallback (model override) | ✅ | ✅ | `req.Model` respected at runtime |
| Multi-modal input (images) | ✅ | ✅ | InlineData (bytes) or FileData (HTTPS URL) |
| File input (PDF, CSV, MP3) | ✅ | ✅ | InlineData → data URI; model support varies |
| Content filtering | ✅ | - | Azure Content Safety + Llama Guard 3 8B |
| Data masking (PII) | ✅ | - | SAP DPI: anonymization/pseudonymization |
| Fabricated data masking | ✅ | - | `FabricatedEntity()` / `ConstantEntity()` strategies |
| Mask file input | ✅ | - | `MaskFileInputMethod` for file+masking |
| Translation | ✅ | - | SAP Document Translation: input/output |
| Module fallback | ✅ | - | Try model A, fall back to model B |
| Prompt caching | ✅ | - | Anthropic cache_control on system + tools |

## API Reference

### Provider Options

| Option | Description |
|--------|-------------|
| `WithEndpoint(url)` | SAP AI Core API base URL (required) |
| `WithAuth(id, secret, authURL)` | OAuth2 credentials (required) |
| `WithOrchestration()` | Orchestration mode: auto-discovers deployment (default) |
| `WithDeploymentID(id)` | Orchestration mode: explicit deployment ID |
| `WithDeployments(map)` | Foundation-models mode: per-model deployment IDs |
| `WithResourceGroup(group)` | Resource group (default: `"default"`) |
| `WithHTTPClient(client)` | Custom `*http.Client` |
| `WithHeaders(headers)` | Extra HTTP headers on every request |
| `WithTimeout(seconds)` | Server-side LLM timeout in seconds (default: 600) |
| `WithMaxRetries(n)` | Server-side retry count (default: 2) |
| `WithFiltering(cfg)` | Content filtering (nil = strict defaults). Orchestration only |
| `WithMasking(cfg)` | PII data masking. Orchestration only |
| `WithTranslation(cfg)` | Input/output translation. Orchestration only |
| `WithFallback(models...)` | Model fallback chain. Orchestration only |
| `WithPromptCaching(ttl...)` | Anthropic prompt caching. Default 5m TTL. Orchestration only |
| `WithStreamOptions(opts)` | Global stream chunk_size/delimiters. Orchestration only |

If none of `WithOrchestration`, `WithDeploymentID`, or `WithDeployments` is specified, orchestration auto-discovery is used. These three options are mutually exclusive.

### Model Options

| Option | Description |
|--------|-------------|
| `WithModelParams(map)` | Extra params forwarded to the model |
| `WithModelFiltering(cfg)` | Override provider filtering (nil = strict defaults) |
| `WithoutFiltering()` | Remove inherited filtering |
| `WithModelMasking(cfg)` | Override provider masking |
| `WithoutMasking()` | Remove inherited masking |
| `WithModelTranslation(cfg)` | Override provider translation |
| `WithoutTranslation()` | Remove inherited translation |
| `WithModelFallback(models...)` | Model-level fallback chain |
| `WithModelPromptCaching(ttl...)` | Enable prompt caching for this model |

### Errors

```go
sapaicore.ErrMissingConfig      // required option not provided
sapaicore.ErrDeploymentNotFound // model name not in Deployments map (foundation mode only)
sapaicore.ErrDiscovery          // orchestration deployment auto-discovery failed
```

## Credentials

From your SAP AI Core service key (BTP cockpit → Instances → Service Keys):

| Option | Service Key Field |
|--------|------------------|
| `WithEndpoint` | `serviceurls.AI_API_URL` |
| `WithAuth` clientID | `uaa.clientid` |
| `WithAuth` clientSecret | `uaa.clientsecret` |
| `WithAuth` authURL | `uaa.url` + `/oauth/token` |

## How It Works

**Orchestration mode** sends all requests to SAP AI Core's harmonized API:
```
POST {endpoint}/v2/inference/deployments/{deploymentId}/v2/completion
```
The model name is embedded in the request body. SAP AI Core routes to the appropriate provider (OpenAI, Anthropic, Google, etc.).

**Foundation-models mode** sends requests directly to per-model deployments:
```
POST {endpoint}/v2/inference/deployments/{perModelId}/v1/chat/completions
```

## Dependencies

Two direct dependencies:

- `google.golang.org/adk/v2` - ADK model interface
- `google.golang.org/genai` - genai types (Content, Schema, Tool, etc.)

## Development

This project uses [mise](https://mise.jdx.dev/) for toolchain management. Run `mise install` to get the correct Go and tooling versions.

```bash
mise install
mise run check  # build + vet + lint + test
```

Or run individual tasks:

```bash
mise run build
mise run lint
mise run test
mise run fix   # auto-fix lint issues (wsl whitespace, gofmt, etc.)
```

### Smoke Tests

Integration tests against a live SAP AI Core instance covering both API modes, streaming, tool calling, extended thinking, and full ADK agent loops.

```bash
go test -tags=smoke ./smoketest/ -v -timeout=5m
```

See [`smoketest/README.md`](smoketest/README.md) for credentials setup and the full test catalog.

## License

MIT
