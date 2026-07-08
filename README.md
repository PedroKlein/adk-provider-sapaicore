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
| System instructions | ✅ | ✅ | |
| Multi-turn conversation | ✅ | ✅ | Full message history |
| Temperature / TopP | ✅ | ✅ | |
| MaxOutputTokens | ✅ | ✅ | |
| StopSequences | ✅ | ✅ | |
| FrequencyPenalty / PresencePenalty | ✅ | ✅ | |
| JSON response format | ✅ | ✅ | With schema validation |
| Extra model params (`WithModelParams`) | ✅ | ✅ | Thinking, reasoning_effort, logprobs, etc. |
| Server-side timeout/retries | ✅ | - | Orchestration only |
| OAuth2 token caching | ✅ | ✅ | Auto-refresh before expiry |
| Auto-discovery | ✅ | - | Finds orchestration deployment automatically |
| Custom HTTP client | ✅ | ✅ | |
| Custom headers | ✅ | ✅ | |
| Resource groups | ✅ | ✅ | |
| Refusal handling | ✅ | ✅ | ErrorCode="refusal" |
| BeforeModelCallback (model override) | ✅ | ✅ | `req.Model` respected at runtime |

### Roadmap

**Phase 1 - ADK field coverage** (extract from `GenerateContentConfig`, no new APIs):

| Feature | ADK Field | Status |
|---------|-----------|--------|
| Seed (deterministic outputs) | `Seed` | Planned |
| TopK sampling | `TopK` | Planned |
| Logprobs in response | `LogprobsResult` | Planned |
| Tool choice (auto/none/required) | `ToolConfig` | Planned (foundation mode only*) |

*\*SAP AI Core orchestration mode doesn't support `tool_choice` yet ([tracking issue](https://github.com/SAP/ai-sdk-js/issues/1500)).*

**Phase 2 - SAP AI Core orchestration modules** (new `With*` provider options):

| Feature | SAP Module | Description |
|---------|-----------|-------------|
| Content filtering | `filtering` | Input/output safety filtering |
| Data masking | `masking` | PII redaction before sending to LLM |
| Document grounding (RAG) | `grounding` | Retrieve from enterprise knowledge bases |
| Translation | `translation` | Input/output language translation |
| Module fallback | fallback chain | Try model A, fall back to model B |
| Prompt caching | `cache_control` | Cost reduction for repeated context |

**Phase 3 - Pending SAP AI Core support:**

| Feature | Notes |
|---------|-------|
| Tool choice in orchestration mode | Waiting on SAP to ship ([#1500](https://github.com/SAP/ai-sdk-js/issues/1500)) |
| Multi-modal input (images) | SAP supports `image_url` in messages |

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

If none of `WithOrchestration`, `WithDeploymentID`, or `WithDeployments` is specified, orchestration auto-discovery is used. These three options are mutually exclusive.

### Model Options

| Option | Description |
|--------|-------------|
| `WithModelParams(map)` | Extra params forwarded to the model |

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
```

### Smoke Tests

Integration tests against a live SAP AI Core instance covering both API modes, streaming, tool calling, extended thinking, and full ADK agent loops.

```bash
go test -tags=smoke ./smoketest/ -v -timeout=5m
```

See [`smoketest/README.md`](smoketest/README.md) for credentials setup and the full test catalog.

## License

MIT
