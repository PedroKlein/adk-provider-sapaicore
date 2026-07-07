# go-adk-sap-ai-core

ADK Go v2 model provider for SAP AI Core.

Implements `model.LLM` from [`google.golang.org/adk/v2`](https://github.com/google/adk-go), letting any ADK Go agent use models deployed on SAP AI Core.

## Install

```bash
go get github.com/PedroKlein/go-adk-sap-ai-core
```

Requires Go 1.25+.

## Quick start (orchestration mode)

Orchestration mode uses a single deployment to access all models. SAP AI Core creates this deployment automatically during onboarding.

```go
package main

import (
	"log"
	"os"

	sapaicore "github.com/PedroKlein/go-adk-sap-ai-core"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/runner"
)

func main() {
	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint(os.Getenv("AI_CORE_ENDPOINT")),
		sapaicore.WithAuth(
			os.Getenv("AI_CORE_CLIENT_ID"),
			os.Getenv("AI_CORE_CLIENT_SECRET"),
			os.Getenv("AI_CORE_AUTH_URL"),
		),
		sapaicore.WithDeploymentID(os.Getenv("AI_CORE_DEPLOYMENT_ID")),
	)
	if err != nil {
		log.Fatal(err)
	}

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		log.Fatal(err)
	}

	agent := llmagent.New(llmagent.Config{
		Name:        "my-agent",
		Model:       llm,
		Instruction: "You are a helpful assistant.",
	})

	_ = runner.New(runner.Config{AppName: "my-app", Agent: agent})
}
```

You can use any model available on SAP AI Core by passing its name directly:

```go
provider.Model("anthropic--claude-4.5-sonnet")
provider.Model("gemini-2.5-flash")
provider.Model("gpt-4.1-mini")
provider.Model("mistralai--mistral-large-instruct")
```

## Foundation-models mode

If you need per-model deployments (e.g., for dedicated capacity), use `WithDeployments` instead:

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

## Extra model parameters

For features beyond what ADK exposes (extended thinking, reasoning effort, 1M context), use `WithModelParams`:

```go
// Claude extended thinking
llm, _ := provider.Model("anthropic--claude-4.5-sonnet",
	sapaicore.WithModelParams(map[string]any{
		"thinking":   map[string]any{"type": "enabled", "budget_tokens": 16384},
		"max_tokens": 64000,
	}),
)

// Claude 1M context window
llm, _ := provider.Model("anthropic--claude-4.5-sonnet",
	sapaicore.WithModelParams(map[string]any{
		"anthropic_beta": []string{"context-1m-2025-08-07"},
		"max_tokens":     200000,
	}),
)

// OpenAI reasoning effort
llm, _ := provider.Model("o4-mini",
	sapaicore.WithModelParams(map[string]any{
		"reasoning_effort": "high",
	}),
)
```

## API reference

### Provider options

| Option | Description |
|--------|-------------|
| `WithEndpoint(url)` | SAP AI Core API base URL (required) |
| `WithAuth(id, secret, authURL)` | OAuth2 credentials (required) |
| `WithDeploymentID(id)` | Orchestration mode: single deployment for all models |
| `WithDeployments(map)` | Foundation-models mode: per-model deployment IDs |
| `WithResourceGroup(group)` | Resource group (default: `"default"`) |
| `WithHTTPClient(client)` | Custom `*http.Client` |
| `WithHeaders(headers)` | Extra HTTP headers on every request |

Exactly one of `WithDeploymentID` or `WithDeployments` is required.

### Model options

| Option | Description |
|--------|-------------|
| `WithModelParams(map)` | Extra params forwarded to the model (thinking, reasoning_effort, etc.) |

### Errors

```go
sapaicore.ErrMissingConfig      // required option not provided
sapaicore.ErrDeploymentNotFound // model name not in Deployments map (foundation mode only)
sapaicore.ErrTokenRefresh       // OAuth2 token fetch failed
sapaicore.ErrInference          // inference request failed
sapaicore.ErrDiscovery          // orchestration deployment auto-discovery failed
```

## Credentials

From your SAP AI Core service key (BTP cockpit):

| Config | Service key field |
|--------|------------------|
| `WithEndpoint` | `serviceurls.AI_API_URL` |
| `WithAuth` clientID | `uaa.clientid` |
| `WithAuth` clientSecret | `uaa.clientsecret` |
| `WithAuth` authURL | `uaa.url` + `/oauth/token` |

With `WithOrchestration()`, that's all you need — no deployment ID required.

## How it works

In orchestration mode, every request goes to:
```
POST {endpoint}/v2/inference/deployments/{deploymentId}/v2/completion
```

The model name and parameters are embedded in the request body. SAP AI Core routes to the appropriate model provider (OpenAI, Anthropic, Google, etc.).

In foundation-models mode, each model has its own deployment and the request goes to:
```
POST {endpoint}/v2/inference/deployments/{perModelId}/chat/completions
```

Both modes support streaming (SSE), tool calling, system prompts, and all standard ADK features.

## Dependencies

Two direct dependencies, everything else is stdlib:

- `google.golang.org/adk/v2`
- `google.golang.org/genai`

## Development

```bash
go build ./...
go vet ./...
golangci-lint run ./...
go test -race ./...
```

## License

MIT
