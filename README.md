# go-adk-sap-ai-core

ADK Go v2 model provider for SAP AI Core.

This module implements the `model.LLM` interface from [`google.golang.org/adk/v2`](https://github.com/google/adk-go), letting any ADK Go agent use models deployed on SAP AI Core without dealing with OAuth2 tokens or deployment IDs directly.

## Install

```bash
go get github.com/PedroKlein/go-adk-sap-ai-core
```

Requires Go 1.25+.

## Quick start

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
	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     os.Getenv("AI_CORE_ENDPOINT"),
		ClientID:     os.Getenv("AI_CORE_CLIENT_ID"),
		ClientSecret: os.Getenv("AI_CORE_CLIENT_SECRET"),
		AuthURL:      os.Getenv("AI_CORE_AUTH_URL"),
		Deployments: map[string]string{
			"gpt-4.1":      "d1234abc",
			"gpt-4.1-mini": "d5678def",
		},
	})
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

	r := runner.New(runner.Config{
		AppName: "my-app",
		Agent:   agent,
	})

	// Use r to run conversations...
	_ = r
}
```

## How it works

SAP AI Core exposes an OpenAI-compatible chat completions API behind OAuth2 authentication. Each model is identified by a deployment ID rather than a name.

This module handles:

1. **OAuth2 client credentials flow** with token caching (refreshes automatically before expiry)
2. **Deployment routing** from logical model names to SAP AI Core deployment URLs
3. **Protocol translation** between ADK's `genai.Content` types and OpenAI's message format
4. **Streaming** via Server-Sent Events, yielding partial responses as they arrive

The inference URL for each request is:

```
POST {endpoint}/v2/inference/deployments/{deploymentId}/chat/completions
```

Every request includes the `AI-Resource-Group` header.

## API reference

### `sapaicore.Config`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `Endpoint` | `string` | yes | SAP AI Core API base URL |
| `ClientID` | `string` | yes | OAuth2 client ID from your service key |
| `ClientSecret` | `string` | yes | OAuth2 client secret |
| `AuthURL` | `string` | yes | Token endpoint URL (usually `{uaa.url}/oauth/token`) |
| `ResourceGroup` | `string` | no | AI Core resource group. Default: `"default"` |
| `Deployments` | `map[string]string` | yes | Logical model name to deployment ID |
| `HTTPClient` | `*http.Client` | no | Custom HTTP client. Default: standard client |

### `sapaicore.NewProvider(cfg Config) (*Provider, error)`

Validates the configuration and returns a `Provider`. Returns `ErrMissingConfig` if any required field is empty or `Deployments` is empty.

### `(*Provider).Model(name string) (model.LLM, error)`

Returns a `model.LLM` for the given logical name. The name must exist in `Config.Deployments`. Returns `ErrDeploymentNotFound` if not found.

### Errors

```go
var (
    ErrMissingConfig      // required config field is empty
    ErrDeploymentNotFound // model name not in Deployments map
    ErrTokenRefresh       // OAuth2 token fetch failed
    ErrInference          // inference request failed
)
```

All errors are wrapped with context and can be inspected with `errors.Is`.

## Credentials

You'll find the values you need in your SAP AI Core service key JSON (from the BTP cockpit):

```json
{
  "clientid": "sb-xxx...",
  "clientsecret": "...",
  "url": "https://api.ai.xxx.aicore.cfapps.xxx.hana.ondemand.com",
  "serviceurls": {
    "AI_API_URL": "https://api.ai.xxx.aicore.cfapps.xxx.hana.ondemand.com"
  },
  "uaa": {
    "url": "https://xxx.authentication.xxx.hana.ondemand.com",
    "clientid": "sb-xxx...",
    "clientsecret": "..."
  }
}
```

Mapping to `Config`:

- `Endpoint` = `serviceurls.AI_API_URL`
- `ClientID` = `uaa.clientid`
- `ClientSecret` = `uaa.clientsecret`
- `AuthURL` = `uaa.url` + `/oauth/token`

## Deployment IDs

Each model you want to use must have a running deployment in SAP AI Core. You can find deployment IDs in the AI Launchpad or via the AI Core API:

```
GET {endpoint}/v2/lm/deployments
```

The `Deployments` map lets you use friendly names in your code while the module handles the routing internally.

## Streaming

Both streaming and non-streaming modes are supported. ADK controls which mode to use based on the agent configuration. When streaming:

- Each text chunk yields a partial `LLMResponse` with `Partial: true`
- Tool call arguments are accumulated across chunks
- A final aggregated `LLMResponse` with `TurnComplete: true` is yielded at the end

## Dependencies

Two direct dependencies, everything else is stdlib:

- `google.golang.org/adk/v2` (the `model.LLM` interface)
- `google.golang.org/genai` (content types, transitive from ADK)

No SAP Cloud SDK. No third-party HTTP or OAuth2 libraries.

## Development

```bash
go build ./...
go vet ./...
golangci-lint run ./...
go test -race ./...
```

## License

MIT
