//go:build smoke

package smoketest_test

import (
	"context"
	"os"
	"testing"
	"time"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
	"google.golang.org/adk/v2/model"
)

// testModels defines the models tested in multi-model scenarios.
var testModels = []string{
	"gpt-4.1-mini",
	"anthropic--claude-4.5-sonnet",
	"gemini-2.5-flash",
}

func newProvider(t *testing.T) *sapaicore.Provider {
	t.Helper()

	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")

	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithOrchestration(),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	return provider
}

func envOrSkip(t *testing.T, key string) string {
	t.Helper()

	v := os.Getenv(key)
	if v == "" {
		t.Skipf("skipping: %s not set", key)
	}

	return v
}

func withTimeout(t *testing.T, d time.Duration) context.Context {
	t.Helper()

	ctx, cancel := context.WithTimeout(t.Context(), d)
	t.Cleanup(cancel)

	return ctx
}

//nolint:revive //  t must be first in test helpers for t.Helper() semantics
func generateOne(t *testing.T, ctx context.Context, llm model.LLM, req *model.LLMRequest) *model.LLMResponse {
	t.Helper()

	var result *model.LLMResponse

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}

		result = resp
	}

	if result == nil {
		t.Fatal("no response received")
	}

	return result
}

//nolint:revive //  t must be first in test helpers for t.Helper() semantics
func generateStream(t *testing.T, ctx context.Context, llm model.LLM, req *model.LLMRequest) (partials []*model.LLMResponse, final *model.LLMResponse) {
	t.Helper()

	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("GenerateContent stream: %v", err)
		}

		if resp.TurnComplete {
			final = resp
		} else {
			partials = append(partials, resp)
		}
	}

	if final == nil {
		t.Fatal("no final response in stream")
	}

	return partials, final
}

func requireText(t *testing.T, resp *model.LLMResponse) string {
	t.Helper()

	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		t.Fatal("response has no content parts")
	}

	return resp.Content.Parts[0].Text
}

func requireFunctionCalls(t *testing.T, resp *model.LLMResponse) []*genai.FunctionCall {
	t.Helper()

	if resp.Content == nil || len(resp.Content.Parts) == 0 {
		t.Fatal("response has no content parts")
	}

	var calls []*genai.FunctionCall

	for _, part := range resp.Content.Parts {
		if part.FunctionCall != nil {
			calls = append(calls, part.FunctionCall)
		}
	}

	if len(calls) == 0 {
		t.Fatalf("expected function calls, got text: %q", resp.Content.Parts[0].Text)
	}

	return calls
}

func simpleReq(text string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: text}}, Role: "user"},
		},
	}
}
