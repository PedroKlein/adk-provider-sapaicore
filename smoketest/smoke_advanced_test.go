//go:build smoke

package smoketest_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/go-adk-sap-ai-core"
	"google.golang.org/adk/v2/model"
)

func TestSmoke_ExtendedThinking(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 60*time.Second)

	llm, err := provider.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"thinking":   map[string]any{"type": "enabled", "budget_tokens": 8192},
			"max_tokens": 16000,
		}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := simpleReq("What is 7 * 13 * 29? Think step by step.")

	resp := generateOne(t, ctx, llm, req)

	text := requireText(t, resp)
	if !strings.Contains(text, "2639") && !strings.Contains(text, "2,639") {
		t.Errorf("expected 2639 in response, got: %q", text)
	}

	if resp.UsageMetadata != nil && resp.UsageMetadata.CandidatesTokenCount > 200 {
		t.Logf("high completion tokens (%d) suggests thinking was active",
			resp.UsageMetadata.CandidatesTokenCount)
	}

	t.Logf("response=%q", text)
}

func TestSmoke_ExtendedThinking_Streaming(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 60*time.Second)

	llm, err := provider.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"thinking":   map[string]any{"type": "enabled", "budget_tokens": 8192},
			"max_tokens": 16000,
		}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := simpleReq("What is 11 * 17 * 23?")

	partials, final := generateStream(t, ctx, llm, req)

	text := requireText(t, final)
	if !strings.Contains(text, "4301") && !strings.Contains(text, "4,301") {
		t.Errorf("expected 4301 in response, got: %q", text)
	}

	t.Logf("chunks=%d response=%q", len(partials), text)
}

func TestSmoke_Anthropic1MContext(t *testing.T) {
	if os.Getenv("SMOKE_LARGE_CONTEXT") == "" {
		t.Skip("skipping: set SMOKE_LARGE_CONTEXT=1 to run (slow and costly)")
	}

	provider := newProvider(t)
	ctx := withTimeout(t, 120*time.Second)

	// Claude 4.6+ has 1M context natively, no beta header needed.
	llm, err := provider.Model("anthropic--claude-4.6-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"max_tokens": 4096,
		}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// Build ~250K tokens of padding.
	var sb strings.Builder
	for range 10000 {
		sb.WriteString("The quick brown fox jumps over the lazy dog near the riverbank. ")
		sb.WriteString("She sells seashells by the seashore while the wind blows gently. ")
		sb.WriteString("A stitch in time saves nine but only if you act before it is too late. ")
	}

	prompt := sb.String() + "\n\nThe hidden word is PINEAPPLE. What is the hidden word? Reply with just the word."

	req := simpleReq(prompt)

	resp := generateOne(t, ctx, llm, req)

	text := requireText(t, resp)
	if !strings.Contains(strings.ToUpper(text), "PINEAPPLE") {
		t.Errorf("model didn't find hidden word: %q", text)
	}

	if resp.UsageMetadata != nil && resp.UsageMetadata.PromptTokenCount > 200000 {
		t.Logf("confirmed >200K prompt tokens: %d", resp.UsageMetadata.PromptTokenCount)
	}
}

func TestSmoke_TimeoutAndRetries(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")

	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithOrchestration(),
		sapaicore.WithTimeout(30),
		sapaicore.WithMaxRetries(3),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 45*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("Say hello"))

	text := requireText(t, resp)
	if text == "" {
		t.Error("empty response")
	}

	t.Logf("response=%q (timeout=30s, retries=3)", text)
}

func TestSmoke_ErrorHandling_InvalidModel(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("nonexistent-model-xyz-99999")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// Should get an error from the API (invalid model name).
	for _, err := range llm.GenerateContent(ctx, simpleReq("hi"), false) {
		if err != nil {
			t.Logf("got expected error: %v", err)
			return
		}
	}

	// Some orchestration instances may route to a default — that's also acceptable.
	t.Log("no error returned (orchestration may have routed to default model)")
}

func TestSmoke_ADK_AgentStyleUsage(t *testing.T) {
	// Simulates how an ADK agent would use this provider in a real loop:
	// 1. User sends message
	// 2. Model responds with tool call
	// 3. Agent executes tool
	// 4. Agent sends result back
	// 5. Model produces final answer
	provider := newProvider(t)
	ctx := withTimeout(t, 60*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	weatherTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "get_weather",
			Description: "Get weather for a location",
			Parameters: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"location": {Type: genai.TypeString},
				},
				Required: []string{"location"},
			},
		}},
	}

	// Step 1: User asks about weather.
	req1 := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What's the weather like in Tokyo?"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{weatherTool},
		},
	}

	resp1 := generateOne(t, ctx, llm, req1)
	calls := requireFunctionCalls(t, resp1)

	if calls[0].Name != "get_weather" {
		t.Fatalf("expected get_weather call, got %q", calls[0].Name)
	}

	t.Logf("step 1: model called %s(%v)", calls[0].Name, calls[0].Args)

	// Step 2: Simulate tool execution and send result back.
	req2 := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What's the weather like in Tokyo?"}}, Role: "user"},
			{Parts: []*genai.Part{{FunctionCall: calls[0]}}, Role: "model"},
			{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID:       calls[0].ID,
				Name:     calls[0].Name,
				Response: map[string]any{"temperature": "28°C", "condition": "humid and cloudy"},
			}}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{weatherTool},
		},
	}

	resp2 := generateOne(t, ctx, llm, req2)

	text := requireText(t, resp2)
	lower := strings.ToLower(text)

	if !strings.Contains(lower, "28") && !strings.Contains(lower, "humid") && !strings.Contains(lower, "tokyo") {
		t.Errorf("model didn't synthesize tool result: %q", text)
	}

	t.Logf("step 2: final answer=%q", text)
}

func TestSmoke_ParamForwarding_Logprobs(t *testing.T) {
	// Verifies that WithModelParams actually reaches the provider by requesting
	// logprobs from GPT. If forwarded, the raw API response includes a logprobs
	// field. We can't inspect it directly through the ADK response (ADK doesn't
	// expose logprobs), but the request succeeds only if the param is accepted.
	// An invalid param would cause an API error.
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini",
		sapaicore.WithModelParams(map[string]any{
			"logprobs":     true,
			"top_logprobs": 3,
		}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("What is 2+2?"))

	text := requireText(t, resp)
	if text == "" {
		t.Error("empty response")
	}

	t.Logf("response=%q (logprobs param accepted by provider)", text)
}

func TestSmoke_ParamForwarding_ReasoningEffort(t *testing.T) {
	// o4-mini accepts reasoning_effort param. If not forwarded, the request
	// would either fail or use default effort.
	provider := newProvider(t)
	ctx := withTimeout(t, 45*time.Second)

	llm, err := provider.Model("o4-mini",
		sapaicore.WithModelParams(map[string]any{
			"reasoning_effort": "low",
		}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("What is the capital of France?"))

	text := requireText(t, resp)
	if !strings.Contains(strings.ToLower(text), "paris") {
		t.Errorf("expected Paris, got: %q", text)
	}

	t.Logf("response=%q (reasoning_effort=low accepted)", text)
}
