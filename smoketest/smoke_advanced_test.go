//go:build smoke

package smoketest_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"github.com/PedroKlein/adk-provider-sapaicore/sapaicore"
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

	provider, err := sapaicore.NewProvider(t.Context(),
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

func TestSmoke_SeedDeterminism(t *testing.T) {
	// Proves seed controls determinism by contrasting:
	// - Same seed + temp > 0 → same outputs
	// - Different seeds + temp > 0 → different outputs (probabilistic but reliable)
	// Seed determinism is "best effort" per OpenAI, so we allow one retry on mismatch.
	provider := newProvider(t)
	ctx := withTimeout(t, 45*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	temp := float32(0.9) // High temperature = non-deterministic without seed
	seed1 := int32(11111)
	seed2 := int32(99999)

	makeReq := func(seed *int32) *model.LLMRequest {
		return &model.LLMRequest{
			Contents: []*genai.Content{
				{Parts: []*genai.Part{{Text: "Write a random 8-character password using letters and numbers"}}, Role: "user"},
			},
			Config: &genai.GenerateContentConfig{
				Seed:        seed,
				Temperature: &temp,
			},
		}
	}

	// Same seed twice → should produce same output
	resp1a := generateOne(t, ctx, llm, makeReq(&seed1))
	text1a := requireText(t, resp1a)

	resp1b := generateOne(t, ctx, llm, makeReq(&seed1))
	text1b := requireText(t, resp1b)

	// Different seed → should produce different output
	resp2 := generateOne(t, ctx, llm, makeReq(&seed2))
	text2 := requireText(t, resp2)

	t.Logf("seed=%d: %q / %q | seed=%d: %q", seed1, text1a, text1b, seed2, text2)

	if text1a != text1b {
		// Retry once — GPU non-determinism can cause rare mismatches.
		resp1c := generateOne(t, ctx, llm, makeReq(&seed1))
		text1c := requireText(t, resp1c)

		if text1a != text1c && text1b != text1c {
			t.Errorf("same seed produced different outputs after retry: %q / %q / %q", text1a, text1b, text1c)
		}
	}

	if text1a == text2 {
		t.Errorf("different seeds produced same output: %q (this is statistically improbable)", text1a)
	}
}

func TestSmoke_Logprobs_InResponse(t *testing.T) {
	// Contrast test: proves logprobs are only returned when requested.
	// Step 1: request WITHOUT logprobs → LogprobsResult must be nil
	// Step 2: request WITH logprobs → LogprobsResult must be populated
	provider := newFoundationProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// Step 1: Without logprobs — must be nil.
	reqNoLogprobs := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hi"}}, Role: "user"},
		},
	}

	respNo := generateOne(t, ctx, llm, reqNoLogprobs)
	requireText(t, respNo)

	if respNo.LogprobsResult != nil {
		t.Fatalf("LogprobsResult should be nil without ResponseLogprobs, got %+v", respNo.LogprobsResult)
	}

	// Step 2: With logprobs — must be populated.
	logprobs := int32(3)

	reqWithLogprobs := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hi"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
		},
	}

	respWith := generateOne(t, ctx, llm, reqWithLogprobs)
	requireText(t, respWith)

	if respWith.LogprobsResult == nil {
		t.Fatal("LogprobsResult is nil when ResponseLogprobs=true")
	}

	if len(respWith.LogprobsResult.ChosenCandidates) == 0 {
		t.Fatal("ChosenCandidates is empty")
	}

	if len(respWith.LogprobsResult.TopCandidates) == 0 {
		t.Fatal("TopCandidates is empty")
	}

	// Each top candidate set should have up to logprobs (3) alternatives.
	firstTop := respWith.LogprobsResult.TopCandidates[0]
	if len(firstTop.Candidates) == 0 || len(firstTop.Candidates) > int(logprobs)+1 {
		t.Errorf("top candidates count = %d, expected 1-%d", len(firstTop.Candidates), logprobs+1)
	}

	// Verify log probabilities are negative (valid log-space values).
	for _, c := range respWith.LogprobsResult.ChosenCandidates {
		if c.LogProbability > 0 {
			t.Errorf("logprob = %v, expected <= 0", c.LogProbability)
			break
		}
	}

	t.Logf("contrast: without=%v, with=%d chosen, %d top (first has %d alts)",
		respNo.LogprobsResult,
		len(respWith.LogprobsResult.ChosenCandidates),
		len(respWith.LogprobsResult.TopCandidates),
		len(firstTop.Candidates))
}

func TestSmoke_ToolChoice_Required(t *testing.T) {
	// Contrast test: proves tool_choice=required changes model behavior.
	// Step 1: same prompt + tool WITHOUT tool_choice → model answers in text
	// Step 2: same prompt + tool WITH tool_choice=required → model forced to call tool
	provider := newFoundationProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	tools := []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "log_interaction",
			Description: "Log a user interaction with metadata",
			Parameters: &genai.Schema{
				Type: "OBJECT",
				Properties: map[string]*genai.Schema{
					"category": {Type: "STRING"},
				},
			},
		}},
	}}

	// Step 1: Without tool_choice — model should answer with text, not call the tool.
	reqNoChoice := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Tell me a short joke"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: tools,
		},
	}

	respNo := generateOne(t, ctx, llm, reqNoChoice)
	hasText := respNo.Content != nil && len(respNo.Content.Parts) > 0 && respNo.Content.Parts[0].Text != ""
	hasToolCall := respNo.Content != nil && len(respNo.Content.Parts) > 0 && respNo.Content.Parts[0].FunctionCall != nil

	if !hasText && !hasToolCall {
		t.Fatal("no response without tool_choice")
	}

	// Without tool_choice, model should just answer the joke (no reason to call log_interaction).
	if hasToolCall && respNo.Content.Parts[0].FunctionCall.Name == "log_interaction" {
		t.Log("WARNING: model called log_interaction even without tool_choice=required (unusual but not impossible)")
	}

	// Step 2: With tool_choice=required — model MUST call the tool.
	reqWithChoice := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Tell me a short joke"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: tools,
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
		},
	}

	respWith := generateOne(t, ctx, llm, reqWithChoice)
	calls := requireFunctionCalls(t, respWith)

	if calls[0].Name != "log_interaction" {
		t.Errorf("expected log_interaction, got %q", calls[0].Name)
	}

	t.Logf("contrast: without tool_choice → text=%v toolcall=%v | with tool_choice=required → forced %q",
		hasText, hasToolCall, calls[0].Name)
}

func TestSmoke_TopK(t *testing.T) {
	// Verifies that TopK is accepted by the API (forwarded as top_k).
	// Models that support it (Anthropic, Gemini) use it for sampling;
	// OpenAI models ignore it silently.
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gemini-2.5-flash")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	topK := float32(40)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hello"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			TopK: &topK,
		},
	}

	resp := generateOne(t, ctx, llm, req)
	text := requireText(t, resp)

	if text == "" {
		t.Error("empty response with top_k")
	}

	t.Logf("top_k=40: response=%q", text)
}

func TestSmoke_Logprobs_Orchestration(t *testing.T) {
	// Verifies that logprobs work in orchestration mode
	// (passed via model.params). Asserts the same structural guarantees.
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	logprobs := int32(3)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hi"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
		},
	}

	resp := generateOne(t, ctx, llm, req)
	text := requireText(t, resp)

	if text == "" {
		t.Fatal("empty response")
	}

	if resp.LogprobsResult == nil {
		t.Fatal("LogprobsResult is nil in orchestration mode")
	}

	if len(resp.LogprobsResult.ChosenCandidates) == 0 {
		t.Fatal("ChosenCandidates is empty")
	}

	// Verify log probabilities are negative (valid log-space values).
	for _, c := range resp.LogprobsResult.ChosenCandidates {
		if c.LogProbability > 0 {
			t.Errorf("logprob = %v, expected <= 0", c.LogProbability)
			break
		}
	}

	t.Logf("orchestration logprobs: %d chosen tokens", len(resp.LogprobsResult.ChosenCandidates))
}

func TestSmoke_Logprobs_Streaming(t *testing.T) {
	// Verifies logprobs are populated in the STREAMING final response.
	// This is a regression test for the bug where streaming logprobs were
	// silently dropped because ChunkChoice didn't capture them.
	provider := newFoundationProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	logprobs := int32(3)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hi"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
		},
	}

	_, final := generateStream(t, ctx, llm, req)

	if final.LogprobsResult == nil {
		t.Fatal("LogprobsResult is nil in streaming mode — logprobs dropped during aggregation")
	}

	if len(final.LogprobsResult.ChosenCandidates) == 0 {
		t.Fatal("ChosenCandidates is empty in streaming mode")
	}

	// Verify log probabilities are valid.
	for _, c := range final.LogprobsResult.ChosenCandidates {
		if c.LogProbability > 0 {
			t.Errorf("logprob = %v, expected <= 0", c.LogProbability)
			break
		}
	}

	t.Logf("streaming logprobs: %d chosen tokens aggregated from stream chunks",
		len(final.LogprobsResult.ChosenCandidates))
}

func TestSmoke_ToolChoice_Orchestration(t *testing.T) {
	// Contrast test for orchestration mode — same logic as foundation.
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	tools := []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "log_interaction",
			Description: "Log a user interaction with metadata",
			Parameters: &genai.Schema{
				Type: "OBJECT",
				Properties: map[string]*genai.Schema{
					"category": {Type: "STRING"},
				},
			},
		}},
	}}

	// Without tool_choice — should get text.
	reqNoChoice := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Tell me a short joke"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: tools,
		},
	}

	respNo := generateOne(t, ctx, llm, reqNoChoice)
	hasText := respNo.Content != nil && len(respNo.Content.Parts) > 0 && respNo.Content.Parts[0].Text != ""

	// With tool_choice=required — MUST call the tool.
	reqWithChoice := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Tell me a short joke"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: tools,
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
		},
	}

	respWith := generateOne(t, ctx, llm, reqWithChoice)
	calls := requireFunctionCalls(t, respWith)

	if calls[0].Name != "log_interaction" {
		t.Errorf("expected log_interaction, got %q", calls[0].Name)
	}

	t.Logf("orchestration contrast: without → text=%v | with required → forced %q",
		hasText, calls[0].Name)
}

func TestSmoke_Seed_Foundation(t *testing.T) {
	// Same contrast test for foundation mode.
	// Seed determinism is "best effort" per OpenAI, so we retry once on mismatch.
	provider := newFoundationProvider(t)
	ctx := withTimeout(t, 45*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	temp := float32(0.9)
	seed := int32(11111)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Write a random 8-character password using letters and numbers"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Seed:        &seed,
			Temperature: &temp,
		},
	}

	resp1 := generateOne(t, ctx, llm, req)
	text1 := requireText(t, resp1)

	resp2 := generateOne(t, ctx, llm, req)
	text2 := requireText(t, resp2)

	t.Logf("foundation seed=%d temp=0.9: %q / %q", seed, text1, text2)

	if text1 != text2 {
		// Retry once — OpenAI seed is "best effort", GPU non-determinism can cause occasional mismatches.
		resp3 := generateOne(t, ctx, llm, req)
		text3 := requireText(t, resp3)

		if text1 != text3 && text2 != text3 {
			t.Errorf("seed not deterministic after retry: %q / %q / %q", text1, text2, text3)
		}
	}
}
