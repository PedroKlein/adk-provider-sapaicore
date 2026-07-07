package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/go-adk-sap-ai-core"
	"google.golang.org/adk/v2/model"
)

func main() {
	endpoint := requireEnv("AI_CORE_ENDPOINT")
	clientID := requireEnv("AI_CORE_CLIENT_ID")
	clientSecret := requireEnv("AI_CORE_CLIENT_SECRET")
	authURL := requireEnv("AI_CORE_AUTH_URL")

	headers := http.Header{}
	headers.Set("X-Custom-Test", "smoke-test")

	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithOrchestration(),
		sapaicore.WithHeaders(headers),
	)
	if err != nil {
		log.Fatalf("NewProvider: %v", err)
	}

	fmt.Println("✅ Provider created (orchestration deployment auto-discovered)")

	ctx := context.Background()

	models := []string{
		"gpt-4.1-mini",
		"anthropic--claude-4.5-sonnet",
		"gemini-2.5-flash",
	}

	// --- Non-streaming ---
	fmt.Println("=== Non-streaming ===")

	for _, name := range models {
		fmt.Printf("\n--- %s ---\n", name)
		testNonStreaming(ctx, provider, name)
	}

	// --- Streaming ---
	fmt.Println("\n\n=== Streaming ===")

	for _, name := range models {
		fmt.Printf("\n--- %s ---\n", name)
		testStreaming(ctx, provider, name)
	}

	// --- With model params ---
	fmt.Println("\n\n=== Model Params (max_tokens cap) ===")
	testWithParams(ctx, provider)

	// --- Anthropic 1M context ---
	fmt.Println("\n\n=== Anthropic 1M Context Window ===")
	testAnthropic1M(ctx, provider)

	// --- Tool calling ---
	fmt.Println("\n\n=== Tool Calling ===")
	testToolCalling(ctx, provider)

	// --- Tool calling streaming ---
	fmt.Println("\n\n=== Tool Calling (Streaming) ===")
	testToolCallingStream(ctx, provider)

	// --- Multi-turn conversation ---
	fmt.Println("\n\n=== Multi-turn Conversation ===")
	testMultiTurn(ctx, provider)

	// --- Extended thinking (Claude) ---
	fmt.Println("\n\n=== Extended Thinking (Claude) ===")
	testExtendedThinking(ctx, provider)

	// --- Function response round-trip ---
	// NOTE: SAP AI Core orchestration has a known limitation where tool call
	// results cannot be passed back in a single multi-turn request
	// (see https://github.com/SAP/ai-sdk-js/issues/1479).
	// In practice this isn't an issue because the ADK runner handles tool loops
	// automatically — each tool call is a separate GenerateContent request.
	fmt.Println("\n\n=== Function Response Round-trip ===")
	fmt.Println("  SKIPPED: SAP AI Core orchestration does not support tool role in messages_history.")
	fmt.Println("  This is a known API limitation (github.com/SAP/ai-sdk-js/issues/1479).")
	fmt.Println("  ADK agents work because tool loops use separate requests per turn.")

	// --- Error handling ---
	fmt.Println("\n\n=== Error Handling ===")
	testErrorHandling(ctx, provider)

	fmt.Println("\n\n✅ All done!")
}

func testNonStreaming(ctx context.Context, provider *sapaicore.Provider, modelName string) {
	llm, err := provider.Model(modelName)
	if err != nil {
		fmt.Printf("  ERROR creating model: %v\n", err)
		return
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Reply with exactly: hello from <your model name>"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a concise assistant. Reply in one line."}},
			},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			return
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Response: %s\n", resp.Content.Parts[0].Text)
		}

		if resp.UsageMetadata != nil {
			fmt.Printf("  Tokens: prompt=%d completion=%d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount)
		}

		fmt.Printf("  ModelVersion: %s\n", resp.ModelVersion)
	}
}

func testStreaming(ctx context.Context, provider *sapaicore.Provider, modelName string) {
	llm, err := provider.Model(modelName)
	if err != nil {
		fmt.Printf("  ERROR creating model: %v\n", err)
		return
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Count from 1 to 5, one number per word."}}, Role: "user"},
		},
	}

	var chunks int

	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			return
		}

		if resp.Partial {
			chunks++
			continue
		}

		// Final response.
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Final text: %s\n", resp.Content.Parts[0].Text)
		}

		fmt.Printf("  Chunks received: %d\n", chunks)
		fmt.Printf("  TurnComplete: %v\n", resp.TurnComplete)
	}
}

func testWithParams(ctx context.Context, provider *sapaicore.Provider) {
	llm, err := provider.Model("gpt-4.1-mini",
		sapaicore.WithModelParams(map[string]any{
			"max_tokens": 50,
		}),
	)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hi in exactly 3 words."}}, Role: "user"},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			return
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Response: %s\n", resp.Content.Parts[0].Text)
		}
	}
}

func testAnthropic1M(ctx context.Context, provider *sapaicore.Provider) {
	llm, err := provider.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"anthropic_beta": []string{"context-1m-2025-08-07"},
			"max_tokens":     4096,
		}),
	)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	fmt.Printf("  Model: %s\n", llm.Name())
	fmt.Printf("  Beta: context-1m-2025-08-07\n")

	// If SMOKE_LARGE_CONTEXT is set, send >200K tokens to prove 1M is active.
	// Without it, we just verify the beta flag is accepted (no error).
	var prompt string

	if os.Getenv("SMOKE_LARGE_CONTEXT") != "" {
		fmt.Printf("  Mode: LARGE CONTEXT (>200K tokens) — this will be slow and costly\n")

		var sb strings.Builder

		// ~250K tokens worth of text (~1MB). Each word ≈ 1.3 tokens.
		// 200K tokens ≈ 150K words ≈ 750KB of English text.
		for range 10000 {
			sb.WriteString("The quick brown fox jumps over the lazy dog near the riverbank. ")
			sb.WriteString("She sells seashells by the seashore while the wind blows gently. ")
			sb.WriteString("A stitch in time saves nine but only if you act before it is too late. ")
			sb.WriteString("Knowledge is power and power comes from understanding the world around you. ")
			sb.WriteString("Every journey of a thousand miles begins with a single determined step forward. ")
		}

		fmt.Printf("  Prompt size: ~%d KB (~%d estimated tokens)\n", sb.Len()/1024, sb.Len()/4)
		prompt = sb.String() + "\n\nThe hidden word is PINEAPPLE. What is the hidden word? Reply with just the word."
	} else {
		fmt.Printf("  Mode: basic (set SMOKE_LARGE_CONTEXT=1 to verify >200K tokens)\n")

		prompt = "Reply with exactly: 1M beta accepted"
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: prompt}}, Role: "user"},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			fmt.Printf("  (If this is a context length error, the beta flag may not be active)\n")

			return
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Response: %s\n", resp.Content.Parts[0].Text)
		}

		if resp.UsageMetadata != nil {
			fmt.Printf("  Tokens: prompt=%d completion=%d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount)

			if resp.UsageMetadata.PromptTokenCount > 200000 {
				fmt.Printf("  ✅ CONFIRMED: >200K prompt tokens accepted — 1M context is active!\n")
			}
		}
	}
}

func testToolCalling(ctx context.Context, provider *sapaicore.Provider) {
	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What's the weather in Berlin?"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get the current weather for a city",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"city": {
								Type:        genai.TypeString,
								Description: "The city name",
							},
						},
						Required: []string{"city"},
					},
				}},
			}},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			return
		}

		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			fmt.Printf("  No content parts returned\n")
			return
		}

		for i, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				fmt.Printf("  Part[%d] FunctionCall:\n", i)
				fmt.Printf("    ID:   %s\n", part.FunctionCall.ID)
				fmt.Printf("    Name: %s\n", part.FunctionCall.Name)
				fmt.Printf("    Args: %v\n", part.FunctionCall.Args)
			} else if part.Text != "" {
				fmt.Printf("  Part[%d] Text: %s\n", i, part.Text)
			}
		}

		fmt.Printf("  FinishReason: %v\n", resp.FinishReason)
	}
}

func testToolCallingStream(ctx context.Context, provider *sapaicore.Provider) {
	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Get the weather in Tokyo and the population of France."}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "get_weather",
						Description: "Get current weather for a city",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"city": {Type: genai.TypeString, Description: "City name"},
							},
							Required: []string{"city"},
						},
					},
					{
						Name:        "get_population",
						Description: "Get the population of a country",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"country": {Type: genai.TypeString, Description: "Country name"},
							},
							Required: []string{"country"},
						},
					},
				},
			}},
		},
	}

	var chunks int

	for resp, err := range llm.GenerateContent(ctx, req, true) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			return
		}

		if resp.Partial {
			chunks++
			continue
		}

		// Final response with assembled tool calls.
		fmt.Printf("  Chunks received: %d\n", chunks)

		if resp.Content != nil {
			for i, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					fmt.Printf("  Part[%d] FunctionCall: %s(%v)\n", i, part.FunctionCall.Name, part.FunctionCall.Args)
				}
			}
		}

		fmt.Printf("  TurnComplete: %v\n", resp.TurnComplete)
	}
}

func testMultiTurn(ctx context.Context, provider *sapaicore.Provider) {
	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	// Simulate a conversation: user sets context, assistant acknowledges, user asks recall.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "My name is Pedro. Remember it."}}, Role: "user"},
			{Parts: []*genai.Part{{Text: "Got it! Your name is Pedro. I'll remember that."}}, Role: "model"},
			{Parts: []*genai.Part{{Text: "What is my name?"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a helpful assistant. Be concise."}},
			},
		},
	}

	fmt.Printf("  Messages: [user→assistant→user] testing context recall\n")

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			return
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			text := resp.Content.Parts[0].Text
			fmt.Printf("  Response: %s\n", text)

			if strings.Contains(strings.ToLower(text), "pedro") {
				fmt.Printf("  ✅ Multi-turn context preserved\n")
			} else {
				fmt.Printf("  ⚠️  Response doesn't mention 'Pedro' — context may be lost\n")
			}
		}
	}
}

func testExtendedThinking(ctx context.Context, provider *sapaicore.Provider) {
	llm, err := provider.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"thinking":   map[string]any{"type": "enabled", "budget_tokens": 8192},
			"max_tokens": 16000,
		}),
	)
	if err != nil {
		fmt.Printf("  ERROR: %v\n", err)
		return
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What is 7 * 13 * 29? Think step by step."}}, Role: "user"},
		},
	}

	fmt.Printf("  Model: %s\n", llm.Name())
	fmt.Printf("  Thinking: enabled (budget: 8192 tokens)\n")

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
			fmt.Printf("  (Extended thinking may not be supported in orchestration mode)\n")

			return
		}

		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Response: %s\n", resp.Content.Parts[0].Text)
		}

		if resp.UsageMetadata != nil {
			fmt.Printf("  Tokens: prompt=%d completion=%d\n",
				resp.UsageMetadata.PromptTokenCount,
				resp.UsageMetadata.CandidatesTokenCount)

			// Extended thinking typically uses more completion tokens.
			if resp.UsageMetadata.CandidatesTokenCount > 200 {
				fmt.Printf("  ✅ High completion tokens suggests thinking was active\n")
			}
		}
	}
}

func testErrorHandling(ctx context.Context, provider *sapaicore.Provider) {
	testInvalidModel(ctx, provider)
	testEmptyMessage(ctx, provider)
	testStopSequences(ctx, provider)
	testTemperatureZero(ctx, provider)
}

func testInvalidModel(ctx context.Context, provider *sapaicore.Provider) {
	fmt.Printf("\n  --- Invalid model name ---\n")

	llm, err := provider.Model("nonexistent-model-xyz-12345")
	if err != nil {
		fmt.Printf("  Model() error (unexpected in orchestration mode): %v\n", err)
		return
	}

	for _, err := range llm.GenerateContent(ctx, newSimpleReq("hi"), false) {
		if err != nil {
			fmt.Printf("  ✅ Got expected error: %v\n", err)
		} else {
			fmt.Printf("  ⚠️  No error — orchestration may have routed to a default model\n")
		}

		break
	}
}

func testEmptyMessage(ctx context.Context, provider *sapaicore.Provider) {
	fmt.Printf("\n  --- Empty message ---\n")

	llm, _ := provider.Model("gpt-4.1-mini")

	emptyReq := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: ""}}, Role: "user"},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, emptyReq, false) {
		if err != nil {
			fmt.Printf("  Error (may be expected): %v\n", err)
		} else if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Response: %s\n", resp.Content.Parts[0].Text)
		}

		break
	}
}

func testStopSequences(ctx context.Context, provider *sapaicore.Provider) {
	fmt.Printf("\n  --- Stop sequences ---\n")

	llm, _ := provider.Model("gpt-4.1-mini")

	stopReq := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Count from 1 to 20, writing each number on a new line."}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			StopSequences: []string{"5"},
		},
	}

	for resp, err := range llm.GenerateContent(ctx, stopReq, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
		} else if resp.Content != nil && len(resp.Content.Parts) > 0 {
			text := resp.Content.Parts[0].Text
			fmt.Printf("  Response: %s\n", text)

			if !strings.Contains(text, "6") {
				fmt.Printf("  ✅ Stop sequence worked (stopped before '6')\n")
			} else {
				fmt.Printf("  ⚠️  Stop sequence may not have been honored\n")
			}
		}

		break
	}
}

func testTemperatureZero(ctx context.Context, provider *sapaicore.Provider) {
	fmt.Printf("\n  --- Temperature 0 (deterministic) ---\n")

	llm, _ := provider.Model("gpt-4.1-mini")

	temp := float32(0)

	tempReq := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What is 2+2? Reply with just the number."}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Temperature: &temp,
		},
	}

	for resp, err := range llm.GenerateContent(ctx, tempReq, false) {
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
		} else if resp.Content != nil && len(resp.Content.Parts) > 0 {
			fmt.Printf("  Response: %s\n", resp.Content.Parts[0].Text)
			fmt.Printf("  ✅ Temperature 0 accepted\n")
		}

		break
	}
}

func newSimpleReq(text string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: text}}, Role: "user"},
		},
	}
}

func requireEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("missing required env var: %s", key)
	}

	return v
}
