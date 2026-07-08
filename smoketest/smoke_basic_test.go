//go:build smoke

package smoketest_test

import (
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
	"google.golang.org/adk/v2/model"
)

func TestSmoke_NonStreaming(t *testing.T) {
	provider := newProvider(t)

	for _, modelName := range testModels {
		t.Run(modelName, func(t *testing.T) {
			ctx := withTimeout(t, 30*time.Second)

			llm, err := provider.Model(modelName)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{Parts: []*genai.Part{{Text: "What is the capital of France? Reply with just the city name."}}, Role: "user"},
				},
			}

			resp := generateOne(t, ctx, llm, req)

			text := requireText(t, resp)
			if !strings.Contains(strings.ToLower(text), "paris") {
				t.Errorf("expected 'paris' in response, got: %q", text)
			}

			if resp.UsageMetadata == nil {
				t.Error("missing usage metadata")
			} else {
				if resp.UsageMetadata.PromptTokenCount == 0 {
					t.Error("PromptTokenCount = 0")
				}
				if resp.UsageMetadata.CandidatesTokenCount == 0 {
					t.Error("CandidatesTokenCount = 0")
				}
			}

			if resp.FinishReason != genai.FinishReasonStop {
				t.Errorf("FinishReason = %v, want Stop", resp.FinishReason)
			}

			t.Logf("response=%q model=%s tokens=%+v", text, resp.ModelVersion, resp.UsageMetadata)
		})
	}
}

func TestSmoke_Streaming(t *testing.T) {
	provider := newProvider(t)

	for _, modelName := range testModels {
		t.Run(modelName, func(t *testing.T) {
			ctx := withTimeout(t, 30*time.Second)

			llm, err := provider.Model(modelName)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			req := simpleReq("Count from 1 to 5, separating each number with a comma.")

			partials, final := generateStream(t, ctx, llm, req)

			if len(partials) == 0 {
				t.Error("expected at least one partial chunk")
			}

			text := requireText(t, final)
			if !strings.Contains(text, "3") && !strings.Contains(strings.ToLower(text), "three") {
				t.Errorf("expected text to contain '3' or 'three', got: %q", text)
			}

			if !final.TurnComplete {
				t.Error("final response not marked TurnComplete")
			}

			t.Logf("chunks=%d final=%q", len(partials), text)
		})
	}
}

func TestSmoke_ModelParams(t *testing.T) {
	// Contrast test: proves max_tokens param is actually forwarded and respected.
	// A long essay prompt WITHOUT max_tokens should produce >50 words.
	// Same prompt WITH max_tokens=50 should be truncated with FinishReason=MaxTokens.
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	prompt := "Write a detailed essay about the history of computing from the 1940s to today."

	// Step 1: Without max_tokens limit — should be verbose.
	llmUnlimited, _ := provider.Model("gpt-4.1-mini")
	respLong := generateOne(t, ctx, llmUnlimited, simpleReq(prompt))
	textLong := requireText(t, respLong)
	wordsLong := len(strings.Fields(textLong))

	// Step 2: With max_tokens=50 — should be much shorter.
	llmLimited, err := provider.Model("gpt-4.1-mini",
		sapaicore.WithModelParams(map[string]any{"max_tokens": 50}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	respShort := generateOne(t, ctx, llmLimited, simpleReq(prompt))
	textShort := requireText(t, respShort)
	wordsShort := len(strings.Fields(textShort))

	t.Logf("unlimited=%d words, limited=%d words", wordsLong, wordsShort)

	if wordsShort >= wordsLong {
		t.Errorf("max_tokens not effective: limited=%d words >= unlimited=%d words", wordsShort, wordsLong)
	}

	if respShort.FinishReason != genai.FinishReasonMaxTokens {
		t.Errorf("FinishReason = %v, want MaxTokens (proves truncation)", respShort.FinishReason)
	}
}

func TestSmoke_MultiTurn(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "My name is Pedro. Remember it."}}, Role: "user"},
			{Parts: []*genai.Part{{Text: "Got it! Your name is Pedro."}}, Role: "model"},
			{Parts: []*genai.Part{{Text: "What is my name?"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "Be concise."}},
			},
		},
	}

	resp := generateOne(t, ctx, llm, req)

	text := requireText(t, resp)
	if !strings.Contains(strings.ToLower(text), "pedro") {
		t.Errorf("multi-turn context lost: response=%q", text)
	}

	t.Logf("response=%q", text)
}

func TestSmoke_StopSequences(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	temp := float32(0)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Count from 1 to 20, each number on a new line."}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			StopSequences: []string{"5"},
			Temperature:   &temp,
		},
	}

	resp := generateOne(t, ctx, llm, req)

	text := requireText(t, resp)
	if strings.Contains(text, "6") {
		t.Errorf("stop sequence not honored, text contains '6': %q", text)
	}

	t.Logf("response=%q", text)
}

func TestSmoke_ResponseFormat_JSON(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Give me a person with name and age."}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Type: "OBJECT",
				Properties: map[string]*genai.Schema{
					"name": {Type: "STRING"},
					"age":  {Type: "INTEGER"},
				},
				Required: []string{"name", "age"},
			},
		},
	}

	resp := generateOne(t, ctx, llm, req)

	text := requireText(t, resp)
	if !strings.HasPrefix(strings.TrimSpace(text), "{") {
		t.Errorf("expected JSON object, got: %q", text)
	}

	t.Logf("response=%q", text)
}
