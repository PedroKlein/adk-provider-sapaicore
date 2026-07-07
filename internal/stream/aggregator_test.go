package stream_test

import (
	"testing"

	"google.golang.org/genai"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/stream"
)

func TestParseSSELine(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		line    string
		wantStr string
		wantOK  bool
	}{
		{"valid data line", "data: hello", "hello", true},
		{"done marker", "data: [DONE]", "[DONE]", true},
		{"json payload", `data: {"id":"c1"}`, `{"id":"c1"}`, true},
		{"empty prefix", "event: message", "", false},
		{"empty line", "", "", false},
		{"no space after colon", "data:nospace", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, ok := stream.ParseSSELine(tt.line)

			if ok != tt.wantOK {
				t.Errorf("ok = %v, want %v", ok, tt.wantOK)
			}

			if got != tt.wantStr {
				t.Errorf("data = %q, want %q", got, tt.wantStr)
			}
		})
	}
}

func TestAggregator_TextStreaming(t *testing.T) {
	t.Parallel()

	chunks := []string{
		`{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
	}

	var agg stream.Aggregator

	var partials []string

	for _, chunk := range chunks {
		resp := agg.ProcessChunk(stream.ModeFoundation, chunk)
		if resp != nil {
			partials = append(partials, resp.Content.Parts[0].Text)
		}
	}

	if len(partials) != 3 {
		t.Fatalf("partials = %d, want 3", len(partials))
	}

	wantTexts := []string{"Hello", " world", "!"}
	for i, want := range wantTexts {
		if partials[i] != want {
			t.Errorf("partial[%d] = %q, want %q", i, partials[i], want)
		}
	}

	final := agg.Finalize()

	if !final.TurnComplete {
		t.Error("expected TurnComplete=true")
	}

	if final.Content.Parts[0].Text != "Hello world!" {
		t.Errorf("final text = %q, want %q", final.Content.Parts[0].Text, "Hello world!")
	}

	if final.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want Stop", final.FinishReason)
	}

	if final.UsageMetadata == nil || final.UsageMetadata.PromptTokenCount != 5 {
		t.Errorf("usage = %+v, want prompt=5", final.UsageMetadata)
	}
}

func TestAggregator_ToolCallStreaming(t *testing.T) {
	t.Parallel()

	chunks := []string{
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}`,
		`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hi\"}"}}]},"finish_reason":"tool_calls"}]}`,
	}

	var agg stream.Aggregator

	for _, chunk := range chunks {
		agg.ProcessChunk(stream.ModeFoundation, chunk)
	}

	final := agg.Finalize()

	if len(final.Content.Parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(final.Content.Parts))
	}

	fc := final.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall")
	}

	if fc.ID != "call_1" {
		t.Errorf("ID = %q, want call_1", fc.ID)
	}

	if fc.Name != "search" {
		t.Errorf("Name = %q, want search", fc.Name)
	}

	q, _ := fc.Args["q"].(string)
	if q != "hi" {
		t.Errorf("q = %q, want hi", q)
	}
}

func TestAggregator_OrchestrationMode(t *testing.T) {
	t.Parallel()

	chunks := []string{
		`{"request_id":"r1","final_result":{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"Hi"}}]}}`,
		`{"request_id":"r1","final_result":{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}}`,
	}

	var agg stream.Aggregator

	for _, chunk := range chunks {
		agg.ProcessChunk(stream.ModeOrchestration, chunk)
	}

	final := agg.Finalize()

	if final.Content.Parts[0].Text != "Hi!" {
		t.Errorf("text = %q, want %q", final.Content.Parts[0].Text, "Hi!")
	}

	if final.ModelVersion != "gpt-4.1" {
		t.Errorf("model = %q, want gpt-4.1", final.ModelVersion)
	}
}

func TestAggregator_InvalidJSON(t *testing.T) {
	t.Parallel()

	var agg stream.Aggregator

	resp := agg.ProcessChunk(stream.ModeFoundation, "not json")
	if resp != nil {
		t.Error("expected nil for invalid JSON")
	}

	resp = agg.ProcessChunk(stream.ModeOrchestration, "also not json")
	if resp != nil {
		t.Error("expected nil for invalid JSON in orchestration mode")
	}
}

func TestAggregator_EmptyFinalize(t *testing.T) {
	t.Parallel()

	var agg stream.Aggregator

	final := agg.Finalize()

	if !final.TurnComplete {
		t.Error("expected TurnComplete=true")
	}

	if len(final.Content.Parts) != 0 {
		t.Errorf("expected empty parts, got %d", len(final.Content.Parts))
	}
}
