package sapaicore_test

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/go-adk-sap-ai-core"
	"google.golang.org/adk/v2/model"
)

func TestOrchestration_Streaming(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")

		// Orchestration wraps chunks in {request_id, final_result: {...}}
		chunks := []string{
			`{"request_id":"r1","final_result":{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}}`,
			`{"request_id":"r1","final_result":{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":" world"}}]}}`,
			`{"request_id":"r1","final_result":{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}}`,
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("gpt-4.1")

	var responses []*model.LLMResponse

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("Hi"), true) {
		if err != nil {
			t.Fatalf("GenerateContent stream error: %v", err)
		}

		responses = append(responses, resp)
	}

	// 3 partial + 1 final.
	if len(responses) != 4 {
		t.Fatalf("expected 4 responses, got %d", len(responses))
	}

	expectedTexts := []string{"Hello", " world", "!"}

	for i, p := range responses[:3] {
		if !p.Partial {
			t.Errorf("response %d: expected Partial=true", i)
		}

		if p.Content.Parts[0].Text != expectedTexts[i] {
			t.Errorf("response %d: text=%q, want %q", i, p.Content.Parts[0].Text, expectedTexts[i])
		}
	}

	final := responses[3]

	if final.Partial {
		t.Error("final should not be partial")
	}

	if !final.TurnComplete {
		t.Error("final should have TurnComplete=true")
	}

	if final.Content.Parts[0].Text != "Hello world!" {
		t.Errorf("final text = %q, want %q", final.Content.Parts[0].Text, "Hello world!")
	}

	if final.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want Stop", final.FinishReason)
	}

	if final.UsageMetadata == nil || final.UsageMetadata.PromptTokenCount != 5 {
		t.Errorf("unexpected usage metadata: %+v", final.UsageMetadata)
	}
}

func TestOrchestration_StreamingToolCalls(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		chunks := []string{
			`{"request_id":"r1","final_result":{"id":"c2","model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"search","arguments":""}}]}}]}}`,
			`{"request_id":"r1","final_result":{"id":"c2","model":"gpt-4.1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":"}}]}}]}}`,
			`{"request_id":"r1","final_result":{"id":"c2","model":"gpt-4.1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hello\"}"}}]},"finish_reason":"tool_calls"}]}}`,
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("m")

	var final *model.LLMResponse

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("search"), true) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}

		if resp.TurnComplete {
			final = resp
		}
	}

	if final == nil {
		t.Fatal("no final response")
	}

	fc := final.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall in final")
	}

	if fc.ID != "call_abc" {
		t.Errorf("ID = %q, want %q", fc.ID, "call_abc")
	}

	if fc.Name != "search" {
		t.Errorf("name = %q, want %q", fc.Name, "search")
	}

	query, _ := fc.Args["query"].(string)
	if query != "hello" {
		t.Errorf("args.query = %q, want %q", query, "hello")
	}
}

func TestFoundation_Streaming(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		// Foundation mode: flat chunks (no request_id wrapper).
		chunks := []string{
			`{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
			`{"id":"c1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}`,
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"m": "d"}),
	)

	llm, _ := provider.Model("m")

	var responses []*model.LLMResponse

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("hey"), true) {
		if err != nil {
			t.Fatalf("error: %v", err)
		}

		responses = append(responses, resp)
	}

	// 2 partial + 1 final.
	if len(responses) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(responses))
	}

	final := responses[2]
	if final.Content.Parts[0].Text != "Hi!" {
		t.Errorf("final text = %q, want %q", final.Content.Parts[0].Text, "Hi!")
	}
}

func TestStreaming_ErrorResponse(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limited","code":429}}`))
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("m")

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("hi"), true) {
		if err == nil {
			t.Fatal("expected error for 429")
		}

		return
	}

	t.Fatal("expected at least one yield")
}
