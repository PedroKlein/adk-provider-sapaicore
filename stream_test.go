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

func TestGenerateContent_Streaming(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		chunks := []string{
			`{"id":"chatcmpl-1","model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-1","model":"gpt-4.1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     inferenceServer.URL,
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      authServer.URL + "/oauth/token",
		Deployments:  map[string]string{"gpt-4.1": "deploy-1"},
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, _ := provider.Model("gpt-4.1")

	var responses []*model.LLMResponse

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("Hi"), true) {
		if err != nil {
			t.Fatalf("GenerateContent stream error: %v", err)
		}

		responses = append(responses, resp)
	}

	// Expect 3 partial responses + 1 final aggregated response.
	if len(responses) != 4 {
		t.Fatalf("expected 4 responses (3 partial + 1 final), got %d", len(responses))
	}

	// Verify partial responses.
	partials := responses[:3]
	expectedTexts := []string{"Hello", " world", "!"}

	for i, p := range partials {
		if !p.Partial {
			t.Errorf("response %d: expected Partial=true", i)
		}

		if p.Content == nil || len(p.Content.Parts) == 0 {
			t.Fatalf("response %d: no content", i)
		}

		if got := p.Content.Parts[0].Text; got != expectedTexts[i] {
			t.Errorf("response %d: text=%q, want %q", i, got, expectedTexts[i])
		}
	}

	// Verify final aggregated response.
	final := responses[3]

	if final.Partial {
		t.Error("final response should not be partial")
	}

	if !final.TurnComplete {
		t.Error("final response should have TurnComplete=true")
	}

	if final.Content == nil || len(final.Content.Parts) == 0 {
		t.Fatal("final response has no content")
	}

	if got := final.Content.Parts[0].Text; got != "Hello world!" {
		t.Errorf("final text = %q, want %q", got, "Hello world!")
	}

	if final.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want Stop", final.FinishReason)
	}

	if final.UsageMetadata == nil {
		t.Error("expected usage metadata in final response")
	} else if final.UsageMetadata.PromptTokenCount != 5 {
		t.Errorf("prompt tokens = %d, want 5", final.UsageMetadata.PromptTokenCount)
	}
}

func TestGenerateContent_StreamingToolCalls(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")

		chunks := []string{
			// First chunk: tool call start with name.
			`{"id":"chatcmpl-2","model":"gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}`,
			// Second chunk: tool call arguments.
			`{"id":"chatcmpl-2","model":"gpt-4.1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"query\":"}}]},"finish_reason":null}]}`,
			// Third chunk: more arguments.
			`{"id":"chatcmpl-2","model":"gpt-4.1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"hello\"}"}}]},"finish_reason":"tool_calls"}]}`,
		}

		for _, chunk := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", chunk)
		}

		fmt.Fprint(w, "data: [DONE]\n\n")
		w.(http.Flusher).Flush()
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     inferenceServer.URL,
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      authServer.URL + "/oauth/token",
		Deployments:  map[string]string{"m": "d"},
	})

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
		t.Fatal("no final response received")
	}

	if final.Content == nil || len(final.Content.Parts) == 0 {
		t.Fatal("final response has no parts")
	}

	fc := final.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall in final response")
	}

	if fc.ID != "call_abc" {
		t.Errorf("function call ID = %q, want %q", fc.ID, "call_abc")
	}

	if fc.Name != "search" {
		t.Errorf("function name = %q, want %q", fc.Name, "search")
	}

	query, _ := fc.Args["query"].(string)
	if query != "hello" {
		t.Errorf("args.query = %q, want %q", query, "hello")
	}

	if final.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want Stop (tool_calls maps to Stop)", final.FinishReason)
	}
}

func TestGenerateContent_StreamingError(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limited","type":"rate_limit","code":"429"}}`))
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     inferenceServer.URL,
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      authServer.URL + "/oauth/token",
		Deployments:  map[string]string{"m": "d"},
	})

	llm, _ := provider.Model("m")

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("hi"), true) {
		if err == nil {
			t.Fatal("expected error for 429 response")
		}

		return
	}

	t.Fatal("expected at least one yield from GenerateContent")
}
