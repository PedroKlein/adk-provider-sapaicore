package sapaicore_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/go-adk-sap-ai-core"
	"google.golang.org/adk/v2/model"
)

func TestNewProvider_ValidatesConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  sapaicore.Config
	}{
		{
			name: "missing endpoint",
			cfg: sapaicore.Config{
				ClientID:     "id",
				ClientSecret: "secret",
				AuthURL:      "https://auth.example.com/token",
				Deployments:  map[string]string{"m": "d"},
			},
		},
		{
			name: "missing client ID",
			cfg: sapaicore.Config{
				Endpoint:     "https://api.example.com",
				ClientSecret: "secret",
				AuthURL:      "https://auth.example.com/token",
				Deployments:  map[string]string{"m": "d"},
			},
		},
		{
			name: "missing client secret",
			cfg: sapaicore.Config{
				Endpoint:    "https://api.example.com",
				ClientID:    "id",
				AuthURL:     "https://auth.example.com/token",
				Deployments: map[string]string{"m": "d"},
			},
		},
		{
			name: "missing auth URL",
			cfg: sapaicore.Config{
				Endpoint:     "https://api.example.com",
				ClientID:     "id",
				ClientSecret: "secret",
				Deployments:  map[string]string{"m": "d"},
			},
		},
		{
			name: "empty deployments",
			cfg: sapaicore.Config{
				Endpoint:     "https://api.example.com",
				ClientID:     "id",
				ClientSecret: "secret",
				AuthURL:      "https://auth.example.com/token",
				Deployments:  map[string]string{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := sapaicore.NewProvider(tt.cfg)
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			if !errors.Is(err, sapaicore.ErrMissingConfig) {
				t.Errorf("expected ErrMissingConfig, got: %v", err)
			}
		})
	}
}

func TestProvider_Model_NotFound(t *testing.T) {
	t.Parallel()

	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     "https://api.example.com",
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      "https://auth.example.com/token",
		Deployments:  map[string]string{"gpt-4.1": "deploy-1"},
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	_, err = provider.Model("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent model")
	}

	if !errors.Is(err, sapaicore.ErrDeploymentNotFound) {
		t.Errorf("expected ErrDeploymentNotFound, got: %v", err)
	}
}

func TestProvider_Model_DefaultResourceGroup(t *testing.T) {
	t.Parallel()

	var capturedResourceGroup string

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedResourceGroup = r.Header.Get("AI-Resource-Group")

		writeSimpleResponse(w)
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     inferenceServer.URL,
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      authServer.URL + "/oauth/token",
		Deployments:  map[string]string{"m": "d"},
		// ResourceGroup intentionally omitted — should default to "default".
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, _ := provider.Model("m")

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("test"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	if capturedResourceGroup != "default" {
		t.Errorf("resource group = %q, want %q", capturedResourceGroup, "default")
	}
}

func TestGenerateContent_NonStreaming(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request structure.
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}

		if ct := r.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("expected application/json, got %s", ct)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "chatcmpl-test",
			"model": "gpt-4.1",
			"choices": []map[string]any{{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello from SAP AI Core!",
				},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens":     12,
				"completion_tokens": 8,
				"total_tokens":      20,
			},
		})
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:      inferenceServer.URL,
		ClientID:      "id",
		ClientSecret:  "secret",
		AuthURL:       authServer.URL + "/oauth/token",
		ResourceGroup: "my-group",
		Deployments:   map[string]string{"gpt-4.1": "deploy-xyz"},
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{{Text: "Hello"}},
				Role:  "user",
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are helpful."}},
			},
			Temperature: ptrFloat32(0.7),
		},
	}

	var responses []*model.LLMResponse

	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent error: %v", err)
		}

		responses = append(responses, resp)
	}

	if len(responses) != 1 {
		t.Fatalf("expected 1 response, got %d", len(responses))
	}

	resp := responses[0]

	// Verify response content.
	if resp.Content == nil {
		t.Fatal("response content is nil")
	}

	if len(resp.Content.Parts) == 0 {
		t.Fatal("response has no parts")
	}

	if got := resp.Content.Parts[0].Text; got != "Hello from SAP AI Core!" {
		t.Errorf("text = %q, want %q", got, "Hello from SAP AI Core!")
	}

	if resp.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish reason = %v, want Stop", resp.FinishReason)
	}

	if !resp.TurnComplete {
		t.Error("expected TurnComplete=true")
	}

	// Verify request body was correctly converted.
	messages, _ := capturedBody["messages"].([]any)
	if len(messages) < 2 {
		t.Fatalf("expected >= 2 messages, got %d", len(messages))
	}

	// First message should be system instruction.
	sysMsg, _ := messages[0].(map[string]any)
	if sysMsg["role"] != "system" {
		t.Errorf("first message role = %v, want system", sysMsg["role"])
	}

	if sysMsg["content"] != "You are helpful." {
		t.Errorf("system content = %v, want %q", sysMsg["content"], "You are helpful.")
	}

	// Verify stream=false in request.
	if stream, _ := capturedBody["stream"].(bool); stream {
		t.Error("expected stream=false in request body")
	}
}

func TestGenerateContent_ToolCalls(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "chatcmpl-tc",
			"model": "gpt-4.1",
			"choices": []map[string]any{{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"tool_calls": []map[string]any{{
						"id":   "call_123",
						"type": "function",
						"function": map[string]any{
							"name":      "get_weather",
							"arguments": `{"location":"Berlin"}`,
						},
					}},
				},
				"finish_reason": "tool_calls",
			}},
		})
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

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("weather?"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}

		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatal("expected parts with function call")
		}

		fc := resp.Content.Parts[0].FunctionCall
		if fc == nil {
			t.Fatal("expected FunctionCall part")
		}

		if fc.Name != "get_weather" {
			t.Errorf("function name = %q, want %q", fc.Name, "get_weather")
		}

		if fc.ID != "call_123" {
			t.Errorf("function call ID = %q, want %q", fc.ID, "call_123")
		}

		loc, _ := fc.Args["location"].(string)
		if loc != "Berlin" {
			t.Errorf("args.location = %q, want %q", loc, "Berlin")
		}
	}
}

func TestGenerateContent_FunctionResponse(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)
		writeSimpleResponse(w)
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

	// Simulate a conversation with a function response.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{{Text: "What's the weather?"}},
				Role:  "user",
			},
			{
				Parts: []*genai.Part{{
					FunctionCall: &genai.FunctionCall{
						ID:   "call_456",
						Name: "get_weather",
						Args: map[string]any{"location": "Munich"},
					},
				}},
				Role: "model",
			},
			{
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       "call_456",
						Name:     "get_weather",
						Response: map[string]any{"temperature": 22, "unit": "celsius"},
					},
				}},
				Role: "user",
			},
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	// Verify the function response was converted to a tool message.
	messages, _ := capturedBody["messages"].([]any)
	if len(messages) < 3 {
		t.Fatalf("expected >= 3 messages, got %d", len(messages))
	}

	// Find the tool message.
	var toolMsg map[string]any

	for _, m := range messages {
		msg, _ := m.(map[string]any)
		if msg["role"] == "tool" {
			toolMsg = msg

			break
		}
	}

	if toolMsg == nil {
		t.Fatal("no tool message found in request")
	}

	if toolMsg["tool_call_id"] != "call_456" {
		t.Errorf("tool_call_id = %v, want %q", toolMsg["tool_call_id"], "call_456")
	}
}

// --- Helpers ---

func newMockAuthServer(t *testing.T) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "mock-token",
			"expires_in":   3600,
			"token_type":   "bearer",
		})
	}))
}

func newSimpleRequest(text string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{{Text: text}},
				Role:  "user",
			},
		},
	}
}

func writeSimpleResponse(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id":    "chatcmpl-1",
		"model": "gpt-4.1",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": "ok"},
			"finish_reason": "stop",
		}},
	})
}

func ptrFloat32(f float32) *float32 {
	return &f
}
