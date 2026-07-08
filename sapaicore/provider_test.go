package sapaicore_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"

	"github.com/PedroKlein/adk-provider-sapaicore/sapaicore"
	"google.golang.org/adk/v2/model"
)

// --- Provider validation tests ---

func TestNewProvider_ValidatesConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		opts []sapaicore.Option
	}{
		{
			name: "missing endpoint",
			opts: []sapaicore.Option{
				sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
				sapaicore.WithDeploymentID("d"),
			},
		},
		{
			name: "missing auth",
			opts: []sapaicore.Option{
				sapaicore.WithEndpoint("https://api.example.com"),
				sapaicore.WithDeploymentID("d"),
			},
		},
		{
			name: "both deployment modes",
			opts: []sapaicore.Option{
				sapaicore.WithEndpoint("https://api.example.com"),
				sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
				sapaicore.WithDeploymentID("d"),
				sapaicore.WithDeployments(map[string]string{"m": "d"}),
			},
		},
		{
			name: "orchestration and deployment ID",
			opts: []sapaicore.Option{
				sapaicore.WithEndpoint("https://api.example.com"),
				sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
				sapaicore.WithOrchestration(),
				sapaicore.WithDeploymentID("d"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := sapaicore.NewProvider(t.Context(), tt.opts...)
			if err == nil {
				t.Fatal("expected error, got nil")
			}

			if !errors.Is(err, sapaicore.ErrMissingConfig) {
				t.Errorf("expected ErrMissingConfig, got: %v", err)
			}
		})
	}
}

func TestProvider_Model_NotFound_FoundationMode(t *testing.T) {
	t.Parallel()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint("https://api.example.com"),
		sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
		sapaicore.WithDeployments(map[string]string{"gpt-4.1": "deploy-1"}),
	)
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

func TestProvider_Model_AnyName_OrchestrationMode(t *testing.T) {
	t.Parallel()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint("https://api.example.com"),
		sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
		sapaicore.WithDeploymentID("orch-deploy"),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	// Any model name should work in orchestration mode.
	llm, err := provider.Model("anthropic--claude-4.5-sonnet")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm.Name() != "anthropic--claude-4.5-sonnet" {
		t.Errorf("name = %q, want %q", llm.Name(), "anthropic--claude-4.5-sonnet")
	}
}

func TestProvider_WithOrchestration_AutoDiscovers(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)

	// Mock the deployments API.
	deploymentsServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v2/lm/deployments" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)

			return
		}

		if scenario := r.URL.Query().Get("scenarioId"); scenario != "orchestration" {
			t.Errorf("scenarioId = %q, want %q", scenario, "orchestration")
		}

		if status := r.URL.Query().Get("status"); status != "RUNNING" {
			t.Errorf("status = %q, want %q", status, "RUNNING")
		}

		if rg := r.Header.Get("AI-Resource-Group"); rg != "default" {
			t.Errorf("resource group = %q, want %q", rg, "default")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resources": []map[string]any{
				{"id": "discovered-deploy-id", "scenarioId": "orchestration", "status": "RUNNING"},
			},
		})
	}))
	defer deploymentsServer.Close()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(deploymentsServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithOrchestration(),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	// Should be able to create models (deployment was discovered).
	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm.Name() != "gpt-4.1" {
		t.Errorf("name = %q, want %q", llm.Name(), "gpt-4.1")
	}
}

func TestProvider_WithOrchestration_NoDeploymentFound(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)

	deploymentsServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resources": []map[string]any{},
		})
	}))
	defer deploymentsServer.Close()

	_, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(deploymentsServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithOrchestration(),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	if !errors.Is(err, sapaicore.ErrDiscovery) {
		t.Errorf("expected ErrDiscovery, got: %v", err)
	}
}

// --- Orchestration mode tests ---

func TestOrchestration_NonStreaming(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}

		// Verify orchestration URL path.
		if want := "/v2/inference/deployments/orch-123/v2/completion"; r.URL.Path != want {
			t.Errorf("path = %q, want %q", r.URL.Path, want)
		}

		if rg := r.Header.Get("AI-Resource-Group"); rg != "my-group" {
			t.Errorf("resource group = %q, want %q", rg, "my-group")
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "Hello from orchestration!", "stop")
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("orch-123"),
		sapaicore.WithResourceGroup("my-group"),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, _ := provider.Model("gpt-4.1")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Hello"}}, Role: "user"},
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

	if resp.Content.Parts[0].Text != "Hello from orchestration!" {
		t.Errorf("text = %q, want %q", resp.Content.Parts[0].Text, "Hello from orchestration!")
	}

	if !resp.TurnComplete {
		t.Error("expected TurnComplete=true")
	}

	// Verify orchestration request structure.
	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	modelCfg, _ := pt["model"].(map[string]any)

	if modelCfg["name"] != "gpt-4.1" {
		t.Errorf("model name = %v, want gpt-4.1", modelCfg["name"])
	}

	params, _ := modelCfg["params"].(map[string]any)
	if params["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", params["temperature"])
	}

	prompt, _ := pt["prompt"].(map[string]any)
	template, _ := prompt["template"].([]any)

	// All messages (system + user) go in template.
	if len(template) != 2 {
		t.Fatalf("expected 2 template messages (system + user), got %d", len(template))
	}

	sysMsg, _ := template[0].(map[string]any)
	if sysMsg["role"] != "system" {
		t.Errorf("template[0] role = %v, want system", sysMsg["role"])
	}

	userMsg, _ := template[1].(map[string]any)
	if userMsg["role"] != "user" {
		t.Errorf("template[1] role = %v, want user", userMsg["role"])
	}

	// No messages_history field should be present.
	if _, exists := capturedBody["messages_history"]; exists {
		t.Error("expected no messages_history field, but it was present")
	}
}

func TestOrchestration_WithModelParams(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"thinking":       map[string]any{"type": "enabled", "budget_tokens": 16384},
			"anthropic_beta": []string{"context-1m-2025-08-07"},
			"max_tokens":     200000,
		}),
	)

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("test"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	// Verify extra params were passed in model.params.
	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	modelCfg, _ := pt["model"].(map[string]any)
	params, _ := modelCfg["params"].(map[string]any)

	if params["max_tokens"] == nil {
		t.Error("expected max_tokens in model params")
	}

	thinking, _ := params["thinking"].(map[string]any)
	if thinking["type"] != "enabled" {
		t.Errorf("thinking.type = %v, want enabled", thinking["type"])
	}

	beta, _ := params["anthropic_beta"].([]any)
	if len(beta) == 0 || beta[0] != "context-1m-2025-08-07" {
		t.Errorf("anthropic_beta = %v, want [context-1m-2025-08-07]", beta)
	}
}

// --- Foundation-models mode tests ---

func TestFoundation_NonStreaming(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify foundation-models URL path.
		if want := "/v2/inference/deployments/deploy-xyz/v1/chat/completions"; r.URL.Path != want {
			t.Errorf("path = %q, want %q", r.URL.Path, want)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeFoundationResponse(w, "Hello from foundation!", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"gpt-4.1": "deploy-xyz"}),
	)

	llm, _ := provider.Model("gpt-4.1")

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("Hi"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}

		if resp.Content.Parts[0].Text != "Hello from foundation!" {
			t.Errorf("text = %q", resp.Content.Parts[0].Text)
		}
	}

	// Verify flat OpenAI format (no nested config.modules).
	if _, hasConfig := capturedBody["config"]; hasConfig {
		t.Error("foundation mode should not have nested config")
	}

	if capturedBody["model"] != "gpt-4.1" {
		t.Errorf("model = %v, want gpt-4.1", capturedBody["model"])
	}
}

func TestFoundation_ToolCalls(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)

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

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"m": "d"}),
	)

	llm, _ := provider.Model("m")

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("weather?"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
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
	}
}

// --- Helpers ---

func newMockAuthServer(t *testing.T) *httptest.Server {
	t.Helper()

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "mock-token",
			"expires_in":   3600,
			"token_type":   "bearer",
		})
	}))
	t.Cleanup(s.Close)

	return s
}

func newSimpleRequest(text string) *model.LLMRequest {
	return &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: text}}, Role: "user"},
		},
	}
}

func writeOrchestrationResponse(w http.ResponseWriter, content, finishReason string) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"request_id": "req-123",
		"final_result": map[string]any{
			"id":    "chatcmpl-1",
			"model": "gpt-4.1",
			"choices": []map[string]any{{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": content},
				"finish_reason": finishReason,
			}},
			"usage": map[string]any{
				"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
			},
		},
	})
}

func writeFoundationResponse(w http.ResponseWriter, content, finishReason string) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id":    "chatcmpl-1",
		"model": "gpt-4.1",
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": content},
			"finish_reason": finishReason,
		}},
	})
}

func ptrFloat32(f float32) *float32 {
	return &f
}

func TestOrchestration_FunctionResponseFormat(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "The weather in Berlin is 22°C and sunny.", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("orch-123"),
	)

	llm, _ := provider.Model("gpt-4.1-mini")

	// Simulate a tool call round-trip: user → assistant(tool_call) → tool result.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What is the weather in Berlin?"}}, Role: "user"},
			{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID:   "call_abc",
				Name: "get_weather",
				Args: map[string]any{"city": "Berlin"},
			}}}, Role: "model"},
			{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID:       "call_abc",
				Name:     "get_weather",
				Response: map[string]any{"temp": "22°C", "condition": "sunny"},
			}}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are a weather assistant."}},
			},
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get weather",
				}},
			}},
		},
	}

	ctx := t.Context()

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}

		_ = resp
	}

	// Verify the wire format: all messages in template, tool role preserved.
	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	prompt, _ := pt["prompt"].(map[string]any)
	template, _ := prompt["template"].([]any)

	// All messages in template: system, user, assistant(tool_calls), tool.
	if len(template) != 4 {
		t.Fatalf("expected 4 template messages, got %d", len(template))
	}

	assistantMsg, _ := template[2].(map[string]any)
	if assistantMsg["role"] != "assistant" {
		t.Errorf("template[2] role = %v, want assistant", assistantMsg["role"])
	}

	// Assistant message must have content field (even if empty).
	if _, hasContent := assistantMsg["content"]; !hasContent {
		t.Error("assistant message missing content field")
	}

	toolCalls, _ := assistantMsg["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool_call, got %d", len(toolCalls))
	}

	toolMsg, _ := template[3].(map[string]any)
	if toolMsg["role"] != "tool" {
		t.Errorf("template[3] role = %v, want tool", toolMsg["role"])
	}

	if toolMsg["tool_call_id"] != "call_abc" {
		t.Errorf("tool_call_id = %v, want call_abc", toolMsg["tool_call_id"])
	}

	// Tool message should NOT have a name field (SDK doesn't include it).
	if _, hasName := toolMsg["name"]; hasName {
		t.Error("tool message should not have name field")
	}

	// No messages_history.
	if _, exists := capturedBody["messages_history"]; exists {
		t.Error("expected no messages_history, but it was present")
	}
}

func TestOrchestration_RefusalHandling(t *testing.T) {
	t.Parallel()

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"request_id": "r1",
			"final_result": map[string]any{
				"id":    "c1",
				"model": "gpt-4.1",
				"choices": []map[string]any{{
					"index":         0,
					"message":       map[string]any{"role": "assistant", "content": "", "refusal": "I cannot help with that."},
					"finish_reason": "stop",
				}},
				"usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
			},
		})
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("gpt-4.1")

	for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("do something bad"), false) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if resp.ErrorCode != "refusal" {
			t.Errorf("ErrorCode = %q, want \"refusal\"", resp.ErrorCode)
		}

		if resp.ErrorMessage != "I cannot help with that." {
			t.Errorf("ErrorMessage = %q, want \"I cannot help with that.\"", resp.ErrorMessage)
		}
	}
}

func TestOrchestration_ResponseFormat(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, `{"name":"test"}`, "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("gpt-4.1")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Give me a person"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema: &genai.Schema{
				Type: "OBJECT",
				Properties: map[string]*genai.Schema{
					"name": {Type: "STRING"},
					"age":  {Type: "INTEGER"},
				},
				Required: []string{"name"},
			},
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	prompt, _ := pt["prompt"].(map[string]any)
	rf, _ := prompt["response_format"].(map[string]any)

	if rf == nil {
		t.Fatal("expected response_format in prompt, got nil")
	}

	if rf["type"] != "json_schema" {
		t.Errorf("response_format.type = %v, want json_schema", rf["type"])
	}

	js, _ := rf["json_schema"].(map[string]any)
	if js == nil {
		t.Fatal("expected json_schema object")
	}

	schema, _ := js["schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Errorf("schema.type = %v, want object", schema["type"])
	}

	props, _ := schema["properties"].(map[string]any)
	if props["name"] == nil {
		t.Error("expected name property in schema")
	}
}

func TestOrchestration_TimeoutAndRetries(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
		sapaicore.WithTimeout(30),
		sapaicore.WithMaxRetries(3),
	)

	llm, _ := provider.Model("gpt-4.1")

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("hi"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	modelCfg, _ := pt["model"].(map[string]any)

	if modelCfg["timeout"] != float64(30) {
		t.Errorf("timeout = %v, want 30", modelCfg["timeout"])
	}

	if modelCfg["max_retries"] != float64(3) {
		t.Errorf("max_retries = %v, want 3", modelCfg["max_retries"])
	}
}

func TestOrchestration_DefaultTimeoutAndRetries(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("gpt-4.1")

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("hi"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	modelCfg, _ := pt["model"].(map[string]any)

	if modelCfg["timeout"] != float64(600) {
		t.Errorf("timeout = %v, want 600 (default)", modelCfg["timeout"])
	}

	if modelCfg["max_retries"] != float64(2) {
		t.Errorf("max_retries = %v, want 2 (default)", modelCfg["max_retries"])
	}
}

func TestFoundation_WithModelParams(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeFoundationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"m": "d"}),
	)

	llm, _ := provider.Model("m",
		sapaicore.WithModelParams(map[string]any{
			"reasoning_effort": "high",
			"logprobs":         true,
		}),
	)

	for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("test"), false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	// Extra params must appear at the top level of the foundation request body.
	if capturedBody["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v, want \"high\"", capturedBody["reasoning_effort"])
	}

	if capturedBody["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", capturedBody["logprobs"])
	}

	// Standard fields should still be present.
	if capturedBody["model"] != "m" {
		t.Errorf("model = %v, want \"m\"", capturedBody["model"])
	}
}

func TestFoundation_SeedAndTopK(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeFoundationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"m": "d"}),
	)

	seed := int32(42)
	topK := float32(10)

	llm, _ := provider.Model("m")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "test"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Seed: &seed,
			TopK: &topK,
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	if capturedBody["seed"] != float64(42) {
		t.Errorf("seed = %v, want 42", capturedBody["seed"])
	}

	if capturedBody["top_k"] != float64(10) {
		t.Errorf("top_k = %v, want 10", capturedBody["top_k"])
	}
}

func TestFoundation_Logprobs(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		// Return response with logprobs.
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "chatcmpl-1",
			"model": "gpt-4.1",
			"choices": []map[string]any{{
				"index":         0,
				"message":       map[string]any{"role": "assistant", "content": "Hi"},
				"finish_reason": "stop",
				"logprobs": map[string]any{
					"content": []map[string]any{
						{
							"token":   "Hi",
							"logprob": -0.5,
							"top_logprobs": []map[string]any{
								{"token": "Hi", "logprob": -0.5},
								{"token": "Hello", "logprob": -1.2},
							},
						},
					},
				},
			}},
		})
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"m": "d"}),
	)

	logprobs := int32(3)

	llm, _ := provider.Model("m")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "test"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
		},
	}

	var result *model.LLMResponse

	for resp, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}

		result = resp
	}

	// Verify request sent logprobs fields.
	if capturedBody["logprobs"] != true {
		t.Errorf("request logprobs = %v, want true", capturedBody["logprobs"])
	}

	if capturedBody["top_logprobs"] != float64(3) {
		t.Errorf("request top_logprobs = %v, want 3", capturedBody["top_logprobs"])
	}

	// Verify response has LogprobsResult populated.
	if result.LogprobsResult == nil {
		t.Fatal("expected LogprobsResult in response")
	}

	if len(result.LogprobsResult.ChosenCandidates) != 1 {
		t.Fatalf("ChosenCandidates len = %d, want 1", len(result.LogprobsResult.ChosenCandidates))
	}

	if result.LogprobsResult.ChosenCandidates[0].Token != "Hi" {
		t.Errorf("token = %q, want Hi", result.LogprobsResult.ChosenCandidates[0].Token)
	}

	if len(result.LogprobsResult.TopCandidates) != 1 {
		t.Fatalf("TopCandidates len = %d, want 1", len(result.LogprobsResult.TopCandidates))
	}

	if len(result.LogprobsResult.TopCandidates[0].Candidates) != 2 {
		t.Errorf("alternatives = %d, want 2", len(result.LogprobsResult.TopCandidates[0].Candidates))
	}
}

func TestFoundation_ToolChoice(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeFoundationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeployments(map[string]string{"m": "d"}),
	)

	llm, _ := provider.Model("m")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "test"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get weather",
				}},
			}},
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"get_weather"},
				},
			},
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	toolChoice, ok := capturedBody["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice = %v (%T), want map", capturedBody["tool_choice"], capturedBody["tool_choice"])
	}

	if toolChoice["type"] != "function" {
		t.Errorf("tool_choice.type = %v, want function", toolChoice["type"])
	}

	fn, ok := toolChoice["function"].(map[string]any)
	if !ok {
		t.Fatal("tool_choice.function not a map")
	}

	if fn["name"] != "get_weather" {
		t.Errorf("tool_choice.function.name = %v, want get_weather", fn["name"])
	}
}

func TestOrchestration_SeedTopKLogprobs(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	seed := int32(99)
	topK := float32(5)
	logprobs := int32(2)

	llm, _ := provider.Model("gpt-4.1")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "test"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Seed:             &seed,
			TopK:             &topK,
			ResponseLogprobs: true,
			Logprobs:         &logprobs,
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	// In orchestration mode, these go into config.modules.prompt_templating.model.params.
	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	modelDef, _ := pt["model"].(map[string]any)
	params, _ := modelDef["params"].(map[string]any)

	if params["seed"] != float64(99) {
		t.Errorf("seed = %v, want 99", params["seed"])
	}

	if params["top_k"] != float64(5) {
		t.Errorf("top_k = %v, want 5", params["top_k"])
	}

	if params["logprobs"] != true {
		t.Errorf("logprobs = %v, want true", params["logprobs"])
	}

	if params["top_logprobs"] != float64(2) {
		t.Errorf("top_logprobs = %v, want 2", params["top_logprobs"])
	}
}

func TestOrchestration_ToolChoice(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)

	llm, _ := provider.Model("gpt-4.1")

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "test"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name: "fn",
				}},
			}},
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	// In orchestration mode, tool_choice goes into model.params.
	cfg, _ := capturedBody["config"].(map[string]any)
	modules, _ := cfg["modules"].(map[string]any)
	pt, _ := modules["prompt_templating"].(map[string]any)
	modelDef, _ := pt["model"].(map[string]any)
	params, _ := modelDef["params"].(map[string]any)

	if params["tool_choice"] != "none" {
		t.Errorf("tool_choice = %v, want \"none\"", params["tool_choice"])
	}
}
