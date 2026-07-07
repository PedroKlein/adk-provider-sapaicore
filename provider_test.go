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
			name: "no deployment config",
			opts: []sapaicore.Option{
				sapaicore.WithEndpoint("https://api.example.com"),
				sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
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

			_, err := sapaicore.NewProvider(tt.opts...)
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

	provider, err := sapaicore.NewProvider(
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

	provider, err := sapaicore.NewProvider(
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
	defer authServer.Close()

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

	provider, err := sapaicore.NewProvider(
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
	defer authServer.Close()

	deploymentsServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"resources": []map[string]any{},
		})
	}))
	defer deploymentsServer.Close()

	_, err := sapaicore.NewProvider(
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
	defer authServer.Close()

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

	provider, err := sapaicore.NewProvider(
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

	// Only system message goes in template.
	if len(template) != 1 {
		t.Fatalf("expected 1 template message (system), got %d", len(template))
	}

	sysMsg, _ := template[0].(map[string]any)
	if sysMsg["role"] != "system" {
		t.Errorf("first message role = %v, want system", sysMsg["role"])
	}

	// User messages go in messages_history.
	history, _ := capturedBody["messages_history"].([]any)
	if len(history) != 1 {
		t.Fatalf("expected 1 message in history, got %d", len(history))
	}

	userMsg, _ := history[0].(map[string]any)
	if userMsg["role"] != "user" {
		t.Errorf("history[0] role = %v, want user", userMsg["role"])
	}
}

func TestOrchestration_WithModelParams(t *testing.T) {
	t.Parallel()

	var capturedBody map[string]any

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(
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
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify foundation-models URL path.
		if want := "/v2/inference/deployments/deploy-xyz/chat/completions"; r.URL.Path != want {
			t.Errorf("path = %q, want %q", r.URL.Path, want)
		}

		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)

		writeFoundationResponse(w, "Hello from foundation!", "stop")
	}))
	defer inferenceServer.Close()

	provider, _ := sapaicore.NewProvider(
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

	provider, _ := sapaicore.NewProvider(
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
