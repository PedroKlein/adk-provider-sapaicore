package sapaicore_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	sapaicore "github.com/PedroKlein/go-adk-sap-ai-core"
)

func TestTokenCache_CachesToken(t *testing.T) {
	t.Parallel()

	var callCount atomic.Int32

	authServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)

		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}

		if ct := r.Header.Get("Content-Type"); ct != "application/x-www-form-urlencoded" {
			t.Errorf("expected form content type, got %s", ct)
		}

		user, pass, ok := r.BasicAuth()
		if !ok || user != "test-client-id" || pass != "test-client-secret" {
			t.Errorf("unexpected basic auth: user=%q pass=%q ok=%v", user, pass, ok)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "test-token-123",
			"expires_in":   3600,
			"token_type":   "bearer",
		})
	}))
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-token-123" {
			t.Errorf("expected Bearer token, got %q", auth)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":    "chatcmpl-1",
			"model": "gpt-4.1",
			"choices": []map[string]any{{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello!",
				},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		})
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:      inferenceServer.URL,
		ClientID:      "test-client-id",
		ClientSecret:  "test-client-secret",
		AuthURL:       authServer.URL + "/oauth/token",
		ResourceGroup: "default",
		Deployments:   map[string]string{"gpt-4.1": "deploy-abc"},
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// Call GenerateContent multiple times; token should be fetched only once.
	for i := range 3 {
		for resp, err := range llm.GenerateContent(t.Context(), newSimpleRequest("Hi"), false) {
			if err != nil {
				t.Fatalf("iteration %d: GenerateContent error: %v", i, err)
			}

			if resp.Content == nil || len(resp.Content.Parts) == 0 {
				t.Fatalf("iteration %d: empty content", i)
			}
		}
	}

	if got := callCount.Load(); got != 1 {
		t.Errorf("token fetched %d times, want 1 (should cache)", got)
	}
}

func TestTokenCache_RefreshOnExpiry(t *testing.T) {
	t.Parallel()

	var callCount atomic.Int32

	authServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount.Add(1)

		w.Header().Set("Content-Type", "application/json")
		// Token expires in 30 seconds (less than the 60s buffer), so it's always "expired".
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": fmt.Sprintf("token-%d", callCount.Load()),
			"expires_in":   30,
			"token_type":   "bearer",
		})
	}))
	defer authServer.Close()

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
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
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(sapaicore.Config{
		Endpoint:     inferenceServer.URL,
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      authServer.URL + "/oauth/token",
		Deployments:  map[string]string{"m": "d"},
	})
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, _ := provider.Model("m")

	// Each call should trigger a new token fetch since expiry < buffer.
	for range 3 {
		for _, err := range llm.GenerateContent(t.Context(), newSimpleRequest("hi"), false) {
			if err != nil {
				t.Fatalf("GenerateContent: %v", err)
			}
		}
	}

	if got := callCount.Load(); got < 3 {
		t.Errorf("token fetched %d times, want >= 3 (should refresh on each call)", got)
	}
}
