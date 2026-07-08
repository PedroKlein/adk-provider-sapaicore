package sapaicore_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
)

func TestTokenCache_CachesToken(t *testing.T) {
	t.Parallel()

	var callCount atomic.Int32

	authServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount.Add(1)

		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
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

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-token-123" {
			t.Errorf("expected Bearer token, got %q", auth)
		}

		writeOrchestrationResponse(w, "Hello!", "stop")
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("test-client-id", "test-client-secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("orch-deploy"),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

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
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": fmt.Sprintf("token-%d", callCount.Load()),
			"expires_in":   30,
			"token_type":   "bearer",
		})
	}))

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("d"),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, _ := provider.Model("m")

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
