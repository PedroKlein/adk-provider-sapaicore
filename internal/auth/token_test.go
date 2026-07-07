package auth_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/PedroKlein/go-adk-sap-ai-core/internal/auth"
)

func TestTokenCache_CachesValidToken(t *testing.T) {
	t.Parallel()

	var callCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount.Add(1)
		writeTokenResponse(w, "token-1", 3600)
	}))
	defer server.Close()

	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	cache := auth.NewTokenCache(auth.TokenCacheConfig{
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      server.URL,
		HTTPClient:   http.DefaultClient,
		Clock:        func() time.Time { return now },
	})

	ctx := context.Background()

	for range 5 {
		token, err := cache.GetToken(ctx)
		if err != nil {
			t.Fatalf("GetToken: %v", err)
		}

		if token != "token-1" {
			t.Fatalf("token = %q, want %q", token, "token-1")
		}
	}

	if got := callCount.Load(); got != 1 {
		t.Errorf("fetch count = %d, want 1", got)
	}
}

func TestTokenCache_RefreshesExpiredToken(t *testing.T) {
	t.Parallel()

	var callCount atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		callCount.Add(1)
		writeTokenResponse(w, fmt.Sprintf("token-%d", callCount.Load()), 120)
	}))
	defer server.Close()

	now := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	cache := auth.NewTokenCache(auth.TokenCacheConfig{
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      server.URL,
		HTTPClient:   http.DefaultClient,
		Clock:        func() time.Time { return now },
	})

	ctx := context.Background()

	// First call fetches.
	_, err := cache.GetToken(ctx)
	if err != nil {
		t.Fatalf("GetToken: %v", err)
	}

	// Advance time past expiry (120s - 60s buffer = 60s effective).
	now = now.Add(61 * time.Second)

	// Should refetch.
	_, err = cache.GetToken(ctx)
	if err != nil {
		t.Fatalf("GetToken: %v", err)
	}

	if got := callCount.Load(); got != 2 {
		t.Errorf("fetch count = %d, want 2", got)
	}
}

func TestTokenCache_SendsBasicAuth(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, pass, ok := r.BasicAuth()
		if !ok {
			t.Error("missing basic auth")
		}

		if user != "my-client" || pass != "my-secret" {
			t.Errorf("basic auth = %q:%q, want my-client:my-secret", user, pass)
		}

		writeTokenResponse(w, "t", 3600)
	}))
	defer server.Close()

	cache := auth.NewTokenCache(auth.TokenCacheConfig{
		ClientID:     "my-client",
		ClientSecret: "my-secret",
		AuthURL:      server.URL,
		HTTPClient:   http.DefaultClient,
	})

	if _, err := cache.GetToken(context.Background()); err != nil {
		t.Fatalf("GetToken: %v", err)
	}
}

func TestTokenCache_ErrorOnNon200(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer server.Close()

	cache := auth.NewTokenCache(auth.TokenCacheConfig{
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      server.URL,
		HTTPClient:   http.DefaultClient,
	})

	_, err := cache.GetToken(context.Background())
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
}

func TestTokenCache_ErrorOnEmptyToken(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeTokenResponse(w, "", 3600)
	}))
	defer server.Close()

	cache := auth.NewTokenCache(auth.TokenCacheConfig{
		ClientID:     "id",
		ClientSecret: "secret",
		AuthURL:      server.URL,
		HTTPClient:   http.DefaultClient,
	})

	_, err := cache.GetToken(context.Background())
	if err == nil {
		t.Fatal("expected error for empty token")
	}
}

func writeTokenResponse(w http.ResponseWriter, token string, expiresIn int) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"access_token": token,
		"expires_in":   expiresIn,
		"token_type":   "bearer",
	})
}
