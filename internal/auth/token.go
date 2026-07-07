// Package auth manages OAuth2 client-credentials tokens with thread-safe caching.
package auth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// ErrTokenFetch indicates the OAuth2 token endpoint returned an error.
var ErrTokenFetch = errors.New("token fetch")

const tokenExpiryBuffer = 60 * time.Second

// Clock returns the current time. Inject a custom implementation for testing.
type Clock func() time.Time

// HTTPDoer executes HTTP requests. *http.Client satisfies this interface.
type HTTPDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// TokenCache manages OAuth2 access tokens with thread-safe caching and automatic refresh.
type TokenCache struct {
	mu     sync.Mutex
	token  string
	expiry time.Time

	clientID     string
	clientSecret string
	authURL      string
	httpClient   HTTPDoer
	clock        Clock
}

// TokenCacheConfig holds the parameters needed to create a TokenCache.
type TokenCacheConfig struct {
	ClientID     string
	ClientSecret string
	AuthURL      string
	HTTPClient   HTTPDoer
	Clock        Clock
}

// NewTokenCache creates a token cache from the given configuration.
// If Clock is nil, time.Now is used.
func NewTokenCache(cfg TokenCacheConfig) *TokenCache {
	clock := cfg.Clock
	if clock == nil {
		clock = time.Now
	}

	return &TokenCache{
		clientID:     cfg.ClientID,
		clientSecret: cfg.ClientSecret,
		authURL:      cfg.AuthURL,
		httpClient:   cfg.HTTPClient,
		clock:        clock,
	}
}

type tokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
	TokenType   string `json:"token_type"`
}

// GetToken returns a valid access token, fetching a new one if expired.
func (tc *TokenCache) GetToken(ctx context.Context) (string, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if tc.token != "" && tc.clock().Before(tc.expiry) {
		return tc.token, nil
	}

	token, expiry, err := tc.fetchToken(ctx)
	if err != nil {
		return "", err
	}

	tc.token = token
	tc.expiry = expiry

	return tc.token, nil
}

func (tc *TokenCache) fetchToken(ctx context.Context) (string, time.Time, error) {
	form := url.Values{
		"grant_type": {"client_credentials"},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tc.authURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", time.Time{}, fmt.Errorf("creating token request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.SetBasicAuth(tc.clientID, tc.clientSecret)

	resp, err := tc.httpClient.Do(req)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("executing token request: %w", err)
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("token endpoint returned %d: %w", resp.StatusCode, ErrTokenFetch)
	}

	var tokenResp tokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return "", time.Time{}, fmt.Errorf("decoding token response: %w", err)
	}

	if tokenResp.AccessToken == "" {
		return "", time.Time{}, fmt.Errorf("empty access token in response: %w", ErrTokenFetch)
	}

	expiry := tc.clock().Add(time.Duration(tokenResp.ExpiresIn)*time.Second - tokenExpiryBuffer)

	return tokenResp.AccessToken, expiry, nil
}
