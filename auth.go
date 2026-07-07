package sapaicore

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const tokenExpiryBuffer = 60 * time.Second

// tokenCache manages OAuth2 access tokens with thread-safe caching.
type tokenCache struct {
	mu     sync.Mutex
	token  string
	expiry time.Time

	clientID     string
	clientSecret string
	authURL      string
	httpClient   *http.Client
}

// tokenResponse is the OAuth2 token endpoint response.
type tokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
	TokenType   string `json:"token_type"`
}

// getToken returns a valid access token, fetching a new one if expired.
func (tc *tokenCache) getToken(ctx context.Context) (string, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if tc.token != "" && time.Now().Before(tc.expiry) {
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

// fetchToken performs the OAuth2 client credentials exchange.
func (tc *tokenCache) fetchToken(ctx context.Context) (string, time.Time, error) {
	form := url.Values{
		"grant_type": {"client_credentials"},
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tc.authURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", time.Time{}, fmt.Errorf("creating token request: %w", ErrTokenRefresh)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.SetBasicAuth(tc.clientID, tc.clientSecret)

	resp, err := tc.httpClient.Do(req)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("executing token request: %w", ErrTokenRefresh)
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("token endpoint returned %d: %w", resp.StatusCode, ErrTokenRefresh)
	}

	var tokenResp tokenResponse
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return "", time.Time{}, fmt.Errorf("decoding token response: %w", ErrTokenRefresh)
	}

	if tokenResp.AccessToken == "" {
		return "", time.Time{}, fmt.Errorf("empty access token in response: %w", ErrTokenRefresh)
	}

	expiry := time.Now().Add(time.Duration(tokenResp.ExpiresIn)*time.Second - tokenExpiryBuffer)

	return tokenResp.AccessToken, expiry, nil
}
