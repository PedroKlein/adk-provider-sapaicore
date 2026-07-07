// Package request provides shared HTTP execution for SAP AI Core inference requests.
package request

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
)

// TokenGetter retrieves a valid OAuth2 access token.
type TokenGetter interface {
	GetToken(ctx context.Context) (string, error)
}

// Config holds the shared parameters for executing inference HTTP requests.
type Config struct {
	Endpoint      string
	DeploymentID  string
	ResourceGroup string
	Headers       http.Header
	Auth          TokenGetter
	HTTPClient    *http.Client
}

// Do executes an HTTP POST to the given URL path with the provided body.
// It attaches the authorization header, resource group, and custom headers.
func Do(ctx context.Context, cfg *Config, url string, body []byte) (*http.Response, error) {
	token, err := cfg.Auth.GetToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("AI-Resource-Group", cfg.ResourceGroup)

	for key, values := range cfg.Headers {
		for _, v := range values {
			req.Header.Add(key, v)
		}
	}

	resp, err := cfg.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}

	return resp, nil
}
