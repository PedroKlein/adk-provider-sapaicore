// Package request provides shared HTTP execution for SAP AI Core inference requests.
package request

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

// TokenProvider retrieves a valid OAuth2 access token.
type TokenProvider interface {
	GetToken(ctx context.Context) (string, error)
}

// Client wraps shared HTTP execution and error handling for inference requests.
type Client struct {
	Endpoint      string
	ResourceGroup string
	Headers       http.Header
	HTTPClient    *http.Client
	Auth          TokenProvider
}

// Execute sends a POST request to the given URL with the provided body.
// It handles token acquisition, authorization, and custom headers.
func (c *Client) Execute(ctx context.Context, url string, body []byte) (*http.Response, error) {
	token, err := c.Auth.GetToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("AI-Resource-Group", c.ResourceGroup)

	for key, values := range c.Headers {
		for _, v := range values {
			req.Header.Add(key, v)
		}
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}

	return resp, nil
}

// HandleError reads a limited portion of an error response body and attempts
// to extract a structured API error message. Returns a formatted error wrapping
// the given sentinel.
func (c *Client) HandleError(resp *http.Response, sentinel error) error {
	var errResp oai.FoundationResponse

	limited := io.LimitReader(resp.Body, 1<<20)
	if err := json.NewDecoder(limited).Decode(&errResp); err == nil && errResp.Error != nil {
		return fmt.Errorf("API error %d: %s: %w", resp.StatusCode, errResp.Error.Message, sentinel)
	}

	return fmt.Errorf("API returned status %d: %w", resp.StatusCode, sentinel)
}
