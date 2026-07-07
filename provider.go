// Package sapaicore implements the ADK Go v2 model.LLM interface for SAP AI Core.
//
// Two modes are supported:
//   - Orchestration (default): single deployment handles all models via harmonized API
//   - Foundation-models: per-model deployment IDs with direct OpenAI-compatible API
package sapaicore

import (
	"errors"
	"fmt"
	"net/http"

	"google.golang.org/adk/v2/model"
)

// Sentinel errors.
var (
	ErrMissingConfig      = errors.New("sapaicore: missing required configuration")
	ErrDeploymentNotFound = errors.New("sapaicore: deployment not found for model")
	ErrTokenRefresh       = errors.New("sapaicore: token refresh failed")
	ErrInference          = errors.New("sapaicore: inference request failed")
)

const defaultResourceGroup = "default"

type mode int

const (
	modeOrchestration mode = iota
	modeFoundation
)

// providerConfig holds validated provider settings.
type providerConfig struct {
	endpoint      string
	clientID      string
	clientSecret  string
	authURL       string
	resourceGroup string
	httpClient    *http.Client
	headers       http.Header
	deploymentID  string
	deployments   map[string]string
}

// Option configures a Provider.
type Option func(*providerConfig)

// WithEndpoint sets the SAP AI Core API base URL.
// Example: "https://api.ai.xxx.aicore.cfapps.xxx.hana.ondemand.com"
func WithEndpoint(endpoint string) Option {
	return func(c *providerConfig) {
		c.endpoint = endpoint
	}
}

// WithAuth sets the OAuth2 client credentials.
// authURL is the token endpoint, e.g. "https://xxx.authentication.xxx.hana.ondemand.com/oauth/token"
func WithAuth(clientID, clientSecret, authURL string) Option {
	return func(c *providerConfig) {
		c.clientID = clientID
		c.clientSecret = clientSecret
		c.authURL = authURL
	}
}

// WithResourceGroup sets the SAP AI Core resource group.
// If not set, defaults to "default".
func WithResourceGroup(group string) Option {
	return func(c *providerConfig) {
		c.resourceGroup = group
	}
}

// WithHTTPClient sets a custom HTTP client for all requests.
func WithHTTPClient(client *http.Client) Option {
	return func(c *providerConfig) {
		c.httpClient = client
	}
}

// WithHeaders adds custom HTTP headers to every request.
func WithHeaders(headers http.Header) Option {
	return func(c *providerConfig) {
		c.headers = headers
	}
}

// WithDeploymentID enables orchestration mode using a single deployment
// that routes to all models. The model name is passed in the request body.
func WithDeploymentID(id string) Option {
	return func(c *providerConfig) {
		c.deploymentID = id
	}
}

// WithDeployments enables foundation-models mode with per-model deployment IDs.
func WithDeployments(deployments map[string]string) Option {
	return func(c *providerConfig) {
		c.deployments = deployments
	}
}

// Provider creates model.LLM instances for SAP AI Core.
type Provider struct {
	cfg  providerConfig
	mode mode
	auth *tokenCache
}

// NewProvider validates options and returns a Provider.
func NewProvider(opts ...Option) (*Provider, error) {
	cfg := providerConfig{
		resourceGroup: defaultResourceGroup,
	}

	for _, opt := range opts {
		opt(&cfg)
	}

	if err := validateProviderConfig(&cfg); err != nil {
		return nil, err
	}

	if cfg.httpClient == nil {
		cfg.httpClient = &http.Client{}
	}

	m := modeOrchestration
	if len(cfg.deployments) > 0 {
		m = modeFoundation
	}

	auth := &tokenCache{
		clientID:     cfg.clientID,
		clientSecret: cfg.clientSecret,
		authURL:      cfg.authURL,
		httpClient:   cfg.httpClient,
	}

	return &Provider{
		cfg:  cfg,
		mode: m,
		auth: auth,
	}, nil
}

// ModelOption configures a specific model instance.
type ModelOption func(*modelConfig)

type modelConfig struct {
	extraParams map[string]any
}

// WithModelParams adds extra parameters passed to the model.
// In orchestration mode these go into model.params (e.g. thinking, reasoning_effort, anthropic_beta).
// In foundation-models mode these are merged into the request body.
func WithModelParams(params map[string]any) ModelOption {
	return func(c *modelConfig) {
		c.extraParams = params
	}
}

// Model returns a model.LLM for the given model name.
//
// In orchestration mode, name is the SAP AI Core model identifier
// (e.g. "gpt-4.1", "anthropic--claude-4.5-sonnet").
//
// In foundation-models mode, name must exist in the Deployments map.
func (p *Provider) Model(name string, opts ...ModelOption) (model.LLM, error) {
	mc := modelConfig{}

	for _, opt := range opts {
		opt(&mc)
	}

	var deploymentID string

	switch p.mode {
	case modeFoundation:
		id, ok := p.cfg.deployments[name]
		if !ok {
			return nil, fmt.Errorf("model %q: %w", name, ErrDeploymentNotFound)
		}

		deploymentID = id

	case modeOrchestration:
		deploymentID = p.cfg.deploymentID
	}

	return &sapModel{
		name:          name,
		deploymentID:  deploymentID,
		endpoint:      p.cfg.endpoint,
		resourceGroup: p.cfg.resourceGroup,
		headers:       p.cfg.headers,
		auth:          p.auth,
		httpClient:    p.cfg.httpClient,
		mode:          p.mode,
		extraParams:   mc.extraParams,
	}, nil
}

func validateProviderConfig(cfg *providerConfig) error {
	switch {
	case cfg.endpoint == "":
		return fmt.Errorf("endpoint: %w", ErrMissingConfig)
	case cfg.clientID == "":
		return fmt.Errorf("client ID: %w", ErrMissingConfig)
	case cfg.clientSecret == "":
		return fmt.Errorf("client secret: %w", ErrMissingConfig)
	case cfg.authURL == "":
		return fmt.Errorf("auth URL: %w", ErrMissingConfig)
	}

	hasDeploymentID := cfg.deploymentID != ""
	hasDeployments := len(cfg.deployments) > 0

	if !hasDeploymentID && !hasDeployments {
		return fmt.Errorf("either WithDeploymentID or WithDeployments required: %w", ErrMissingConfig)
	}

	if hasDeploymentID && hasDeployments {
		return fmt.Errorf("WithDeploymentID and WithDeployments are mutually exclusive: %w", ErrMissingConfig)
	}

	return nil
}
