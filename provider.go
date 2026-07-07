// Package sapaicore implements the ADK Go v2 [model.LLM] interface for SAP AI Core.
//
// Two modes are supported:
//   - Orchestration (default): a single deployment handles all models via SAP AI Core's
//     harmonized API. Use [WithDeploymentID] to enable this mode.
//   - Foundation-models: per-model deployment IDs with a direct OpenAI-compatible API.
//     Use [WithDeployments] to enable this mode.
//
// Create a [Provider] with [NewProvider], then call [Provider.Model] to obtain an
// [model.LLM] you can pass to any ADK agent.
package sapaicore

import (
	"context"
	"encoding/json"
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
	ErrDiscovery          = errors.New("sapaicore: orchestration deployment discovery failed")
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
	autoDiscover  bool
	timeout       int
	maxRetries    int
}

// Option configures a [Provider]. Pass options to [NewProvider].
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

// WithOrchestration enables orchestration mode by automatically discovering
// the orchestration deployment. It queries the SAP AI Core deployments API
// at provider creation time to find the running orchestration deployment.
//
// This is the simplest way to use orchestration mode — no deployment ID needed.
func WithOrchestration() Option {
	return func(c *providerConfig) {
		c.autoDiscover = true
	}
}

// WithDeploymentID enables orchestration mode using a specific deployment ID.
// Use this if you know the deployment ID, or use [WithOrchestration] for auto-discovery.
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

// WithTimeout sets the server-side LLM request timeout in seconds (1–1200).
// Zero means use the server default (600s).
func WithTimeout(seconds int) Option {
	return func(c *providerConfig) {
		c.timeout = seconds
	}
}

// WithMaxRetries sets the server-side retry count for LLM requests (0–5).
// Zero means use the server default (2 retries).
func WithMaxRetries(n int) Option {
	return func(c *providerConfig) {
		c.maxRetries = n
	}
}

// Provider creates [model.LLM] instances backed by SAP AI Core deployments.
// Use [NewProvider] to construct a valid Provider.
type Provider struct {
	cfg  providerConfig
	mode mode
	auth *tokenCache
}

// NewProvider validates the given options and returns a ready-to-use [Provider].
// It returns [ErrMissingConfig] if required options are absent or conflicting.
//
// When [WithOrchestration] is used, NewProvider makes an HTTP call to discover
// the orchestration deployment ID. This requires network access at creation time.
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

	auth := &tokenCache{
		clientID:     cfg.clientID,
		clientSecret: cfg.clientSecret,
		authURL:      cfg.authURL,
		httpClient:   cfg.httpClient,
	}

	// Auto-discover orchestration deployment if requested.
	if cfg.autoDiscover {
		deploymentID, err := discoverOrchestrationDeployment(auth, &cfg)
		if err != nil {
			return nil, err
		}

		cfg.deploymentID = deploymentID
	}

	m := modeOrchestration
	if len(cfg.deployments) > 0 {
		m = modeFoundation
	}

	return &Provider{
		cfg:  cfg,
		mode: m,
		auth: auth,
	}, nil
}

// ModelOption configures a specific model instance returned by [Provider.Model].
type ModelOption func(*modelConfig)

type modelConfig struct {
	extraParams map[string]any
}

// WithModelParams adds extra parameters forwarded directly to the model.
// In orchestration mode these go into model.params (e.g. thinking, reasoning_effort, anthropic_beta).
// In foundation-models mode these are merged into the top-level request body.
//
// Use this for provider-specific features that ADK's [genai.GenerateContentConfig]
// does not expose (extended thinking, 1M context windows, reasoning effort, etc.).
func WithModelParams(params map[string]any) ModelOption {
	return func(c *modelConfig) {
		c.extraParams = params
	}
}

// Model returns a [model.LLM] for the given model name.
//
// In orchestration mode, name is any SAP AI Core model identifier
// (e.g. "gpt-4.1", "anthropic--claude-4.5-sonnet", "gemini-2.5-flash").
//
// In foundation-models mode, name must exist in the map provided to [WithDeployments].
// Returns [ErrDeploymentNotFound] if the name is not registered.
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
		timeout:       p.cfg.timeout,
		maxRetries:    p.cfg.maxRetries,
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
	hasAutoDiscover := cfg.autoDiscover

	modes := 0
	if hasDeploymentID {
		modes++
	}

	if hasDeployments {
		modes++
	}

	if hasAutoDiscover {
		modes++
	}

	if modes == 0 {
		return fmt.Errorf("one of WithOrchestration, WithDeploymentID, or WithDeployments required: %w", ErrMissingConfig)
	}

	if modes > 1 {
		return fmt.Errorf("WithOrchestration, WithDeploymentID, and WithDeployments are mutually exclusive: %w", ErrMissingConfig)
	}

	return nil
}

// discoverOrchestrationDeployment queries the SAP AI Core API to find
// the running orchestration deployment.
func discoverOrchestrationDeployment(auth *tokenCache, cfg *providerConfig) (string, error) {
	ctx := context.Background()

	token, err := auth.getToken(ctx)
	if err != nil {
		return "", fmt.Errorf("getting token for discovery: %w", ErrDiscovery)
	}

	url := cfg.endpoint + "/v2/lm/deployments?scenarioId=orchestration&status=RUNNING"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return "", fmt.Errorf("creating discovery request: %w", ErrDiscovery)
	}

	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("AI-Resource-Group", cfg.resourceGroup)

	resp, err := cfg.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("executing discovery request: %w", ErrDiscovery)
	}

	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("discovery API returned status %d: %w", resp.StatusCode, ErrDiscovery)
	}

	var result deploymentsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decoding discovery response: %w", ErrDiscovery)
	}

	if len(result.Resources) == 0 {
		return "", fmt.Errorf("no running orchestration deployment found: %w", ErrDiscovery)
	}

	return result.Resources[0].ID, nil
}

// deploymentsResponse is the SAP AI Core deployments list response.
type deploymentsResponse struct {
	Resources []deploymentResource `json:"resources"`
}

type deploymentResource struct {
	ID         string `json:"id"`
	ScenarioID string `json:"scenarioId"`
	Status     string `json:"status"`
}
