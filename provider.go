package sapaicore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/auth"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/foundation"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/orchestration"
)

// Sentinel errors.
var (
	ErrMissingConfig      = errors.New("sapaicore: missing required configuration")
	ErrDeploymentNotFound = errors.New("sapaicore: deployment not found for model")
	ErrDiscovery          = errors.New("sapaicore: orchestration deployment discovery")
)

const defaultResourceGroup = "default"

const (
	defaultTimeout    = 600
	defaultMaxRetries = 2
)

type providerMode int

const (
	providerModeUnspecified providerMode = iota
	providerModeOrchestration
	providerModeFoundation
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
func WithEndpoint(endpoint string) Option {
	return func(c *providerConfig) {
		c.endpoint = endpoint
	}
}

// WithAuth sets the OAuth2 client credentials.
// authURL is the token endpoint, e.g. "https://xxx.authentication.xxx.hana.ondemand.com/oauth/token".
func WithAuth(clientID, clientSecret, authURL string) Option {
	return func(c *providerConfig) {
		c.clientID = clientID
		c.clientSecret = clientSecret
		c.authURL = authURL
	}
}

// WithResourceGroup sets the SAP AI Core resource group. Defaults to "default".
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
func WithOrchestration() Option {
	return func(c *providerConfig) {
		c.autoDiscover = true
	}
}

// WithDeploymentID enables orchestration mode using a specific deployment ID.
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

// WithTimeout sets the server-side LLM request timeout in seconds.
// Default is 600 seconds.
func WithTimeout(seconds int) Option {
	return func(c *providerConfig) {
		c.timeout = seconds
	}
}

// WithMaxRetries sets the server-side retry count for LLM requests.
// Default is 2 retries.
func WithMaxRetries(n int) Option {
	return func(c *providerConfig) {
		c.maxRetries = n
	}
}

// Provider creates [model.LLM] instances backed by SAP AI Core deployments.
type Provider struct {
	cfg  providerConfig
	mode providerMode
	auth *auth.TokenCache
}

// NewProvider validates the given options and returns a ready-to-use [Provider].
// It returns [ErrMissingConfig] if required options are absent.
//
// If no mode is specified, orchestration auto-discovery is used by default.
// NewProvider makes an HTTP call to discover the orchestration deployment ID
// when auto-discovery is active.
//
// ctx is used for any HTTP calls made during provider initialization
// (e.g. orchestration deployment discovery). It does not affect subsequent
// Model() or GenerateContent() calls.
func NewProvider(ctx context.Context, opts ...Option) (*Provider, error) {
	cfg := providerConfig{
		resourceGroup: defaultResourceGroup,
		timeout:       defaultTimeout,
		maxRetries:    defaultMaxRetries,
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

	tokenCache := auth.NewTokenCache(auth.TokenCacheConfig{
		ClientID:     cfg.clientID,
		ClientSecret: cfg.clientSecret,
		AuthURL:      cfg.authURL,
		HTTPClient:   cfg.httpClient,
	})

	if cfg.autoDiscover {
		deploymentID, err := discoverOrchestrationDeployment(ctx, tokenCache, &cfg)
		if err != nil {
			return nil, err
		}

		cfg.deploymentID = deploymentID
	}

	m := providerModeOrchestration
	if len(cfg.deployments) > 0 {
		m = providerModeFoundation
	}

	return &Provider{
		cfg:  cfg,
		mode: m,
		auth: tokenCache,
	}, nil
}

// ModelOption configures a specific model instance returned by [Provider.Model].
type ModelOption func(*modelConfig)

type modelConfig struct {
	extraParams map[string]any
}

// WithModelParams adds extra parameters forwarded directly to the model.
// In orchestration mode these go into model.params (e.g. thinking, reasoning_effort).
// In foundation-models mode these are merged into the top-level request body.
//
// If a key conflicts with a field set via [genai.GenerateContentConfig]
// (e.g. "seed", "top_k", "logprobs"), the WithModelParams value takes precedence.
// This allows overriding any first-class parameter when needed.
func WithModelParams(params map[string]any) ModelOption {
	return func(c *modelConfig) {
		c.extraParams = params
	}
}

// Model returns a [model.LLM] for the given model name.
//
// In orchestration mode, name is any SAP AI Core model identifier
// (e.g. "gpt-4.1", "anthropic--claude-4.5-sonnet").
//
// In foundation-models mode, name must exist in the map provided to [WithDeployments].
// Returns [ErrDeploymentNotFound] if the name is not registered.
func (p *Provider) Model(name string, opts ...ModelOption) (model.LLM, error) {
	mc := modelConfig{}

	for _, opt := range opts {
		opt(&mc)
	}

	switch p.mode {
	case providerModeFoundation:
		deploymentID, ok := p.cfg.deployments[name]
		if !ok {
			return nil, fmt.Errorf("model %q: %w", name, ErrDeploymentNotFound)
		}

		return &foundation.Model{
			ModelName:     name,
			DeploymentID:  deploymentID,
			Endpoint:      p.cfg.endpoint,
			ResourceGroup: p.cfg.resourceGroup,
			Headers:       p.cfg.headers,
			Auth:          p.auth,
			HTTPClient:    p.cfg.httpClient,
			ExtraParams:   mc.extraParams,
		}, nil

	default:
		return &orchestration.Model{
			ModelName:     name,
			DeploymentID:  p.cfg.deploymentID,
			Endpoint:      p.cfg.endpoint,
			ResourceGroup: p.cfg.resourceGroup,
			Headers:       p.cfg.headers,
			Auth:          p.auth,
			HTTPClient:    p.cfg.httpClient,
			ExtraParams:   mc.extraParams,
			Timeout:       p.cfg.timeout,
			MaxRetries:    p.cfg.maxRetries,
		}, nil
	}
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

	// Default to orchestration auto-discovery when no mode is specified.
	if modes == 0 {
		cfg.autoDiscover = true
	}

	if modes > 1 {
		return fmt.Errorf("WithOrchestration, WithDeploymentID, and WithDeployments are mutually exclusive: %w", ErrMissingConfig)
	}

	return nil
}

func discoverOrchestrationDeployment(ctx context.Context, tokenCache *auth.TokenCache, cfg *providerConfig) (string, error) {
	token, err := tokenCache.GetToken(ctx)
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

type deploymentsResponse struct {
	Resources []deploymentResource `json:"resources"`
}

type deploymentResource struct {
	ID         string `json:"id"`
	ScenarioID string `json:"scenarioId"`
	Status     string `json:"status"`
}
