package sapaicore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/auth"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/foundation"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/orchestration"
)

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
	_ providerMode = iota
	providerModeOrchestration
	providerModeFoundation
)

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
	modules       moduleConfigs
}

// Option configures a [Provider].
type Option func(*providerConfig)

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

func WithResourceGroup(group string) Option {
	return func(c *providerConfig) {
		c.resourceGroup = group
	}
}

func WithHTTPClient(client *http.Client) Option {
	return func(c *providerConfig) {
		c.httpClient = client
	}
}

func WithHeaders(headers http.Header) Option {
	return func(c *providerConfig) {
		c.headers = headers.Clone()
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
		c.deployments = maps.Clone(deployments)
	}
}

// WithTimeout sets the server-side LLM request timeout in seconds.
// Default is 600 seconds. Values <= 0 are ignored.
func WithTimeout(seconds int) Option {
	return func(c *providerConfig) {
		if seconds > 0 {
			c.timeout = seconds
		}
	}
}

// WithMaxRetries sets the server-side retry count for LLM requests.
// Default is 2 retries. Negative values are ignored.
func WithMaxRetries(n int) Option {
	return func(c *providerConfig) {
		if n >= 0 {
			c.maxRetries = n
		}
	}
}

// WithFiltering enables content filtering. Orchestration-mode only.
//
// Pass nil for sensible defaults: Azure Content Safety ALLOW_SAFE on all
// categories with prompt_shield, applied to both input and output.
func WithFiltering(cfg *FilteringConfig) Option {
	return func(c *providerConfig) {
		if cfg == nil {
			c.modules.filtering = defaultFilteringConfig()
		} else {
			c.modules.filtering = cfg
		}
	}
}

// WithMasking enables PII redaction before messages reach the LLM.
// Orchestration-mode only. Method defaults to [Anonymization] if empty.
func WithMasking(cfg MaskingConfig) Option {
	return func(c *providerConfig) {
		if cfg.Method == "" {
			cfg.Method = Anonymization
		}

		c.modules.masking = &cfg
	}
}

// WithTranslation enables input/output translation. Orchestration-mode only.
func WithTranslation(cfg TranslationConfig) Option {
	return func(c *providerConfig) {
		c.modules.translation = &cfg
	}
}

// WithFallback configures model fallback. The service tries the primary model
// first, then each fallback in order. Fallback models inherit all module configs.
// Orchestration-mode only.
func WithFallback(models ...string) Option {
	return func(c *providerConfig) {
		c.modules.fallbackModels = models
	}
}

// WithPromptCaching adds cache_control annotations to the system message and
// tool definitions. Orchestration-mode only.
// Non-Anthropic models ignore the annotation.
// Default TTL is 5m. Pass [CacheTTL1h] for 1-hour caching on supported models.
func WithPromptCaching(ttl ...CacheTTL) Option {
	return func(c *providerConfig) {
		c.modules.promptCaching = true
		if len(ttl) > 0 {
			c.modules.cacheTTL = ttl[0]
		}
	}
}

func WithStreamOptions(opts StreamOptions) Option {
	return func(c *providerConfig) {
		c.modules.streamOptions = &opts
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
	modules     moduleConfigs
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

// WithModelFiltering overrides provider-level filtering for this model.
// Pass nil for the same defaults as [WithFiltering].
func WithModelFiltering(cfg *FilteringConfig) ModelOption {
	return func(c *modelConfig) {
		if cfg == nil {
			c.modules.filtering = defaultFilteringConfig()
		} else {
			c.modules.filtering = cfg
		}
	}
}

func WithoutFiltering() ModelOption {
	return func(c *modelConfig) {
		c.modules.noFiltering = true
	}
}

// WithModelMasking overrides provider-level masking for this model.
func WithModelMasking(cfg MaskingConfig) ModelOption {
	return func(c *modelConfig) {
		if cfg.Method == "" {
			cfg.Method = Anonymization
		}

		c.modules.masking = &cfg
	}
}

func WithoutMasking() ModelOption {
	return func(c *modelConfig) {
		c.modules.noMasking = true
	}
}

// WithModelTranslation overrides provider-level translation for this model.
func WithModelTranslation(cfg TranslationConfig) ModelOption {
	return func(c *modelConfig) {
		c.modules.translation = &cfg
	}
}

func WithoutTranslation() ModelOption {
	return func(c *modelConfig) {
		c.modules.noTranslation = true
	}
}

func WithModelFallback(models ...string) ModelOption {
	return func(c *modelConfig) {
		c.modules.fallbackModels = models
	}
}

func WithModelPromptCaching(ttl ...CacheTTL) ModelOption {
	return func(c *modelConfig) {
		c.modules.promptCaching = true
		if len(ttl) > 0 {
			c.modules.cacheTTL = ttl[0]
		}
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

		resolved := resolveModules(&p.cfg.modules, &mc.modules)
		if err := validateNoModulesForFoundation(resolved); err != nil {
			return nil, err
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
		resolved := resolveModules(&p.cfg.modules, &mc.modules)

		if err := validateModuleConfigs(resolved); err != nil {
			return nil, err
		}

		return &orchestration.Model{
			ModelName:      name,
			DeploymentID:   p.cfg.deploymentID,
			Endpoint:       p.cfg.endpoint,
			ResourceGroup:  p.cfg.resourceGroup,
			Headers:        p.cfg.headers,
			Auth:           p.auth,
			HTTPClient:     p.cfg.httpClient,
			ExtraParams:    mc.extraParams,
			Timeout:        p.cfg.timeout,
			MaxRetries:     p.cfg.maxRetries,
			Filtering:      buildFilteringWire(resolved.Filtering),
			Masking:        buildMaskingWire(resolved.Masking),
			Translation:    buildTranslationWire(resolved.Translation),
			FallbackModels: resolved.FallbackModels,
			PromptCaching:  resolved.PromptCaching,
			CacheTTL:       string(resolved.CacheTTL),
			StreamOptions:  buildStreamOptionsWire(resolved.StreamOptions),
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

	switch {
	case hasDeploymentID && hasDeployments,
		hasDeploymentID && hasAutoDiscover,
		hasDeployments && hasAutoDiscover:
		return fmt.Errorf("WithOrchestration, WithDeploymentID, and WithDeployments are mutually exclusive: %w", ErrMissingConfig)
	case !hasDeploymentID && !hasDeployments && !hasAutoDiscover:
		// Default to orchestration auto-discovery when no mode is specified.
		cfg.autoDiscover = true
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

func validateNoModulesForFoundation(r resolvedModules) error {
	if r.Filtering != nil || r.Masking != nil || r.Translation != nil || len(r.FallbackModels) > 0 || r.PromptCaching {
		return fmt.Errorf("orchestration modules (filtering, masking, translation, fallback, caching) require orchestration mode: %w", ErrMissingConfig)
	}

	return nil
}

func validateModuleConfigs(r resolvedModules) error {
	if r.Translation != nil && r.Translation.Input == nil && r.Translation.Output == nil {
		return fmt.Errorf("translation config requires at least Input or Output: %w", ErrMissingConfig)
	}

	if r.Masking != nil && len(r.Masking.Entities) == 0 {
		return fmt.Errorf("masking config requires at least one entity: %w", ErrMissingConfig)
	}

	return nil
}
