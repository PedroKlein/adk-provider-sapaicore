// Package sapaicore implements the ADK Go v2 model.LLM interface for SAP AI Core.
//
// It enables any ADK Go v2 agent to use models deployed on SAP AI Core by handling
// OAuth2 authentication, deployment ID mapping, and protocol translation between
// ADK's genai-based types and SAP AI Core's OpenAI-compatible inference API.
package sapaicore

import (
	"errors"
	"fmt"
	"net/http"

	"google.golang.org/adk/v2/model"
)

// Sentinel errors for the sapaicore package.
var (
	ErrMissingConfig      = errors.New("sapaicore: missing required configuration")
	ErrDeploymentNotFound = errors.New("sapaicore: deployment not found for model")
	ErrTokenRefresh       = errors.New("sapaicore: token refresh failed")
	ErrInference          = errors.New("sapaicore: inference request failed")
)

const defaultResourceGroup = "default"

// Config holds the SAP AI Core connection configuration.
type Config struct {
	// Endpoint is the SAP AI Core API base URL.
	// Example: "https://api.ai.xxx.aicore.cfapps.xxx.hana.ondemand.com"
	Endpoint string

	// ClientID for OAuth2 client credentials flow.
	ClientID string

	// ClientSecret for OAuth2 client credentials flow.
	ClientSecret string

	// AuthURL is the OAuth2 token endpoint.
	// Example: "https://xxx.authentication.xxx.hana.ondemand.com/oauth/token"
	AuthURL string

	// ResourceGroup is the SAP AI Core resource group.
	// If empty, defaults to "default".
	ResourceGroup string

	// Deployments maps logical model names to SAP AI Core deployment IDs.
	// Example: {"gpt-4.1": "d1234abc", "gpt-4.1-mini": "d5678def"}
	Deployments map[string]string

	// HTTPClient is an optional HTTP client for custom transport configuration.
	// If nil, a default client is used.
	HTTPClient *http.Client
}

// Provider creates model.LLM instances for SAP AI Core deployments.
type Provider struct {
	endpoint      string
	resourceGroup string
	deployments   map[string]string
	auth          *tokenCache
	httpClient    *http.Client
}

// NewProvider validates the configuration and returns a Provider.
func NewProvider(cfg Config) (*Provider, error) {
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}

	resourceGroup := cfg.ResourceGroup
	if resourceGroup == "" {
		resourceGroup = defaultResourceGroup
	}

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}

	auth := &tokenCache{
		clientID:     cfg.ClientID,
		clientSecret: cfg.ClientSecret,
		authURL:      cfg.AuthURL,
		httpClient:   httpClient,
	}

	return &Provider{
		endpoint:      cfg.Endpoint,
		resourceGroup: resourceGroup,
		deployments:   cfg.Deployments,
		auth:          auth,
		httpClient:    httpClient,
	}, nil
}

// Model returns a model.LLM for the given logical model name.
// The name must exist in Config.Deployments.
func (p *Provider) Model(name string) (model.LLM, error) {
	deploymentID, ok := p.deployments[name]
	if !ok {
		return nil, fmt.Errorf("model %q: %w", name, ErrDeploymentNotFound)
	}

	return &sapModel{
		name:          name,
		deploymentID:  deploymentID,
		endpoint:      p.endpoint,
		resourceGroup: p.resourceGroup,
		auth:          p.auth,
		httpClient:    p.httpClient,
	}, nil
}

// validateConfig checks that all required fields are present.
func validateConfig(cfg Config) error {
	switch {
	case cfg.Endpoint == "":
		return fmt.Errorf("endpoint: %w", ErrMissingConfig)
	case cfg.ClientID == "":
		return fmt.Errorf("client ID: %w", ErrMissingConfig)
	case cfg.ClientSecret == "":
		return fmt.Errorf("client secret: %w", ErrMissingConfig)
	case cfg.AuthURL == "":
		return fmt.Errorf("auth URL: %w", ErrMissingConfig)
	case len(cfg.Deployments) == 0:
		return fmt.Errorf("deployments: %w", ErrMissingConfig)
	}

	return nil
}
