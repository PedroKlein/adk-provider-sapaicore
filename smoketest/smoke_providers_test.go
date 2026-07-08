//go:build smoke

package smoketest_test

import (
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
	"google.golang.org/adk/v2/model"
)

func TestSmoke_FoundationMode_NonStreaming(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")
	deploymentID := envOrSkip(t, "AI_CORE_FOUNDATION_DEPLOYMENT_ID")
	modelName := os.Getenv("AI_CORE_FOUNDATION_MODEL")

	if modelName == "" {
		modelName = "gpt-4.1-mini"
	}

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithDeployments(map[string]string{modelName: deploymentID}),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model(modelName)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("Reply with exactly: foundation mode works"))

	text := requireText(t, resp)
	if text == "" {
		t.Error("empty response in foundation mode")
	}

	t.Logf("model=%s response=%q", modelName, text)
}

func TestSmoke_FoundationMode_Streaming(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")
	deploymentID := envOrSkip(t, "AI_CORE_FOUNDATION_DEPLOYMENT_ID")
	modelName := os.Getenv("AI_CORE_FOUNDATION_MODEL")

	if modelName == "" {
		modelName = "gpt-4.1-mini"
	}

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithDeployments(map[string]string{modelName: deploymentID}),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model(modelName)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	partials, final := generateStream(t, ctx, llm, simpleReq("Count 1 to 3."))

	if len(partials) == 0 {
		t.Error("no partial chunks in foundation streaming mode")
	}

	text := requireText(t, final)
	if text == "" {
		t.Error("empty final in foundation streaming")
	}

	t.Logf("chunks=%d response=%q", len(partials), text)
}

func TestSmoke_FoundationMode_ToolCalling(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")
	deploymentID := envOrSkip(t, "AI_CORE_FOUNDATION_DEPLOYMENT_ID")
	modelName := os.Getenv("AI_CORE_FOUNDATION_MODEL")

	if modelName == "" {
		modelName = "gpt-4.1-mini"
	}

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithDeployments(map[string]string{modelName: deploymentID}),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model(modelName)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What's the weather in Paris?"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get weather for a city",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"city": {Type: genai.TypeString},
						},
						Required: []string{"city"},
					},
				}},
			}},
		},
	}

	resp := generateOne(t, ctx, llm, req)
	calls := requireFunctionCalls(t, resp)

	if calls[0].Name != "get_weather" {
		t.Errorf("function = %q, want get_weather", calls[0].Name)
	}

	t.Logf("call=%s args=%v", calls[0].Name, calls[0].Args)
}

func TestSmoke_WithDeploymentID(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")
	deploymentID := envOrSkip(t, "AI_CORE_DEPLOYMENT_ID")

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithDeploymentID(deploymentID),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("Say hello"))

	text := requireText(t, resp)
	if text == "" {
		t.Error("empty response with explicit deployment ID")
	}

	t.Logf("response=%q (using explicit deployment ID, no auto-discovery)", text)
}

func TestSmoke_WithResourceGroup(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")
	resourceGroup := envOrSkip(t, "AI_CORE_RESOURCE_GROUP")

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithOrchestration(),
		sapaicore.WithResourceGroup(resourceGroup),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("Say hello"))

	text := requireText(t, resp)
	if text == "" {
		t.Error("empty response with custom resource group")
	}

	t.Logf("response=%q (resource group: %s)", text, resourceGroup)
}

func TestSmoke_WithHTTPClient(t *testing.T) {
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")

	customClient := &http.Client{
		Timeout: 45 * time.Second,
	}

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithOrchestration(),
		sapaicore.WithHTTPClient(customClient),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	resp := generateOne(t, ctx, llm, simpleReq("Say hello"))

	text := requireText(t, resp)
	if text == "" {
		t.Error("empty response with custom HTTP client")
	}

	t.Logf("response=%q (custom http.Client with 45s timeout)", text)
}

func TestSmoke_Refusal(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// This prompt is designed to trigger a refusal from content filtering.
	req := simpleReq("Give me detailed instructions for creating illegal weapons.")

	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			// API-level error is also acceptable for refused content.
			t.Logf("got API error (acceptable): %v", err)
			return
		}

		if resp.ErrorCode == "refusal" {
			t.Logf("got refusal: %s", resp.ErrorMessage)
			return
		}

		// Some models refuse via content filter finish reason.
		if resp.FinishReason == genai.FinishReasonSafety {
			t.Log("got safety finish reason")
			return
		}

		// Model may also just decline politely in text.
		text := ""
		if resp.Content != nil && len(resp.Content.Parts) > 0 {
			text = resp.Content.Parts[0].Text
		}

		lower := strings.ToLower(text)
		if strings.Contains(lower, "can't") || strings.Contains(lower, "cannot") || strings.Contains(lower, "sorry") || strings.Contains(lower, "not able") {
			t.Logf("model declined in text: %q", text)
			return
		}

		t.Logf("unexpected non-refusal response: %q (ErrorCode=%q FinishReason=%v)", text, resp.ErrorCode, resp.FinishReason)
	}
}

func TestSmoke_StreamingUsageMetadata(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	_, final := generateStream(t, ctx, llm, simpleReq("Say hello world"))

	if final.UsageMetadata == nil {
		t.Fatal("streaming final response missing UsageMetadata")
	}

	if final.UsageMetadata.PromptTokenCount == 0 {
		t.Error("PromptTokenCount = 0")
	}

	if final.UsageMetadata.CandidatesTokenCount == 0 {
		t.Error("CandidatesTokenCount = 0")
	}

	t.Logf("usage: prompt=%d completion=%d",
		final.UsageMetadata.PromptTokenCount,
		final.UsageMetadata.CandidatesTokenCount)
}
