//go:build smoke

package smoketest_test

import (
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
	"google.golang.org/adk/v2/model"
)

func TestSmoke_ToolCalling(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What's the weather in Berlin?"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get the current weather for a city",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"city": {Type: genai.TypeString, Description: "The city name"},
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
		t.Errorf("function name = %q, want get_weather", calls[0].Name)
	}

	city, _ := calls[0].Args["city"].(string)
	if !strings.Contains(strings.ToLower(city), "berlin") {
		t.Errorf("city = %q, want something containing 'berlin'", city)
	}

	t.Logf("call=%s args=%v", calls[0].Name, calls[0].Args)
}

func TestSmoke_ToolCalling_Streaming(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Get the weather in Tokyo and the population of France."}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "get_weather",
						Description: "Get current weather for a city",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"city": {Type: genai.TypeString},
							},
							Required: []string{"city"},
						},
					},
					{
						Name:        "get_population",
						Description: "Get the population of a country",
						Parameters: &genai.Schema{
							Type: genai.TypeObject,
							Properties: map[string]*genai.Schema{
								"country": {Type: genai.TypeString},
							},
							Required: []string{"country"},
						},
					},
				},
			}},
		},
	}

	_, final := generateStream(t, ctx, llm, req)
	calls := requireFunctionCalls(t, final)

	if len(calls) < 2 {
		t.Fatalf("expected at least 2 function calls, got %d", len(calls))
	}

	names := make(map[string]bool)
	for _, call := range calls {
		names[call.Name] = true
		t.Logf("call=%s args=%v", call.Name, call.Args)
	}

	if !names["get_weather"] {
		t.Error("missing get_weather call")
	}

	if !names["get_population"] {
		t.Error("missing get_population call")
	}
}

func TestSmoke_ToolRoundTrip(t *testing.T) {
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// Full round-trip: user → model(tool_call) → tool_result → final answer.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "What is the weather in Berlin?"}}, Role: "user"},
			{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID:   "call_123",
				Name: "get_weather",
				Args: map[string]any{"city": "Berlin"},
			}}}, Role: "model"},
			{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID:       "call_123",
				Name:     "get_weather",
				Response: map[string]any{"temperature": "22°C", "condition": "sunny"},
			}}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "Be concise."}},
			},
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get current weather for a city",
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

	text := requireText(t, resp)
	lower := strings.ToLower(text)

	if !strings.Contains(lower, "22") && !strings.Contains(lower, "sunny") {
		t.Errorf("model didn't use tool result: %q", text)
	}

	t.Logf("response=%q", text)
}

func TestSmoke_ToolCalling_MultiModel(t *testing.T) {
	provider := newProvider(t)

	for _, modelName := range testModels {
		t.Run(modelName, func(t *testing.T) {
			ctx := withTimeout(t, 45*time.Second)

			llm, err := provider.Model(modelName)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{Parts: []*genai.Part{{Text: "Search for 'hello world'"}}, Role: "user"},
				},
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{
						FunctionDeclarations: []*genai.FunctionDeclaration{{
							Name:        "search",
							Description: "Search for a query",
							Parameters: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"query": {Type: genai.TypeString},
								},
								Required: []string{"query"},
							},
						}},
					}},
				},
			}

			resp := generateOne(t, ctx, llm, req)
			calls := requireFunctionCalls(t, resp)

			if calls[0].Name != "search" {
				t.Errorf("function = %q, want search", calls[0].Name)
			}

			t.Logf("model=%s call=%s args=%v", modelName, calls[0].Name, calls[0].Args)
		})
	}
}

func TestSmoke_ADK_BeforeModelCallback(t *testing.T) {
	// Proves req.Model override (set by ADK's BeforeModelCallback) actually
	// routes to a different model. We create the LLM as gpt-4.1-mini but
	// override to gemini. The response's ModelVersion should reflect gemini.
	provider := newProvider(t)
	ctx := withTimeout(t, 30*time.Second)

	// Create model as gpt-4.1-mini but the request will override to gemini.
	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	// Step 1: Normal request (no override) — should come from gpt-4.1-mini.
	reqNormal := simpleReq("Say hello")
	respNormal := generateOne(t, ctx, llm, reqNormal)

	// Step 2: Override to gemini — should come from gemini.
	reqOverride := &model.LLMRequest{
		Model: "gemini-2.5-flash", // Simulates BeforeModelCallback override.
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Say hello"}}, Role: "user"},
		},
	}
	respOverride := generateOne(t, ctx, llm, reqOverride)

	// ModelVersion from SAP AI Core reflects the actual model used.
	normalModel := respNormal.ModelVersion
	overrideModel := respOverride.ModelVersion

	t.Logf("normal ModelVersion=%q, override ModelVersion=%q", normalModel, overrideModel)

	if normalModel == overrideModel {
		t.Errorf("model override had no effect: both returned ModelVersion=%q", normalModel)
	}

	if !strings.Contains(strings.ToLower(overrideModel), "gemini") {
		t.Errorf("override ModelVersion=%q, expected to contain 'gemini'", overrideModel)
	}
}

func TestSmoke_CustomHeaders(t *testing.T) {
	// Verifies custom headers don't break the request.
	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithOrchestration(),
		sapaicore.WithHeaders(map[string][]string{
			"X-Custom-Test": {"smoke-integration"},
		}),
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
		t.Error("empty response with custom headers")
	}
}
