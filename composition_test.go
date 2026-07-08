package sapaicore_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
	"google.golang.org/adk/v2/model"
)

func TestComposition_FilteringInRequestBody(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1")
	}, sapaicore.WithFiltering(nil))

	modules := extractModules(t, body)

	filtering, ok := modules["filtering"].(map[string]any)
	if !ok {
		t.Fatal("filtering module missing from request body")
	}

	input := filtering["input"].(map[string]any)
	filters := input["filters"].([]any)

	if len(filters) != 1 {
		t.Fatalf("input filters len = %d, want 1", len(filters))
	}

	f := filters[0].(map[string]any)
	if f["type"] != "azure_content_safety" {
		t.Errorf("filter type = %v, want azure_content_safety", f["type"])
	}

	cfg := f["config"].(map[string]any)
	if cfg["hate"] != float64(0) {
		t.Errorf("hate = %v, want 0", cfg["hate"])
	}

	if cfg["prompt_shield"] != true {
		t.Errorf("prompt_shield = %v, want true", cfg["prompt_shield"])
	}

	output := filtering["output"].(map[string]any)
	outFilters := output["filters"].([]any)

	if len(outFilters) != 1 {
		t.Fatalf("output filters len = %d, want 1", len(outFilters))
	}
}

func TestComposition_ModelFilteringReplacesProvider(t *testing.T) {
	t.Parallel()

	permissive := &sapaicore.FilteringConfig{
		Input: &sapaicore.InputFilterConfig{
			AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
				Hate:     sapaicore.AzureThresholdAllowAll,
				SelfHarm: sapaicore.AzureThresholdAllowAll,
				Sexual:   sapaicore.AzureThresholdAllowAll,
				Violence: sapaicore.AzureThresholdAllowAll,
			},
		},
	}

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1", sapaicore.WithModelFiltering(permissive))
	}, sapaicore.WithFiltering(nil))

	modules := extractModules(t, body)
	filtering := modules["filtering"].(map[string]any)
	input := filtering["input"].(map[string]any)
	filters := input["filters"].([]any)
	cfg := filters[0].(map[string]any)["config"].(map[string]any)

	if cfg["hate"] != float64(6) {
		t.Errorf("hate = %v, want 6 (model should override provider)", cfg["hate"])
	}

	if filtering["output"] != nil {
		t.Errorf("output should be nil (model config has no output), got %v", filtering["output"])
	}
}

func TestComposition_WithoutFilteringRemovesInherited(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1", sapaicore.WithoutFiltering())
	}, sapaicore.WithFiltering(nil))

	modules := extractModules(t, body)

	if modules["filtering"] != nil {
		t.Errorf("filtering should be nil after WithoutFiltering, got %v", modules["filtering"])
	}
}

func TestComposition_MaskingInRequestBody(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1")
	}, sapaicore.WithMasking(sapaicore.MaskingConfig{
		Entities:  sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
		Allowlist: []string{"SAP"},
	}))

	modules := extractModules(t, body)

	masking, ok := modules["masking"].(map[string]any)
	if !ok {
		t.Fatal("masking module missing")
	}

	providers := masking["providers"].([]any)
	if len(providers) != 1 {
		t.Fatalf("providers len = %d, want 1", len(providers))
	}

	p := providers[0].(map[string]any)

	if p["type"] != "sap_data_privacy_integration" {
		t.Errorf("type = %v, want sap_data_privacy_integration", p["type"])
	}

	if p["method"] != "anonymization" {
		t.Errorf("method = %v, want anonymization", p["method"])
	}

	entities := p["entities"].([]any)
	if len(entities) != 5 {
		t.Errorf("entities len = %d, want 5", len(entities))
	}

	allowlist := p["allowlist"].([]any)
	if allowlist[0] != "SAP" {
		t.Errorf("allowlist[0] = %v, want SAP", allowlist[0])
	}
}

func TestComposition_TranslationInRequestBody(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1")
	}, sapaicore.WithTranslation(sapaicore.TranslationConfig{
		Output: &sapaicore.TranslationOutputConfig{
			TargetLanguage: "de-DE",
		},
	}))

	modules := extractModules(t, body)

	translation, ok := modules["translation"].(map[string]any)
	if !ok {
		t.Fatal("translation module missing")
	}

	output := translation["output"].(map[string]any)

	if output["type"] != "sap_document_translation" {
		t.Errorf("type = %v, want sap_document_translation", output["type"])
	}

	cfg := output["config"].(map[string]any)

	if cfg["target_language"] != "de-DE" {
		t.Errorf("target_language = %v, want de-DE", cfg["target_language"])
	}
}

func TestComposition_FallbackProducesArray(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1",
			sapaicore.WithModelFallback("gpt-4.1-mini"),
		)
	}, sapaicore.WithFiltering(nil))

	cfg := body["config"].(map[string]any)
	modulesRaw := cfg["modules"]

	arr, ok := modulesRaw.([]any)
	if !ok {
		t.Fatalf("modules should be array for fallback, got %T", modulesRaw)
	}

	if len(arr) != 2 {
		t.Fatalf("modules array len = %d, want 2", len(arr))
	}

	first := arr[0].(map[string]any)
	pt := first["prompt_templating"].(map[string]any)
	m := pt["model"].(map[string]any)

	if m["name"] != "gpt-4.1" {
		t.Errorf("first model = %v, want gpt-4.1", m["name"])
	}

	second := arr[1].(map[string]any)
	pt2 := second["prompt_templating"].(map[string]any)
	m2 := pt2["model"].(map[string]any)

	if m2["name"] != "gpt-4.1-mini" {
		t.Errorf("second model = %v, want gpt-4.1-mini", m2["name"])
	}

	if second["filtering"] == nil {
		t.Error("filtering should be replicated to fallback entry")
	}
}

func TestComposition_PromptCachingAnnotatesSystem(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("anthropic--claude-4.5-sonnet",
			sapaicore.WithModelPromptCaching(),
		)
	})

	modules := extractModules(t, body)
	pt := modules["prompt_templating"].(map[string]any)
	prompt := pt["prompt"].(map[string]any)
	template := prompt["template"].([]any)

	systemMsg := template[0].(map[string]any)

	if systemMsg["role"] != "system" {
		t.Fatalf("first message role = %v, want system", systemMsg["role"])
	}

	// Content should be an array of text blocks with cache_control.
	contentBlocks, ok := systemMsg["content"].([]any)
	if !ok {
		t.Fatalf("system message content should be array, got %T", systemMsg["content"])
	}

	block := contentBlocks[0].(map[string]any)
	if block["type"] != "text" {
		t.Errorf("block type = %v, want text", block["type"])
	}

	cc, ok := block["cache_control"].(map[string]any)
	if !ok {
		t.Fatal("text block missing cache_control")
	}

	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control type = %v, want ephemeral", cc["type"])
	}
}

func TestComposition_FoundationModeRejectsModules(t *testing.T) {
	t.Parallel()

	provider, err := sapaicore.NewProvider(t.Context(),
		sapaicore.WithEndpoint("https://api.example.com"),
		sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
		sapaicore.WithDeployments(map[string]string{"gpt-4.1-mini": "d1"}),
		sapaicore.WithFiltering(nil),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	_, err = provider.Model("gpt-4.1-mini")
	if err == nil {
		t.Fatal("expected error for foundation + modules")
	}
}

// --- Test helpers ---

func captureOrchestrationBody(t *testing.T, modelFn func(*sapaicore.Provider) (model.LLM, error), providerOpts ...sapaicore.Option) map[string]any {
	t.Helper()

	authServer := newMockAuthServer(t)
	defer authServer.Close()

	var capturedBody map[string]any

	inferenceServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &capturedBody)
		writeOrchestrationResponse(w, "ok", "stop")
	}))
	defer inferenceServer.Close()

	opts := []sapaicore.Option{
		sapaicore.WithEndpoint(inferenceServer.URL),
		sapaicore.WithAuth("id", "secret", authServer.URL+"/oauth/token"),
		sapaicore.WithDeploymentID("orch-test"),
	}
	opts = append(opts, providerOpts...)

	provider, err := sapaicore.NewProvider(t.Context(), opts...)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, err := modelFn(provider)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{Parts: []*genai.Part{{Text: "Hello"}}, Role: "user"},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{{Text: "You are helpful."}},
			},
		},
	}

	for _, err := range llm.GenerateContent(t.Context(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}

	if capturedBody == nil {
		t.Fatal("no request body captured")
	}

	return capturedBody
}

func extractModules(t *testing.T, body map[string]any) map[string]any {
	t.Helper()

	cfg, ok := body["config"].(map[string]any)
	if !ok {
		t.Fatal("config missing from body")
	}

	modules, ok := cfg["modules"].(map[string]any)
	if !ok {
		t.Fatal("modules missing or is array (use array test for fallback)")
	}

	return modules
}

func TestComposition_FallbackStreamOverlapOnAllEntries(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1",
			sapaicore.WithModelFallback("gpt-4.1-mini"),
			sapaicore.WithModelFiltering(&sapaicore.FilteringConfig{
				Output: &sapaicore.OutputFilterConfig{
					AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
						Hate: sapaicore.AzureThresholdAllowSafe,
					},
					Overlap: 50,
				},
			}),
		)
	})

	cfg := body["config"].(map[string]any)
	arr := cfg["modules"].([]any)

	if len(arr) != 2 {
		t.Fatalf("modules array len = %d, want 2", len(arr))
	}

	// Both entries should have output filtering with stream_options.overlap.
	for i, entry := range arr {
		m := entry.(map[string]any)
		filtering := m["filtering"].(map[string]any)
		output := filtering["output"].(map[string]any)
		streamOpts, ok := output["stream_options"].(map[string]any)

		if !ok {
			t.Fatalf("entry %d: missing stream_options on output filtering", i)
		}

		if streamOpts["overlap"] != float64(50) {
			t.Errorf("entry %d: overlap = %v, want 50", i, streamOpts["overlap"])
		}
	}
}

func TestComposition_FabricatedEntityInRequestBody(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1")
	}, sapaicore.WithMasking(sapaicore.MaskingConfig{
		Method: sapaicore.Pseudonymization,
		Entities: []sapaicore.MaskingEntity{
			sapaicore.StandardEntity(sapaicore.EntityEmail),
			sapaicore.FabricatedEntity(sapaicore.EntityPerson),
			sapaicore.ConstantEntity(sapaicore.EntityPhone, "PHONE_REDACTED"),
		},
	}))

	modules := extractModules(t, body)
	masking := modules["masking"].(map[string]any)
	providers := masking["providers"].([]any)
	p := providers[0].(map[string]any)
	entities := p["entities"].([]any)

	if len(entities) != 3 {
		t.Fatalf("entities len = %d, want 3", len(entities))
	}

	// Entity 0: StandardEntity(EntityEmail) — no replacement_strategy.
	e0 := entities[0].(map[string]any)
	if e0["type"] != "profile-email" {
		t.Errorf("entity[0].type = %v, want profile-email", e0["type"])
	}

	if e0["replacement_strategy"] != nil {
		t.Errorf("entity[0] should have no replacement_strategy, got %v", e0["replacement_strategy"])
	}

	// Entity 1: FabricatedEntity(EntityPerson) — fabricated_data.
	e1 := entities[1].(map[string]any)
	if e1["type"] != "profile-person" {
		t.Errorf("entity[1].type = %v, want profile-person", e1["type"])
	}

	strat1 := e1["replacement_strategy"].(map[string]any)
	if strat1["method"] != "fabricated_data" {
		t.Errorf("entity[1].replacement_strategy.method = %v, want fabricated_data", strat1["method"])
	}

	if strat1["value"] != nil {
		t.Errorf("fabricated_data should not have value, got %v", strat1["value"])
	}

	// Entity 2: ConstantEntity(EntityPhone, "PHONE_REDACTED") — constant.
	e2 := entities[2].(map[string]any)
	if e2["type"] != "profile-phone" {
		t.Errorf("entity[2].type = %v, want profile-phone", e2["type"])
	}

	strat2 := e2["replacement_strategy"].(map[string]any)
	if strat2["method"] != "constant" {
		t.Errorf("entity[2].replacement_strategy.method = %v, want constant", strat2["method"])
	}

	if strat2["value"] != "PHONE_REDACTED" {
		t.Errorf("entity[2].replacement_strategy.value = %v, want PHONE_REDACTED", strat2["value"])
	}
}

func TestComposition_MaskFileInputMethodInRequestBody(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1")
	}, sapaicore.WithMasking(sapaicore.MaskingConfig{
		Method:              sapaicore.Anonymization,
		Entities:            sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
		MaskFileInputMethod: sapaicore.MaskFileSkip,
	}))

	modules := extractModules(t, body)
	masking := modules["masking"].(map[string]any)
	providers := masking["providers"].([]any)
	p := providers[0].(map[string]any)

	if p["mask_file_input_method"] != "skip" {
		t.Errorf("mask_file_input_method = %v, want skip", p["mask_file_input_method"])
	}
}

func TestComposition_MaskFileInputMethodOmittedWhenEmpty(t *testing.T) {
	t.Parallel()

	body := captureOrchestrationBody(t, func(p *sapaicore.Provider) (model.LLM, error) {
		return p.Model("gpt-4.1")
	}, sapaicore.WithMasking(sapaicore.MaskingConfig{
		Method:   sapaicore.Anonymization,
		Entities: sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
	}))

	modules := extractModules(t, body)
	masking := modules["masking"].(map[string]any)
	providers := masking["providers"].([]any)
	p := providers[0].(map[string]any)

	if p["mask_file_input_method"] != nil {
		t.Errorf("mask_file_input_method should be omitted when empty, got %v", p["mask_file_input_method"])
	}
}
