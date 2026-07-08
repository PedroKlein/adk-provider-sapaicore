package sapaicore_test

import (
	"errors"
	"testing"

	"github.com/PedroKlein/adk-provider-sapaicore/sapaicore"
)

func newTestProvider(t *testing.T, extraOpts ...sapaicore.Option) *sapaicore.Provider {
	t.Helper()

	opts := []sapaicore.Option{
		sapaicore.WithEndpoint("https://api.example.com"),
		sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
		sapaicore.WithDeploymentID("d-test-orch"),
	}
	opts = append(opts, extraOpts...)

	p, err := sapaicore.NewProvider(t.Context(), opts...)
	if err != nil {
		t.Fatalf("newTestProvider: %v", err)
	}

	return p
}

func newFoundationTestProvider(t *testing.T, extraOpts ...sapaicore.Option) *sapaicore.Provider {
	t.Helper()

	opts := []sapaicore.Option{
		sapaicore.WithEndpoint("https://api.example.com"),
		sapaicore.WithAuth("id", "secret", "https://auth.example.com/token"),
		sapaicore.WithDeployments(map[string]string{"gpt-4.1-mini": "d-foundation"}),
	}
	opts = append(opts, extraOpts...)

	p, err := sapaicore.NewProvider(t.Context(), opts...)
	if err != nil {
		t.Fatalf("newFoundationTestProvider: %v", err)
	}

	return p
}

func TestResolveModules_FilteringComposition(t *testing.T) {
	t.Parallel()

	strict := &sapaicore.FilteringConfig{
		Input: &sapaicore.InputFilterConfig{
			AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
				Hate: sapaicore.AzureThresholdAllowSafe,
			},
		},
	}
	permissive := &sapaicore.FilteringConfig{
		Input: &sapaicore.InputFilterConfig{
			AzureContentSafety: &sapaicore.AzureContentSafetyConfig{
				Hate: sapaicore.AzureThresholdAllowAll,
			},
		},
	}

	tests := []struct {
		name         string
		providerOpts []sapaicore.Option
		modelOpts    []sapaicore.ModelOption
	}{
		{
			name: "no modules configured",
		},
		{
			name:         "provider filtering only",
			providerOpts: []sapaicore.Option{sapaicore.WithFiltering(strict)},
		},
		{
			name:         "model filtering replaces provider",
			providerOpts: []sapaicore.Option{sapaicore.WithFiltering(strict)},
			modelOpts:    []sapaicore.ModelOption{sapaicore.WithModelFiltering(permissive)},
		},
		{
			name:         "WithoutFiltering removes inherited",
			providerOpts: []sapaicore.Option{sapaicore.WithFiltering(strict)},
			modelOpts:    []sapaicore.ModelOption{sapaicore.WithoutFiltering()},
		},
		{
			name:         "nil filtering uses defaults",
			providerOpts: []sapaicore.Option{sapaicore.WithFiltering(nil)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider := newTestProvider(t, tt.providerOpts...)

			llm, err := provider.Model("gpt-4.1", tt.modelOpts...)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			if llm == nil {
				t.Fatal("Model returned nil")
			}
		})
	}
}

func TestResolveModules_MaskingComposition(t *testing.T) {
	t.Parallel()

	masking := sapaicore.MaskingConfig{
		Method:   sapaicore.Anonymization,
		Entities: sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
	}

	tests := []struct {
		name         string
		providerOpts []sapaicore.Option
		modelOpts    []sapaicore.ModelOption
	}{
		{
			name:         "provider masking only",
			providerOpts: []sapaicore.Option{sapaicore.WithMasking(masking)},
		},
		{
			name:      "model masking only",
			modelOpts: []sapaicore.ModelOption{sapaicore.WithModelMasking(masking)},
		},
		{
			name:         "WithoutMasking removes inherited",
			providerOpts: []sapaicore.Option{sapaicore.WithMasking(masking)},
			modelOpts:    []sapaicore.ModelOption{sapaicore.WithoutMasking()},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider := newTestProvider(t, tt.providerOpts...)

			llm, err := provider.Model("gpt-4.1", tt.modelOpts...)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			if llm == nil {
				t.Fatal("Model returned nil")
			}
		})
	}
}

func TestResolveModules_DifferentModulesMerge(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t, sapaicore.WithFiltering(nil))

	llm, err := provider.Model("gpt-4.1",
		sapaicore.WithModelMasking(sapaicore.MaskingConfig{
			Entities: sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
		}),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm == nil {
		t.Fatal("Model returned nil")
	}
}

func TestResolveModules_Fallback(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t)

	llm, err := provider.Model("gpt-4.1",
		sapaicore.WithModelFallback("gpt-4.1-mini", "gpt-4.1-nano"),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm == nil {
		t.Fatal("Model returned nil")
	}
}

func TestResolveModules_PromptCaching(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t, sapaicore.WithPromptCaching())

	llm, err := provider.Model("anthropic--claude-4.5-sonnet")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm == nil {
		t.Fatal("Model returned nil")
	}

	provider2 := newTestProvider(t)

	llm2, err := provider2.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelPromptCaching(),
	)
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm2 == nil {
		t.Fatal("Model returned nil")
	}
}

func TestResolveModules_FoundationRejectsModules(t *testing.T) {
	t.Parallel()

	provider := newFoundationTestProvider(t,
		sapaicore.WithFiltering(nil),
	)

	_, err := provider.Model("gpt-4.1-mini")
	if err == nil {
		t.Fatal("expected error for foundation mode + modules")
	}

	if !errors.Is(err, sapaicore.ErrMissingConfig) {
		t.Errorf("expected ErrMissingConfig, got: %v", err)
	}
}

func TestMaskingMethod_DefaultsToAnonymization(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t,
		sapaicore.WithMasking(sapaicore.MaskingConfig{
			Entities: sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
		}),
	)

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	if llm == nil {
		t.Fatal("Model returned nil")
	}
}

func TestStandardEntities(t *testing.T) {
	t.Parallel()

	entities := sapaicore.StandardEntities(sapaicore.CommonPIIEntities)
	if len(entities) != 5 {
		t.Errorf("len(StandardEntities(CommonPIIEntities)) = %d, want 5", len(entities))
	}
}

func TestCustomMaskingEntity(t *testing.T) {
	t.Parallel()

	entity := sapaicore.CustomMaskingEntity(`\b[0-9]{2}-SAP-[0-9]{3}\b`, "REDACTED_ID")
	entities := []sapaicore.MaskingEntity{entity}

	if len(entities) != 1 {
		t.Errorf("len = %d, want 1", len(entities))
	}
}

func TestValidation_TranslationRequiresInputOrOutput(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t,
		sapaicore.WithTranslation(sapaicore.TranslationConfig{}),
	)

	_, err := provider.Model("gpt-4.1")
	if err == nil {
		t.Fatal("expected error for empty translation config")
	}

	if !errors.Is(err, sapaicore.ErrMissingConfig) {
		t.Errorf("expected ErrMissingConfig, got: %v", err)
	}
}

func TestValidation_MaskingRequiresEntities(t *testing.T) {
	t.Parallel()

	provider := newTestProvider(t,
		sapaicore.WithMasking(sapaicore.MaskingConfig{
			Method: sapaicore.Anonymization,
			// Entities intentionally empty.
		}),
	)

	_, err := provider.Model("gpt-4.1")
	if err == nil {
		t.Fatal("expected error for empty masking entities")
	}

	if !errors.Is(err, sapaicore.ErrMissingConfig) {
		t.Errorf("expected ErrMissingConfig, got: %v", err)
	}
}
