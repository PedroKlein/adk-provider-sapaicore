package sapaicore

// moduleConfigs holds all orchestration module configurations.
type moduleConfigs struct {
	filtering      *FilteringConfig
	masking        *MaskingConfig
	translation    *TranslationConfig
	noFiltering    bool
	noMasking      bool
	noTranslation  bool
	fallbackModels []string
	promptCaching  bool
	cacheTTL       CacheTTL
	streamOptions  *StreamOptions
}

type resolvedModules struct {
	Filtering      *FilteringConfig
	Masking        *MaskingConfig
	Translation    *TranslationConfig
	FallbackModels []string
	PromptCaching  bool
	CacheTTL       CacheTTL
	StreamOptions  *StreamOptions
}

// resolveModules applies per-module replace composition:
//   - Same module at both levels → model wins
//   - Different modules → both apply
//   - Without*() → removes inherited module
func resolveModules(provider, model *moduleConfigs) resolvedModules {
	return resolvedModules{
		Filtering:      resolveFiltering(provider, model),
		Masking:        resolveMasking(provider, model),
		Translation:    resolveTranslation(provider, model),
		FallbackModels: resolveFallback(provider, model),
		PromptCaching:  (model != nil && model.promptCaching) || (provider != nil && provider.promptCaching),
		CacheTTL:       resolveCacheTTL(provider, model),
		StreamOptions:  resolveStreamOptions(provider, model),
	}
}

func resolveFiltering(provider, model *moduleConfigs) *FilteringConfig {
	switch {
	case model != nil && model.noFiltering:
		return nil
	case model != nil && model.filtering != nil:
		return model.filtering
	case provider != nil && provider.filtering != nil:
		return provider.filtering
	default:
		return nil
	}
}

func resolveMasking(provider, model *moduleConfigs) *MaskingConfig {
	switch {
	case model != nil && model.noMasking:
		return nil
	case model != nil && model.masking != nil:
		return model.masking
	case provider != nil && provider.masking != nil:
		return provider.masking
	default:
		return nil
	}
}

func resolveTranslation(provider, model *moduleConfigs) *TranslationConfig {
	switch {
	case model != nil && model.noTranslation:
		return nil
	case model != nil && model.translation != nil:
		return model.translation
	case provider != nil && provider.translation != nil:
		return provider.translation
	default:
		return nil
	}
}

func resolveFallback(provider, model *moduleConfigs) []string {
	if model != nil && len(model.fallbackModels) > 0 {
		return model.fallbackModels
	}

	if provider != nil && len(provider.fallbackModels) > 0 {
		return provider.fallbackModels
	}

	return nil
}

func resolveStreamOptions(provider, model *moduleConfigs) *StreamOptions {
	if model != nil && model.streamOptions != nil {
		return model.streamOptions
	}

	if provider != nil && provider.streamOptions != nil {
		return provider.streamOptions
	}

	return nil
}

func resolveCacheTTL(provider, model *moduleConfigs) CacheTTL {
	if model != nil && model.cacheTTL != "" {
		return model.cacheTTL
	}

	if provider != nil && provider.cacheTTL != "" {
		return provider.cacheTTL
	}

	return ""
}

// defaultFilteringConfig returns the zero-config default:
// Azure Content Safety ALLOW_SAFE on all categories + prompt_shield, input+output.
func defaultFilteringConfig() *FilteringConfig {
	return &FilteringConfig{
		Input: &InputFilterConfig{
			AzureContentSafety: &AzureContentSafetyConfig{
				Hate:         AzureThresholdAllowSafe,
				SelfHarm:     AzureThresholdAllowSafe,
				Sexual:       AzureThresholdAllowSafe,
				Violence:     AzureThresholdAllowSafe,
				PromptShield: true,
			},
		},
		Output: &OutputFilterConfig{
			AzureContentSafety: &AzureContentSafetyConfig{
				Hate:     AzureThresholdAllowSafe,
				SelfHarm: AzureThresholdAllowSafe,
				Sexual:   AzureThresholdAllowSafe,
				Violence: AzureThresholdAllowSafe,
			},
		},
	}
}
