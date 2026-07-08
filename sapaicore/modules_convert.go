package sapaicore

import oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"

func buildFilteringWire(cfg *FilteringConfig) *oai.FilteringModuleConfig {
	if cfg == nil {
		return nil
	}

	out := &oai.FilteringModuleConfig{}

	if cfg.Input != nil {
		out.Input = &oai.InputFilteringConfig{}

		if cfg.Input.AzureContentSafety != nil {
			a := cfg.Input.AzureContentSafety
			out.Input.Filters = append(out.Input.Filters, oai.InputFilterConfig{
				Type: "azure_content_safety",
				Config: oai.AzureContentSafetyFilterInput{
					Hate:         int(a.Hate),
					SelfHarm:     int(a.SelfHarm),
					Sexual:       int(a.Sexual),
					Violence:     int(a.Violence),
					PromptShield: a.PromptShield,
				},
			})
		}

		if cfg.Input.LlamaGuard != nil {
			out.Input.Filters = append(out.Input.Filters, oai.InputFilterConfig{
				Type:   "llama_guard_3_8b",
				Config: llamaGuardWire(cfg.Input.LlamaGuard),
			})
		}
	}

	if cfg.Output != nil {
		out.Output = &oai.OutputFilteringConfig{}

		if cfg.Output.AzureContentSafety != nil {
			a := cfg.Output.AzureContentSafety
			out.Output.Filters = append(out.Output.Filters, oai.OutputFilterConfig{
				Type: "azure_content_safety",
				Config: oai.AzureContentSafetyFilterOutput{
					Hate:     int(a.Hate),
					SelfHarm: int(a.SelfHarm),
					Sexual:   int(a.Sexual),
					Violence: int(a.Violence),
				},
			})
		}

		if cfg.Output.LlamaGuard != nil {
			out.Output.Filters = append(out.Output.Filters, oai.OutputFilterConfig{
				Type:   "llama_guard_3_8b",
				Config: llamaGuardWire(cfg.Output.LlamaGuard),
			})
		}

		if cfg.Output.Overlap > 0 {
			out.Output.StreamOptions = &oai.FilteringStreamOpts{Overlap: cfg.Output.Overlap}
		}
	}

	return out
}

func llamaGuardWire(cfg *LlamaGuardConfig) oai.LlamaGuardFilterConfig {
	return oai.LlamaGuardFilterConfig{
		ViolentCrimes:         cfg.ViolentCrimes,
		NonViolentCrimes:      cfg.NonViolentCrimes,
		SexCrimes:             cfg.SexCrimes,
		ChildExploitation:     cfg.ChildExploitation,
		Defamation:            cfg.Defamation,
		SpecializedAdvice:     cfg.SpecializedAdvice,
		Privacy:               cfg.Privacy,
		IntellectualProperty:  cfg.IntellectualProperty,
		IndiscriminateWeapons: cfg.IndiscriminateWeapons,
		Hate:                  cfg.Hate,
		SelfHarm:              cfg.SelfHarm,
		SexualContent:         cfg.SexualContent,
		Elections:             cfg.Elections,
		CodeInterpreterAbuse:  cfg.CodeInterpreterAbuse,
	}
}

func buildMaskingWire(cfg *MaskingConfig) *oai.MaskingModuleConfig {
	if cfg == nil {
		return nil
	}

	entities := make([]oai.MaskingEntityConfig, len(cfg.Entities))
	for i, e := range cfg.Entities {
		switch {
		case e.custom != nil:
			entities[i] = oai.MaskingEntityConfig{
				Regex: e.custom.Regex,
				ReplacementStrategy: &oai.ReplacementStrategy{
					Method: "constant",
					Value:  e.custom.Replacement,
				},
			}
		case e.standard != nil:
			ec := oai.MaskingEntityConfig{Type: string(*e.standard)}

			switch e.strategy {
			case strategyFabricated:
				ec.ReplacementStrategy = &oai.ReplacementStrategy{Method: "fabricated_data"}
			case strategyConstant:
				ec.ReplacementStrategy = &oai.ReplacementStrategy{Method: "constant", Value: e.value}
			case strategyNone:
				// No replacement_strategy — use DPI default.
			}

			entities[i] = ec
		}
	}

	method := string(cfg.Method)
	if method == "" {
		method = string(Anonymization)
	}

	provider := oai.MaskingProviderConfig{
		Type:     "sap_data_privacy_integration",
		Method:   method,
		Entities: entities,
	}

	if len(cfg.Allowlist) > 0 {
		provider.Allowlist = cfg.Allowlist
	}

	if cfg.MaskGroundingInput {
		provider.MaskGroundingInput = &oai.MaskGroundingInput{Enabled: true}
	}

	if cfg.MaskFileInputMethod != "" {
		provider.MaskFileInputMethod = string(cfg.MaskFileInputMethod)
	}

	return &oai.MaskingModuleConfig{Providers: []oai.MaskingProviderConfig{provider}}
}

func buildTranslationWire(cfg *TranslationConfig) *oai.TranslationModuleConfig {
	if cfg == nil {
		return nil
	}

	out := &oai.TranslationModuleConfig{}

	if cfg.Input != nil {
		out.Input = &oai.SAPDocumentTranslationInput{
			Type: "sap_document_translation",
			Config: oai.TranslationInputWireConfig{
				SourceLanguage: cfg.Input.SourceLanguage,
				TargetLanguage: cfg.Input.TargetLanguage,
			},
		}
	}

	if cfg.Output != nil {
		out.Output = &oai.SAPDocumentTranslationOutput{
			Type: "sap_document_translation",
			Config: oai.TranslationOutputWireConfig{
				SourceLanguage: cfg.Output.SourceLanguage,
				TargetLanguage: cfg.Output.TargetLanguage,
			},
		}
	}

	return out
}

func buildStreamOptionsWire(cfg *StreamOptions) *oai.StreamConfig {
	if cfg == nil {
		return nil
	}

	return &oai.StreamConfig{
		ChunkSize:  cfg.ChunkSize,
		Delimiters: cfg.Delimiters,
	}
}
