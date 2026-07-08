package openai

// --- Filtering module ---

// FilteringModuleConfig configures content filtering for orchestration mode.
type FilteringModuleConfig struct {
	Input  *InputFilteringConfig  `json:"input,omitempty"`
	Output *OutputFilteringConfig `json:"output,omitempty"`
}

// InputFilteringConfig holds the list of input content filters.
type InputFilteringConfig struct {
	Filters []InputFilterConfig `json:"filters"`
}

// OutputFilteringConfig holds the list of output content filters.
type OutputFilteringConfig struct {
	Filters       []OutputFilterConfig `json:"filters"`
	StreamOptions *FilteringStreamOpts `json:"stream_options,omitempty"`
}

// FilteringStreamOpts controls streaming overlap for output filtering.
type FilteringStreamOpts struct {
	Overlap int `json:"overlap,omitempty"`
}

// InputFilterConfig defines a single input filter provider and its configuration.
type InputFilterConfig struct {
	Type   string `json:"type"`
	Config any    `json:"config,omitempty"`
}

// OutputFilterConfig defines a single output filter provider and its configuration.
type OutputFilterConfig struct {
	Type   string `json:"type"`
	Config any    `json:"config,omitempty"`
}

// AzureContentSafetyFilterInput configures Azure Content Safety thresholds for input.
type AzureContentSafetyFilterInput struct {
	Hate         int  `json:"hate"`
	SelfHarm     int  `json:"self_harm"`
	Sexual       int  `json:"sexual"`
	Violence     int  `json:"violence"`
	PromptShield bool `json:"prompt_shield,omitempty"`
}

// AzureContentSafetyFilterOutput configures Azure Content Safety thresholds for output.
type AzureContentSafetyFilterOutput struct {
	Hate     int `json:"hate"`
	SelfHarm int `json:"self_harm"`
	Sexual   int `json:"sexual"`
	Violence int `json:"violence"`
}

// LlamaGuardFilterConfig enables specific Llama Guard 3 8B safety categories.
type LlamaGuardFilterConfig struct {
	ViolentCrimes         bool `json:"violent_crimes,omitempty"`
	NonViolentCrimes      bool `json:"non_violent_crimes,omitempty"`
	SexCrimes             bool `json:"sex_crimes,omitempty"`
	ChildExploitation     bool `json:"child_exploitation,omitempty"`
	Defamation            bool `json:"defamation,omitempty"`
	SpecializedAdvice     bool `json:"specialized_advice,omitempty"`
	Privacy               bool `json:"privacy,omitempty"`
	IntellectualProperty  bool `json:"intellectual_property,omitempty"`
	IndiscriminateWeapons bool `json:"indiscriminate_weapons,omitempty"`
	Hate                  bool `json:"hate,omitempty"`
	SelfHarm              bool `json:"self_harm,omitempty"`
	SexualContent         bool `json:"sexual_content,omitempty"`
	Elections             bool `json:"elections,omitempty"`
	CodeInterpreterAbuse  bool `json:"code_interpreter_abuse,omitempty"`
}

// --- Masking module ---

// MaskingModuleConfig configures data masking for orchestration mode.
type MaskingModuleConfig struct {
	Providers []MaskingProviderConfig `json:"providers"`
}

// MaskingProviderConfig defines a masking provider and its entity configuration.
type MaskingProviderConfig struct {
	Type                string                `json:"type"`
	Method              string                `json:"method"`
	Entities            []MaskingEntityConfig `json:"entities"`
	Allowlist           []string              `json:"allowlist,omitempty"`
	MaskGroundingInput  *MaskGroundingInput   `json:"mask_grounding_input,omitempty"`
	MaskFileInputMethod string                `json:"mask_file_input_method,omitempty"`
}

// MaskingEntityConfig defines a single entity to mask.
type MaskingEntityConfig struct {
	Type                string               `json:"type,omitempty"`
	Regex               string               `json:"regex,omitempty"`
	ReplacementStrategy *ReplacementStrategy `json:"replacement_strategy,omitempty"`
}

// ReplacementStrategy represents a DPI replacement method.
// For "constant": Method="constant", Value is required.
// For "fabricated_data": Method="fabricated_data", Value is omitted.
type ReplacementStrategy struct {
	Method string `json:"method"`
	Value  string `json:"value,omitempty"`
}

// MaskGroundingInput controls whether grounding module input is also masked.
type MaskGroundingInput struct {
	Enabled bool `json:"enabled"`
}

// --- Translation module ---

// TranslationModuleConfig configures translation for orchestration mode.
type TranslationModuleConfig struct {
	Input  *SAPDocumentTranslationInput  `json:"input,omitempty"`
	Output *SAPDocumentTranslationOutput `json:"output,omitempty"`
}

// SAPDocumentTranslationInput configures input translation.
type SAPDocumentTranslationInput struct {
	Type   string                     `json:"type"`
	Config TranslationInputWireConfig `json:"config"`
}

// TranslationInputWireConfig holds input translation parameters.
type TranslationInputWireConfig struct {
	SourceLanguage string `json:"source_language,omitempty"`
	TargetLanguage string `json:"target_language"`
}

// SAPDocumentTranslationOutput configures output translation.
type SAPDocumentTranslationOutput struct {
	Type   string                      `json:"type"`
	Config TranslationOutputWireConfig `json:"config"`
}

// TranslationOutputWireConfig holds output translation parameters.
type TranslationOutputWireConfig struct {
	SourceLanguage string `json:"source_language,omitempty"`
	TargetLanguage string `json:"target_language"`
}
