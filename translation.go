package sapaicore

// TranslationInputConfig configures input translation before messages reach the LLM.
type TranslationInputConfig struct {
	// SourceLanguage (e.g. "de-DE"). Empty means auto-detect.
	SourceLanguage string
	// TargetLanguage (e.g. "en-US"). Required.
	TargetLanguage string
}

// TranslationOutputConfig configures output translation of the LLM response.
type TranslationOutputConfig struct {
	// SourceLanguage of the LLM output. Empty means auto-detect.
	SourceLanguage string
	// TargetLanguage for the translated output (e.g. "fr-FR"). Required.
	TargetLanguage string
}

// TranslationConfig configures translation. Orchestration-mode only.
// At least one of Input or Output must be set.
type TranslationConfig struct {
	Input  *TranslationInputConfig
	Output *TranslationOutputConfig
}

// StreamOptions configures global streaming behavior for orchestration modules.
// These options control how post-LLM modules (translation, filtering) process chunks.
type StreamOptions struct {
	// ChunkSize is the minimum characters per chunk that post-LLM modules operate on.
	ChunkSize int
	// Delimiters to split text into chunks. Required when translation is active with streaming.
	Delimiters []string
}

// CacheTTL controls the time-to-live for prompt cache entries.
type CacheTTL string

const (
	CacheTTL5m CacheTTL = "5m"
	CacheTTL1h CacheTTL = "1h" // Only supported on select Anthropic models.
)
