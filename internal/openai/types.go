// Package openai defines wire types for the OpenAI-compatible chat completions
// format used by both SAP AI Core API modes.
package openai

import "encoding/json"

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	Refusal    string     `json:"refusal,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// --- Content blocks (multi-block content arrays) ---

// ContentBlock is implemented by all types that can appear in a multimodal
// message content array. Prevents arbitrary types from being added to the slice.
type ContentBlock interface {
	contentBlock()
}

type TextContentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (TextContentBlock) contentBlock() {}

type ImageURLContentBlock struct {
	Type         string        `json:"type"`
	ImageURL     ImageURL      `json:"image_url"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (ImageURLContentBlock) contentBlock() {}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type FileContentBlock struct {
	Type         string        `json:"type"`
	File         FileContent   `json:"file"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (FileContentBlock) contentBlock() {}

type FileContent struct {
	FileData string `json:"file_data"`
	Filename string `json:"filename,omitempty"`
}

type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type ToolDef struct {
	Type         string        `json:"type"`
	Function     FunctionDef   `json:"function"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type ChatChoice struct {
	Index        int           `json:"index"`
	Message      ChatMessage   `json:"message"`
	FinishReason string        `json:"finish_reason"`
	Logprobs     *ChatLogprobs `json:"logprobs,omitempty"`
}

type ChatLogprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
}

type TokenLogprob struct {
	Token       string            `json:"token"`
	Logprob     float64           `json:"logprob"`
	TokenID     int32             `json:"token_id,omitempty"`
	TopLogprobs []TopTokenLogprob `json:"top_logprobs,omitempty"`
}

type TopTokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	TokenID int32   `json:"token_id,omitempty"`
}

type ChatUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

type ChatError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

type ChunkChoice struct {
	Index        int           `json:"index"`
	Delta        ChatDelta     `json:"delta"`
	FinishReason string        `json:"finish_reason,omitempty"`
	Logprobs     *ChatLogprobs `json:"logprobs,omitempty"`
}

type ChatDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// --- Foundation-models mode ---

type FoundationRequest struct {
	Model            string          `json:"model"`
	Messages         []ChatMessage   `json:"messages"`
	Tools            []ToolDef       `json:"tools,omitempty"`
	ToolChoice       any             `json:"tool_choice,omitempty"`
	Stream           bool            `json:"stream"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Temperature      *float32        `json:"temperature,omitempty"`
	MaxTokens        *int32          `json:"max_tokens,omitempty"`
	TopP             *float32        `json:"top_p,omitempty"`
	TopK             *float32        `json:"top_k,omitempty"`
	Seed             *int32          `json:"seed,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	FrequencyPenalty *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float32        `json:"presence_penalty,omitempty"`
	Logprobs         *bool           `json:"logprobs,omitempty"`
	TopLogprobs      *int32          `json:"top_logprobs,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type FoundationResponse struct {
	ID      string       `json:"id"`
	Choices []ChatChoice `json:"choices"`
	Usage   *ChatUsage   `json:"usage,omitempty"`
	Model   string       `json:"model"`
	Error   *ChatError   `json:"error,omitempty"`
}

type FoundationChunk struct {
	ID      string        `json:"id"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *ChatUsage    `json:"usage,omitempty"`
	Model   string        `json:"model"`
}

// --- Orchestration mode ---

type OrchestrationRequest struct {
	Config OrchestrationConfig `json:"config"`
}

type OrchestrationConfig struct {
	Stream  *StreamConfig   `json:"stream,omitempty"`
	Modules json.RawMessage `json:"modules"`
}

type StreamConfig struct {
	Enabled    bool     `json:"enabled"`
	ChunkSize  int      `json:"chunk_size,omitempty"`
	Delimiters []string `json:"delimiters,omitempty"`
}

type ModuleConfigs struct {
	PromptTemplating PromptTemplatingModule   `json:"prompt_templating"`
	Filtering        *FilteringModuleConfig   `json:"filtering,omitempty"`
	Masking          *MaskingModuleConfig     `json:"masking,omitempty"`
	Translation      *TranslationModuleConfig `json:"translation,omitempty"`
}

type PromptTemplatingModule struct {
	Prompt PromptConfig `json:"prompt"`
	Model  ModelDef     `json:"model"`
}

type PromptConfig struct {
	Template       []ChatMessage   `json:"template"`
	Tools          []ToolDef       `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type ModelDef struct {
	Name       string         `json:"name"`
	Version    string         `json:"version,omitempty"`
	Params     map[string]any `json:"params,omitempty"`
	Timeout    int            `json:"timeout,omitempty"`
	MaxRetries int            `json:"max_retries,omitempty"`
}

type OrchestrationResponse struct {
	RequestID   string              `json:"request_id"`
	FinalResult *FoundationResponse `json:"final_result"`
}

type OrchestrationChunk struct {
	RequestID   string           `json:"request_id"`
	FinalResult *FoundationChunk `json:"final_result"`
}

type RequestParams struct {
	ModelName        string
	Messages         []ChatMessage
	Tools            []ToolDef
	ToolChoice       any
	Temperature      *float32
	MaxTokens        int32
	TopP             *float32
	TopK             *float32
	Seed             *int32
	Stop             []string
	FrequencyPenalty *float32
	PresencePenalty  *float32
	ResponseLogprobs bool
	Logprobs         *int32
	ResponseFormat   *ResponseFormat
	Stream           bool
	ExtraParams      map[string]any
	Timeout          int
	MaxRetries       int
}

// --- Cache control ---

type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// --- Filtering module ---

type FilteringModuleConfig struct {
	Input  *InputFilteringConfig  `json:"input,omitempty"`
	Output *OutputFilteringConfig `json:"output,omitempty"`
}

type InputFilteringConfig struct {
	Filters []InputFilterConfig `json:"filters"`
}

type OutputFilteringConfig struct {
	Filters       []OutputFilterConfig `json:"filters"`
	StreamOptions *FilteringStreamOpts `json:"stream_options,omitempty"`
}

type FilteringStreamOpts struct {
	Overlap int `json:"overlap,omitempty"`
}

type InputFilterConfig struct {
	Type   string `json:"type"`
	Config any    `json:"config,omitempty"`
}

type OutputFilterConfig struct {
	Type   string `json:"type"`
	Config any    `json:"config,omitempty"`
}

type AzureContentSafetyFilterInput struct {
	Hate         int  `json:"hate"`
	SelfHarm     int  `json:"self_harm"`
	Sexual       int  `json:"sexual"`
	Violence     int  `json:"violence"`
	PromptShield bool `json:"prompt_shield,omitempty"`
}

type AzureContentSafetyFilterOutput struct {
	Hate     int `json:"hate"`
	SelfHarm int `json:"self_harm"`
	Sexual   int `json:"sexual"`
	Violence int `json:"violence"`
}

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

type MaskingModuleConfig struct {
	Providers []MaskingProviderConfig `json:"providers"`
}

type MaskingProviderConfig struct {
	Type                string                `json:"type"`
	Method              string                `json:"method"`
	Entities            []MaskingEntityConfig `json:"entities"`
	Allowlist           []string              `json:"allowlist,omitempty"`
	MaskGroundingInput  *MaskGroundingInput   `json:"mask_grounding_input,omitempty"`
	MaskFileInputMethod string                `json:"mask_file_input_method,omitempty"`
}

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

type MaskGroundingInput struct {
	Enabled bool `json:"enabled"`
}

// --- Translation module ---

type TranslationModuleConfig struct {
	Input  *SAPDocumentTranslationInput  `json:"input,omitempty"`
	Output *SAPDocumentTranslationOutput `json:"output,omitempty"`
}

type SAPDocumentTranslationInput struct {
	Type   string                     `json:"type"`
	Config TranslationInputWireConfig `json:"config"`
}

type TranslationInputWireConfig struct {
	SourceLanguage string `json:"source_language,omitempty"`
	TargetLanguage string `json:"target_language"`
}

type SAPDocumentTranslationOutput struct {
	Type   string                      `json:"type"`
	Config TranslationOutputWireConfig `json:"config"`
}

type TranslationOutputWireConfig struct {
	SourceLanguage string `json:"source_language,omitempty"`
	TargetLanguage string `json:"target_language"`
}
