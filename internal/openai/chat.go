// Package openai defines wire types for the OpenAI-compatible chat completions
// format used by both SAP AI Core API modes.
package openai

// ChatMessage represents a message in the chat completions format.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	Refusal    string     `json:"refusal,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ContentBlock is implemented by all types that can appear in a multimodal
// message content array. Prevents arbitrary types from being added to the slice.
type ContentBlock interface {
	contentBlock()
}

// TextContentBlock represents a text segment in a multimodal content array.
type TextContentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (TextContentBlock) contentBlock() {}

// ImageURLContentBlock represents an image in a multimodal content array.
type ImageURLContentBlock struct {
	Type         string        `json:"type"`
	ImageURL     ImageURL      `json:"image_url"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (ImageURLContentBlock) contentBlock() {}

// ImageURL holds the URL and optional detail level for an image block.
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// FileContentBlock represents a file attachment in a multimodal content array.
type FileContentBlock struct {
	Type         string        `json:"type"`
	File         FileContent   `json:"file"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

func (FileContentBlock) contentBlock() {}

// FileContent holds the data URI or URL and optional filename for a file block.
type FileContent struct {
	FileData string `json:"file_data"`
	Filename string `json:"filename,omitempty"`
}

// ToolCall represents a function call requested by the model.
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the name and serialized arguments of a tool call.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ToolDef defines a tool available to the model.
type ToolDef struct {
	Type         string        `json:"type"`
	Function     FunctionDef   `json:"function"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// FunctionDef describes a function's name, description, and parameter schema.
type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ChatChoice represents a single completion choice in a non-streaming response.
type ChatChoice struct {
	Index        int           `json:"index"`
	Message      ChatMessage   `json:"message"`
	FinishReason string        `json:"finish_reason"`
	Logprobs     *ChatLogprobs `json:"logprobs,omitempty"`
}

// ChatLogprobs contains per-token log probability information.
type ChatLogprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
}

// TokenLogprob holds the log probability for a single generated token.
type TokenLogprob struct {
	Token       string            `json:"token"`
	Logprob     float64           `json:"logprob"`
	TokenID     int32             `json:"token_id,omitempty"`
	TopLogprobs []TopTokenLogprob `json:"top_logprobs,omitempty"`
}

// TopTokenLogprob holds an alternative token candidate with its log probability.
type TopTokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	TokenID int32   `json:"token_id,omitempty"`
}

// ChatUsage reports token usage for a completion request.
type ChatUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// ChatError represents a structured error from the API.
type ChatError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// ChunkChoice represents a single choice in a streaming chunk.
type ChunkChoice struct {
	Index        int           `json:"index"`
	Delta        ChatDelta     `json:"delta"`
	FinishReason string        `json:"finish_reason,omitempty"`
	Logprobs     *ChatLogprobs `json:"logprobs,omitempty"`
}

// ChatDelta holds the incremental content in a streaming chunk.
type ChatDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// CacheControl represents a cache control annotation for prompt caching.
type CacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

// ResponseFormat specifies the output format constraint for the model.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema defines a JSON Schema constraint for structured output.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

// RequestParams holds the per-request LLM parameters shared between
// foundation and orchestration mode request builders.
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
}
