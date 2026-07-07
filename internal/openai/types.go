// Package openai defines wire types for the OpenAI-compatible chat completions
// format used by both SAP AI Core API modes.
package openai

// ChatMessage represents a single message in the OpenAI chat format.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	Refusal    string     `json:"refusal,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall represents a tool invocation in a chat completion response or delta.
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the function name and serialized arguments.
type FunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ToolDef defines a tool available for the model to call.
type ToolDef struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef describes a callable function's name, description, and parameter schema.
type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ChatChoice represents a single completion choice in a non-streaming response.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatUsage reports token consumption for a request.
type ChatUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// ChatError represents an API error embedded in a response body.
type ChatError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// ChunkChoice represents a single choice in a streaming chunk.
type ChunkChoice struct {
	Index        int       `json:"index"`
	Delta        ChatDelta `json:"delta"`
	FinishReason string    `json:"finish_reason,omitempty"`
}

// ChatDelta is the incremental content within a streaming chunk choice.
type ChatDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// --- Foundation-models mode ---

// FoundationRequest is the request body for foundation-models mode (direct OpenAI format).
type FoundationRequest struct {
	Model            string          `json:"model"`
	Messages         []ChatMessage   `json:"messages"`
	Tools            []ToolDef       `json:"tools,omitempty"`
	Stream           bool            `json:"stream"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Temperature      *float32        `json:"temperature,omitempty"`
	MaxTokens        *int32          `json:"max_tokens,omitempty"`
	TopP             *float32        `json:"top_p,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	FrequencyPenalty *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float32        `json:"presence_penalty,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
}

// StreamOptions configures streaming behavior.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// FoundationResponse is the non-streaming response from foundation-models mode.
type FoundationResponse struct {
	ID      string       `json:"id"`
	Choices []ChatChoice `json:"choices"`
	Usage   *ChatUsage   `json:"usage,omitempty"`
	Model   string       `json:"model"`
	Error   *ChatError   `json:"error,omitempty"`
}

// FoundationChunk is a single streaming chunk from foundation-models mode.
type FoundationChunk struct {
	ID      string        `json:"id"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *ChatUsage    `json:"usage,omitempty"`
	Model   string        `json:"model"`
}

// --- Orchestration mode ---

// OrchestrationRequest is the top-level request envelope for orchestration mode.
type OrchestrationRequest struct {
	Config OrchestrationConfig `json:"config"`
}

// OrchestrationConfig holds the orchestration configuration with modules and streaming.
type OrchestrationConfig struct {
	Stream  *StreamConfig `json:"stream,omitempty"`
	Modules ModuleConfigs `json:"modules"`
}

// StreamConfig enables or disables streaming in orchestration mode.
type StreamConfig struct {
	Enabled bool `json:"enabled"`
}

// ModuleConfigs holds the orchestration module configurations.
type ModuleConfigs struct {
	PromptTemplating PromptTemplatingModule `json:"prompt_templating"`
}

// PromptTemplatingModule configures the prompt and model for orchestration.
type PromptTemplatingModule struct {
	Prompt PromptConfig `json:"prompt"`
	Model  ModelDef     `json:"model"`
}

// PromptConfig defines the prompt template, tools, and response format.
type PromptConfig struct {
	Template       []ChatMessage   `json:"template"`
	Tools          []ToolDef       `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat specifies the desired output format for the model.
type ResponseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// JSONSchema defines a structured output schema for json_schema response format.
type JSONSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

// ModelDef identifies the model, version, and parameters for orchestration mode.
type ModelDef struct {
	Name       string         `json:"name"`
	Version    string         `json:"version,omitempty"`
	Params     map[string]any `json:"params,omitempty"`
	Timeout    int            `json:"timeout,omitempty"`
	MaxRetries int            `json:"max_retries,omitempty"`
}

// OrchestrationResponse is the non-streaming response from orchestration mode.
type OrchestrationResponse struct {
	RequestID   string              `json:"request_id"`
	FinalResult *FoundationResponse `json:"final_result"`
}

// OrchestrationChunk is a single streaming chunk from orchestration mode.
type OrchestrationChunk struct {
	RequestID   string           `json:"request_id"`
	FinalResult *FoundationChunk `json:"final_result"`
}

// RequestParams holds all generation parameters extracted from the ADK request config.
// Used by strategy implementations to build mode-specific request bodies.
type RequestParams struct {
	ModelName        string
	Messages         []ChatMessage
	Tools            []ToolDef
	Temperature      *float32
	MaxTokens        int32
	TopP             *float32
	Stop             []string
	FrequencyPenalty *float32
	PresencePenalty  *float32
	ResponseFormat   *ResponseFormat
	Stream           bool
	ExtraParams      map[string]any
	Timeout          int
	MaxRetries       int
}
