package sapaicore

// Request/response types for both API modes. All types are unexported.

// --- Shared types (OpenAI chat completions format) ---

type chatMessage struct {
	Role       string     `json:"role"`
	Content    *string    `json:"content"`
	Refusal    string     `json:"refusal,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type toolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function functionCall `json:"function"`
}

type functionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type toolDef struct {
	Type     string      `json:"type"`
	Function functionDef `json:"function"`
}

type functionDef struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

type chatError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

type chunkChoice struct {
	Index        int       `json:"index"`
	Delta        chatDelta `json:"delta"`
	FinishReason string    `json:"finish_reason,omitempty"`
}

type chatDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

// --- Foundation-models mode ---

type foundationRequest struct {
	Model            string          `json:"model"`
	Messages         []chatMessage   `json:"messages"`
	Tools            []toolDef       `json:"tools,omitempty"`
	Stream           bool            `json:"stream"`
	StreamOptions    *streamOptions  `json:"stream_options,omitempty"`
	Temperature      *float32        `json:"temperature,omitempty"`
	MaxTokens        *int32          `json:"max_tokens,omitempty"`
	TopP             *float32        `json:"top_p,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	FrequencyPenalty *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float32        `json:"presence_penalty,omitempty"`
	ResponseFormat   *responseFormat `json:"response_format,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type foundationResponse struct {
	ID      string       `json:"id"`
	Choices []chatChoice `json:"choices"`
	Usage   *chatUsage   `json:"usage,omitempty"`
	Model   string       `json:"model"`
	Error   *chatError   `json:"error,omitempty"`
}

type foundationChunk struct {
	ID      string        `json:"id"`
	Choices []chunkChoice `json:"choices"`
	Usage   *chatUsage    `json:"usage,omitempty"`
	Model   string        `json:"model"`
}

// --- Orchestration mode ---

type orchestrationRequest struct {
	Config orchestrationConfig `json:"config"`
}

type orchestrationConfig struct {
	Stream  *streamConfig `json:"stream,omitempty"`
	Modules moduleConfigs `json:"modules"`
}

type streamConfig struct {
	Enabled bool `json:"enabled"`
}

type moduleConfigs struct {
	PromptTemplating promptTemplatingModule `json:"prompt_templating"`
}

type promptTemplatingModule struct {
	Prompt promptConfig `json:"prompt"`
	Model  modelDef     `json:"model"`
}

type promptConfig struct {
	Template       []chatMessage   `json:"template"`
	Tools          []toolDef       `json:"tools,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type responseFormat struct {
	Type       string      `json:"type"`
	JSONSchema *jsonSchema `json:"json_schema,omitempty"`
}

type jsonSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Schema      map[string]any `json:"schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type modelDef struct {
	Name       string         `json:"name"`
	Version    string         `json:"version,omitempty"`
	Params     map[string]any `json:"params,omitempty"`
	Timeout    int            `json:"timeout,omitempty"`
	MaxRetries int            `json:"max_retries,omitempty"`
}

type orchestrationResponse struct {
	RequestID   string              `json:"request_id"`
	FinalResult *foundationResponse `json:"final_result"`
}

type orchestrationChunk struct {
	RequestID   string           `json:"request_id"`
	FinalResult *foundationChunk `json:"final_result"`
}
