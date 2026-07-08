package openai

// FoundationRequest is the wire format for SAP AI Core foundation-models mode.
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

// StreamOptions controls stream behavior for foundation-models mode.
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
