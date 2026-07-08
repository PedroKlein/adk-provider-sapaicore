package openai

import "encoding/json"

// OrchestrationRequest is the wire format for SAP AI Core orchestration mode.
type OrchestrationRequest struct {
	Config OrchestrationConfig `json:"config"`
}

// OrchestrationConfig holds the stream and module configuration.
type OrchestrationConfig struct {
	Stream  *StreamConfig   `json:"stream,omitempty"`
	Modules json.RawMessage `json:"modules"`
}

// StreamConfig controls streaming behavior for orchestration mode.
type StreamConfig struct {
	Enabled    bool     `json:"enabled"`
	ChunkSize  int      `json:"chunk_size,omitempty"`
	Delimiters []string `json:"delimiters,omitempty"`
}

// ModuleConfigs holds all orchestration module configurations.
type ModuleConfigs struct {
	PromptTemplating PromptTemplatingModule   `json:"prompt_templating"`
	Filtering        *FilteringModuleConfig   `json:"filtering,omitempty"`
	Masking          *MaskingModuleConfig     `json:"masking,omitempty"`
	Translation      *TranslationModuleConfig `json:"translation,omitempty"`
}

// PromptTemplatingModule configures the prompt template and model.
type PromptTemplatingModule struct {
	Prompt PromptConfig `json:"prompt"`
	Model  ModelDef     `json:"model"`
}

// PromptConfig holds the message template, tools, and response format.
type PromptConfig struct {
	Template       []ChatMessage   `json:"template"`
	Tools          []ToolDef       `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ModelDef identifies the model and its parameters for orchestration.
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
