package orchestration

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/convert"
	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

func (m *Model) buildRequestBody(req *model.LLMRequest, doStream bool) ([]byte, error) {
	params, err := m.extractParams(req, doStream)
	if err != nil {
		return nil, err
	}

	modelParams := buildModelParams(params, doStream)

	template := params.Messages
	if len(template) == 0 {
		defaultContent := "You are a helpful assistant."
		template = []oai.ChatMessage{{Role: "system", Content: &defaultContent}}
	}

	if m.PromptCaching {
		template = annotateCacheControl(template, m.CacheTTL)
		params.Tools = annotateToolsCacheControl(params.Tools, m.CacheTTL)
	}

	moduleConfig := oai.ModuleConfigs{
		PromptTemplating: oai.PromptTemplatingModule{
			Prompt: oai.PromptConfig{
				Template:       template,
				Tools:          params.Tools,
				ResponseFormat: params.ResponseFormat,
			},
			Model: oai.ModelDef{
				Name:       params.ModelName,
				Version:    "latest",
				Params:     modelParams,
				Timeout:    params.Timeout,
				MaxRetries: params.MaxRetries,
			},
		},
		Filtering:   m.Filtering,
		Masking:     m.Masking,
		Translation: m.Translation,
	}

	modulesJSON, err := m.marshalModules(moduleConfig, params, doStream)
	if err != nil {
		return nil, err
	}

	orchReq := oai.OrchestrationRequest{
		Config: oai.OrchestrationConfig{
			Modules: modulesJSON,
		},
	}

	if doStream {
		streamCfg := &oai.StreamConfig{Enabled: true}
		if m.StreamOptions != nil {
			streamCfg.ChunkSize = m.StreamOptions.ChunkSize
			streamCfg.Delimiters = m.StreamOptions.Delimiters
		}

		orchReq.Config.Stream = streamCfg
	}

	body, err := json.Marshal(orchReq)
	if err != nil {
		return nil, fmt.Errorf("marshaling orchestration request: %w", err)
	}

	return body, nil
}

func (m *Model) marshalModules(primary oai.ModuleConfigs, params oai.RequestParams, doStream bool) (json.RawMessage, error) {
	if len(m.FallbackModels) == 0 {
		data, err := json.Marshal(primary)
		if err != nil {
			return nil, fmt.Errorf("marshaling modules: %w", err)
		}

		return data, nil
	}

	configs := make([]oai.ModuleConfigs, 0, 1+len(m.FallbackModels))
	configs = append(configs, primary)

	for _, fallbackModel := range m.FallbackModels {
		fallbackParams := buildModelParams(params, doStream)

		fb := oai.ModuleConfigs{
			PromptTemplating: oai.PromptTemplatingModule{
				Prompt: primary.PromptTemplating.Prompt,
				Model: oai.ModelDef{
					Name:       fallbackModel,
					Version:    "latest",
					Params:     fallbackParams,
					Timeout:    params.Timeout,
					MaxRetries: params.MaxRetries,
				},
			},
			Filtering:   primary.Filtering,
			Masking:     primary.Masking,
			Translation: primary.Translation,
		}
		configs = append(configs, fb)
	}

	data, err := json.Marshal(configs)
	if err != nil {
		return nil, fmt.Errorf("marshaling fallback modules: %w", err)
	}

	return data, nil
}

func annotateCacheControl(messages []oai.ChatMessage, ttl string) []oai.ChatMessage {
	result := make([]oai.ChatMessage, len(messages))
	copy(result, messages)

	for i, msg := range slices.Backward(result) {
		if msg.Role == "system" {
			// Convert content from plain string to content blocks with cache_control.
			var text string

			switch c := msg.Content.(type) {
			case *string:
				if c != nil {
					text = *c
				}
			case string:
				text = c
			}

			// Construct a new message to avoid aliasing the original.
			result[i] = oai.ChatMessage{
				Role: msg.Role,
				Content: []oai.TextContentBlock{{
					Type:         "text",
					Text:         text,
					CacheControl: &oai.CacheControl{Type: "ephemeral", TTL: ttl},
				}},
				ToolCalls:  msg.ToolCalls,
				ToolCallID: msg.ToolCallID,
			}

			break
		}
	}

	return result
}

func annotateToolsCacheControl(tools []oai.ToolDef, ttl string) []oai.ToolDef {
	if len(tools) == 0 {
		return tools
	}

	result := make([]oai.ToolDef, len(tools))
	copy(result, tools)

	result[len(result)-1].CacheControl = &oai.CacheControl{Type: "ephemeral", TTL: ttl}

	return result
}

func buildModelParams(params oai.RequestParams, doStream bool) map[string]any {
	modelParams := make(map[string]any)

	if params.Temperature != nil {
		modelParams["temperature"] = *params.Temperature
	}

	if params.MaxTokens > 0 {
		modelParams["max_tokens"] = params.MaxTokens
	}

	if params.TopP != nil {
		modelParams["top_p"] = *params.TopP
	}

	if params.TopK != nil {
		modelParams["top_k"] = *params.TopK
	}

	if params.Seed != nil {
		modelParams["seed"] = *params.Seed
	}

	if len(params.Stop) > 0 {
		modelParams["stop"] = params.Stop
	}

	if params.FrequencyPenalty != nil {
		modelParams["frequency_penalty"] = *params.FrequencyPenalty
	}

	if params.PresencePenalty != nil {
		modelParams["presence_penalty"] = *params.PresencePenalty
	}

	if params.ResponseLogprobs {
		modelParams["logprobs"] = true
	}

	if params.Logprobs != nil {
		modelParams["top_logprobs"] = *params.Logprobs
	}

	if params.ToolChoice != nil {
		modelParams["tool_choice"] = params.ToolChoice
	}

	maps.Copy(modelParams, params.ExtraParams)

	if doStream {
		modelParams["stream_options"] = map[string]any{"include_usage": true}
	}

	return modelParams
}

func (m *Model) extractParams(req *model.LLMRequest, doStream bool) (oai.RequestParams, error) {
	params, err := convert.ExtractParams(req.Config, req.Contents, req.Model, m.ModelName, m.ExtraParams)
	if err != nil {
		return oai.RequestParams{}, fmt.Errorf("converting request content: %w", err)
	}

	params.Stream = doStream
	params.Timeout = m.Timeout
	params.MaxRetries = m.MaxRetries

	return params, nil
}
