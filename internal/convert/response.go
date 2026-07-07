package convert

import (
	"encoding/json"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

// ChoiceToResponse converts a chat completion choice, usage, and model version
// into an ADK LLMResponse.
func ChoiceToResponse(choice oai.ChatChoice, usage *oai.ChatUsage, modelVersion string) *model.LLMResponse {
	resp := &model.LLMResponse{
		Content:       responseMessage2Content(choice.Message),
		FinishReason:  MapFinishReason(choice.FinishReason),
		UsageMetadata: Usage(usage),
		ModelVersion:  modelVersion,
	}

	if choice.Message.Refusal != "" {
		resp.ErrorCode = "refusal"
		resp.ErrorMessage = choice.Message.Refusal
	}

	return resp
}

func responseMessage2Content(msg oai.ChatMessage) *genai.Content {
	var parts []*genai.Part

	if msg.Content != nil && *msg.Content != "" {
		parts = append(parts, &genai.Part{Text: *msg.Content})
	}

	for _, tc := range msg.ToolCalls {
		parts = append(parts, ToolCallToPart(tc))
	}

	if len(parts) == 0 {
		parts = []*genai.Part{}
	}

	return &genai.Content{
		Parts: parts,
		Role:  "model",
	}
}

// ToolCallToPart converts a single tool call into a genai FunctionCall part.
func ToolCallToPart(tc oai.ToolCall) *genai.Part {
	var args map[string]any

	_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)

	return &genai.Part{
		FunctionCall: &genai.FunctionCall{
			ID:   tc.ID,
			Name: tc.Function.Name,
			Args: args,
		},
	}
}

// Usage converts OpenAI usage info to genai usage metadata.
func Usage(usage *oai.ChatUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     usage.PromptTokens,
		CandidatesTokenCount: usage.CompletionTokens,
	}
}

// MapFinishReason maps an OpenAI finish reason string to a genai FinishReason.
func MapFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "tool_calls":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonOther
	}
}
