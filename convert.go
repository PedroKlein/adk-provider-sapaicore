package sapaicore

import (
	"encoding/json"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func convertMessages(systemInstruction *genai.Content, contents []*genai.Content) []chatMessage {
	var messages []chatMessage

	if systemInstruction != nil {
		messages = append(messages, convertSystemInstruction(systemInstruction))
	}

	for _, content := range contents {
		if content == nil {
			continue
		}

		messages = append(messages, convertContent(content)...)
	}

	return messages
}

func convertSystemInstruction(content *genai.Content) chatMessage {
	var texts []string

	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	return chatMessage{
		Role:    "system",
		Content: strings.Join(texts, "\n"),
	}
}

// convertContent transforms a single genai.Content into one or more messages.
// A Content with FunctionResponse parts produces separate tool messages.
func convertContent(content *genai.Content) []chatMessage {
	role := mapRole(content.Role)

	var textParts []string

	var toolCalls []toolCall

	var messages []chatMessage

	for _, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			messages = append(messages, convertFunctionResponse(part.FunctionResponse))

		case part.FunctionCall != nil:
			toolCalls = append(toolCalls, convertFunctionCall(part.FunctionCall))

		case part.Text != "" && !part.Thought:
			textParts = append(textParts, part.Text)
		}
	}

	if len(textParts) > 0 || len(toolCalls) > 0 {
		msg := chatMessage{
			Role:      role,
			Content:   strings.Join(textParts, ""),
			ToolCalls: toolCalls,
		}

		messages = append([]chatMessage{msg}, messages...)
	}

	return messages
}

func convertFunctionCall(fc *genai.FunctionCall) toolCall {
	argsJSON, _ := json.Marshal(fc.Args)

	return toolCall{
		ID:   fc.ID,
		Type: "function",
		Function: functionCall{
			Name:      fc.Name,
			Arguments: string(argsJSON),
		},
	}
}

func convertFunctionResponse(fr *genai.FunctionResponse) chatMessage {
	responseJSON, _ := json.Marshal(fr.Response)

	return chatMessage{
		Role:       "tool",
		Content:    string(responseJSON),
		ToolCallID: fr.ID,
		Name:       fr.Name,
	}
}

func convertTools(tools []*genai.Tool) []toolDef {
	var defs []toolDef

	for _, tool := range tools {
		if tool == nil || tool.FunctionDeclarations == nil {
			continue
		}

		for _, decl := range tool.FunctionDeclarations {
			defs = append(defs, convertFunctionDeclaration(decl))
		}
	}

	return defs
}

func convertFunctionDeclaration(decl *genai.FunctionDeclaration) toolDef {
	var params any

	if decl.Parameters != nil {
		params = decl.Parameters
	}

	return toolDef{
		Type: "function",
		Function: functionDef{
			Name:        decl.Name,
			Description: decl.Description,
			Parameters:  params,
		},
	}
}

func convertChoiceToResponse(choice chatChoice, usage *chatUsage, modelVersion string) *model.LLMResponse {
	return &model.LLMResponse{
		Content:       convertResponseMessage(choice.Message),
		FinishReason:  mapFinishReason(choice.FinishReason),
		UsageMetadata: convertUsage(usage),
		ModelVersion:  modelVersion,
	}
}

func convertResponseMessage(msg chatMessage) *genai.Content {
	var parts []*genai.Part

	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}

	for _, tc := range msg.ToolCalls {
		parts = append(parts, convertToolCallToPart(tc))
	}

	if len(parts) == 0 {
		parts = []*genai.Part{}
	}

	return &genai.Content{
		Parts: parts,
		Role:  "model",
	}
}

func convertToolCallToPart(tc toolCall) *genai.Part {
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

func convertUsage(usage *chatUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     usage.PromptTokens,
		CandidatesTokenCount: usage.CompletionTokens,
	}
}

func mapRole(role string) string {
	switch role {
	case "model":
		return "assistant"
	case "user":
		return "user"
	default:
		return role
	}
}

func mapFinishReason(reason string) genai.FinishReason {
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
