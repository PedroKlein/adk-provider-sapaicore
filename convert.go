package sapaicore

import (
	"encoding/json"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

// convertRequest transforms an ADK LLMRequest into an OpenAI chat request.
func convertRequest(req *model.LLMRequest, deploymentModel string) chatRequest {
	cr := chatRequest{
		Model: deploymentModel,
	}

	if req.Config != nil {
		cr.Temperature = req.Config.Temperature
		cr.TopP = req.Config.TopP
		cr.Stop = req.Config.StopSequences

		if req.Config.MaxOutputTokens > 0 {
			maxTokens := req.Config.MaxOutputTokens
			cr.MaxTokens = &maxTokens
		}

		if req.Config.SystemInstruction != nil {
			cr.Messages = append(cr.Messages, convertSystemInstruction(req.Config.SystemInstruction))
		}

		cr.Tools = convertTools(req.Config.Tools)
	}

	cr.Messages = append(cr.Messages, convertContents(req.Contents)...)

	return cr
}

// convertSystemInstruction extracts text from a system instruction Content.
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

// convertContents transforms genai.Content messages into OpenAI chat messages.
func convertContents(contents []*genai.Content) []chatMessage {
	var messages []chatMessage

	for _, content := range contents {
		if content == nil {
			continue
		}

		msgs := convertContent(content)
		messages = append(messages, msgs...)
	}

	return messages
}

// convertContent transforms a single genai.Content into one or more OpenAI messages.
// A single Content may produce multiple messages when it contains function responses.
func convertContent(content *genai.Content) []chatMessage {
	role := mapRole(content.Role)

	var textParts []string

	var toolCalls []toolCall

	var messages []chatMessage

	for _, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			// Function responses become separate tool messages.
			messages = append(messages, convertFunctionResponse(part.FunctionResponse))

		case part.FunctionCall != nil:
			toolCalls = append(toolCalls, convertFunctionCall(part.FunctionCall))

		case part.Text != "" && !part.Thought:
			textParts = append(textParts, part.Text)
		}
	}

	// Emit the main message if it has content or tool calls.
	if len(textParts) > 0 || len(toolCalls) > 0 {
		msg := chatMessage{
			Role:      role,
			Content:   strings.Join(textParts, ""),
			ToolCalls: toolCalls,
		}
		// Prepend main message before any tool responses.
		messages = append([]chatMessage{msg}, messages...)
	}

	return messages
}

// convertFunctionCall transforms a genai FunctionCall into an OpenAI tool call.
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

// convertFunctionResponse transforms a genai FunctionResponse into an OpenAI tool message.
func convertFunctionResponse(fr *genai.FunctionResponse) chatMessage {
	responseJSON, _ := json.Marshal(fr.Response)

	return chatMessage{
		Role:       "tool",
		Content:    string(responseJSON),
		ToolCallID: fr.ID,
		Name:       fr.Name,
	}
}

// convertTools transforms genai Tools into OpenAI tool definitions.
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

// convertFunctionDeclaration transforms a genai FunctionDeclaration into an OpenAI tool def.
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

// convertResponse transforms an OpenAI chat response into an ADK LLMResponse.
func convertResponse(resp *chatResponse) *model.LLMResponse {
	if resp.Error != nil {
		return &model.LLMResponse{
			ErrorCode:    resp.Error.Code,
			ErrorMessage: resp.Error.Message,
		}
	}

	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{}, Role: "model"},
		}
	}

	choice := resp.Choices[0]

	return &model.LLMResponse{
		Content:       convertResponseMessage(choice.Message),
		FinishReason:  mapFinishReason(choice.FinishReason),
		UsageMetadata: convertUsage(resp.Usage),
		ModelVersion:  resp.Model,
	}
}

// convertResponseMessage transforms an OpenAI response message into genai Content.
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

// convertToolCallToPart transforms an OpenAI tool call into a genai FunctionCall part.
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

// convertUsage transforms OpenAI usage into genai usage metadata.
func convertUsage(usage *chatUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage == nil {
		return nil
	}

	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     usage.PromptTokens,
		CandidatesTokenCount: usage.CompletionTokens,
	}
}

// mapRole converts genai role names to OpenAI role names.
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

// mapFinishReason converts OpenAI finish reasons to genai FinishReason.
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
