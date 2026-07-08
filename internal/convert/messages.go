// Package convert transforms between Google genai types and OpenAI-compatible wire types.
package convert

import (
	"encoding/json"
	"strings"

	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

// Messages converts a system instruction and content list into OpenAI chat messages.
func Messages(systemInstruction *genai.Content, contents []*genai.Content) []oai.ChatMessage {
	var messages []oai.ChatMessage

	if systemInstruction != nil {
		messages = append(messages, systemInstruction2Message(systemInstruction))
	}

	for _, content := range contents {
		if content == nil {
			continue
		}

		messages = append(messages, content2Messages(content)...)
	}

	return messages
}

func systemInstruction2Message(content *genai.Content) oai.ChatMessage {
	var texts []string

	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}

	return oai.ChatMessage{
		Role:    "system",
		Content: strPtr(strings.Join(texts, "\n")),
	}
}

// content2Messages transforms a single genai.Content into one or more messages.
// A Content with FunctionResponse parts produces separate tool messages.
func content2Messages(content *genai.Content) []oai.ChatMessage {
	role := MapRole(content.Role)

	var textParts []string

	var toolCalls []oai.ToolCall

	var messages []oai.ChatMessage

	for _, part := range content.Parts {
		switch {
		case part.FunctionResponse != nil:
			messages = append(messages, functionResponse2Message(part.FunctionResponse))

		case part.FunctionCall != nil:
			toolCalls = append(toolCalls, functionCall2ToolCall(part.FunctionCall))

		case part.Text != "" && !part.Thought:
			textParts = append(textParts, part.Text)
		}
	}

	if len(textParts) > 0 || len(toolCalls) > 0 {
		msg := oai.ChatMessage{
			Role:      role,
			Content:   strPtr(strings.Join(textParts, "")),
			ToolCalls: toolCalls,
		}

		messages = append([]oai.ChatMessage{msg}, messages...)
	}

	return messages
}

func functionCall2ToolCall(fc *genai.FunctionCall) oai.ToolCall {
	argsJSON, _ := json.Marshal(fc.Args)

	return oai.ToolCall{
		ID:   fc.ID,
		Type: toolTypeFunction,
		Function: oai.FunctionCall{
			Name:      fc.Name,
			Arguments: string(argsJSON),
		},
	}
}

func functionResponse2Message(fr *genai.FunctionResponse) oai.ChatMessage {
	responseJSON, _ := json.Marshal(fr.Response)

	return oai.ChatMessage{
		Role:       "tool",
		Content:    strPtr(string(responseJSON)),
		ToolCallID: fr.ID,
	}
}

// MapRole converts a genai role to an OpenAI role.
func MapRole(role string) string {
	switch role {
	case "model":
		return "assistant"
	case "user":
		return "user"
	default:
		return role
	}
}

func strPtr(s string) *string { return &s }
