// Package convert transforms between Google genai types and OpenAI-compatible wire types.
package convert

import (
	"encoding/json"
	"fmt"
	"strings"

	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

func Messages(systemInstruction *genai.Content, contents []*genai.Content) ([]oai.ChatMessage, error) {
	var messages []oai.ChatMessage

	if systemInstruction != nil {
		messages = append(messages, systemInstruction2Message(systemInstruction))
	}

	for _, content := range contents {
		if content == nil {
			continue
		}

		msgs, err := content2Messages(content)
		if err != nil {
			return nil, err
		}

		messages = append(messages, msgs...)
	}

	correlateEmptyToolCallIDs(messages)

	return messages, nil
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

// content2Messages handles three branching paths:
//   - Multimodal (InlineData/FileData present) → content array + separate tool messages
//   - Tool-calling (FunctionCall/FunctionResponse) → assistant message with tool_calls + tool messages
//   - Text-only → plain string content
func content2Messages(content *genai.Content) ([]oai.ChatMessage, error) {
	role := MapRole(content.Role)

	blocks, err := ContentBlocks(content.Parts)
	if err != nil {
		return nil, err
	}

	if blocks != nil {
		return multimodalMessages(role, blocks, content.Parts)
	}

	return textAndToolMessages(role, content.Parts)
}

func multimodalMessages(role string, blocks []oai.ContentBlock, parts []*genai.Part) ([]oai.ChatMessage, error) {
	var messages []oai.ChatMessage

	var toolCalls []oai.ToolCall

	for _, part := range parts {
		if part.FunctionCall != nil {
			tc, err := functionCall2ToolCall(part.FunctionCall)
			if err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, tc)
		}
	}

	messages = append(messages, oai.ChatMessage{
		Role:      role,
		Content:   blocks,
		ToolCalls: toolCalls,
	})

	for _, part := range parts {
		if part.FunctionResponse != nil {
			msg, err := functionResponse2Message(part.FunctionResponse)
			if err != nil {
				return nil, err
			}

			messages = append(messages, msg)
		}
	}

	return messages, nil
}

func textAndToolMessages(role string, parts []*genai.Part) ([]oai.ChatMessage, error) {
	var textParts []string

	var toolCalls []oai.ToolCall

	var messages []oai.ChatMessage

	for _, part := range parts {
		switch {
		case part.FunctionResponse != nil:
			msg, err := functionResponse2Message(part.FunctionResponse)
			if err != nil {
				return nil, err
			}

			messages = append(messages, msg)

		case part.FunctionCall != nil:
			tc, err := functionCall2ToolCall(part.FunctionCall)
			if err != nil {
				return nil, err
			}

			toolCalls = append(toolCalls, tc)

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

	return messages, nil
}

func functionCall2ToolCall(fc *genai.FunctionCall) (oai.ToolCall, error) {
	argsJSON, err := json.Marshal(fc.Args)
	if err != nil {
		return oai.ToolCall{}, fmt.Errorf("marshaling function call args for %q: %w", fc.Name, err)
	}

	return oai.ToolCall{
		ID:   fc.ID,
		Type: toolTypeFunction,
		Function: oai.FunctionCall{
			Name:      fc.Name,
			Arguments: string(argsJSON),
		},
	}, nil
}

func functionResponse2Message(fr *genai.FunctionResponse) (oai.ChatMessage, error) {
	responseJSON, err := json.Marshal(fr.Response)
	if err != nil {
		return oai.ChatMessage{}, fmt.Errorf("marshaling function response for %q: %w", fr.Name, err)
	}

	return oai.ChatMessage{
		Role:       "tool",
		Content:    strPtr(string(responseJSON)),
		ToolCallID: fr.ID,
	}, nil
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

// correlateEmptyToolCallIDs assigns synthetic IDs when tool_calls or tool
// result messages have empty IDs (stripped by ADK's RemoveClientFunctionCallID).
func correlateEmptyToolCallIDs(messages []oai.ChatMessage) {
	seq := 0

	for i := range messages {
		if messages[i].Role != "assistant" || len(messages[i].ToolCalls) == 0 {
			continue
		}

		needsIDs := false

		for _, tc := range messages[i].ToolCalls {
			if tc.ID == "" {
				needsIDs = true
				break
			}
		}

		if !needsIDs {
			continue
		}

		for j := range messages[i].ToolCalls {
			if messages[i].ToolCalls[j].ID == "" {
				messages[i].ToolCalls[j].ID = fmt.Sprintf("call_%d", seq)
				seq++
			}
		}

		toolIdx := 0
		for k := i + 1; k < len(messages) && toolIdx < len(messages[i].ToolCalls); k++ {
			if messages[k].Role == "assistant" {
				break
			}

			if messages[k].Role != "tool" {
				continue
			}

			if messages[k].ToolCallID == "" {
				messages[k].ToolCallID = messages[i].ToolCalls[toolIdx].ID
			}

			toolIdx++
		}
	}
}
