package convert

import (
	"strings"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

const sanitizeReplacement = "__"

// ToolNameMapping is a reverse lookup (sanitized → original) for function names
// that were rewritten to satisfy the orchestration API's ^[a-zA-Z0-9-_]+$ constraint.
type ToolNameMapping map[string]string

// Restore returns the original function name for a sanitized name.
// Returns the input unchanged if the name is not in the mapping or the mapping is nil.
func (m ToolNameMapping) Restore(name string) string {
	if m == nil {
		return name
	}

	if original, ok := m[name]; ok {
		return original
	}

	return name
}

// Sanitize rewrites a name using the forward direction (original → sanitized).
// Returns the input unchanged if it contains no dots.
func (m ToolNameMapping) Sanitize(name string) string {
	if !strings.Contains(name, ".") {
		return name
	}

	return strings.ReplaceAll(name, ".", sanitizeReplacement)
}

// SanitizeToolNames replaces dots with "__" in function names that violate
// the orchestration API's ^[a-zA-Z0-9-_]+$ constraint.
// Returns a copy of the tools slice with sanitized names, plus a reverse mapping.
// The input slice is not modified.
func SanitizeToolNames(tools []oai.ToolDef) ([]oai.ToolDef, ToolNameMapping) {
	var mapping ToolNameMapping

	result := make([]oai.ToolDef, len(tools))
	copy(result, tools)

	for i, t := range result {
		if !strings.Contains(t.Function.Name, ".") {
			continue
		}

		if mapping == nil {
			mapping = make(ToolNameMapping)
		}

		sanitized := strings.ReplaceAll(t.Function.Name, ".", sanitizeReplacement)
		mapping[sanitized] = t.Function.Name
		result[i].Function.Name = sanitized
	}

	return result, mapping
}

// SanitizeMessageToolNames rewrites function names in tool_calls within messages
// to match the sanitized tool definitions. This ensures multi-turn conversations
// (where prior assistant messages reference dotted tool names) remain consistent
// with the declared tools.
func SanitizeMessageToolNames(messages []oai.ChatMessage, mapping ToolNameMapping) []oai.ChatMessage {
	if mapping == nil {
		return messages
	}

	for i, msg := range messages {
		if len(msg.ToolCalls) == 0 {
			continue
		}

		for j := range messages[i].ToolCalls {
			name := messages[i].ToolCalls[j].Function.Name
			messages[i].ToolCalls[j].Function.Name = mapping.Sanitize(name)
		}
	}

	return messages
}
