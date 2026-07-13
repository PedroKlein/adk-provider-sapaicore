package convert

import (
	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

const toolTypeFunction = "function"

// Tools converts genai Tool declarations into OpenAI-compatible tool definitions.
func Tools(tools []*genai.Tool) []oai.ToolDef {
	var defs []oai.ToolDef

	for _, tool := range tools {
		if tool == nil || tool.FunctionDeclarations == nil {
			continue
		}

		for _, decl := range tool.FunctionDeclarations {
			defs = append(defs, functionDecl2ToolDef(decl))
		}
	}

	return defs
}

// ToolChoice converts a genai ToolConfig into the OpenAI tool_choice format.
// Returns nil when no tool config is set (letting the model decide).
//
// Mapping:
//   - AUTO / unspecified → "auto"
//   - NONE → "none"
//   - ANY with no names → "required"
//   - ANY with one name → {"type":"function","function":{"name":"X"}}
//   - ANY with multiple names → "required" (OpenAI doesn't support a name list)
func ToolChoice(cfg *genai.ToolConfig) any {
	if cfg == nil || cfg.FunctionCallingConfig == nil {
		return nil
	}

	fcc := cfg.FunctionCallingConfig

	switch fcc.Mode {
	case genai.FunctionCallingConfigModeNone:
		return "none"
	case genai.FunctionCallingConfigModeAny:
		names := fcc.AllowedFunctionNames
		if len(names) == 1 {
			return map[string]any{
				"type":     toolTypeFunction,
				"function": map[string]any{"name": names[0]},
			}
		}

		return "required"
	case genai.FunctionCallingConfigModeAuto:
		return "auto"
	default:
		// MODE_UNSPECIFIED and any future modes: omit tool_choice
		// (server defaults to auto behavior).
		return nil
	}
}

func functionDecl2ToolDef(decl *genai.FunctionDeclaration) oai.ToolDef {
	var params any

	switch {
	case decl.Parameters != nil:
		params = Schema(decl.Parameters)
	case decl.ParametersJsonSchema != nil:
		// Already JSON-serializable (*jsonschema.Schema from ADK v2 functiontool).
		params = decl.ParametersJsonSchema
	}

	return oai.ToolDef{
		Type: toolTypeFunction,
		Function: oai.FunctionDef{
			Name:        decl.Name,
			Description: decl.Description,
			Parameters:  params,
		},
	}
}
