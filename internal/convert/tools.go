package convert

import (
	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

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

func functionDecl2ToolDef(decl *genai.FunctionDeclaration) oai.ToolDef {
	var params any

	if decl.Parameters != nil {
		params = Schema(decl.Parameters)
	}

	return oai.ToolDef{
		Type: "function",
		Function: oai.FunctionDef{
			Name:        decl.Name,
			Description: decl.Description,
			Parameters:  params,
		},
	}
}
