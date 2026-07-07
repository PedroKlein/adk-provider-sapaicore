package convert

import (
	"strings"

	"google.golang.org/genai"

	oai "github.com/PedroKlein/go-adk-sap-ai-core/internal/openai"
)

// Schema transforms a genai.Schema into an OpenAI-compatible JSON Schema map.
// genai uses uppercase types ("STRING", "OBJECT") while OpenAI/SAP expects lowercase.
// Object types automatically include "additionalProperties": false for strict schema compliance.
func Schema(s *genai.Schema) map[string]any {
	result := make(map[string]any)

	if s.Type != "" {
		result["type"] = strings.ToLower(string(s.Type))
	}

	if s.Description != "" {
		result["description"] = s.Description
	}

	if len(s.Enum) > 0 {
		result["enum"] = s.Enum
	}

	if len(s.Required) > 0 {
		result["required"] = s.Required
	}

	if len(s.Properties) > 0 {
		props := make(map[string]any, len(s.Properties))

		for name, prop := range s.Properties {
			props[name] = Schema(prop)
		}

		result["properties"] = props
		result["additionalProperties"] = false
	}

	if s.Items != nil {
		result["items"] = Schema(s.Items)
	}

	return result
}

// ResponseFormat maps genai's ResponseMIMEType/ResponseSchema to the
// OpenAI-compatible response_format field.
func ResponseFormat(mimeType string, schema *genai.Schema) *oai.ResponseFormat {
	if mimeType != "application/json" {
		return nil
	}

	if schema == nil {
		return &oai.ResponseFormat{Type: "json_object"}
	}

	strict := true

	return &oai.ResponseFormat{
		Type: "json_schema",
		JSONSchema: &oai.JSONSchema{
			Name:   "response",
			Schema: Schema(schema),
			Strict: &strict,
		},
	}
}
