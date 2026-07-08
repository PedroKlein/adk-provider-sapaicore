package convert

import (
	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

// ExtractParams maps ADK request config fields to the shared RequestParams
// used by both foundation and orchestration mode request builders.
func ExtractParams(req *genai.GenerateContentConfig, contents []*genai.Content, modelName, fallbackModel string, extraParams map[string]any) oai.RequestParams {
	var (
		systemInstruction *genai.Content
		tools             []*genai.Tool
		temperature       *float32
		maxTokens         int32
		topP              *float32
		stop              []string
		frequencyPenalty  *float32
		presencePenalty   *float32
		responseMIME      string
		responseSchema    *genai.Schema
	)

	if req != nil {
		systemInstruction = req.SystemInstruction
		tools = req.Tools
		temperature = req.Temperature
		maxTokens = req.MaxOutputTokens
		topP = req.TopP
		stop = req.StopSequences
		frequencyPenalty = req.FrequencyPenalty
		presencePenalty = req.PresencePenalty
		responseMIME = req.ResponseMIMEType
		responseSchema = req.ResponseSchema
	}

	if modelName == "" {
		modelName = fallbackModel
	}

	return oai.RequestParams{
		ModelName:        modelName,
		Messages:         Messages(systemInstruction, contents),
		Tools:            Tools(tools),
		Temperature:      temperature,
		MaxTokens:        maxTokens,
		TopP:             topP,
		Stop:             stop,
		FrequencyPenalty: frequencyPenalty,
		PresencePenalty:  presencePenalty,
		ResponseFormat:   ResponseFormat(responseMIME, responseSchema),
		ExtraParams:      extraParams,
	}
}
