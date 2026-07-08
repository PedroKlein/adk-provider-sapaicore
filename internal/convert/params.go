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
		toolConfig        *genai.ToolConfig
		temperature       *float32
		maxTokens         int32
		topP              *float32
		topK              *float32
		seed              *int32
		stop              []string
		frequencyPenalty  *float32
		presencePenalty   *float32
		responseLogprobs  bool
		logprobs          *int32
		responseMIME      string
		responseSchema    *genai.Schema
	)

	if req != nil {
		systemInstruction = req.SystemInstruction
		tools = req.Tools
		toolConfig = req.ToolConfig
		temperature = req.Temperature
		maxTokens = req.MaxOutputTokens
		topP = req.TopP
		topK = req.TopK
		seed = req.Seed
		stop = req.StopSequences
		frequencyPenalty = req.FrequencyPenalty
		presencePenalty = req.PresencePenalty
		responseLogprobs = req.ResponseLogprobs
		logprobs = req.Logprobs
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
		ToolChoice:       ToolChoice(toolConfig),
		Temperature:      temperature,
		MaxTokens:        maxTokens,
		TopP:             topP,
		TopK:             topK,
		Seed:             seed,
		Stop:             stop,
		FrequencyPenalty: frequencyPenalty,
		PresencePenalty:  presencePenalty,
		ResponseLogprobs: responseLogprobs,
		Logprobs:         logprobs,
		ResponseFormat:   ResponseFormat(responseMIME, responseSchema),
		ExtraParams:      extraParams,
	}
}
