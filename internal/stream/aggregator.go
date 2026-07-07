// Package stream handles Server-Sent Events parsing and incremental aggregation
// of streaming chat completion responses.
package stream

import (
	"encoding/json"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/go-adk-sap-ai-core/internal/convert"
	oai "github.com/PedroKlein/go-adk-sap-ai-core/internal/openai"
)

// Mode distinguishes how chunks are wrapped on the wire.
type Mode int

const (
	_ Mode = iota
	ModeOrchestration
	ModeFoundation
)

// Aggregator accumulates streaming chunks and produces partial + final LLMResponses.
type Aggregator struct {
	textBuf   strings.Builder
	toolCalls []oai.ToolCall
	usage     *oai.ChatUsage
	modelVer  string
	finishRsn string
}

// ProcessChunk parses a single SSE data payload and returns a partial response
// if the chunk contains text content, or nil if it should be silently accumulated.
func (a *Aggregator) ProcessChunk(m Mode, data string) *model.LLMResponse {
	var (
		choices  []oai.ChunkChoice
		usage    *oai.ChatUsage
		modelVer string
	)

	switch m {
	case ModeOrchestration:
		var chunk oai.OrchestrationChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil || chunk.FinalResult == nil {
			return nil
		}

		choices = chunk.FinalResult.Choices
		usage = chunk.FinalResult.Usage
		modelVer = chunk.FinalResult.Model

	default:
		var chunk oai.FoundationChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil
		}

		choices = chunk.Choices
		usage = chunk.Usage
		modelVer = chunk.Model
	}

	if modelVer != "" {
		a.modelVer = modelVer
	}

	if usage != nil {
		a.usage = usage
	}

	if len(choices) == 0 {
		return nil
	}

	choice := choices[0]

	if choice.FinishReason != "" {
		a.finishRsn = choice.FinishReason
	}

	if len(choice.Delta.ToolCalls) > 0 {
		a.toolCalls = mergeToolCallDeltas(a.toolCalls, choice.Delta.ToolCalls)
	}

	if choice.Delta.Content == "" {
		return nil
	}

	a.textBuf.WriteString(choice.Delta.Content)

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: []*genai.Part{{Text: choice.Delta.Content}},
			Role:  "model",
		},
		Partial: true,
	}
}

// Finalize produces the final aggregated response after all chunks have been processed.
func (a *Aggregator) Finalize() *model.LLMResponse {
	var parts []*genai.Part

	if a.textBuf.Len() > 0 {
		parts = append(parts, &genai.Part{Text: a.textBuf.String()})
	}

	for _, tc := range a.toolCalls {
		parts = append(parts, convert.ToolCallToPart(tc))
	}

	if len(parts) == 0 {
		parts = []*genai.Part{}
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  "model",
		},
		FinishReason:  convert.MapFinishReason(a.finishRsn),
		UsageMetadata: convert.Usage(a.usage),
		ModelVersion:  a.modelVer,
		TurnComplete:  true,
	}
}

// ParseSSELine extracts the data payload from an SSE line.
// Returns the data string and true if the line is a valid "data: " line.
func ParseSSELine(line string) (string, bool) {
	if !strings.HasPrefix(line, "data: ") {
		return "", false
	}

	return strings.TrimPrefix(line, "data: "), true
}

func mergeToolCallDeltas(accumulated, deltas []oai.ToolCall) []oai.ToolCall {
	for _, delta := range deltas {
		idx := delta.Index

		for idx >= len(accumulated) {
			accumulated = append(accumulated, oai.ToolCall{Type: "function"})
		}

		tc := &accumulated[idx]

		if delta.ID != "" {
			tc.ID = delta.ID
		}

		if delta.Function.Name != "" {
			tc.Function.Name += delta.Function.Name
		}

		tc.Function.Arguments += delta.Function.Arguments
	}

	return accumulated
}
