// Package foundation implements the SAP AI Core foundation-models mode strategy.
package foundation

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"maps"
	"net/http"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/convert"
	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/request"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/stream"
)

// ErrInference indicates an inference request error.
var ErrInference = errors.New("foundation: inference")

// Model implements the ADK model.LLM interface for SAP AI Core foundation-models mode.
type Model struct {
	ModelName   string
	ExtraParams map[string]any
	URL         string
	Client      *request.Client
}

var _ model.LLM = (*Model)(nil)

func (m *Model) Name() string {
	return m.ModelName
}

func (m *Model) GenerateContent(ctx context.Context, req *model.LLMRequest, doStream bool) iter.Seq2[*model.LLMResponse, error] {
	if doStream {
		return m.generateStream(ctx, req)
	}

	return m.generate(ctx, req)
}

func (m *Model) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		body, err := m.buildRequestBody(req, false)
		if err != nil {
			yield(nil, err)
			return
		}

		httpResp, err := m.doHTTPRequest(ctx, body)
		if err != nil {
			yield(nil, err)
			return
		}

		defer func() { _ = httpResp.Body.Close() }()

		if httpResp.StatusCode != http.StatusOK {
			yield(nil, m.handleErrorResponse(httpResp))
			return
		}

		llmResp, err := m.parseResponse(httpResp)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp.TurnComplete = true

		yield(llmResp, nil)
	}
}

func (m *Model) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		body, err := m.buildRequestBody(req, true)
		if err != nil {
			yield(nil, err)
			return
		}

		httpResp, err := m.doHTTPRequest(ctx, body)
		if err != nil {
			yield(nil, err)
			return
		}

		defer func() { _ = httpResp.Body.Close() }()

		if httpResp.StatusCode != http.StatusOK {
			yield(nil, m.handleErrorResponse(httpResp))
			return
		}

		var agg stream.Aggregator

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		for scanner.Scan() {
			if ctx.Err() != nil {
				return
			}

			data, ok := stream.ParseSSELine(scanner.Bytes())
			if !ok {
				continue
			}

			if bytes.Equal(data, stream.DoneMarker) {
				break
			}

			partial := agg.ProcessChunk(stream.ModeFoundation, data)
			if partial != nil {
				if !yield(partial, nil) {
					return
				}
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("reading stream: %w", err))
			return
		}

		yield(agg.Finalize(), nil)
	}
}

func (m *Model) buildRequestBody(req *model.LLMRequest, doStream bool) ([]byte, error) {
	params, err := m.extractParams(req, doStream)
	if err != nil {
		return nil, err
	}

	if len(params.ExtraParams) > 0 {
		return m.buildRequestBodyWithExtras(params, doStream)
	}

	fr := oai.FoundationRequest{
		Model:            params.ModelName,
		Messages:         params.Messages,
		Tools:            params.Tools,
		ToolChoice:       params.ToolChoice,
		Stream:           doStream,
		Temperature:      params.Temperature,
		TopP:             params.TopP,
		TopK:             params.TopK,
		Seed:             params.Seed,
		Stop:             params.Stop,
		FrequencyPenalty: params.FrequencyPenalty,
		PresencePenalty:  params.PresencePenalty,
		ResponseFormat:   params.ResponseFormat,
	}

	if params.MaxTokens > 0 {
		fr.MaxTokens = &params.MaxTokens
	}

	if params.ResponseLogprobs {
		lp := true
		fr.Logprobs = &lp
	}

	if params.Logprobs != nil {
		fr.TopLogprobs = params.Logprobs
	}

	if doStream {
		fr.StreamOptions = &oai.StreamOptions{IncludeUsage: true}
	}

	body, err := json.Marshal(fr)
	if err != nil {
		return nil, fmt.Errorf("marshaling foundation request: %w", err)
	}

	return body, nil
}

// buildRequestBodyWithExtras builds the request as a map to merge extra params
// directly, avoiding a marshal → unmarshal → re-marshal round-trip.
func (m *Model) buildRequestBodyWithExtras(params oai.RequestParams, doStream bool) ([]byte, error) {
	obj := map[string]any{
		"model":    params.ModelName,
		"messages": params.Messages,
		"stream":   doStream,
	}

	if len(params.Tools) > 0 {
		obj["tools"] = params.Tools
	}

	if params.ToolChoice != nil {
		obj["tool_choice"] = params.ToolChoice
	}

	setIfNotNil(obj, "temperature", params.Temperature)
	setIfNotNil(obj, "top_p", params.TopP)
	setIfNotNil(obj, "top_k", params.TopK)
	setIfNotNil(obj, "seed", params.Seed)
	setIfNotNil(obj, "frequency_penalty", params.FrequencyPenalty)
	setIfNotNil(obj, "presence_penalty", params.PresencePenalty)

	if params.MaxTokens > 0 {
		obj["max_tokens"] = params.MaxTokens
	}

	if len(params.Stop) > 0 {
		obj["stop"] = params.Stop
	}

	if params.ResponseLogprobs {
		obj["logprobs"] = true
	}

	if params.Logprobs != nil {
		obj["top_logprobs"] = *params.Logprobs
	}

	if params.ResponseFormat != nil {
		obj["response_format"] = params.ResponseFormat
	}

	if doStream {
		obj["stream_options"] = map[string]any{"include_usage": true}
	}

	maps.Copy(obj, params.ExtraParams)

	body, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("marshaling foundation request: %w", err)
	}

	return body, nil
}

func setIfNotNil[T any](m map[string]any, key string, ptr *T) {
	if ptr != nil {
		m[key] = *ptr
	}
}

func (m *Model) extractParams(req *model.LLMRequest, doStream bool) (oai.RequestParams, error) {
	params, err := convert.ExtractParams(req.Config, req.Contents, req.Model, m.ModelName, m.ExtraParams)
	if err != nil {
		return oai.RequestParams{}, fmt.Errorf("converting request content: %w", err)
	}

	params.Stream = doStream

	return params, nil
}

func (m *Model) parseResponse(resp *http.Response) (*model.LLMResponse, error) {
	var fResp oai.FoundationResponse
	if err := json.NewDecoder(resp.Body).Decode(&fResp); err != nil {
		return nil, fmt.Errorf("decoding foundation response: %w", err)
	}

	if fResp.Error != nil {
		return &model.LLMResponse{
			ErrorCode:    fResp.Error.Code,
			ErrorMessage: fResp.Error.Message,
		}, nil
	}

	if len(fResp.Choices) == 0 {
		return &model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{}, Role: "model"},
		}, nil
	}

	return convert.ChoiceToResponse(fResp.Choices[0], fResp.Usage, fResp.Model), nil
}

func (m *Model) doHTTPRequest(ctx context.Context, body []byte) (*http.Response, error) {
	resp, err := m.Client.Execute(ctx, m.URL, body)
	if err != nil {
		return nil, fmt.Errorf("inference request: %w", err)
	}

	return resp, nil
}

func (m *Model) handleErrorResponse(resp *http.Response) error {
	//nolint:wrapcheck // request is an internal package; error is already wrapped with sentinel
	return m.Client.HandleError(resp, ErrInference)
}
