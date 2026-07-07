// Package foundation implements the SAP AI Core foundation-models mode strategy.
package foundation

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/go-adk-sap-ai-core/internal/convert"
	oai "github.com/PedroKlein/go-adk-sap-ai-core/internal/openai"
	"github.com/PedroKlein/go-adk-sap-ai-core/internal/request"
	"github.com/PedroKlein/go-adk-sap-ai-core/internal/stream"
)

// ErrInference indicates an inference request error.
var ErrInference = errors.New("foundation: inference")

// TokenGetter retrieves a valid OAuth2 access token.
type TokenGetter = request.TokenGetter

// Model implements the ADK model.LLM interface for SAP AI Core foundation-models mode.
type Model struct {
	ModelName     string
	DeploymentID  string
	Endpoint      string
	ResourceGroup string
	Headers       http.Header
	Auth          TokenGetter
	HTTPClient    *http.Client
	ExtraParams   map[string]any
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

		for scanner.Scan() {
			data, ok := stream.ParseSSELine(scanner.Text())
			if !ok {
				continue
			}

			if data == "[DONE]" {
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
	params := m.extractParams(req, doStream)

	fr := oai.FoundationRequest{
		Model:            params.ModelName,
		Messages:         params.Messages,
		Tools:            params.Tools,
		Stream:           doStream,
		Temperature:      params.Temperature,
		TopP:             params.TopP,
		Stop:             params.Stop,
		FrequencyPenalty: params.FrequencyPenalty,
		PresencePenalty:  params.PresencePenalty,
		ResponseFormat:   params.ResponseFormat,
	}

	if params.MaxTokens > 0 {
		fr.MaxTokens = &params.MaxTokens
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

func (m *Model) extractParams(req *model.LLMRequest, doStream bool) oai.RequestParams {
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

	if req.Config != nil {
		systemInstruction = req.Config.SystemInstruction
		tools = req.Config.Tools
		temperature = req.Config.Temperature
		maxTokens = req.Config.MaxOutputTokens
		topP = req.Config.TopP
		stop = req.Config.StopSequences
		frequencyPenalty = req.Config.FrequencyPenalty
		presencePenalty = req.Config.PresencePenalty
		responseMIME = req.Config.ResponseMIMEType
		responseSchema = req.Config.ResponseSchema
	}

	modelName := req.Model
	if modelName == "" {
		modelName = m.ModelName
	}

	return oai.RequestParams{
		ModelName:        modelName,
		Messages:         convert.Messages(systemInstruction, req.Contents),
		Tools:            convert.Tools(tools),
		Temperature:      temperature,
		MaxTokens:        maxTokens,
		TopP:             topP,
		Stop:             stop,
		FrequencyPenalty: frequencyPenalty,
		PresencePenalty:  presencePenalty,
		ResponseFormat:   convert.ResponseFormat(responseMIME, responseSchema),
		Stream:           doStream,
		ExtraParams:      m.ExtraParams,
	}
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

func (m *Model) requestURL() string {
	return fmt.Sprintf("%s/v2/inference/deployments/%s/v1/chat/completions", m.Endpoint, m.DeploymentID)
}

func (m *Model) reqConfig() *request.Config {
	return &request.Config{
		Endpoint:      m.Endpoint,
		DeploymentID:  m.DeploymentID,
		ResourceGroup: m.ResourceGroup,
		Headers:       m.Headers,
		Auth:          m.Auth,
		HTTPClient:    m.HTTPClient,
	}
}

func (m *Model) doHTTPRequest(ctx context.Context, body []byte) (*http.Response, error) {
	resp, err := request.Do(ctx, m.reqConfig(), m.requestURL(), body)
	if err != nil {
		return nil, fmt.Errorf("inference request: %w", err)
	}

	return resp, nil
}

func (m *Model) handleErrorResponse(resp *http.Response) error {
	var errResp oai.FoundationResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != nil {
		return fmt.Errorf("API error %d: %s: %w", resp.StatusCode, errResp.Error.Message, ErrInference)
	}

	return fmt.Errorf("API returned status %d: %w", resp.StatusCode, ErrInference)
}
