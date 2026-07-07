// Package orchestration implements the SAP AI Core orchestration mode strategy.
package orchestration

import (
	"bufio"
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

// ErrInference indicates an orchestration inference request error.
var ErrInference = errors.New("orchestration: inference")

// TokenGetter retrieves a valid OAuth2 access token.
type TokenGetter = request.TokenGetter

// Model implements the ADK model.LLM interface for SAP AI Core orchestration mode.
type Model struct {
	ModelName     string
	DeploymentID  string
	Endpoint      string
	ResourceGroup string
	Headers       http.Header
	Auth          TokenGetter
	HTTPClient    *http.Client
	ExtraParams   map[string]any
	Timeout       int
	MaxRetries    int
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

			partial := agg.ProcessChunk(stream.ModeOrchestration, data)
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

	modelParams := make(map[string]any)

	if params.Temperature != nil {
		modelParams["temperature"] = *params.Temperature
	}

	if params.MaxTokens > 0 {
		modelParams["max_tokens"] = params.MaxTokens
	}

	if params.TopP != nil {
		modelParams["top_p"] = *params.TopP
	}

	if len(params.Stop) > 0 {
		modelParams["stop"] = params.Stop
	}

	if params.FrequencyPenalty != nil {
		modelParams["frequency_penalty"] = *params.FrequencyPenalty
	}

	if params.PresencePenalty != nil {
		modelParams["presence_penalty"] = *params.PresencePenalty
	}

	maps.Copy(modelParams, params.ExtraParams)

	if doStream {
		modelParams["stream_options"] = map[string]any{"include_usage": true}
	}

	template := params.Messages
	if len(template) == 0 {
		defaultContent := "You are a helpful assistant."
		template = []oai.ChatMessage{{Role: "system", Content: &defaultContent}}
	}

	orchReq := oai.OrchestrationRequest{
		Config: oai.OrchestrationConfig{
			Modules: oai.ModuleConfigs{
				PromptTemplating: oai.PromptTemplatingModule{
					Prompt: oai.PromptConfig{
						Template:       template,
						Tools:          params.Tools,
						ResponseFormat: params.ResponseFormat,
					},
					Model: oai.ModelDef{
						Name:       params.ModelName,
						Version:    "latest",
						Params:     modelParams,
						Timeout:    params.Timeout,
						MaxRetries: params.MaxRetries,
					},
				},
			},
		},
	}

	if doStream {
		orchReq.Config.Stream = &oai.StreamConfig{Enabled: true}
	}

	body, err := json.Marshal(orchReq)
	if err != nil {
		return nil, fmt.Errorf("marshaling orchestration request: %w", err)
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
		Timeout:          m.Timeout,
		MaxRetries:       m.MaxRetries,
	}
}

func (m *Model) parseResponse(resp *http.Response) (*model.LLMResponse, error) {
	var orchResp oai.OrchestrationResponse
	if err := json.NewDecoder(resp.Body).Decode(&orchResp); err != nil {
		return nil, fmt.Errorf("decoding orchestration response: %w", err)
	}

	if orchResp.FinalResult == nil {
		return &model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{}, Role: "model"},
		}, nil
	}

	return convertFoundationResponse(orchResp.FinalResult), nil
}

func convertFoundationResponse(resp *oai.FoundationResponse) *model.LLMResponse {
	if resp.Error != nil {
		return &model.LLMResponse{
			ErrorCode:    resp.Error.Code,
			ErrorMessage: resp.Error.Message,
		}
	}

	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			Content: &genai.Content{Parts: []*genai.Part{}, Role: "model"},
		}
	}

	return convert.ChoiceToResponse(resp.Choices[0], resp.Usage, resp.Model)
}

func (m *Model) requestURL() string {
	return fmt.Sprintf("%s/v2/inference/deployments/%s/v2/completion", m.Endpoint, m.DeploymentID)
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
	var errResp struct {
		Error struct {
			Message string `json:"message"`
			Code    int    `json:"code"`
		} `json:"error"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error.Message != "" {
		return fmt.Errorf("orchestration error %d: %s: %w", resp.StatusCode, errResp.Error.Message, ErrInference)
	}

	return fmt.Errorf("API returned status %d: %w", resp.StatusCode, ErrInference)
}
