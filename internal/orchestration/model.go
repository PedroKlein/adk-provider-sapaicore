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

// tokenGetter retrieves a valid OAuth2 access token.
type tokenGetter interface {
	GetToken(ctx context.Context) (string, error)
}

// Model implements the ADK model.LLM interface for SAP AI Core orchestration mode.
type Model struct {
	ModelName     string
	DeploymentID  string
	Endpoint      string
	ResourceGroup string
	Headers       http.Header
	Auth          tokenGetter
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
	modelParams := buildModelParams(params, doStream)

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

func buildModelParams(params oai.RequestParams, doStream bool) map[string]any {
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

	if params.TopK != nil {
		modelParams["top_k"] = *params.TopK
	}

	if params.Seed != nil {
		modelParams["seed"] = *params.Seed
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

	if params.ResponseLogprobs {
		modelParams["logprobs"] = true
	}

	if params.Logprobs != nil {
		modelParams["top_logprobs"] = *params.Logprobs
	}

	if params.ToolChoice != nil {
		modelParams["tool_choice"] = params.ToolChoice
	}

	maps.Copy(modelParams, params.ExtraParams)

	if doStream {
		modelParams["stream_options"] = map[string]any{"include_usage": true}
	}

	return modelParams
}

func (m *Model) extractParams(req *model.LLMRequest, doStream bool) oai.RequestParams {
	params := convert.ExtractParams(req.Config, req.Contents, req.Model, m.ModelName, m.ExtraParams)
	params.Stream = doStream
	params.Timeout = m.Timeout
	params.MaxRetries = m.MaxRetries

	return params
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
		HTTPClient:    m.HTTPClient,
	}
}

func (m *Model) doHTTPRequest(ctx context.Context, body []byte) (*http.Response, error) {
	token, err := m.Auth.GetToken(ctx)
	if err != nil {
		return nil, fmt.Errorf("getting auth token: %w", err)
	}

	resp, err := request.Do(ctx, m.reqConfig(), m.requestURL(), body, token)
	if err != nil {
		return nil, fmt.Errorf("inference request: %w", err)
	}

	return resp, nil
}

func (m *Model) handleErrorResponse(resp *http.Response) error {
	var errResp oai.FoundationResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != nil {
		return fmt.Errorf("orchestration error %d: %s: %w", resp.StatusCode, errResp.Error.Message, ErrInference)
	}

	return fmt.Errorf("API returned status %d: %w", resp.StatusCode, ErrInference)
}
