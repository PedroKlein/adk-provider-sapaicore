// Package orchestration implements the SAP AI Core orchestration mode strategy.
package orchestration

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/convert"
	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/request"
	"github.com/PedroKlein/adk-provider-sapaicore/internal/stream"
)

var ErrInference = errors.New("orchestration: inference")

// Model implements the ADK model.LLM interface for SAP AI Core orchestration mode.
type Model struct {
	ModelName      string
	ExtraParams    map[string]any
	Timeout        int
	MaxRetries     int
	Filtering      *oai.FilteringModuleConfig
	Masking        *oai.MaskingModuleConfig
	Translation    *oai.TranslationModuleConfig
	FallbackModels []string
	PromptCaching  bool
	CacheTTL       string
	StreamOptions  *oai.StreamConfig
	URL            string
	Client         *request.Client
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
