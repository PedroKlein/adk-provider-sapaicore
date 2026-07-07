package sapaicore

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

var _ model.LLM = (*sapModel)(nil)

// sapModel implements model.LLM for SAP AI Core deployments.
type sapModel struct {
	name          string
	deploymentID  string
	endpoint      string
	resourceGroup string
	auth          *tokenCache
	httpClient    *http.Client
}

func (m *sapModel) Name() string {
	return m.name
}

// GenerateContent calls SAP AI Core's OpenAI-compatible chat completions endpoint.
func (m *sapModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	chatReq := convertRequest(req, m.requestModel(req))
	chatReq.Stream = stream

	if stream {
		return m.generateStream(ctx, chatReq)
	}

	return m.generate(ctx, chatReq)
}

// requestModel determines the model name to send in the request.
// Prefers req.Model (set by BeforeModelCallback) over the construction-time name.
func (m *sapModel) requestModel(req *model.LLMRequest) string {
	if req.Model != "" {
		return req.Model
	}

	return m.name
}

// generate performs a non-streaming chat completion request.
func (m *sapModel) generate(ctx context.Context, chatReq chatRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.doRequest(ctx, chatReq)
		if err != nil {
			yield(nil, err)
			return
		}

		llmResp := convertResponse(resp)
		llmResp.TurnComplete = true

		yield(llmResp, nil)
	}
}

// generateStream performs a streaming chat completion request using SSE.
func (m *sapModel) generateStream(ctx context.Context, chatReq chatRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		httpResp, err := m.doHTTPRequest(ctx, chatReq)
		if err != nil {
			yield(nil, err)
			return
		}

		defer func() { _ = httpResp.Body.Close() }()

		if httpResp.StatusCode != http.StatusOK {
			yield(nil, m.handleErrorResponse(httpResp))
			return
		}

		var agg streamAggregator

		scanner := bufio.NewScanner(httpResp.Body)

		for scanner.Scan() {
			line := scanner.Text()

			data, ok := parseSSELine(line)
			if !ok {
				continue
			}

			if data == "[DONE]" {
				break
			}

			partial := agg.processChunk(data)
			if partial != nil {
				if !yield(partial, nil) {
					return
				}
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("reading stream: %w", ErrInference))
			return
		}

		yield(agg.finalize(), nil)
	}
}

// streamAggregator accumulates streaming chunks into a final response.
type streamAggregator struct {
	textBuf   strings.Builder
	toolCalls []toolCall
	usage     *chatUsage
	modelVer  string
	finishRsn string
}

// processChunk handles a single SSE data payload. Returns a partial response if
// there is text content to emit, otherwise returns nil.
func (a *streamAggregator) processChunk(data string) *model.LLMResponse {
	var chunk chatChunk
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return nil
	}

	if chunk.Model != "" {
		a.modelVer = chunk.Model
	}

	if chunk.Usage != nil {
		a.usage = chunk.Usage
	}

	if len(chunk.Choices) == 0 {
		return nil
	}

	choice := chunk.Choices[0]

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

// finalize returns the final aggregated response after streaming completes.
func (a *streamAggregator) finalize() *model.LLMResponse {
	return buildFinalResponse(a.textBuf.String(), a.toolCalls, a.finishRsn, a.usage, a.modelVer)
}

// parseSSELine extracts the data payload from an SSE line.
func parseSSELine(line string) (string, bool) {
	if !strings.HasPrefix(line, "data: ") {
		return "", false
	}

	return strings.TrimPrefix(line, "data: "), true
}

// doRequest performs a non-streaming HTTP request and decodes the response.
func (m *sapModel) doRequest(ctx context.Context, chatReq chatRequest) (*chatResponse, error) {
	httpResp, err := m.doHTTPRequest(ctx, chatReq)
	if err != nil {
		return nil, err
	}

	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode != http.StatusOK {
		return nil, m.handleErrorResponse(httpResp)
	}

	var resp chatResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("decoding response: %w", ErrInference)
	}

	return &resp, nil
}

// doHTTPRequest builds and sends the HTTP request to SAP AI Core.
func (m *sapModel) doHTTPRequest(ctx context.Context, chatReq chatRequest) (*http.Response, error) {
	token, err := m.auth.getToken(ctx)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", ErrInference)
	}

	url := fmt.Sprintf("%s/v2/inference/deployments/%s/chat/completions", m.endpoint, m.deploymentID)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", ErrInference)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("AI-Resource-Group", m.resourceGroup)

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", ErrInference)
	}

	return resp, nil
}

// handleErrorResponse reads the error body and returns a structured error.
func (m *sapModel) handleErrorResponse(resp *http.Response) error {
	var errResp chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != nil {
		return fmt.Errorf("API error %d: %s: %w", resp.StatusCode, errResp.Error.Message, ErrInference)
	}

	return fmt.Errorf("API returned status %d: %w", resp.StatusCode, ErrInference)
}

// mergeToolCallDeltas merges streaming tool call deltas into accumulated tool calls.
func mergeToolCallDeltas(accumulated, deltas []toolCall) []toolCall {
	for _, delta := range deltas {
		idx := delta.Index

		for idx >= len(accumulated) {
			accumulated = append(accumulated, toolCall{Type: "function"})
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

// buildFinalResponse creates the final aggregated LLMResponse after streaming completes.
func buildFinalResponse(text string, toolCalls []toolCall, finishReason string, usage *chatUsage, modelVersion string) *model.LLMResponse {
	var parts []*genai.Part

	if text != "" {
		parts = append(parts, &genai.Part{Text: text})
	}

	for _, tc := range toolCalls {
		parts = append(parts, convertToolCallToPart(tc))
	}

	if len(parts) == 0 {
		parts = []*genai.Part{}
	}

	return &model.LLMResponse{
		Content: &genai.Content{
			Parts: parts,
			Role:  "model",
		},
		FinishReason:  mapFinishReason(finishReason),
		UsageMetadata: convertUsage(usage),
		ModelVersion:  modelVersion,
		TurnComplete:  true,
	}
}
