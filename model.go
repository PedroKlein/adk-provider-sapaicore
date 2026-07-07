package sapaicore

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"strings"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

var _ model.LLM = (*sapModel)(nil)

type sapModel struct {
	name          string
	deploymentID  string
	endpoint      string
	resourceGroup string
	headers       http.Header
	auth          *tokenCache
	httpClient    *http.Client
	mode          mode
	extraParams   map[string]any
	timeout       int
	maxRetries    int
}

func (m *sapModel) Name() string {
	return m.name
}

func (m *sapModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		return m.generateStream(ctx, req)
	}

	return m.generate(ctx, req)
}

func (m *sapModel) generate(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		body := m.buildRequestBody(req, false)

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

func (m *sapModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		body := m.buildRequestBody(req, true)

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

		var agg streamAggregator

		scanner := bufio.NewScanner(httpResp.Body)

		for scanner.Scan() {
			data, ok := parseSSELine(scanner.Text())
			if !ok {
				continue
			}

			if data == "[DONE]" {
				break
			}

			partial := agg.processChunk(m.mode, data)
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

func (m *sapModel) buildRequestBody(req *model.LLMRequest, stream bool) []byte {
	var (
		systemInstruction *genai.Content
		tools             []*genai.Tool
		temperature       *float32
		maxTokens         int32
		topP              *float32
		stop              []string
		frequencyPenalty  *float32
		presencePenalty   *float32
		respFmt           *responseFormat
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
		respFmt = convertResponseFormat(req.Config.ResponseMIMEType, req.Config.ResponseSchema)
	}

	messages := convertMessages(systemInstruction, req.Contents)
	toolDefs := convertTools(tools)
	modelName := m.requestModel(req)

	switch m.mode {
	case modeOrchestration:
		return m.buildOrchestrationBody(modelName, messages, toolDefs, temperature, maxTokens, topP, stop, frequencyPenalty, presencePenalty, respFmt, stream)
	default:
		return m.buildFoundationBody(modelName, messages, toolDefs, temperature, maxTokens, topP, stop, frequencyPenalty, presencePenalty, respFmt, stream)
	}
}

func (m *sapModel) buildOrchestrationBody(
	modelName string,
	messages []chatMessage,
	tools []toolDef,
	temperature *float32,
	maxTokens int32,
	topP *float32,
	stop []string,
	frequencyPenalty *float32,
	presencePenalty *float32,
	respFmt *responseFormat,
	stream bool,
) []byte {
	params := make(map[string]any)

	if temperature != nil {
		params["temperature"] = *temperature
	}

	if maxTokens > 0 {
		params["max_tokens"] = maxTokens
	}

	if topP != nil {
		params["top_p"] = *topP
	}

	if len(stop) > 0 {
		params["stop"] = stop
	}

	if frequencyPenalty != nil {
		params["frequency_penalty"] = *frequencyPenalty
	}

	if presencePenalty != nil {
		params["presence_penalty"] = *presencePenalty
	}

	maps.Copy(params, m.extraParams)

	if stream {
		params["stream_options"] = map[string]any{"include_usage": true}
	}

	// All messages go into prompt.template. The orchestration service merges
	// template with messages_history before sending to the LLM. The JS SDK
	// (sap-ai-provider, pi-sap-aicore) uses this same approach.
	template := messages
	if len(template) == 0 {
		template = []chatMessage{{Role: "system", Content: strPtr("You are a helpful assistant.")}}
	}

	orchReq := orchestrationRequest{
		Config: orchestrationConfig{
			Modules: moduleConfigs{
				PromptTemplating: promptTemplatingModule{
					Prompt: promptConfig{
						Template:       template,
						Tools:          tools,
						ResponseFormat: respFmt,
					},
					Model: modelDef{
						Name:       modelName,
						Version:    "latest",
						Params:     params,
						Timeout:    m.timeout,
						MaxRetries: m.maxRetries,
					},
				},
			},
		},
	}

	if stream {
		orchReq.Config.Stream = &streamConfig{Enabled: true}
	}

	body, _ := json.Marshal(orchReq)

	return body
}

func (m *sapModel) buildFoundationBody(
	modelName string,
	messages []chatMessage,
	tools []toolDef,
	temperature *float32,
	maxTokens int32,
	topP *float32,
	stop []string,
	frequencyPenalty *float32,
	presencePenalty *float32,
	respFmt *responseFormat,
	stream bool,
) []byte {
	fr := foundationRequest{
		Model:            modelName,
		Messages:         messages,
		Tools:            tools,
		Stream:           stream,
		Temperature:      temperature,
		TopP:             topP,
		Stop:             stop,
		FrequencyPenalty: frequencyPenalty,
		PresencePenalty:  presencePenalty,
		ResponseFormat:   respFmt,
	}

	if maxTokens > 0 {
		fr.MaxTokens = &maxTokens
	}

	if stream {
		fr.StreamOptions = &streamOptions{IncludeUsage: true}
	}

	body, _ := json.Marshal(fr)

	return body
}

func (m *sapModel) parseResponse(resp *http.Response) (*model.LLMResponse, error) {
	switch m.mode {
	case modeOrchestration:
		var orchResp orchestrationResponse
		if err := json.NewDecoder(resp.Body).Decode(&orchResp); err != nil {
			return nil, fmt.Errorf("decoding orchestration response: %w", ErrInference)
		}

		if orchResp.FinalResult == nil {
			return &model.LLMResponse{
				Content: &genai.Content{Parts: []*genai.Part{}, Role: "model"},
			}, nil
		}

		return m.convertFoundationResponse(orchResp.FinalResult), nil

	default:
		var fResp foundationResponse
		if err := json.NewDecoder(resp.Body).Decode(&fResp); err != nil {
			return nil, fmt.Errorf("decoding response: %w", ErrInference)
		}

		return m.convertFoundationResponse(&fResp), nil
	}
}

func (m *sapModel) convertFoundationResponse(resp *foundationResponse) *model.LLMResponse {
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

	return convertChoiceToResponse(resp.Choices[0], resp.Usage, resp.Model)
}

// requestModel prefers req.Model (set by BeforeModelCallback) over the construction-time name.
func (m *sapModel) requestModel(req *model.LLMRequest) string {
	if req.Model != "" {
		return req.Model
	}

	return m.name
}

func (m *sapModel) requestURL() string {
	switch m.mode {
	case modeOrchestration:
		return fmt.Sprintf("%s/v2/inference/deployments/%s/v2/completion", m.endpoint, m.deploymentID)
	default:
		return fmt.Sprintf("%s/v2/inference/deployments/%s/chat/completions", m.endpoint, m.deploymentID)
	}
}

func (m *sapModel) doHTTPRequest(ctx context.Context, body []byte) (*http.Response, error) {
	token, err := m.auth.getToken(ctx)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.requestURL(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", ErrInference)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("AI-Resource-Group", m.resourceGroup)

	for key, values := range m.headers {
		for _, v := range values {
			req.Header.Add(key, v)
		}
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", ErrInference)
	}

	return resp, nil
}

func (m *sapModel) handleErrorResponse(resp *http.Response) error {
	switch m.mode {
	case modeOrchestration:
		var errResp struct {
			Error struct {
				Message string `json:"message"`
				Code    int    `json:"code"`
			} `json:"error"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error.Message != "" {
			return fmt.Errorf("orchestration error %d: %s: %w", resp.StatusCode, errResp.Error.Message, ErrInference)
		}

	default:
		var errResp foundationResponse
		if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != nil {
			return fmt.Errorf("API error %d: %s: %w", resp.StatusCode, errResp.Error.Message, ErrInference)
		}
	}

	return fmt.Errorf("API returned status %d: %w", resp.StatusCode, ErrInference)
}

// --- Streaming ---

type streamAggregator struct {
	textBuf   strings.Builder
	toolCalls []toolCall
	usage     *chatUsage
	modelVer  string
	finishRsn string
}

func (a *streamAggregator) processChunk(m mode, data string) *model.LLMResponse {
	var (
		choices  []chunkChoice
		usage    *chatUsage
		modelVer string
	)

	switch m {
	case modeOrchestration:
		var chunk orchestrationChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil || chunk.FinalResult == nil {
			return nil
		}

		choices = chunk.FinalResult.Choices
		usage = chunk.FinalResult.Usage
		modelVer = chunk.FinalResult.Model

	default:
		var chunk foundationChunk
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

func (a *streamAggregator) finalize() *model.LLMResponse {
	var parts []*genai.Part

	if a.textBuf.Len() > 0 {
		parts = append(parts, &genai.Part{Text: a.textBuf.String()})
	}

	for _, tc := range a.toolCalls {
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
		FinishReason:  mapFinishReason(a.finishRsn),
		UsageMetadata: convertUsage(a.usage),
		ModelVersion:  a.modelVer,
		TurnComplete:  true,
	}
}

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

func parseSSELine(line string) (string, bool) {
	if !strings.HasPrefix(line, "data: ") {
		return "", false
	}

	return strings.TrimPrefix(line, "data: "), true
}
