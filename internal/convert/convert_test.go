package convert_test

import (
	"testing"

	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/convert"
)

func TestMapRole(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		want  string
	}{
		{"model", "assistant"},
		{"user", "user"},
		{"system", "system"},
		{"tool", "tool"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			t.Parallel()

			if got := convert.MapRole(tt.input); got != tt.want {
				t.Errorf("MapRole(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestMapFinishReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		want  genai.FinishReason
	}{
		{"stop", genai.FinishReasonStop},
		{"tool_calls", genai.FinishReasonStop},
		{"length", genai.FinishReasonMaxTokens},
		{"content_filter", genai.FinishReasonSafety},
		{"unknown", genai.FinishReasonOther},
		{"", genai.FinishReasonOther},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			t.Parallel()

			if got := convert.MapFinishReason(tt.input); got != tt.want {
				t.Errorf("MapFinishReason(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestMessages_SystemAndUser(t *testing.T) {
	t.Parallel()

	sys := &genai.Content{
		Parts: []*genai.Part{{Text: "Be helpful."}},
	}

	contents := []*genai.Content{
		{Parts: []*genai.Part{{Text: "Hello"}}, Role: "user"},
	}

	msgs := convert.Messages(sys, contents)

	if len(msgs) != 2 {
		t.Fatalf("len = %d, want 2", len(msgs))
	}

	if msgs[0].Role != "system" {
		t.Errorf("msgs[0].Role = %q, want system", msgs[0].Role)
	}

	if c, ok := msgs[0].Content.(*string); !ok || *c != "Be helpful." {
		t.Errorf("msgs[0].Content = %v, want %q", msgs[0].Content, "Be helpful.")
	}

	if msgs[1].Role != "user" {
		t.Errorf("msgs[1].Role = %q, want user", msgs[1].Role)
	}
}

func TestMessages_SkipsNilContent(t *testing.T) {
	t.Parallel()

	contents := []*genai.Content{
		nil,
		{Parts: []*genai.Part{{Text: "Hi"}}, Role: "user"},
		nil,
	}

	msgs := convert.Messages(nil, contents)

	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}
}

func TestMessages_FunctionCallAndResponse(t *testing.T) {
	t.Parallel()

	contents := []*genai.Content{
		{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			ID:   "call_1",
			Name: "search",
			Args: map[string]any{"q": "test"},
		}}}, Role: "model"},
		{Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID:       "call_1",
			Name:     "search",
			Response: map[string]any{"result": "found"},
		}}}, Role: "user"},
	}

	msgs := convert.Messages(nil, contents)

	if len(msgs) != 2 {
		t.Fatalf("len = %d, want 2", len(msgs))
	}

	if msgs[0].Role != "assistant" {
		t.Errorf("msgs[0].Role = %q, want assistant", msgs[0].Role)
	}

	if len(msgs[0].ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(msgs[0].ToolCalls))
	}

	if msgs[0].ToolCalls[0].Function.Name != "search" {
		t.Errorf("function name = %q, want search", msgs[0].ToolCalls[0].Function.Name)
	}

	if msgs[1].Role != "tool" {
		t.Errorf("msgs[1].Role = %q, want tool", msgs[1].Role)
	}

	if msgs[1].ToolCallID != "call_1" {
		t.Errorf("tool_call_id = %q, want call_1", msgs[1].ToolCallID)
	}
}

func TestMessages_SkipsThoughtParts(t *testing.T) {
	t.Parallel()

	contents := []*genai.Content{
		{Parts: []*genai.Part{
			{Text: "thinking...", Thought: true},
			{Text: "Hello!"},
		}, Role: "model"},
	}

	msgs := convert.Messages(nil, contents)

	if len(msgs) != 1 {
		t.Fatalf("len = %d, want 1", len(msgs))
	}

	if c, ok := msgs[0].Content.(*string); !ok || *c != "Hello!" {
		t.Errorf("content = %v, want %q", msgs[0].Content, "Hello!")
	}
}

func TestSchema_ConvertsTypesToLowercase(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		schema   *genai.Schema
		wantType string
	}{
		{"object", &genai.Schema{Type: "OBJECT"}, "object"},
		{"string", &genai.Schema{Type: "STRING"}, "string"},
		{"integer", &genai.Schema{Type: "INTEGER"}, "integer"},
		{"array", &genai.Schema{Type: "ARRAY"}, "array"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convert.Schema(tt.schema)

			if result["type"] != tt.wantType {
				t.Errorf("type = %v, want %q", result["type"], tt.wantType)
			}
		})
	}
}

func TestSchema_NestedProperties(t *testing.T) {
	t.Parallel()

	schema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"name": {Type: "STRING", Description: "The name"},
			"age":  {Type: "INTEGER"},
		},
		Required: []string{"name"},
	}

	result := convert.Schema(schema)

	props, ok := result["properties"].(map[string]any)
	if !ok {
		t.Fatal("properties not a map")
	}

	nameSchema, ok := props["name"].(map[string]any)
	if !ok {
		t.Fatal("name property not a map")
	}

	if nameSchema["type"] != "string" {
		t.Errorf("name.type = %v, want string", nameSchema["type"])
	}

	if nameSchema["description"] != "The name" {
		t.Errorf("name.description = %v, want %q", nameSchema["description"], "The name")
	}

	required, ok := result["required"].([]string)
	if !ok {
		t.Fatal("required not a string slice")
	}

	if len(required) != 1 || required[0] != "name" {
		t.Errorf("required = %v, want [name]", required)
	}
}

func TestTools_ConvertsDeclarations(t *testing.T) {
	t.Parallel()

	tools := []*genai.Tool{{
		FunctionDeclarations: []*genai.FunctionDeclaration{
			{
				Name:        "search",
				Description: "Search for something",
				Parameters: &genai.Schema{
					Type: "OBJECT",
					Properties: map[string]*genai.Schema{
						"query": {Type: "STRING"},
					},
				},
			},
		},
	}}

	defs := convert.Tools(tools)

	if len(defs) != 1 {
		t.Fatalf("len = %d, want 1", len(defs))
	}

	if defs[0].Type != "function" {
		t.Errorf("type = %q, want function", defs[0].Type)
	}

	if defs[0].Function.Name != "search" {
		t.Errorf("name = %q, want search", defs[0].Function.Name)
	}
}

func TestTools_SkipsNilTools(t *testing.T) {
	t.Parallel()

	tools := []*genai.Tool{nil, {FunctionDeclarations: nil}}

	defs := convert.Tools(tools)

	if len(defs) != 0 {
		t.Errorf("len = %d, want 0", len(defs))
	}
}

func TestResponseFormat_JSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		mime     string
		schema   *genai.Schema
		wantNil  bool
		wantType string
	}{
		{"non-json returns nil", "text/plain", nil, true, ""},
		{"json without schema", "application/json", nil, false, "json_object"},
		{"json with schema", "application/json", &genai.Schema{Type: "OBJECT"}, false, "json_schema"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convert.ResponseFormat(tt.mime, tt.schema)

			if tt.wantNil {
				if result != nil {
					t.Errorf("expected nil, got %+v", result)
				}

				return
			}

			if result == nil {
				t.Fatal("expected non-nil result")
			}

			if result.Type != tt.wantType {
				t.Errorf("type = %q, want %q", result.Type, tt.wantType)
			}
		})
	}
}

func TestUsage_Nil(t *testing.T) {
	t.Parallel()

	if got := convert.Usage(nil); got != nil {
		t.Errorf("expected nil, got %+v", got)
	}
}

func TestUsage_Values(t *testing.T) {
	t.Parallel()

	usage := &oai.ChatUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}

	got := convert.Usage(usage)

	if got.PromptTokenCount != 10 {
		t.Errorf("prompt = %d, want 10", got.PromptTokenCount)
	}

	if got.CandidatesTokenCount != 5 {
		t.Errorf("completion = %d, want 5", got.CandidatesTokenCount)
	}
}

func TestChoiceToResponse_TextContent(t *testing.T) {
	t.Parallel()

	choice := oai.ChatChoice{
		Index: 0,
		Message: oai.ChatMessage{
			Role:    "assistant",
			Content: strPtr("Hello!"),
		},
		FinishReason: "stop",
	}

	usage := &oai.ChatUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}

	resp := convert.ChoiceToResponse(choice, usage, "gpt-4.1")

	if resp.Content.Role != "model" {
		t.Errorf("role = %q, want model", resp.Content.Role)
	}

	if resp.Content.Parts[0].Text != "Hello!" {
		t.Errorf("text = %q, want Hello!", resp.Content.Parts[0].Text)
	}

	if resp.FinishReason != genai.FinishReasonStop {
		t.Errorf("finish = %v, want Stop", resp.FinishReason)
	}

	if resp.ModelVersion != "gpt-4.1" {
		t.Errorf("model = %q, want gpt-4.1", resp.ModelVersion)
	}

	if resp.UsageMetadata.PromptTokenCount != 10 {
		t.Errorf("prompt tokens = %d, want 10", resp.UsageMetadata.PromptTokenCount)
	}
}

func TestChoiceToResponse_Refusal(t *testing.T) {
	t.Parallel()

	choice := oai.ChatChoice{
		Message: oai.ChatMessage{
			Role:    "assistant",
			Content: strPtr(""),
			Refusal: "I cannot help with that.",
		},
		FinishReason: "stop",
	}

	resp := convert.ChoiceToResponse(choice, nil, "gpt-4.1")

	if resp.ErrorCode != "refusal" {
		t.Errorf("ErrorCode = %q, want refusal", resp.ErrorCode)
	}

	if resp.ErrorMessage != "I cannot help with that." {
		t.Errorf("ErrorMessage = %q", resp.ErrorMessage)
	}
}

func TestChoiceToResponse_ToolCalls(t *testing.T) {
	t.Parallel()

	choice := oai.ChatChoice{
		Message: oai.ChatMessage{
			Role:    "assistant",
			Content: strPtr(""),
			ToolCalls: []oai.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: oai.FunctionCall{
					Name:      "search",
					Arguments: `{"q":"hello"}`,
				},
			}},
		},
		FinishReason: "tool_calls",
	}

	resp := convert.ChoiceToResponse(choice, nil, "m")

	if len(resp.Content.Parts) != 1 {
		t.Fatalf("parts = %d, want 1", len(resp.Content.Parts))
	}

	fc := resp.Content.Parts[0].FunctionCall
	if fc == nil {
		t.Fatal("expected FunctionCall")
	}

	if fc.Name != "search" {
		t.Errorf("name = %q, want search", fc.Name)
	}
}

func TestChoiceToResponse_EmptyContent(t *testing.T) {
	t.Parallel()

	choice := oai.ChatChoice{
		Message: oai.ChatMessage{
			Role:    "assistant",
			Content: strPtr(""),
		},
		FinishReason: "stop",
	}

	resp := convert.ChoiceToResponse(choice, nil, "m")

	if len(resp.Content.Parts) != 0 {
		t.Errorf("parts = %d, want 0", len(resp.Content.Parts))
	}
}

func strPtr(s string) *string { return &s }

func TestToolCallToPart(t *testing.T) {
	t.Parallel()

	tc := oai.ToolCall{
		ID:   "call_123",
		Type: "function",
		Function: oai.FunctionCall{
			Name:      "get_weather",
			Arguments: `{"city":"Berlin"}`,
		},
	}

	part := convert.ToolCallToPart(tc)

	if part.FunctionCall == nil {
		t.Fatal("expected FunctionCall part")
	}

	if part.FunctionCall.ID != "call_123" {
		t.Errorf("ID = %q, want call_123", part.FunctionCall.ID)
	}

	if part.FunctionCall.Name != "get_weather" {
		t.Errorf("Name = %q, want get_weather", part.FunctionCall.Name)
	}

	city, _ := part.FunctionCall.Args["city"].(string)
	if city != "Berlin" {
		t.Errorf("city = %q, want Berlin", city)
	}
}

func TestExtractParams(t *testing.T) {
	t.Parallel()

	seed := int32(42)
	topK := float32(10)
	logprobs := int32(5)

	tests := []struct {
		name   string
		cfg    *genai.GenerateContentConfig
		assert func(t *testing.T, p oai.RequestParams)
	}{
		{
			name: "seed and topK",
			cfg:  &genai.GenerateContentConfig{Seed: &seed, TopK: &topK},
			assert: func(t *testing.T, p oai.RequestParams) {
				t.Helper()

				if p.Seed == nil || *p.Seed != 42 {
					t.Errorf("Seed = %v, want 42", p.Seed)
				}

				if p.TopK == nil || *p.TopK != 10 {
					t.Errorf("TopK = %v, want 10", p.TopK)
				}
			},
		},
		{
			name: "logprobs",
			cfg:  &genai.GenerateContentConfig{ResponseLogprobs: true, Logprobs: &logprobs},
			assert: func(t *testing.T, p oai.RequestParams) {
				t.Helper()

				if !p.ResponseLogprobs {
					t.Error("ResponseLogprobs = false, want true")
				}

				if p.Logprobs == nil || *p.Logprobs != 5 {
					t.Errorf("Logprobs = %v, want 5", p.Logprobs)
				}
			},
		},
		{
			name: "tool config",
			cfg: &genai.GenerateContentConfig{
				ToolConfig: &genai.ToolConfig{
					FunctionCallingConfig: &genai.FunctionCallingConfig{
						Mode: genai.FunctionCallingConfigModeAny,
					},
				},
			},
			assert: func(t *testing.T, p oai.RequestParams) {
				t.Helper()

				if p.ToolChoice != "required" {
					t.Errorf("ToolChoice = %v, want \"required\"", p.ToolChoice)
				}
			},
		},
		{
			name: "nil config",
			cfg:  nil,
			assert: func(t *testing.T, p oai.RequestParams) {
				t.Helper()

				if p.Seed != nil {
					t.Errorf("Seed = %v, want nil", p.Seed)
				}

				if p.TopK != nil {
					t.Errorf("TopK = %v, want nil", p.TopK)
				}

				if p.ResponseLogprobs {
					t.Error("ResponseLogprobs = true, want false")
				}

				if p.ToolChoice != nil {
					t.Errorf("ToolChoice = %v, want nil", p.ToolChoice)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			params := convert.ExtractParams(tt.cfg, nil, "test-model", "fallback", nil)
			tt.assert(t, params)
		})
	}
}

func TestToolChoice(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		cfg    *genai.ToolConfig
		assert func(t *testing.T, got any)
	}{
		{
			name: "nil config",
			cfg:  nil,
			assert: func(t *testing.T, got any) {
				t.Helper()

				if got != nil {
					t.Errorf("expected nil, got %v", got)
				}
			},
		},
		{
			name: "auto",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAuto,
				},
			},
			assert: func(t *testing.T, got any) {
				t.Helper()

				if got != "auto" {
					t.Errorf("got %v, want \"auto\"", got)
				}
			},
		},
		{
			name: "none",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeNone,
				},
			},
			assert: func(t *testing.T, got any) {
				t.Helper()

				if got != "none" {
					t.Errorf("got %v, want \"none\"", got)
				}
			},
		},
		{
			name: "any without names",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
				},
			},
			assert: func(t *testing.T, got any) {
				t.Helper()

				if got != "required" {
					t.Errorf("got %v, want \"required\"", got)
				}
			},
		},
		{
			name: "any with one name",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"get_weather"},
				},
			},
			assert: func(t *testing.T, got any) {
				t.Helper()

				m, ok := got.(map[string]any)
				if !ok {
					t.Fatalf("expected map, got %T", got)
				}

				if m["type"] != "function" {
					t.Errorf("type = %v, want function", m["type"])
				}

				fn, ok := m["function"].(map[string]any)
				if !ok {
					t.Fatal("function not a map")
				}

				if fn["name"] != "get_weather" {
					t.Errorf("name = %v, want get_weather", fn["name"])
				}
			},
		},
		{
			name: "any with multiple names falls back to required",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{"fn_a", "fn_b"},
				},
			},
			assert: func(t *testing.T, got any) {
				t.Helper()

				if got != "required" {
					t.Errorf("got %v, want \"required\"", got)
				}
			},
		},
		{
			name: "unspecified mode returns nil",
			cfg: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeUnspecified,
				},
			},
			assert: func(t *testing.T, got any) {
				t.Helper()

				if got != nil {
					t.Errorf("expected nil, got %v", got)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.assert(t, convert.ToolChoice(tt.cfg))
		})
	}
}

func TestLogprobs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		input  *oai.ChatLogprobs
		assert func(t *testing.T, got *genai.LogprobsResult)
	}{
		{
			name:  "nil input",
			input: nil,
			assert: func(t *testing.T, got *genai.LogprobsResult) {
				t.Helper()

				if got != nil {
					t.Errorf("expected nil, got %+v", got)
				}
			},
		},
		{
			name:  "empty content",
			input: &oai.ChatLogprobs{Content: []oai.TokenLogprob{}},
			assert: func(t *testing.T, got *genai.LogprobsResult) {
				t.Helper()

				if got != nil {
					t.Errorf("expected nil for empty content, got %+v", got)
				}
			},
		},
		{
			name: "with tokens and top candidates",
			input: &oai.ChatLogprobs{
				Content: []oai.TokenLogprob{
					{
						Token:   "Hello",
						Logprob: -0.5,
						TokenID: 9906,
						TopLogprobs: []oai.TopTokenLogprob{
							{Token: "Hello", Logprob: -0.5, TokenID: 9906},
							{Token: "Hi", Logprob: -1.2, TokenID: 17250},
						},
					},
					{
						Token:   " world",
						Logprob: -0.3,
						TokenID: 1917,
						TopLogprobs: []oai.TopTokenLogprob{
							{Token: " world", Logprob: -0.3, TokenID: 1917},
						},
					},
				},
			},
			assert: func(t *testing.T, got *genai.LogprobsResult) {
				t.Helper()

				if got == nil {
					t.Fatal("expected non-nil LogprobsResult")
				}

				if len(got.ChosenCandidates) != 2 {
					t.Fatalf("ChosenCandidates len = %d, want 2", len(got.ChosenCandidates))
				}

				if got.ChosenCandidates[0].Token != "Hello" {
					t.Errorf("token[0] = %q, want Hello", got.ChosenCandidates[0].Token)
				}

				if got.ChosenCandidates[0].TokenID != 9906 {
					t.Errorf("tokenID[0] = %d, want 9906", got.ChosenCandidates[0].TokenID)
				}

				if got.ChosenCandidates[0].LogProbability != -0.5 {
					t.Errorf("logprob[0] = %v, want -0.5", got.ChosenCandidates[0].LogProbability)
				}

				if len(got.TopCandidates) != 2 {
					t.Fatalf("TopCandidates len = %d, want 2", len(got.TopCandidates))
				}

				if len(got.TopCandidates[0].Candidates) != 2 {
					t.Errorf("TopCandidates[0] len = %d, want 2", len(got.TopCandidates[0].Candidates))
				}

				if got.TopCandidates[0].Candidates[1].Token != "Hi" {
					t.Errorf("top[0][1].Token = %q, want Hi", got.TopCandidates[0].Candidates[1].Token)
				}

				if got.TopCandidates[0].Candidates[1].TokenID != 17250 {
					t.Errorf("top[0][1].TokenID = %d, want 17250", got.TopCandidates[0].Candidates[1].TokenID)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.assert(t, convert.Logprobs(tt.input))
		})
	}
}

func TestChoiceToResponse_Logprobs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		logprobs *oai.ChatLogprobs
		wantNil  bool
	}{
		{
			name:     "without logprobs",
			logprobs: nil,
			wantNil:  true,
		},
		{
			name: "with logprobs",
			logprobs: &oai.ChatLogprobs{
				Content: []oai.TokenLogprob{
					{Token: "Hi", Logprob: -0.1, TokenID: 100},
				},
			},
			wantNil: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			choice := oai.ChatChoice{
				Message: oai.ChatMessage{
					Role:    "assistant",
					Content: strPtr("Hi"),
				},
				FinishReason: "stop",
				Logprobs:     tt.logprobs,
			}

			resp := convert.ChoiceToResponse(choice, nil, "gpt-4.1")

			if tt.wantNil {
				if resp.LogprobsResult != nil {
					t.Errorf("expected nil LogprobsResult, got %+v", resp.LogprobsResult)
				}

				return
			}

			if resp.LogprobsResult == nil {
				t.Fatal("expected LogprobsResult")
			}

			if len(resp.LogprobsResult.ChosenCandidates) != 1 {
				t.Fatalf("ChosenCandidates len = %d, want 1", len(resp.LogprobsResult.ChosenCandidates))
			}

			if resp.LogprobsResult.ChosenCandidates[0].Token != "Hi" {
				t.Errorf("token = %q, want Hi", resp.LogprobsResult.ChosenCandidates[0].Token)
			}

			if resp.LogprobsResult.ChosenCandidates[0].TokenID != 100 {
				t.Errorf("tokenID = %d, want 100", resp.LogprobsResult.ChosenCandidates[0].TokenID)
			}
		})
	}
}
