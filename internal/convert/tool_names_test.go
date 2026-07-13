package convert_test

import (
	"testing"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/convert"
	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

func TestSanitizeToolNames(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		tools         []oai.ToolDef
		wantNames     []string
		wantMapping   convert.ToolNameMapping
		wantInputName string // verify original slice is unmodified (first tool)
	}{
		{
			name: "replaces dots with double underscore",
			tools: []oai.ToolDef{
				{Type: "function", Function: oai.FunctionDef{Name: "github.read-pr"}},
				{Type: "function", Function: oai.FunctionDef{Name: "github.list-changed-files"}},
				{Type: "function", Function: oai.FunctionDef{Name: "simple_tool"}},
			},
			wantNames: []string{"github__read-pr", "github__list-changed-files", "simple_tool"},
			wantMapping: convert.ToolNameMapping{
				"github__read-pr":            "github.read-pr",
				"github__list-changed-files": "github.list-changed-files",
			},
			wantInputName: "github.read-pr",
		},
		{
			name: "no dots returns nil mapping",
			tools: []oai.ToolDef{
				{Type: "function", Function: oai.FunctionDef{Name: "get_weather"}},
				{Type: "function", Function: oai.FunctionDef{Name: "search"}},
			},
			wantNames:     []string{"get_weather", "search"},
			wantMapping:   nil,
			wantInputName: "get_weather",
		},
		{
			name: "multiple dots in one name",
			tools: []oai.ToolDef{
				{Type: "function", Function: oai.FunctionDef{Name: "namespace.category.action"}},
			},
			wantNames: []string{"namespace__category__action"},
			wantMapping: convert.ToolNameMapping{
				"namespace__category__action": "namespace.category.action",
			},
			wantInputName: "namespace.category.action",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			sanitized, mapping := convert.SanitizeToolNames(tt.tools)

			// Verify output names.
			for i, want := range tt.wantNames {
				if sanitized[i].Function.Name != want {
					t.Errorf("sanitized[%d].Name = %q, want %q", i, sanitized[i].Function.Name, want)
				}
			}

			// Verify mapping.
			if tt.wantMapping == nil {
				if mapping != nil {
					t.Errorf("mapping = %v, want nil", mapping)
				}
			} else {
				if len(mapping) != len(tt.wantMapping) {
					t.Fatalf("mapping len = %d, want %d", len(mapping), len(tt.wantMapping))
				}

				for k, v := range tt.wantMapping {
					if mapping[k] != v {
						t.Errorf("mapping[%q] = %q, want %q", k, mapping[k], v)
					}
				}
			}

			// Verify input was not mutated.
			if tt.tools[0].Function.Name != tt.wantInputName {
				t.Errorf("input tools[0].Name = %q, want %q (should be unmodified)",
					tt.tools[0].Function.Name, tt.wantInputName)
			}
		})
	}
}

func TestToolNameMapping_Restore(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		mapping convert.ToolNameMapping
		input   string
		want    string
	}{
		{
			name:    "known sanitized name",
			mapping: convert.ToolNameMapping{"github__read-pr": "github.read-pr"},
			input:   "github__read-pr",
			want:    "github.read-pr",
		},
		{
			name:    "unknown name returned unchanged",
			mapping: convert.ToolNameMapping{"github__read-pr": "github.read-pr"},
			input:   "other_tool",
			want:    "other_tool",
		},
		{
			name:    "nil mapping returned unchanged",
			mapping: nil,
			input:   "github__read-pr",
			want:    "github__read-pr",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := tt.mapping.Restore(tt.input); got != tt.want {
				t.Errorf("Restore(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestSanitizeMessageToolNames(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		messages []oai.ChatMessage
		mapping  convert.ToolNameMapping
		want     []string // expected function names in tool_calls order
	}{
		{
			name: "rewrites dotted names in tool_calls",
			messages: []oai.ChatMessage{
				{Role: "user", Content: "hello"},
				{
					Role: "assistant",
					ToolCalls: []oai.ToolCall{
						{ID: "call_1", Type: "function", Function: oai.FunctionCall{Name: "github.read-pr"}},
						{ID: "call_2", Type: "function", Function: oai.FunctionCall{Name: "simple_tool"}},
					},
				},
				{Role: "tool", ToolCallID: "call_1", Content: "result"},
			},
			mapping: convert.ToolNameMapping{"github__read-pr": "github.read-pr"},
			want:    []string{"github__read-pr", "simple_tool"},
		},
		{
			name: "nil mapping is no-op",
			messages: []oai.ChatMessage{
				{
					Role: "assistant",
					ToolCalls: []oai.ToolCall{
						{ID: "call_1", Type: "function", Function: oai.FunctionCall{Name: "github.read-pr"}},
					},
				},
			},
			mapping: nil,
			want:    []string{"github.read-pr"},
		},
		{
			name: "no tool_calls unchanged",
			messages: []oai.ChatMessage{
				{Role: "user", Content: "hello"},
				{Role: "assistant", Content: "hi"},
			},
			mapping: convert.ToolNameMapping{"github__read-pr": "github.read-pr"},
			want:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := convert.SanitizeMessageToolNames(tt.messages, tt.mapping)

			var gotNames []string

			for _, msg := range result {
				for _, tc := range msg.ToolCalls {
					gotNames = append(gotNames, tc.Function.Name)
				}
			}

			if len(gotNames) != len(tt.want) {
				t.Fatalf("got %d tool call names %v, want %d %v", len(gotNames), gotNames, len(tt.want), tt.want)
			}

			for i, want := range tt.want {
				if gotNames[i] != want {
					t.Errorf("tool_calls[%d].Name = %q, want %q", i, gotNames[i], want)
				}
			}
		})
	}
}
