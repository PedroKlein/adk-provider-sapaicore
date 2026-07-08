package convert_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"google.golang.org/genai"

	"github.com/PedroKlein/adk-provider-sapaicore/internal/convert"
	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

//nolint:gocognit // table-driven test with per-case assertions requires inline checks
func TestContentBlocks(t *testing.T) {
	t.Parallel()

	pngData := []byte{0x89, 0x50, 0x4E, 0x47}
	pdfData := []byte("%PDF-1.4 test content")

	tests := []struct {
		name      string
		parts     []*genai.Part
		wantNil   bool
		wantErr   bool
		wantCount int
		check     func(t *testing.T, blocks []oai.ContentBlock)
	}{
		{
			name:    "text-only returns nil",
			parts:   []*genai.Part{{Text: "hello"}},
			wantNil: true,
		},
		{
			name:    "thought parts return nil",
			parts:   []*genai.Part{{Text: "thinking...", Thought: true}},
			wantNil: true,
		},
		{
			name:    "function call returns nil",
			parts:   []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "test"}}},
			wantNil: true,
		},
		{
			name: "InlineData image produces image_url block",
			parts: []*genai.Part{{
				InlineData: &genai.Blob{Data: pngData, MIMEType: "image/png"},
			}},
			wantCount: 1,
			check: func(t *testing.T, blocks []oai.ContentBlock) {
				t.Helper()

				b, ok := blocks[0].(oai.ImageURLContentBlock)
				if !ok {
					t.Fatalf("block type = %T, want ImageURLContentBlock", blocks[0])
				}

				if b.Type != "image_url" {
					t.Errorf("type = %q, want image_url", b.Type)
				}

				wantURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngData)
				if b.ImageURL.URL != wantURI {
					t.Errorf("url = %q, want %q", b.ImageURL.URL, wantURI)
				}
			},
		},
		{
			name: "InlineData PDF produces file block",
			parts: []*genai.Part{{
				InlineData: &genai.Blob{Data: pdfData, MIMEType: "application/pdf"},
			}},
			wantCount: 1,
			check: func(t *testing.T, blocks []oai.ContentBlock) {
				t.Helper()

				b, ok := blocks[0].(oai.FileContentBlock)
				if !ok {
					t.Fatalf("block type = %T, want FileContentBlock", blocks[0])
				}

				if b.Type != "file" {
					t.Errorf("type = %q, want file", b.Type)
				}

				wantURI := "data:application/pdf;base64," + base64.StdEncoding.EncodeToString(pdfData)
				if b.File.FileData != wantURI {
					t.Errorf("file_data = %q, want %q", b.File.FileData, wantURI)
				}
			},
		},
		{
			name: "FileData HTTP URL passthrough for image",
			parts: []*genai.Part{{
				FileData: &genai.FileData{
					FileURI:  "https://example.com/image.png",
					MIMEType: "image/png",
				},
			}},
			wantCount: 1,
			check: func(t *testing.T, blocks []oai.ContentBlock) {
				t.Helper()

				b, ok := blocks[0].(oai.ImageURLContentBlock)
				if !ok {
					t.Fatalf("block type = %T, want ImageURLContentBlock", blocks[0])
				}

				if b.ImageURL.URL != "https://example.com/image.png" {
					t.Errorf("url = %q, want https://example.com/image.png", b.ImageURL.URL)
				}
			},
		},
		{
			name: "FileData HTTP URL passthrough for PDF",
			parts: []*genai.Part{{
				FileData: &genai.FileData{
					FileURI:     "https://example.com/doc.pdf",
					MIMEType:    "application/pdf",
					DisplayName: "report.pdf",
				},
			}},
			wantCount: 1,
			check: func(t *testing.T, blocks []oai.ContentBlock) {
				t.Helper()

				b, ok := blocks[0].(oai.FileContentBlock)
				if !ok {
					t.Fatalf("block type = %T, want FileContentBlock", blocks[0])
				}

				if b.File.FileData != "https://example.com/doc.pdf" {
					t.Errorf("file_data = %q, want URL", b.File.FileData)
				}

				if b.File.Filename != "report.pdf" {
					t.Errorf("filename = %q, want report.pdf", b.File.Filename)
				}
			},
		},
		{
			name: "FileData gs:// returns error",
			parts: []*genai.Part{{
				FileData: &genai.FileData{
					FileURI:  "gs://bucket/object.png",
					MIMEType: "image/png",
				},
			}},
			wantErr: true,
		},
		{
			name: "FileData empty URI returns error",
			parts: []*genai.Part{{
				FileData: &genai.FileData{
					FileURI:  "",
					MIMEType: "image/png",
				},
			}},
			wantErr: true,
		},
		{
			name: "FileData custom scheme returns error",
			parts: []*genai.Part{{
				FileData: &genai.FileData{
					FileURI:  "ftp://server/file.pdf",
					MIMEType: "application/pdf",
				},
			}},
			wantErr: true,
		},
		{
			name: "mixed text + image produces array with both",
			parts: []*genai.Part{
				{Text: "Describe this:"},
				{InlineData: &genai.Blob{Data: pngData, MIMEType: "image/png"}},
			},
			wantCount: 2,
			check: func(t *testing.T, blocks []oai.ContentBlock) {
				t.Helper()

				textBlock, ok := blocks[0].(oai.TextContentBlock)
				if !ok {
					t.Fatalf("blocks[0] type = %T, want TextContentBlock", blocks[0])
				}

				if textBlock.Text != "Describe this:" {
					t.Errorf("text = %q, want 'Describe this:'", textBlock.Text)
				}

				_, ok = blocks[1].(oai.ImageURLContentBlock)
				if !ok {
					t.Fatalf("blocks[1] type = %T, want ImageURLContentBlock", blocks[1])
				}
			},
		},
		{
			name: "image MIME case-insensitive",
			parts: []*genai.Part{{
				InlineData: &genai.Blob{Data: pngData, MIMEType: "Image/PNG"},
			}},
			wantCount: 1,
			check: func(t *testing.T, blocks []oai.ContentBlock) {
				t.Helper()

				_, ok := blocks[0].(oai.ImageURLContentBlock)
				if !ok {
					t.Fatalf("block type = %T, want ImageURLContentBlock", blocks[0])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			blocks, err := convert.ContentBlocks(tt.parts)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				if !errors.Is(err, convert.ErrUnsupportedURI) {
					t.Errorf("error = %v, want ErrUnsupportedURI", err)
				}

				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.wantNil {
				if blocks != nil {
					t.Fatalf("expected nil blocks, got %d", len(blocks))
				}

				return
			}

			if len(blocks) != tt.wantCount {
				t.Fatalf("blocks count = %d, want %d", len(blocks), tt.wantCount)
			}

			if tt.check != nil {
				tt.check(t, blocks)
			}
		})
	}
}

func TestMessages_MultimodalProducesContentArray(t *testing.T) {
	t.Parallel()

	pngData := []byte{0x89, 0x50, 0x4E, 0x47}

	contents := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "What is this?"},
				{InlineData: &genai.Blob{Data: pngData, MIMEType: "image/png"}},
			},
		},
	}

	msgs, err := convert.Messages(nil, contents)
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}

	if len(msgs) != 1 {
		t.Fatalf("messages count = %d, want 1", len(msgs))
	}

	msg := msgs[0]

	if msg.Role != "user" {
		t.Errorf("role = %q, want user", msg.Role)
	}

	// Content should be []oai.ContentBlock, not *string.
	blocks, ok := msg.Content.([]oai.ContentBlock)
	if !ok {
		t.Fatalf("content type = %T, want []oai.ContentBlock", msg.Content)
	}

	if len(blocks) != 2 {
		t.Fatalf("blocks = %d, want 2", len(blocks))
	}

	// Verify JSON serialization produces the correct wire format.
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var raw map[string]any

	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	contentArr, ok := raw["content"].([]any)
	if !ok {
		t.Fatalf("JSON content type = %T, want array", raw["content"])
	}

	if len(contentArr) != 2 {
		t.Fatalf("JSON content len = %d, want 2", len(contentArr))
	}

	first := contentArr[0].(map[string]any)
	if first["type"] != "text" {
		t.Errorf("first block type = %v, want text", first["type"])
	}

	second := contentArr[1].(map[string]any)
	if second["type"] != "image_url" {
		t.Errorf("second block type = %v, want image_url", second["type"])
	}
}

func TestMessages_TextOnlyStaysString(t *testing.T) {
	t.Parallel()

	contents := []*genai.Content{
		{
			Role:  "user",
			Parts: []*genai.Part{{Text: "Hello world"}},
		},
	}

	msgs, err := convert.Messages(nil, contents)
	if err != nil {
		t.Fatalf("Messages: %v", err)
	}

	if len(msgs) != 1 {
		t.Fatalf("messages count = %d, want 1", len(msgs))
	}

	s, ok := msgs[0].Content.(*string)
	if !ok {
		t.Fatalf("content type = %T, want *string", msgs[0].Content)
	}

	if *s != "Hello world" {
		t.Errorf("content = %q, want 'Hello world'", *s)
	}
}

func TestMessages_FileDataErrorPropagates(t *testing.T) {
	t.Parallel()

	contents := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{FileData: &genai.FileData{FileURI: "gs://bucket/file", MIMEType: "image/png"}},
			},
		},
	}

	_, err := convert.Messages(nil, contents)
	if err == nil {
		t.Fatal("expected error for gs:// URI")
	}

	if !errors.Is(err, convert.ErrUnsupportedURI) {
		t.Errorf("error = %v, want ErrUnsupportedURI", err)
	}
}
