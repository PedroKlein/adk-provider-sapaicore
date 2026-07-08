//go:build smoke

package smoketest_test

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
	"google.golang.org/adk/v2/model"
)

func testdataPath(name string) string {
	_, file, _, _ := runtime.Caller(0)

	return filepath.Join(filepath.Dir(file), "testdata", name)
}

func loadTestdata(t *testing.T, name string) []byte {
	t.Helper()

	data, err := os.ReadFile(testdataPath(name))
	if err != nil {
		t.Fatalf("loading testdata/%s: %v", name, err)
	}

	return data
}

func TestSmoke_ImageInput_InlineData(t *testing.T) {
	provider := newProvider(t)
	redPNG := loadTestdata(t, "red.png")

	visionModels := []string{
		"gpt-4.1-mini",
		"anthropic--claude-4.5-haiku",
		"gemini-2.5-flash",
	}

	for _, modelName := range visionModels {
		t.Run(modelName, func(t *testing.T) {
			ctx := withTimeout(t, 45*time.Second)

			llm, err := provider.Model(modelName)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "What color is this image? Reply with just the color name, one word."},
							{InlineData: &genai.Blob{Data: redPNG, MIMEType: "image/png"}},
						},
					},
				},
			}

			resp := generateOne(t, ctx, llm, req)
			text := strings.ToLower(requireText(t, resp))

			if !strings.Contains(text, "red") {
				t.Errorf("expected 'red' in response for red pixel image, got: %q", text)
			}

			t.Logf("model=%s response=%q", modelName, text)
		})
	}
}

func TestSmoke_ImageInput_Streaming(t *testing.T) {
	ctx := withTimeout(t, 45*time.Second)
	provider := newProvider(t)
	redPNG := loadTestdata(t, "red.png")

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What color is this image? Reply with just the color name."},
					{InlineData: &genai.Blob{Data: redPNG, MIMEType: "image/png"}},
				},
			},
		},
	}

	partials, final := generateStream(t, ctx, llm, req)
	text := strings.ToLower(requireText(t, final))

	if !strings.Contains(text, "red") {
		t.Errorf("expected 'red' in streaming response, got: %q", text)
	}

	if len(partials) == 0 {
		t.Error("expected partial chunks in streaming, got none")
	}

	t.Logf("partials=%d final=%q", len(partials), text)
}

func TestSmoke_FileInput_PDF(t *testing.T) {
	ctx := withTimeout(t, 60*time.Second)
	provider := newProvider(t)
	testPDF := loadTestdata(t, "test.pdf")

	pdfModels := []string{
		"anthropic--claude-4.5-haiku",
		"gemini-2.5-flash",
	}

	for _, modelName := range pdfModels {
		t.Run(modelName, func(t *testing.T) {
			llm, err := provider.Model(modelName)
			if err != nil {
				t.Fatalf("Model: %v", err)
			}

			req := &model.LLMRequest{
				Contents: []*genai.Content{
					{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "What word is written in this PDF document? Reply with just the word."},
							{InlineData: &genai.Blob{Data: testPDF, MIMEType: "application/pdf"}},
						},
					},
				},
			}

			resp := generateOne(t, ctx, llm, req)
			text := strings.ToUpper(requireText(t, resp))

			if !strings.Contains(text, "BANANA") {
				t.Errorf("expected 'BANANA' in response for PDF content, got: %q", requireText(t, resp))
			}

			t.Logf("model=%s response=%q", modelName, requireText(t, resp))
		})
	}
}

func TestSmoke_ImageInput_FileDataURL(t *testing.T) {
	// Some SAP AI Core configurations may not support fetching external URLs.
	ctx := withTimeout(t, 45*time.Second)
	provider := newProvider(t)

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	imageURL := "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/100px-PNG_transparency_demonstration_1.png"

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Briefly describe this image in one sentence."},
					{FileData: &genai.FileData{FileURI: imageURL, MIMEType: "image/png"}},
				},
			},
		},
	}

	var resp *model.LLMResponse

	for r, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			if strings.Contains(err.Error(), "400") {
				t.Skipf("FileData URL not supported by this deployment: %v", err)
			}

			t.Fatalf("GenerateContent: %v", err)
		}

		resp = r
	}

	text := requireText(t, resp)

	if len(text) < 10 {
		t.Errorf("expected meaningful description (>10 chars), got: %q", text)
	}

	t.Logf("FileData URL response=%q", text)
}

func TestSmoke_ImageInput_FoundationMode(t *testing.T) {
	ctx := withTimeout(t, 45*time.Second)
	provider := newFoundationProvider(t)
	redPNG := loadTestdata(t, "red.png")

	llm, err := provider.Model("gpt-4.1-mini")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What color is this image? Reply with just the color name, one word."},
					{InlineData: &genai.Blob{Data: redPNG, MIMEType: "image/png"}},
				},
			},
		},
	}

	resp := generateOne(t, ctx, llm, req)
	text := strings.ToLower(requireText(t, resp))

	if !strings.Contains(text, "red") {
		t.Errorf("expected 'red' in foundation mode response, got: %q", text)
	}

	t.Logf("foundation mode response=%q", text)
}

func TestSmoke_FabricatedMasking_RedactsPII(t *testing.T) {
	// Verifies fabricated_data replacement strategy is accepted by the API.
	// With anonymization: replaces with placeholders, does NOT unmask output.
	// With pseudonymization: replaces with fakes, DOES unmask output (restores original).
	ctx := withTimeout(t, 60*time.Second)

	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")

	testEmail := "bartholomew.henderson.test42@example-corp.com"
	prompt := "Please repeat this email address exactly as written: " + testEmail

	// anonymization + fabricated_data → email should NOT appear in output.
	maskedAnon, err := sapaicore.NewProvider(ctx,
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithMasking(sapaicore.MaskingConfig{
			Method: sapaicore.Anonymization,
			Entities: []sapaicore.MaskingEntity{
				sapaicore.FabricatedEntity(sapaicore.EntityEmail),
			},
		}),
	)
	if err != nil {
		t.Fatalf("NewProvider (anonymization): %v", err)
	}

	llmAnon, _ := maskedAnon.Model("gpt-4.1-mini")
	respAnon := generateOne(t, ctx, llmAnon, simpleReq(prompt))
	textAnon := requireText(t, respAnon)

	if strings.Contains(textAnon, testEmail) {
		t.Errorf("anonymization masking failed: response contains original email in: %q", textAnon)
	}

	if strings.Contains(textAnon, "example-corp") {
		t.Errorf("anonymization masking failed: response contains 'example-corp' in: %q", textAnon)
	}

	t.Logf("anonymization+fabricated=%q", textAnon)

	// pseudonymization + fabricated_data → API accepts config (output is unmasked back).
	maskedPseudo, err := sapaicore.NewProvider(ctx,
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithMasking(sapaicore.MaskingConfig{
			Method: sapaicore.Pseudonymization,
			Entities: []sapaicore.MaskingEntity{
				sapaicore.FabricatedEntity(sapaicore.EntityEmail),
			},
		}),
	)
	if err != nil {
		t.Fatalf("NewProvider (pseudonymization): %v", err)
	}

	llmPseudo, _ := maskedPseudo.Model("gpt-4.1-mini")
	respPseudo := generateOne(t, ctx, llmPseudo, simpleReq(prompt))
	textPseudo := requireText(t, respPseudo)

	if textPseudo == "" {
		t.Error("pseudonymization+fabricated produced empty response")
	}

	t.Logf("pseudonymization+fabricated=%q", textPseudo)
}

func TestSmoke_FileInput_WithMasking(t *testing.T) {
	ctx := withTimeout(t, 60*time.Second)

	endpoint := envOrSkip(t, "AI_CORE_ENDPOINT")
	clientID := envOrSkip(t, "AI_CORE_CLIENT_ID")
	clientSecret := envOrSkip(t, "AI_CORE_CLIENT_SECRET")
	authURL := envOrSkip(t, "AI_CORE_AUTH_URL")

	testPDF := loadTestdata(t, "test.pdf")

	provider, err := sapaicore.NewProvider(ctx,
		sapaicore.WithEndpoint(endpoint),
		sapaicore.WithAuth(clientID, clientSecret, authURL),
		sapaicore.WithMasking(sapaicore.MaskingConfig{
			Method:              sapaicore.Anonymization,
			Entities:            sapaicore.StandardEntities(sapaicore.CommonPIIEntities),
			MaskFileInputMethod: sapaicore.MaskFileSkip,
		}),
	)
	if err != nil {
		t.Fatalf("NewProvider: %v", err)
	}

	llm, err := provider.Model("gemini-2.5-flash")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "What word is written in this PDF? Reply with just the word."},
					{InlineData: &genai.Blob{Data: testPDF, MIMEType: "application/pdf"}},
				},
			},
		},
	}

	resp := generateOne(t, ctx, llm, req)
	text := strings.ToUpper(requireText(t, resp))

	if !strings.Contains(text, "BANANA") {
		t.Errorf("expected 'BANANA' from PDF with MaskFileSkip, got: %q", requireText(t, resp))
	}

	t.Logf("file+masking response=%q", requireText(t, resp))
}
