package convert

import (
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"

	oai "github.com/PedroKlein/adk-provider-sapaicore/internal/openai"
)

var ErrUnsupportedURI = errors.New("unsupported URI scheme")

// ContentBlocks converts genai parts into typed wire content blocks.
// Returns nil when all parts are text-only, signaling the caller to use plain string content.
func ContentBlocks(parts []*genai.Part) ([]oai.ContentBlock, error) {
	var (
		blocks        []oai.ContentBlock
		hasMultimodal bool
	)

	for _, part := range parts {
		switch {
		case part.InlineData != nil:
			hasMultimodal = true

			blocks = append(blocks, inlineDataBlock(part.InlineData))

		case part.FileData != nil:
			hasMultimodal = true

			block, err := fileDataBlock(part.FileData)
			if err != nil {
				return nil, err
			}

			blocks = append(blocks, block)

		case part.Text != "" && !part.Thought:
			blocks = append(blocks, oai.TextContentBlock{
				Type: "text",
				Text: part.Text,
			})
		}
	}

	if !hasMultimodal {
		return nil, nil
	}

	return blocks, nil
}

func inlineDataBlock(blob *genai.Blob) oai.ContentBlock {
	uri := dataURI(blob.Data, blob.MIMEType)

	if isImageMIME(blob.MIMEType) {
		return oai.ImageURLContentBlock{
			Type:     "image_url",
			ImageURL: oai.ImageURL{URL: uri},
		}
	}

	return oai.FileContentBlock{
		Type: "file",
		File: oai.FileContent{FileData: uri},
	}
}

func fileDataBlock(fd *genai.FileData) (oai.ContentBlock, error) {
	if err := validateFileURI(fd.FileURI); err != nil {
		return nil, err
	}

	if isImageMIME(fd.MIMEType) {
		return oai.ImageURLContentBlock{
			Type:     "image_url",
			ImageURL: oai.ImageURL{URL: fd.FileURI},
		}, nil
	}

	return oai.FileContentBlock{
		Type: "file",
		File: oai.FileContent{
			FileData: fd.FileURI,
			Filename: fd.DisplayName,
		},
	}, nil
}

func dataURI(data []byte, mimeType string) string {
	encoded := base64.StdEncoding.EncodeToString(data)
	return "data:" + mimeType + ";base64," + encoded
}

func isImageMIME(mime string) bool {
	return strings.HasPrefix(strings.ToLower(mime), "image/")
}

func validateFileURI(uri string) error {
	lower := strings.ToLower(uri)

	switch {
	case strings.HasPrefix(lower, "https://"), strings.HasPrefix(lower, "http://"):
		return nil
	case strings.HasPrefix(lower, "gs://"):
		return fmt.Errorf("%w: gs:// URIs are not supported by SAP AI Core, use InlineData with bytes or an HTTP/HTTPS URL", ErrUnsupportedURI)
	case uri == "":
		return fmt.Errorf("%w: empty file URI", ErrUnsupportedURI)
	default:
		return fmt.Errorf("%w: %q is not supported, use an HTTP/HTTPS URL", ErrUnsupportedURI, uri)
	}
}
