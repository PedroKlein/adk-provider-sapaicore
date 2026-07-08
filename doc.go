// Package sapaicore implements the ADK Go v2 [model.LLM] interface for SAP AI Core.
//
// It bridges Google's Agent Development Kit with SAP AI Core's inference API,
// supporting two modes of operation:
//
//   - Orchestration: a single deployment handles all models via the SAP AI Core
//     harmonized API. Supports extended thinking, response format, tool calling,
//     server-side timeout/retry, content filtering, data masking, translation,
//     model fallback, and prompt caching.
//
//   - Foundation-models: per-model deployment IDs with a direct OpenAI-compatible
//     chat completions API.
//
// # Quick Start
//
//	provider, err := sapaicore.NewProvider(
//	    sapaicore.WithEndpoint("https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com"),
//	    sapaicore.WithAuth(clientID, clientSecret, authURL),
//	    sapaicore.WithOrchestration(), // auto-discovers deployment
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	llm, err := provider.Model("gpt-4.1")
//
// Both streaming and non-streaming generation are supported via the standard
// ADK [model.LLM] interface.
package sapaicore
