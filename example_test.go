package sapaicore_test

import (
	"fmt"
	"log"

	sapaicore "github.com/PedroKlein/adk-provider-sapaicore"
)

func ExampleNewProvider_orchestration() {
	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint("https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com"),
		sapaicore.WithAuth("client-id", "client-secret", "https://auth.example.com/oauth/token"),
		sapaicore.WithDeploymentID("d1234abc"), // or use WithOrchestration() for auto-discovery
	)
	if err != nil {
		log.Fatal(err)
	}

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(llm.Name())
	// Output: gpt-4.1
}

func ExampleNewProvider_foundationModels() {
	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint("https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com"),
		sapaicore.WithAuth("client-id", "client-secret", "https://auth.example.com/oauth/token"),
		sapaicore.WithDeployments(map[string]string{
			"gpt-4.1":      "d1234abc",
			"gpt-4.1-mini": "d5678def",
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	llm, err := provider.Model("gpt-4.1")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(llm.Name())
	// Output: gpt-4.1
}

func ExampleWithModelParams() {
	provider, err := sapaicore.NewProvider(
		sapaicore.WithEndpoint("https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com"),
		sapaicore.WithAuth("client-id", "client-secret", "https://auth.example.com/oauth/token"),
		sapaicore.WithDeploymentID("d1234abc"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Enable extended thinking for Claude models.
	llm, err := provider.Model("anthropic--claude-4.5-sonnet",
		sapaicore.WithModelParams(map[string]any{
			"thinking":   map[string]any{"type": "enabled", "budget_tokens": 16384},
			"max_tokens": 64000,
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(llm.Name())
	// Output: anthropic--claude-4.5-sonnet
}
