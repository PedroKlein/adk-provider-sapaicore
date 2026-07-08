package sapaicore

// AzureThreshold controls Azure Content Safety sensitivity.
// Values map to SAP AI Core API thresholds (0/2/4/6): lower = stricter.
type AzureThreshold int

const (
	AzureThresholdAllowSafe          AzureThreshold = 0
	AzureThresholdAllowSafeLow       AzureThreshold = 2
	AzureThresholdAllowSafeLowMedium AzureThreshold = 4
	AzureThresholdAllowAll           AzureThreshold = 6
)

type AzureContentSafetyConfig struct {
	Hate     AzureThreshold
	SelfHarm AzureThreshold
	Sexual   AzureThreshold
	Violence AzureThreshold

	// PromptShield enables jailbreak/prompt-injection detection (input only).
	PromptShield bool
}

// LlamaGuardConfig enables Llama Guard 3 8B categories.
type LlamaGuardConfig struct {
	ViolentCrimes         bool
	NonViolentCrimes      bool
	SexCrimes             bool
	ChildExploitation     bool
	Defamation            bool
	SpecializedAdvice     bool
	Privacy               bool
	IntellectualProperty  bool
	IndiscriminateWeapons bool
	Hate                  bool
	SelfHarm              bool
	SexualContent         bool
	Elections             bool
	CodeInterpreterAbuse  bool
}

// InputFilterConfig defines which content safety providers to apply on input messages.
type InputFilterConfig struct {
	AzureContentSafety *AzureContentSafetyConfig
	LlamaGuard         *LlamaGuardConfig
}

// OutputFilterConfig defines which content safety providers to apply on model output.
type OutputFilterConfig struct {
	AzureContentSafety *AzureContentSafetyConfig
	LlamaGuard         *LlamaGuardConfig

	// Overlap sets the number of characters from previous chunks sent as additional
	// context to the filtering service during streaming. Only relevant when streaming.
	Overlap int
}

// FilteringConfig configures content filtering. Orchestration-mode only.
type FilteringConfig struct {
	Input  *InputFilterConfig
	Output *OutputFilterConfig
}
