package sapaicore

type MaskingMethod string

const (
	Anonymization    MaskingMethod = "anonymization"
	Pseudonymization MaskingMethod = "pseudonymization"
)

// MaskFileInputMethod controls how file inputs interact with masking.
// Required when file parts are present and masking is configured.
type MaskFileInputMethod string

const (
	MaskFileAnonymization MaskFileInputMethod = "anonymization"
	MaskFileSkip          MaskFileInputMethod = "skip"
)

type DPIEntity string

const (
	EntityPerson            DPIEntity = "profile-person"
	EntityOrg               DPIEntity = "profile-org"
	EntityUniversity        DPIEntity = "profile-university"
	EntityLocation          DPIEntity = "profile-location"
	EntityEmail             DPIEntity = "profile-email"
	EntityPhone             DPIEntity = "profile-phone"
	EntityAddress           DPIEntity = "profile-address"
	EntitySAPIDsInternal    DPIEntity = "profile-sapids-internal"
	EntitySAPIDsPublic      DPIEntity = "profile-sapids-public"
	EntityURL               DPIEntity = "profile-url"
	EntityUsernamePassword  DPIEntity = "profile-username-password"
	EntityNationalID        DPIEntity = "profile-nationalid"
	EntityIBAN              DPIEntity = "profile-iban"
	EntitySSN               DPIEntity = "profile-ssn"
	EntityCreditCard        DPIEntity = "profile-credit-card-number"
	EntityPassport          DPIEntity = "profile-passport"
	EntityDriverLicense     DPIEntity = "profile-driverlicense"
	EntityNationality       DPIEntity = "profile-nationality"
	EntityReligiousGroup    DPIEntity = "profile-religious-group"
	EntityPoliticalGroup    DPIEntity = "profile-political-group"
	EntityPronounsGender    DPIEntity = "profile-pronouns-gender"
	EntityEthnicity         DPIEntity = "profile-ethnicity"
	EntityGender            DPIEntity = "profile-gender"
	EntitySexualOrientation DPIEntity = "profile-sexual-orientation"
	EntityTradeUnion        DPIEntity = "profile-trade-union"
	EntitySensitiveData     DPIEntity = "profile-sensitive-data"
)

// CommonPIIEntities covers the most frequently masked PII types:
// person names, organizations, email, phone, and addresses.
var CommonPIIEntities = []DPIEntity{
	EntityPerson,
	EntityOrg,
	EntityEmail,
	EntityPhone,
	EntityAddress,
}

type CustomEntity struct {
	Regex       string
	Replacement string
}

type replacementStrategy int

const (
	strategyNone replacementStrategy = iota
	strategyFabricated
	strategyConstant
)

// MaskingEntity represents a single entity to mask. It is either a standard [DPIEntity]
// (created via [StandardEntity], [FabricatedEntity], or [ConstantEntity]) or a custom
// regex-based entity (created via [CustomMaskingEntity]).
type MaskingEntity struct {
	standard *DPIEntity
	custom   *CustomEntity
	strategy replacementStrategy
	value    string // only for strategyConstant on standard entities
}

func StandardEntity(entity DPIEntity) MaskingEntity {
	return MaskingEntity{standard: &entity}
}

// FabricatedEntity replaces matches with realistic fake data appropriate to the entity type.
func FabricatedEntity(entity DPIEntity) MaskingEntity {
	return MaskingEntity{standard: &entity, strategy: strategyFabricated}
}

// ConstantEntity replaces matches with a fixed value followed by an incrementing number.
func ConstantEntity(entity DPIEntity, replacement string) MaskingEntity {
	return MaskingEntity{standard: &entity, strategy: strategyConstant, value: replacement}
}

func CustomMaskingEntity(regex, replacement string) MaskingEntity {
	return MaskingEntity{custom: &CustomEntity{Regex: regex, Replacement: replacement}}
}

// MaskingConfig configures data masking. Orchestration-mode only.
type MaskingConfig struct {
	// Method defaults to [Anonymization] if empty.
	Method MaskingMethod

	// Entities to mask. Use [StandardEntities]([CommonPIIEntities]) for quick setup.
	Entities []MaskingEntity

	// Allowlist contains strings that should never be masked even if they match.
	Allowlist []string

	// MaskGroundingInput controls whether grounding module input is also masked.
	MaskGroundingInput bool

	// MaskFileInputMethod controls how file inputs are handled by masking.
	// Required when file content blocks are present in the request.
	MaskFileInputMethod MaskFileInputMethod
}

func StandardEntities(entities []DPIEntity) []MaskingEntity {
	result := make([]MaskingEntity, len(entities))
	for i, e := range entities {
		result[i] = StandardEntity(e)
	}

	return result
}
