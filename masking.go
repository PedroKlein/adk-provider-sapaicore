package sapaicore

type MaskingMethod string

const (
	Anonymization    MaskingMethod = "anonymization"
	Pseudonymization MaskingMethod = "pseudonymization"
)

// DPIEntity identifies a standard entity type recognized by SAP Data Privacy Integration.
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

// MaskingEntity is either a standard [DPIEntity] or a [CustomEntity].
type MaskingEntity struct {
	standard *DPIEntity
	custom   *CustomEntity
}

func StandardEntity(entity DPIEntity) MaskingEntity {
	return MaskingEntity{standard: &entity}
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
}

// StandardEntities converts a []DPIEntity to []MaskingEntity for use with [MaskingConfig].
func StandardEntities(entities []DPIEntity) []MaskingEntity {
	result := make([]MaskingEntity, len(entities))
	for i, e := range entities {
		result[i] = StandardEntity(e)
	}

	return result
}
