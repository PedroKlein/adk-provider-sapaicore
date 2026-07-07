// Package smoketest provides integration tests against a live SAP AI Core instance.
//
// Run with: go test -tags=smoke ./smoketest/ -v -timeout=5m
//
// Required environment variables:
//   - AI_CORE_ENDPOINT
//   - AI_CORE_CLIENT_ID
//   - AI_CORE_CLIENT_SECRET
//   - AI_CORE_AUTH_URL
//
// Optional:
//   - SMOKE_LARGE_CONTEXT=1 (enables the >200K token test — slow and costly)
package smoketest
