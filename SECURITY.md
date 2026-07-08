# Security Policy

## Reporting a Vulnerability

This library handles OAuth2 credentials and authentication tokens. If you discover a security vulnerability, please report it responsibly.

**Do NOT open a public issue for security vulnerabilities.**

Instead, please email: [pedroklein1@hotmail.com](mailto:pedroklein1@hotmail.com) or [pedro.klein@sap.com](mailto:pedro.klein@sap.com)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

I'll acknowledge receipt within 48 hours and aim to release a fix within 7 days for critical issues.

## Scope

Security issues in scope:
- Token leakage (credentials exposed in logs, errors, or network)
- Authentication bypass
- Improper TLS handling
- Injection via user-controlled parameters

Out of scope:
- Issues in upstream dependencies (report those upstream)
- Denial of service via malformed API responses (SAP AI Core is the server)
