# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| main    | ✅        |
| develop | ✅        |
| older branches | ❌ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues privately by emailing the repository owner or
using GitHub's private vulnerability reporting feature
(Security → Report a vulnerability).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 48 hours.

## Known Considerations

- The backend API has no authentication by default — do not expose it publicly without adding auth middleware.
- `GROQ_API_KEY` must be stored in `.env` and never committed. See `.env.example`.
- Model checkpoint files (`.pt`) are large binary files — do not store sensitive data in them.
