# Security Policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

If you discover a security issue in Ness Agent, please report it responsibly.

**Preferred:** [Open a private GitHub Security Advisory](https://github.com/Sagnnik/ness-agent/security/advisories/new)

If you cannot use GitHub Security Advisories, open a minimal public issue asking for a private contact channel — do **not** include exploit details in the public issue.

Please include:

- A description of the vulnerability and its impact
- Steps to reproduce (proof-of-concept if available)
- Affected versions and components (SDK, CLI, or both)

We aim to acknowledge reports within **72 hours** and will coordinate disclosure and a fix before public announcement when possible.

## Scope

In scope:

- Remote code execution or sandbox escapes via built-in tools
- Unauthorized filesystem or network access beyond documented permission rules
- Secret or credential leakage (API keys, tokens, `.ness/` secrets)
- Supply-chain issues in published PyPI packages (`ness-agent`)

Out of scope:

- Issues in third-party LLM providers or MCP servers you configure
- Social engineering or phishing
- Denial-of-service against local CLI sessions without a practical exploit path

## Safe defaults

Ness Agent runs tools on your machine with permission rules you control. Review `.ness/permissions.json` and MCP server configuration before running in sensitive environments.
