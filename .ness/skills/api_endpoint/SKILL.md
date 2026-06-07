---
name: api_endpoint
description: Create REST API endpoints with validation and consistent error handling.
triggers:
  - api
  - endpoint
  - route
  - handler
  - controller
  - server
  - backend
---
# API Endpoint

When implementing an API endpoint:

- Inspect adjacent routes before editing.
- Match the framework's existing routing and response style.
- Validate input with the project's existing validation library when one exists.
- Return consistent error shapes and appropriate HTTP status codes.
- Add or update tests when the project has an API test pattern.

For TypeScript projects, define request and response types near the handler unless the project already centralizes API types.
