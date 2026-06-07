---
name: react_component
description: Create React components with TypeScript while matching project conventions.
triggers:
  - react
  - component
  - tsx
  - jsx
  - ui
  - page
  - widget
  - frontend
---
# React Component

When implementing a React component:

- Inspect nearby components before creating new patterns.
- Use functional components and hooks.
- Match the project's styling system instead of assuming Tailwind.
- Add `"use client"` only when the component needs client-side behavior.
- Keep props typed and exports consistent with nearby files.
- Verify layout at realistic mobile and desktop widths when the app has a frontend.
