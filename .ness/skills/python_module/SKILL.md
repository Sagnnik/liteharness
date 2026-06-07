---
name: python_module
description: Modify Python modules with small, typed, testable changes.
triggers:
  - python
  - module
  - pytest
  - cli
  - class
  - function
---
# Python Module

When working in Python:

- Read the relevant module and its tests before editing.
- Prefer small functions with explicit inputs and return values.
- Preserve existing public interfaces unless the user asked for a contract change.
- Use standard library tools before adding dependencies.
- Run the narrowest useful test command, then broaden if shared behavior changed.
