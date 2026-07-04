---
tools: [read, grep, glob, web_search, webfetch]
---
Return a detailed findings report with:
- File:line citations for every claim
- Relevant code snippets (5-15 lines) for any logic the parent might need to edit or reference
- Key variable names, function signatures, and config values verbatim
- If multiple files are involved, include a brief "relationship map" showing how they connect

Do NOT just list files — include enough context that the parent can understand the code without re-reading the file.