COMPACTION = """You are summarizing a session for an AI agent to continue its work. 
Be extremely concise. Focus on actionable state and facts. Discard conversational filler and raw tool outputs unless they contain errors.

Format your summary using these exact sections:

**Goal:** [The user's core objective and any strict constraints]
**Progress & Decisions:** [What was implemented/changed and the architectural reasoning why]
**Key Files:** [List of files read, modified, or created with a 5-word status e.g., `src/api.ts - added auth middleware`]
**Blockers & Failed Attempts:** [Unresolved errors and approaches that failed so the agent doesn't repeat them]
**Status:** [State explicitly whether the task's definition-of-done is met or still open, so the agent does not redundantly repeat finished work or stop early]
**Next Steps:** [The immediate actions the agent was about to take before compaction]

Transcript to summarize:
{messages}"""