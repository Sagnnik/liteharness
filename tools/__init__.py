from tools.fs import read_file, write_file, edit_file, multi_edit, apply_patch, glob_files, list_files, add_to_memory
from tools.search import grep
from tools.shell import bash
from tools.git import (
    git_status, git_diff, git_log, git_show, git_blame,
    git_snapshot, git_commit, git_checkout, git_branch, git_stash,
    git_worktree_add, git_worktree_list, git_worktree_remove,
)
from tools.todo import todo_write, todo_read
from memory import load_project_context

get_project_context = load_project_context


ALL_TOOLS = [
    read_file, write_file, edit_file, multi_edit, apply_patch, glob_files, list_files, add_to_memory,
    grep, bash,
    git_status, git_diff, git_log, git_show, git_blame,
    git_snapshot, git_commit, git_checkout, git_branch, git_stash,
    git_worktree_add, git_worktree_list, git_worktree_remove,
    todo_write, todo_read, get_project_context,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
TOOL_NAMES = list(TOOL_MAP.keys())

DESTRUCTIVE_TOOLS = {
    "write_file", "edit_file", "multi_edit", "apply_patch", "add_to_memory",
    "git_snapshot", "git_commit", "git_checkout", "git_branch", "git_stash",
    "git_worktree_add", "git_worktree_remove", "bash",
}
READ_ONLY_TOOLS = {
    "read_file", "grep", "glob_files", "list_files", "git_status", "git_diff",
    "git_log", "git_show", "git_blame", "git_worktree_list", "todo_read",
    "get_project_context",
}