from langchain_core.tools import tool
import subprocess

@tool
def read_file(path: str) -> str:
    pass

@tool
def write_file(path: str, content: str) -> str:
    pass

@tool
def apply_diff(path: str, old_string: str, new_string: str) -> str:
    pass

@tool
def list_files(path: str = ".") -> list[str]:
    pass

@tool
def get_project_context() -> str:
    pass

@tool
def search_files(path: str, query: str) -> list[str]:
    pass


# Git tools

@tool
def git_snapshot(message: str = "agent: auto-save") -> str:
    pass

@tool
def git_commit(message: str) -> str:
    pass

@tool
def git_diff() -> str:
    pass

# Test Runner
@tool
def run_tests(test_path: str = "") -> str:
    pass