"""Minimal stdio MCP server for mcp_client integration tests."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back to the caller."""
    return message


if __name__ == "__main__":
    mcp.run()
