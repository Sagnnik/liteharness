import json
from pathlib import Path
from config import settings

MCP_DIR = Path(settings.ness_dir)
MCP_FILE = MCP_DIR / "mcp.json"

class MCPManager:
    def __init__(self):
        self.servers: dict = {}
        self.tools: dict = {} # mcp__server__tool -> callable

    async def start(self):
        if not MCP_FILE.exists():
            return
        
        cfg = json.loads(MCP_FILE.read_text())
        for name, spec in cfg.get("servers", {}).items():
            try:
                # from mcp import ClientSession, StdioServerParameters
                # session = await ClientSession(...).connect()
                # tools = await session.list_tools()
                # namespace as mcp__{name}__{tool.name}
                self.servers[name] = spec
            except Exception as e:
                self.servers[name] = {"error": str(e)}

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    async def call(self, full_name:str, args:dict) -> str:
        fn = self.tools.get(full_name)
        if not fn:
            return f"Unknown MCP tool: {full_name}"

        return await fn(**args)


mcp_manager = MCPManager()