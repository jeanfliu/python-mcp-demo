import asyncio
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    url = os.getenv(
        "MCP_SERVER_URL",
        "https://demo01.i3x.net/mcp"
    )

    print(f"Connecting to {url}")

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\nAvailable MCP Tools:")

            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")

            if tools.tools:
                print("\nMCP connection successful")


if __name__ == "__main__":
    asyncio.run(main())
