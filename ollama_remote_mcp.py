import asyncio
import json
import os

from ollama import AsyncClient
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


# Configuration
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://demo01.i3x.net/mcp"
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

MAX_TOOL_ROUNDS = 10


def convert_mcp_tools(mcp_tools):
    """Convert MCP tools to Ollama function tools."""

    ollama_tools = []

    for tool in mcp_tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or tool.name,
                "parameters": tool.inputSchema
            }
        })

    return ollama_tools


def serialize_tool_result(result):
    """Convert MCP CallToolResult into JSON text."""

    output = {
        "isError": result.isError,
        "content": []
    }

    for item in result.content:
        if hasattr(item, "text"):
            output["content"].append(item.text)
        else:
            output["content"].append(
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else str(item)
            )

    if getattr(result, "structuredContent", None):
        output["structuredContent"] = result.structuredContent

    return json.dumps(
        output,
        ensure_ascii=False,
        default=str
    )


async def main():

    ollama = AsyncClient(host=OLLAMA_HOST)

    print(f"Connecting MCP: {MCP_SERVER_URL}")
    print(f"Connecting Ollama: {OLLAMA_HOST}")
    print(f"Model: {MODEL}")

    async with streamablehttp_client(
        MCP_SERVER_URL
    ) as (read, write, _):

        async with ClientSession(read, write) as session:

            # Initialize MCP
            await session.initialize()

            # Discover MCP tools
            response = await session.list_tools()

            tools = convert_mcp_tools(response.tools)

            print(f"\nDiscovered {len(tools)} MCP tools")

            for tool in tools:
                print("-", tool["function"]["name"])

            # Interactive chat
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an i3X industrial knowledge graph assistant. "
                        "Use available MCP tools to retrieve real information. "
                        "Do not invent equipment values, relationships, or history. "
                        "If the server requires initialization, call "
                        "connect_and_index before using search tools. "
                        "Explain tool errors clearly."
                    )
                }
            ]

            while True:

                question = input("\nUser: ").strip()

                if question.lower() in ("exit", "quit"):
                    break

                if not question:
                    continue

                messages.append({
                    "role": "user",
                    "content": question
                })

                # Agent tool-calling loop
                for round_no in range(MAX_TOOL_ROUNDS):

                    response = await ollama.chat(
                        model=MODEL,
                        messages=messages,
                        tools=tools
                    )

                    assistant_message = response.message

                    # Preserve assistant message and tool calls
                    messages.append(
                        assistant_message.model_dump(
                            exclude_none=True
                        )
                    )

                    tool_calls = assistant_message.tool_calls or []

                    # No more tool calls: final answer
                    if not tool_calls:
                        print("\nAssistant:")
                        print(assistant_message.content)
                        break

                    # Execute MCP tools
                    for tool_call in tool_calls:

                        function = tool_call.function
                        name = function.name
                        arguments = function.arguments or {}

                        print(f"\n[MCP Tool] {name}")
                        print(
                            json.dumps(
                                arguments,
                                ensure_ascii=False,
                                indent=2
                            )
                        )

                        try:
                            result = await session.call_tool(
                                name,
                                arguments=arguments
                            )

                            tool_output = serialize_tool_result(
                                result
                            )

                        except Exception as exc:
                            tool_output = json.dumps({
                                "isError": True,
                                "error": str(exc)
                            }, ensure_ascii=False)

                        print("[MCP Result]")
                        print(tool_output[:3000])

                        # Return tool result to Ollama
                        messages.append({
                            "role": "tool",
                            "tool_name": name,
                            "content": tool_output
                        })

                else:
                    print(
                        "\nMaximum tool calling rounds reached."
                    )


if __name__ == "__main__":
    asyncio.run(main())
