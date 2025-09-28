#!/usr/bin/env python

import asyncio
import base64
import json

import httpx

try:
    from litellm import experimental_mcp_client
except ImportError:
    experimental_mcp_client = None

try:
    from aider.mcp.server import LocalServer
except ImportError:
    LocalServer = None


class ToolUseManager:
    """Manages tool usage and MCP server interactions for the Coder class."""

    def __init__(self, coder):
        self.coder = coder
        self.mcp_servers = coder.mcp_servers
        self.mcp_tools = coder.mcp_tools

    async def process_tool_calls(self, tool_call_response):
        """Process tool calls from the LLM response."""
        if tool_call_response is None:
            return False

        # Handle different response structures
        try:
            # Try to get tool calls from the standard OpenAI response format
            if hasattr(tool_call_response, "choices") and tool_call_response.choices:
                message = tool_call_response.choices[0].message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    original_tool_calls = message.tool_calls
                else:
                    return False
            else:
                # Handle other response formats
                return False
        except (AttributeError, IndexError):
            return False

        if not original_tool_calls:
            return False

        # Expand any tool calls that have concatenated JSON in their arguments.
        expanded_tool_calls = []
        for tool_call in original_tool_calls:
            args_string = tool_call.function.arguments.strip()

            # If there are no arguments, or it's not a string that looks like it could
            # be concatenated JSON, just add it and continue.
            if not args_string or not (args_string.startswith("{") or args_string.startswith("[")):
                expanded_tool_calls.append(tool_call)
                continue

            json_chunks = self.coder.utils.split_concatenated_json(args_string)

            # If it's just a single JSON object, there's nothing to expand.
            if len(json_chunks) <= 1:
                expanded_tool_calls.append(tool_call)
                continue

            # We have concatenated JSON, so expand it into multiple tool calls.
            for i, chunk in enumerate(json_chunks):
                if not chunk.strip():
                    continue

                # Create a new tool call for each JSON chunk, with a unique ID.
                new_function = tool_call.function.model_copy(update={"arguments": chunk})
                new_tool_call = tool_call.model_copy(
                    update={"id": f"{tool_call.id}-{i}", "function": new_function}
                )
                expanded_tool_calls.append(new_tool_call)

        # Collect all tool calls grouped by server
        server_tool_calls = self._gather_server_tool_calls(expanded_tool_calls)

        if server_tool_calls and self.coder.num_tool_calls < self.coder.max_tool_calls:
            self._print_tool_call_info(server_tool_calls)

            if await self.coder.io.confirm_ask("Run tools?"):
                tool_responses = await self._execute_tool_calls(server_tool_calls)

                # Add all tool responses
                for tool_response in tool_responses:
                    self.coder.cur_messages.append(tool_response)

                return True
        elif self.coder.num_tool_calls >= self.coder.max_tool_calls:
            self.coder.io.tool_warning(
                f"Only {self.coder.max_tool_calls} tool calls allowed, stopping."
            )

        return False

    def _print_tool_call_info(self, server_tool_calls):
        """Print information about an MCP tool call."""
        self.coder.io.tool_output("Preparing to run MCP tools", bold=True)

        for server, tool_calls in server_tool_calls.items():
            for tool_call in tool_calls:
                self.coder.io.tool_output(f"Tool Call: {tool_call.function.name}")
                self.coder.io.tool_output(f"Arguments: {tool_call.function.arguments}")
                self.coder.io.tool_output(f"MCP Server: {server.name}")

                if self.coder.verbose:
                    self.coder.io.tool_output(f"Tool ID: {tool_call.id}")
                    self.coder.io.tool_output(f"Tool type: {tool_call.type}")

                self.coder.io.tool_output("\n")

    def _gather_server_tool_calls(self, tool_calls):
        """Collect all tool calls grouped by server.
        Args:
            tool_calls: List of tool calls from the LLM response

        Returns:
            dict: Dictionary mapping servers to their respective tool calls
        """
        if not self.mcp_tools or len(self.mcp_tools) == 0:
            return None

        server_tool_calls = {}
        for tool_call in tool_calls:
            # Check if this tool_call matches any MCP tool
            for server_name, server_tools in self.mcp_tools:
                for tool in server_tools:
                    tool_name_from_schema = tool.get("function", {}).get("name")
                    if (
                        tool_name_from_schema
                        and tool_name_from_schema.lower() == tool_call.function.name.lower()
                    ):
                        # Find the McpServer instance that will be used for communication
                        for server in self.mcp_servers:
                            if server.name == server_name:
                                if server not in server_tool_calls:
                                    server_tool_calls[server] = []
                                server_tool_calls[server].append(tool_call)
                                break

        return server_tool_calls

    async def _execute_tool_calls(self, tool_calls):
        """Process tool calls from the response and execute them if they match MCP tools.
        Returns a list of tool response messages."""
        tool_responses = []

        # Define the coroutine to execute all tool calls for a single server
        async def _exec_server_tools(server, tool_calls_list):
            if isinstance(server, LocalServer):
                if hasattr(self.coder, "_execute_local_tool_calls"):
                    return await self.coder._execute_local_tool_calls(tool_calls_list)
                else:
                    # This coder doesn't support local tools, return errors for all calls
                    error_responses = []
                    for tool_call in tool_calls_list:
                        error_responses.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": (
                                    f"Coder does not support local tool: {tool_call.function.name}"
                                ),
                            }
                        )
                    return error_responses

            tool_responses = []
            try:
                # Connect to the server once
                session = await server.connect()
                # Execute all tool calls for this server
                for tool_call in tool_calls_list:
                    try:
                        # Arguments can be a stream of JSON objects.
                        # We need to parse them and run a tool call for each.
                        args_string = tool_call.function.arguments.strip()
                        parsed_args_list = []
                        if args_string:
                            json_chunks = self.coder.utils.split_concatenated_json(args_string)
                            for chunk in json_chunks:
                                try:
                                    parsed_args_list.append(json.loads(chunk))
                                except json.JSONDecodeError:
                                    self.coder.io.tool_warning(
                                        "Could not parse JSON chunk for tool"
                                        f" {tool_call.function.name}: {chunk}"
                                    )
                                    continue

                        if not parsed_args_list and not args_string:
                            parsed_args_list.append({})  # For tool calls with no arguments

                        all_results_content = []
                        for args in parsed_args_list:
                            new_tool_call = tool_call.model_copy(deep=True)
                            new_tool_call.function.arguments = json.dumps(args)

                            call_result = await experimental_mcp_client.call_openai_tool(
                                session=session,
                                openai_tool=new_tool_call,
                            )

                            content_parts = []
                            if call_result.content:
                                for item in call_result.content:
                                    if hasattr(item, "resource"):  # EmbeddedResource
                                        resource = item.resource
                                        if hasattr(resource, "text"):  # TextResourceContents
                                            content_parts.append(resource.text)
                                        elif hasattr(resource, "blob"):  # BlobResourceContents
                                            try:
                                                decoded_blob = base64.b64decode(
                                                    resource.blob
                                                ).decode("utf-8")
                                                content_parts.append(decoded_blob)
                                            except (UnicodeDecodeError, TypeError):
                                                # Handle non-text blobs gracefully
                                                name = getattr(resource, "name", "unnamed")
                                                mime_type = getattr(
                                                    resource, "mimeType", "unknown mime type"
                                                )
                                                content_parts.append(
                                                    "[embedded binary resource:"
                                                    f" {name} ({mime_type})]"
                                                )
                                    elif hasattr(item, "text"):  # TextContent
                                        content_parts.append(item.text)

                            result_text = "".join(content_parts)
                            all_results_content.append(result_text)

                        tool_responses.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "\n\n".join(all_results_content),
                            }
                        )

                    except Exception as e:
                        tool_error = f"Error executing tool call {tool_call.function.name}: \n{e}"
                        self.coder.io.tool_warning(
                            f"Executing {tool_call.function.name} on {server.name} failed: \n "
                            f" Error: {e}\n"
                        )
                        tool_responses.append(
                            {"role": "tool", "tool_call_id": tool_call.id, "content": tool_error}
                        )
            except httpx.RemoteProtocolError as e:
                connection_error = f"Server {server.name} disconnected unexpectedly: {e}"
                self.coder.io.tool_warning(connection_error)
                for tool_call in tool_calls_list:
                    tool_responses.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": connection_error}
                    )
            except Exception as e:
                connection_error = f"Could not connect to server {server.name}\n{e}"
                self.coder.io.tool_warning(connection_error)
                for tool_call in tool_calls_list:
                    tool_responses.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": connection_error}
                    )
            finally:
                await server.disconnect()

            return tool_responses

        # Execute all tool calls concurrently
        async def _execute_all_tool_calls():
            tasks = []
            for server, tool_calls_list in tool_calls.items():
                tasks.append(_exec_server_tools(server, tool_calls_list))
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            return results

        # Run the async execution and collect results
        if tool_calls:
            all_results = []
            max_retries = 3
            for i in range(max_retries):
                try:
                    all_results = await _execute_all_tool_calls()
                    break
                except asyncio.exceptions.CancelledError:
                    if i < max_retries - 1:
                        await asyncio.sleep(0.1)  # Brief pause before retrying
                    else:
                        self.coder.io.tool_warning(
                            "MCP tool execution failed after multiple retries due to cancellation."
                        )
                        all_results = []

            # Flatten the results from all servers
            for server_results in all_results:
                tool_responses.extend(server_results)

        return tool_responses

    async def initialize_mcp_tools(self):
        """
        Initialize tools from all configured MCP servers. MCP Servers that fail to be
        initialized will not be available to the Coder instance.
        """
        tools = []

        async def get_server_tools(server):
            try:
                session = await server.connect()
                server_tools = await experimental_mcp_client.load_mcp_tools(
                    session=session, format="openai"
                )
                return (server.name, server_tools)
            except Exception as e:
                self.coder.io.tool_warning(f"Error initializing MCP server {server.name}:\n{e}")
                return None
            finally:
                await server.disconnect()

        async def get_all_server_tools():
            tasks = [get_server_tools(server) for server in self.mcp_servers]
            results = await asyncio.gather(*tasks)
            return [result for result in results if result is not None]

        if self.mcp_servers:
            # Retry initialization in case of CancelledError
            max_retries = 3
            for i in range(max_retries):
                try:
                    tools = await get_all_server_tools()
                    break
                except asyncio.exceptions.CancelledError:
                    if i < max_retries - 1:
                        await asyncio.sleep(0.1)  # Brief pause before retrying
                    else:
                        self.coder.io.tool_warning(
                            "MCP tool initialization failed after multiple retries due to"
                            " cancellation."
                        )
                        tools = []

        if len(tools) > 0:
            self.coder.io.tool_output("MCP servers configured:")
            for server_name, server_tools in tools:
                self.coder.io.tool_output(f"  - {server_name}")

                if self.coder.verbose:
                    for tool in server_tools:
                        tool_name = tool.get("function", {}).get("name", "unknown")
                        tool_desc = tool.get("function", {}).get("description", "").split("\n")[0]
                        self.coder.io.tool_output(f"    - {tool_name}: {tool_desc}")

        self.mcp_tools = tools

    def get_tool_list(self):
        """Get a flattened list of all MCP tools."""
        tool_list = []
        if self.mcp_tools:
            for _, server_tools in self.mcp_tools:
                tool_list.extend(server_tools)
        return tool_list
