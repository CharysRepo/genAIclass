import math
from fastmcp import FastMCP 
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers  
from fastmcp.exceptions import ToolError
import argparse

TOKEN_DB = {
    "user-token-xyz": {"username": "alice", "roles": ["user"]},
    "admin-token-123": {"username": "bob", "roles": ["admin", "user"]}
}

TOOL_ROLE_MAPPING = {
    "calculate_triangle_area": ["user"],
    "calculate_hypotenuse": ["admin"] 
}
class UserAuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """
        Global security interceptor. Validates the Bearer token sent in 
        the HTTP header before tool execution.
        """
        print("Authenticating user...")
        headers = get_http_headers(include_all=True)
        token = headers.get("authorization")
        if not token:
            raise ValueError("Unauthorized: Missing or malformed Authorization header")

        if token not in TOKEN_DB:
            raise ToolError("Unauthorized: Invalid or expired API token")
        
        user_info = TOKEN_DB[token]
        user_roles = user_info.get("roles", [])

        tool_name = context.message.name            
        required_roles = TOOL_ROLE_MAPPING.get(tool_name, [])
        
        if required_roles and not any(role in user_roles for role in required_roles):
            raise ToolError(f"Forbidden: User '{user_info['username']}' lacks required role privileges for tool '{tool_name}'")

        return await call_next(context)
    
server = FastMCP("Geometry-Helper-Server")
server.add_middleware(UserAuthMiddleware())

@server.tool()
def calculate_hypotenuse(a: float, b: float) -> float:
    """
    Calculates the hypotenuse of a right-angled triangle given sides a and b.
    """
    return math.sqrt(a**2 + b**2)


@server.tool()
def calculate_triangle_area(base: float, height: float) -> float:
    """
    Calculates the area of a triangle given its base and vertical height.
    
    Args:
        base: The length of the triangle's base.
        height: The vertical height of the triangle.
    """
    return 0.5 * base * height


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Geometry Helper MCP Server.")
    parser.add_argument(
        "--transport", 
        type=str, 
        default="streamable-http", 
        choices=["stdio", "streamable-http"],
        help="Transport protocol to use (stdio or streamable-http). Default is stdio."
    )
    
    args = parser.parse_args()
    if args.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(transport="streamable-http")

