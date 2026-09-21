import math
from fastmcp import FastMCP 
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers  
from fastmcp.exceptions import ToolError
import argparse

VALID_TOKENS = {
    "user-token-xyz",
    "admin-token-123"
}

class UserAuthMiddleware(Middleware):
    async def on_request(self, context: MiddlewareContext, call_next):
        """
        Global security interceptor. Validates the Bearer token sent in 
        the HTTP header before tool execution.
        """
        print("Authenticating user...")
        headers = get_http_headers(include_all=True)
        token = headers.get("authorization")
        if not token:
            raise ValueError("Unauthorized: Missing or malformed Authorization header")

        if token not in VALID_TOKENS:
            raise ToolError("Unauthorized: Invalid or expired API token")
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