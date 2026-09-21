import math
from fastmcp import FastMCP
import argparse

server = FastMCP("Geometry-Helper-Server")

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

@server.prompt()
def geometry_helper_prompt():
    return """
    You are a helpful mathematical assistant.
    Use the provided tools to perform calculations related to triangles, such as calculating the hypotenuse or the area.
    Always use the tools for calculations to ensure accuracy.
    """

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
