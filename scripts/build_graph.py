
from src.agents.graph import travel_graph
from pathlib import Path

# Graph is already compiled
app = travel_graph

# Create images folder
images_dir = Path("images")
images_dir.mkdir(parents=True, exist_ok=True)

# Generate graph visualization
image_data = app.get_graph().draw_mermaid_png()

# Save graph image
image_path = images_dir / "graph.png"
image_path.write_bytes(image_data)

print(f"Graph saved successfully: {image_path}")


# from backend import graph
# from pathlib import Path

# # Compile the StateGraph
# app = graph.compile()

# # Create images folder if it doesn't exist
# images_dir = Path("images")
# images_dir.mkdir(parents=True, exist_ok=True)

# # Generate graph visualization
# image_data = app.get_graph().draw_mermaid_png()

# # Save graph image
# image_path = images_dir / "graph.png"
# image_path.write_bytes(image_data)

# print(f"Graph saved successfully: {image_path}")