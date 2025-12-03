# RhinoMCP - Rhino Model Context Protocol Integration

RhinoMCP connects Rhino, Grasshopper and more to Claude AI through the Model Context Protocol (MCP), allowing Claude to directly interact with and control Rhino + Grasshopper. If you provide a replicate.com api key you can also AI render images. This integration enables prompt-assisted 3D modeling, scene creation, and manipulation. (inspired by [blender_mcp](https://github.com/ahujasid/blender-mcp))

## Features

#### Rhino
- **Two-way communication**: Connect Claude AI to Rhino through a socket-based server
- **Object manipulation and management**: Create and modify 3D objects in Rhino including metadata
- **Layer management**: View and interact with Rhino layers
- **Scene inspection**: Get detailed information about the current Rhino scene (incl. screencapture) 
- **Code execution**: Run arbitrary Python code in Rhino from Claude
 
#### Grasshopper
- **Code execution**: Run arbitrary Python code in Grasshopper from Claude - includes the generation of gh components
- **Gh canvas inspection**: Get detailed information about your Grasshopper definition, including component graph and parameters
- **Component management**: Update script components, modify parameters, and manage code references
- **External code integration**: Link script components to external Python files for better code organization
- **Real-time feedback**: Get component states, error messages, and runtime information
- **Non-blocking communication**: Stable two-way communication via HTTP server

##### Replicate
- **AI Models**: replicate offers thousands of AI models via API, implemented here: a stable diffusion variant 

## Components

The system consists of two main components:

1. **Rhino-side Script (`rhino_script.py`)**: A Python script that runs inside Rhino to create a socket server that receives and executes commands
2. **MCP Server (`rhino_mcp/server.py`)**: A Python server that implements the Model Context Protocol and connects to the Rhino script

## Installation

### Prerequisites

- Rhino 7 or newer
- Python 3.10 or newer
- Conda (for environment management)
- A Replicate API token (for AI-powered features)

### Setting up the Python Environment

1. Create a new conda environment with Python 3.10:
   ```bash
   conda create -n rhino_mcp python=3.10
   conda activate rhino_mcp
   ```

2. Install the `uv` package manager:
   ```bash
   pip install uv
   ```

3. Install the package in development mode:
   ```bash
   cd rhino_mcp  # Navigate to the package directory
   uv pip install -e .
   ```

### Installing the Rhino-side Script

1. Open Rhino 7
2. Open the Python Editor:
   - Click on the "Tools" menu
   - Select "Python Script" -> "Run.."
   - Navigate to and select `rhino_mcp_client.py`
4. The script will start automatically and you should see these messages in the Python Editor:
   ```
   RhinoMCP script loaded. Server started automatically.
   To stop the server, run: stop_server()
   ```

### Running the MCP Server

The MCP server will be started automatically by Claude Desktop using the configuration in `claude_desktop_config.json`. You don't need to start it manually.

### Starting the Connection

1. First, start the Rhino script:
   - Open Rhino 7
   - Open the Python Editor
   - Open and run `rhino_mcp_client.py`
   - Verify you see the startup messages in the Python Editor

2. Then start Claude Desktop:
   - It will automatically start the MCP server when needed
   - The connection between Claude and Rhino will be established automatically

### Managing the Connection

- To stop the Rhino script server:
  - In the Python Editor, type `stop_server()` and press Enter
  - You should see "RhinoMCP server stopped" in the outputt

### Claude Integration

To integrate with Claude Desktop:

1. Go to Claude Desktop > Settings > Developer > Edit Config 
2. Open the `claude_desktop_config.json` file and add the following configuration:

```json
{
    "mcpServers": {
        "rhino": {
            "command": "/Users/Joo/miniconda3/envs/rhino_mcp/bin/python",
            "args": [
                "-m", "rhino_mcp.server"
            ]
        }
    }
}
```

Make sure to:
- Replace the Python path with the path to Python in your conda environment
- Save the file and restart Claude Desktop

### Cursor IDE Integration

Using cursor has the potential benifit that you can use it to organise your colelction of python scripts you use for ghpython components (especialyl when you use grasshoppers reference code functionality). Morover, you can utilize its codebase indexing features, add Rhino/Grasshopper SDK references and so on. 

To integrate with Cursor IDE:

1. Locate or create the file `~/.cursor/mcp.json` (in your home directory)
2. Add the same configuration as used for Claude Desktop:

```json
{
    "mcpServers": {
        "rhino": {
            "command": "/Users/Joo/miniconda3/envs/rhino_mcp/bin/python",
            "args": [
                "-m",
                "rhino_mcp.server"
            ]
        }
    }
}
```


> **Important Note:** For both Claude Desktop and Cursor IDE, if you're using a conda environment, you must specify the full path to the Python interpreter as shown above. Using relative paths or commands like `python` or `uvx` might not work properly with conda environments.

### Setting up Replicate Integration

1. Create a `.env` file in the root directory of the project
2. Add your Replicate API token:
   ```
   REPLICATE_API_TOKEN=your_token_here
   ```
3. Make sure to keep this file private and never commit it to version control

### Grasshopper Integration

The package includes enhanced Grasshopper integration:

1. Start the Grasshopper server component (in rhino_mcp/grasshopper_mcp_client.gh)



Key features:
- Non-blocking communication via HTTP
- Support for external Python file references
- error handling and feedback


### Example Commands

Here are some examples of what you can ask Claude to do:

- "Get information about the current Rhino scene"
- "Create a cube at the origin"
- "Get all layers in the Rhino document"
- "Execute this Python code in Rhino: ..."
- "Update a Grasshopper script component with new code"
- "Link a Grasshopper component to an external Python file"
- "Get the current state of selected Grasshopper components"

## Troubleshooting

- **Connection issues**: 
  - Make sure the Rhino script is running (check Python Editor output)
  - Verify port 9876 is not in use by another application
  - Check that both Rhino and Claude Desktop are running

- **Script not starting**:
  - Make sure you're using Rhino 7 or newer
  - Check the Python Editor for any error messages
  - Try closing and reopening the Python Editor

- **Package not found**: 
  - Ensure you're in the correct directory and have installed the package in development mode
  - Verify your conda environment is activated

- **Python path issues**: 
  - Verify that the Python path in `claude_desktop_config.json` matches your conda environment's Python path
  - Make sure you're using the full path to the Python interpreter

## Limitations

- The `execute_rhino_code` and `execute_code_in_gh` tools allow running arbitrary Python code, which can be powerful but potentially dangerous. Use with caution.
- Grasshopper integration requires the HTTP server component to be running
- External code references in Grasshopper require careful file path management
- This is a minimal implementation focused on basic functionality. Advanced features may require additional development.

## Extending

To add new functionality, you need to:

1. Add new command handlers and functions in `rhino_script.py` and the `RhinoMCPServer` class.
2. Add corresponding MCP tools in `server.py` that include tool and arg descriptions

## License

This project is open source and available under the MIT License. 