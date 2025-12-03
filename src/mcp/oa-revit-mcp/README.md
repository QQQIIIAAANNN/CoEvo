# Revit-MCP Documentation

## Introduction

Revit-MCP is a powerful tool that connects Large Language Models (LLMs) with Autodesk Revit, enabling AI-driven BIM automation. It allows you to interact with Revit using natural language commands, making it easier to perform complex tasks and automate repetitive workflows.

The project consists of three main components:

- **mcp:** A monorepo containing the server, command set, and code generation tools.
- **plugin:** A Revit plugin that receives messages from the server, loads command sets, and operates on the Revit application.

## Installation

To get started with Revit-MCP, you need to set up the server and the plugin. Follow these steps to install and configure the project.

### 1. Environment Requirements

- **Revit:** 2019-2025
- **Node.js:** v18 or later
- **.NET SDK:** 8.0 or later

### 2. Set Up the Server

The server is responsible for connecting the AI to the Revit plugin. Here's how to set it up:

1.  **Navigate to the server directory:**
    ```bash
    cd mcp/server
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Build the server:**
    ```bash
    npm run build
    ```

4.  **Configure the client:**
    - Open your MCP-supported client (e.g., Claude).
    - Go to **Settings > Developer > Edit Config**.
    - Add the following configuration to `claude_desktop_config.json`:
      ```json
      {
          "mcpServers": {
              "revit-mcp": {
                  "command": "node",
                  "args": ["<path_to_your_project>/mcp/server/build/index.js"]
              }
          }
      }
      ```
    - Replace `<path_to_your_project>` with the actual path to the `revit-mcp` project on your machine.
    - Restart the client. You should see a hammer icon, which indicates a successful connection to the MCP service.

### 3. Set Up the Revit Plugin and Command Set

The Revit plugin is responsible for executing the commands sent by the server. The command set is now integrated into the plugin's solution.

1.  **Compile the plugin and command set:**
    - Open the `plugin/revit-mcp-plugin.sln` solution in Visual Studio.
    - Build the solution. This will also build the `RevitMCPCommandSet` project and automatically generate the necessary command files from `mcp/commandset/command.json`.

2.  **Register the plugin:**
    - The build process will create an `.addin` file in the appropriate Revit Addins folder for you.

3.  **Restart Revit.**

4.  **Configure commands in Revit:**
    - In Revit, go to **Add-ins > Revit MCP Plugin > Settings**.
    - Click **OpenCommandSetFolder** to open the folder where the command sets are stored.
    - Check the commands you want to load and use.

5.  **Enable the service:**
    - Go to **Add-ins > Revit MCP Plugin > Revit MCP Switch**.
    - Turn on the service to allow the AI to connect to your Revit application.

## Usage

Once you have completed the installation and configuration, you can start using Revit-MCP to interact with Revit. Open your MCP-supported client and start sending commands. For example, you can ask the AI to:

- "Get all the walls in the current view."
- "Create a new door in the selected wall."
- "Delete the selected elements."

The AI will use the available tools to execute your commands in Revit.

## Custom Commands

You can extend the functionality of Revit-MCP by creating your own custom commands. The process has been streamlined to minimize boilerplate and manual configuration.

### Adding a New Command

To add a new command, you only need to do one thing:

1.  **Add an entry to `mcp/commandset/command.json`:**

    Open the `command.json` file and add a new entry for your command. For example:

    ```json
    {
      "commandName": "my_new_command",
      "description": "This is my new custom command.",
      "assemblyPath": "RevitMCPCommandSet.dll"
    }
    ```

That's it! The build process will automatically generate the necessary C# and TypeScript files for you when you build the `plugin/revit-mcp-plugin.sln` solution.

### How it Works

The magic happens in the pre-build step of the `RevitMCPCommandSet` project. A code generation tool (`mcp/tools/CodeGen`) reads the `command.json` file and generates the following files:

-   **C# Event Handler:** A new C# class that implements `IExternalEventHandler` is generated in the `mcp/commandset/revit-mcp-commandset/Services/Generated` directory. This class will have a placeholder for your business logic.
-   **TypeScript Tool:** A new TypeScript file is generated in the `mcp/server/src/tools/generated` directory. This file defines the server-side tool that the AI will use.

You will still need to implement the business logic for your command in the generated C# event handler. The generated file will have a `// TODO:` comment to guide you.

### Important Notes

-   **Do not edit the generated files directly.** If you need to make changes to the command's definition, edit the `command.json` file and rebuild the `plugin` solution.
-   The code generation tool will not overwrite existing files. If you want to regenerate a file, you must delete it first.

## Supported Tools

Here is a list of the supported tools that you can use with Revit-MCP:

| Name                        | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| `get_current_view_info`     | Get information about the current view.              |
| `get_current_view_elements` | Get all elements in the current view.                |
| `get_available_family_types`| Get available family types in the current project.   |
| `get_selected_elements`     | Get the elements that are currently selected.        |
| `create_point_based_element`| Create a point-based element (e.g., door, window).   |
| `create_line_based_element` | Create a line-based element (e.g., wall, beam).      |
| `create_surface_based_element`| Create a surface-based element (e.g., floor).      |
| `delete_elements`           | Delete one or more elements.                         |
| `reset_model`               | Reset the model.                                     |
| `modify_element`            | Modify the properties of an element.                 |
| `search_modules`            | Search for available modules.                        |
| `use_module`                | Use a specific module.                               |
| `send_code_to_revit`        | Send a code snippet to be executed in Revit.         |
| `color_splash`              | Color elements based on a parameter value.           |
| `tag_walls`                 | Tag all walls in the current view.                   |
