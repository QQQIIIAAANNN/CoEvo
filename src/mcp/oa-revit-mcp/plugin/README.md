# revit-mcp-plugin

English | [简体中文](README_zh.md)

## Introduction

revit-mcp-plugin is a Revit plugin based on the MCP protocol, enabling AI to interact with Revit.

This project is part of the revit-mcp project (receives messages, loads command sets, operates Revit), and needs to be used in conjunction with [revit-mcp](https://github.com/revit-mcp/revit-mcp) (provides tools to AI) and [revit-mcp-commandset](https://github.com/revit-mcp/revit-mcp-commandset) (specific feature implementations).

## Environment Requirements

- Revit 2019~2025

## Usage Instructions

### Register Plugin

Register the plugin and restart Revit:

```xml
<?xml version="1.0" encoding="utf-8"?>
<RevitAddIns>
  <AddIn Type="Application">
    <Name>revit-mcp</Name>
    <Assembly>%your_path%\revit-mcp-plugin.dll</Assembly>
    <FullClassName>revit_mcp_plugin.Core.Application</FullClassName>
    <ClientId>090A4C8C-61DC-426D-87DF-E4BAE0F80EC1</ClientId>
    <VendorId>revit-mcp</VendorId>
    <VendorDescription>https://github.com/revit-mcp/revit-mcp-plugin</VendorDescription>
  </AddIn>
</RevitAddIns>
```

`%your_path%` needs to be replaced with the actual path after compilation.

### Configure Commands

Add-in Modules -> Revit MCP Plugin -> Settings

This interface is used to configure the commands to be loaded into Revit. Click OpenCommandSetFolder to open the folder storing command sets. A typical command set folder structure looks like this:

```
CommandSetName/
├── 2019/                                # Compatible executable files for different versions
├── 2020/
├── 2021/
├── 2022/
├── 2023/
├── 2024/
└── command.json                         # Configuration file
```

Successfully identified commands need to be checked to be loaded and used.

### Enable Service

Add-in -> Revit MCP Plugin -> Revit MCP Switch

Open the service to allow AI to discover your Revit program. Now AI can control your Revit!

> Note: If you modify the configured commands after enabling the service, you may need to restart REVIT for the configuration to take effect. This is related to whether the command has already been registered.

## Custom Commands

You can refer to the [revit-mcp-commandset](https://github.com/revit-mcp/revit-mcp-commandset) project to develop custom commands.

## Project File Organization

```
revit-mcp-plugin/
├── Configuration/                            # Configuration management related classes
│   ├── CommandConfig.cs                      # Command configuration
│   ├── ConfigurationManager.cs               # Configuration manager
│   ├── DeveloperInfo.cs                      # Developer information
│   ├── FrameworkConfig.cs                    # Framework configuration
│   └── ServiceSettings.cs                    # Service settings
│
├── Core/                                     # Program entry and core functionality
│   ├── Application.cs                        # Application entry point
│   ├── CommandExecutor.cs                    # Command executor
│   ├── CommandManager.cs                     # Command manager
│   ├── ExternalEventManager.cs               # External event manager
│   ├── MCPServiceConnection.cs               # MCP service connection
│   ├── RevitCommandRegistry.cs               # Revit command registration
│   ├── Settings.cs                           # Application settings
│   └── SocketService.cs                      # Socket service implementation
│
├── Models/                                   # Data models
│   └── ...                                   # Various data model classes
│
├── UI/                                       # WPF form interfaces
│   └── ...                                   # Interface related classes
│
└── Utils/                                    # Utility classes
    ├── Logger.cs                             # Logging utility
    └── PathManager.cs                        # Path management utility
```

### Configuration Directory
Responsible for managing various configuration information for the plugin:

- CommandConfig.cs: Defines command-related configuration
- ConfigurationManager.cs: Manages loading, saving, and accessing configurations
- DeveloperInfo.cs: Stores developer-related information
- FrameworkConfig.cs: Framework-level configuration settings
- ServiceSettings.cs: Service-related settings

### Core Directory
Contains the core functionality and entry point of the plugin:

- Application.cs: Application entry point, responsible for initializing the plugin
- CommandExecutor.cs: Core component responsible for executing Revit commands
- CommandManager.cs: Manages and dispatches various commands in the plugin
- ExternalEventManager.cs: Manages Revit external events
- MCPServiceConnection.cs: MCP service connection
- RevitCommandRegistry.cs: Registers and manages available Revit commands
- Settings.cs: Triggers the display of the settings interface
- SocketService.cs: Implements Socket communication with external clients

### Models Directory
Contains data model classes used to pass data between different parts of the system.

### UI Directory
Contains user interface related components of the plugin, implemented using the WPF framework.

### Utils Directory
Provides various auxiliary tools:

- Logger.cs: Logging tool for debugging and error tracking
- PathManager.cs: Project-related file path management
