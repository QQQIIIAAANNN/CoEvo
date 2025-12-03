using System;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace CodeGen
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Usage: CodeGen <path_to_command.json> <csharp_output_directory> <server_output_directory>");
                return;
            }

            string commandJsonPath = args[0];
            string csharpOutputDir = args[1];
            string serverOutputDir = args[2];

            if (!File.Exists(commandJsonPath))
            {
                Console.WriteLine($"Error: command.json not found at '{commandJsonPath}'");
                return;
            }

            Directory.CreateDirectory(csharpOutputDir);
            Directory.CreateDirectory(serverOutputDir);

            string json = File.ReadAllText(commandJsonPath);
            JObject commandSet = JObject.Parse(json);
            JArray commands = (JArray)commandSet["commands"];

            foreach (JObject command in commands)
            {
                string commandName = command["commandName"].Value<string>();
                string description = command["description"]?.Value<string>() ?? commandName;

                string commandType = command["type"]?.Value<string>() ?? "generated";

                if (commandType != "script")
                {
                    // Generate C# Event Handler
                    string className = ToPascalCase(commandName) + "EventHandler";
                    string csharpFilePath = Path.Combine(csharpOutputDir, className + ".cs");

                    if (File.Exists(csharpFilePath))
                    {
                        Console.WriteLine($"Skipping existing C# file: {className}.cs");
                    }
                    else
                    {
                        string csharpFileContent = GenerateEventHandlerClass(className, commandName);
                        File.WriteAllText(csharpFilePath, csharpFileContent);
                        Console.WriteLine($"Generated {className}.cs");
                    }
                }

                // Generate Server Tool
                string serverToolFilePath = Path.Combine(serverOutputDir, commandName + ".ts");
                if (File.Exists(serverToolFilePath))
                {
                    Console.WriteLine($"Skipping existing server tool file: {commandName}.ts");
                }
                else
                {
                    string serverToolFileContent = GenerateServerTool(commandName, description);
                    File.WriteAllText(serverToolFilePath, serverToolFileContent);
                    Console.WriteLine($"Generated {commandName}.ts");
                }
            }
        }

        static string GenerateEventHandlerClass(string className, string commandName)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("using Autodesk.Revit.UI;");
            sb.AppendLine("using RevitMCPSDK.API.Interfaces;");
            sb.AppendLine("using System;");
            sb.AppendLine();
            sb.AppendLine("namespace RevitMCPCommandSet.Services.Generated");
            sb.AppendLine("{");
            sb.AppendLine($"    public class {className} : IExternalEventHandler, IWaitableExternalEventHandler");
            sb.AppendLine("    {");
            sb.AppendLine("        public object Result { get; private set; }");
            sb.AppendLine("        public bool TaskCompleted { get; private set; }");
            sb.AppendLine("        private readonly System.Threading.ManualResetEvent _resetEvent = new System.Threading.ManualResetEvent(false);");
            sb.AppendLine();
            sb.AppendLine("        public void Execute(UIApplication app)");
            sb.AppendLine("        {");
            sb.AppendLine("            try");
            sb.AppendLine("            {");
            sb.AppendLine("                // TODO: Implement the command logic here.");
            sb.AppendLine("                Result = null;");
            sb.AppendLine("            }");
            sb.AppendLine("            catch (Exception ex)");
            sb.AppendLine("            {");
            sb.AppendLine("                System.Diagnostics.Debug.WriteLine(ex.ToString());");
            sb.AppendLine("            }");
            sb.AppendLine("            finally");
            sb.AppendLine("            {");
            sb.AppendLine("                TaskCompleted = true;");
            sb.AppendLine("                _resetEvent.Set();");
            sb.AppendLine("            }");
            sb.AppendLine("        }");
            sb.AppendLine();
            sb.AppendLine("        public string GetName()");
            sb.AppendLine("        {");
            sb.AppendLine($"            return \"{commandName}\";");
            sb.AppendLine("        }");
            sb.AppendLine();
            sb.AppendLine("        public bool WaitForCompletion(int timeoutMilliseconds = 15000)");
            sb.AppendLine("        {");
            sb.AppendLine("            return _resetEvent.WaitOne(timeoutMilliseconds);");
            sb.AppendLine("        }");
            sb.AppendLine("    }");
            sb.AppendLine("}");
            return sb.ToString();
        }

        static string ToPascalCase(string snakeCase)
        {
            return string.Concat(snakeCase.Split('_').Select(s => char.ToUpper(s[0]) + s.Substring(1)));
        }

        static string GenerateServerTool(string commandName, string description)
        {
            string registerFunctionName = "register" + ToPascalCase(commandName) + "Tool";

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("import { z } from \"zod\";");
            sb.AppendLine("import { McpServer } from \"@modelcontextprotocol/sdk/server/mcp.js\";");
            sb.AppendLine("import { withRevitConnection } from \"../utils/ConnectionManager.js\";");
            sb.AppendLine();
            sb.AppendLine($"export function {registerFunctionName}(server: McpServer) {{");
            sb.AppendLine($"  server.tool(");
            sb.AppendLine($"    \"{commandName}\",");
            sb.AppendLine($"    \"{description}\",");
            sb.AppendLine("    {");
            sb.AppendLine("      // TODO: Define your Zod schema for the arguments here");
            sb.AppendLine("    },");
            sb.AppendLine("    async (args, extra) => {");
            sb.AppendLine("      try {");
            sb.AppendLine("        const response = await withRevitConnection(async (revitClient) => {");
            sb.AppendLine("          return await revitClient.sendCommand(");
            sb.AppendLine($"            \"{commandName}\",");
            sb.AppendLine("            args");
            sb.AppendLine("          );");
            sb.AppendLine("        });");
            sb.AppendLine();
            sb.AppendLine("        return {");
            sb.AppendLine("          content: [");
            sb.AppendLine("            {");
            sb.AppendLine("              type: \"text\",");
            sb.AppendLine("              text: JSON.stringify(response, null, 2),");
            sb.AppendLine("            },");
            sb.AppendLine("          ],");
            sb.AppendLine("        };");
            sb.AppendLine("      } catch (error) {");
            sb.AppendLine("        return {");
            sb.AppendLine("          content: [");
            sb.AppendLine("            {");
            sb.AppendLine("              type: \"text\",");
            sb.AppendLine($"              text: `'{commandName}' failed: ${{error instanceof Error ? error.message : String(error)}}`,");
            sb.AppendLine("            },");
            sb.AppendLine("          ],");
            sb.AppendLine("        };");
            sb.AppendLine("      }");
            sb.AppendLine("    }");
            sb.AppendLine("  );");
            sb.AppendLine("}");
            return sb.ToString();
        }
    }
}
