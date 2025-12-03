import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { withRevitConnection } from "../../utils/ConnectionManager.js";

export function registerSendCodeToRevitTool(server: McpServer) {
  server.tool(
    "send_code_to_revit",
    "Executes a C# code snippet within the Revit application. Your code will be placed inside a method where you have access to a 'doc' variable (the active Revit Document) and a 'parameters' variable (an object[]). You must return a value from your script; this value will be serialized and sent back as the result. IMPORTANT: Do not use any UI-related APIs like 'TaskDialog.Show()' as they will fail in the execution context. The code has access to RevitAPI.dll and RevitAPIUI.dll.",
    {
      code: z
        .string()
        .describe(
          "The C# code to execute. Example: 'return doc.Title;'. This code will be inserted into a method with access to a 'doc' (Document) and 'parameters' (object[]) variable."
        ),
      parameters: z
        .array(z.any())
        .optional()
        .describe(
          "Optional execution parameters that will be passed to your code"
        ),
    },
    async (args, extra) => {
      const params = {
        code: args.code,
        parameters: args.parameters || [],
      };

      try {
        const response = await withRevitConnection(async (revitClient: any) => {
          return await revitClient.sendCommand("send_code_to_revit", params);
        });

        return {
          content: [
            {
              type: "text",
              text: `Code execution successful!\nResult: ${JSON.stringify(
                response,
                null,
                2
              )}`,
            },
          ],
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Code execution failed: ${
                error instanceof Error ? error.message : String(error)
              }`,
            },
          ],
        };
      }
    }
  );
}
