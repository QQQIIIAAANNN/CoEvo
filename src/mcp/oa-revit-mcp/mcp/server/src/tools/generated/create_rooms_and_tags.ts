import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { withRevitConnection } from "../utils/ConnectionManager.js";

export function registerCreateRoomsAndTagsTool(server: McpServer) {
  server.tool(
    "create_rooms_and_tags",
    "在所有封閉區域中建立房間，並在每個房間放 RoomTag",
    {
      // TODO: Define your Zod schema for the arguments here
    },
    async (args, extra) => {
      try {
        const response = await withRevitConnection(async (revitClient) => {
          return await revitClient.sendCommand(
            "create_rooms_and_tags",
            args
          );
        });

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(response, null, 2),
            },
          ],
        };
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `'create_rooms_and_tags' failed: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
        };
      }
    }
  );
}
