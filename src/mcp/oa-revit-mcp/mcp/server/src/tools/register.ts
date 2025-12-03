import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

export async function registerTools(server: McpServer) {
  // 获取当前文件的目录路径
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);

  const generatedDir = path.join(__dirname, "generated");

  // Read all files from the generated directory
  const files = fs.readdirSync(generatedDir);

  // Filter for .ts or .js files, excluding index and register files
  const toolFiles = files.filter(
    (file) =>
      (file.endsWith(".ts") || file.endsWith(".js")) &&
      !file.startsWith("index.") &&
      !file.startsWith("register.")
  );

  // Dynamically import and register each tool
  for (const file of toolFiles) {
    try {
      // Construct the import path
      const importPath = `./generated/${file.replace(/\.(ts|js)$/, ".js")}`;

      // Dynamically import the module
      const module = await import(importPath);

      // Find and execute the registration function
      const registerFunctionName = Object.keys(module).find(
        (key) => key.startsWith("register") && typeof module[key] === "function"
      );

      if (registerFunctionName) {
        module[registerFunctionName](server);
        console.error(`Registered tool: ${file}`);
      } else {
        console.warn(`Warning: No registration function found in file ${file}`);
      }
    } catch (error) {
      console.error(`Error registering tool ${file}:`, error);
    }
  }
}
