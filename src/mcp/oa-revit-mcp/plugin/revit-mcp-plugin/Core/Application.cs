using System;
using Autodesk.Revit.UI;
using System.Reflection;
using System.Windows.Media.Imaging;
using revit_mcp_plugin.Utils;

namespace revit_mcp_plugin.Core
{
    public class Application : IExternalApplication
    {
        public Result OnStartup(UIControlledApplication application)
        {
            RibbonPanel mcpPanel = application.CreateRibbonPanel("Revit MCP Plugin");

            PushButtonData pushButtonData = new("ID_EXCMD_TOGGLE_REVIT_MCP", "Revit MCP\r\n Switch",
                Assembly.GetExecutingAssembly().Location, "revit_mcp_plugin.Core.MCPServiceConnection")
            {
                ToolTip = "Open / Close mcp server",
                Image = new BitmapImage(new Uri("/revit-mcp-plugin;component/Core/Ressources/icon-16.png", UriKind.RelativeOrAbsolute)),
                LargeImage = new BitmapImage(new Uri("/revit-mcp-plugin;component/Core/Ressources/icon-32.png", UriKind.RelativeOrAbsolute))
            };
            mcpPanel.AddItem(pushButtonData);

            PushButtonData mcp_settings_pushButtonData = new("ID_EXCMD_MCP_SETTINGS", "Settings",
                Assembly.GetExecutingAssembly().Location, "revit_mcp_plugin.Core.Settings")
            {
                ToolTip = "MCP Settings",
                Image = new BitmapImage(new Uri("/revit-mcp-plugin;component/Core/Ressources/settings-16.png", UriKind.RelativeOrAbsolute)),
                LargeImage = new BitmapImage(new Uri("/revit-mcp-plugin;component/Core/Ressources/settings-32.png", UriKind.RelativeOrAbsolute))
            };
            mcpPanel.AddItem(mcp_settings_pushButtonData);

            return Result.Succeeded;
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            try
            {
                if (SocketService.Instance.IsRunning)
                {
                    SocketService.Instance.Stop();
                }

                // Dispose the event manager
                ExternalEventManager.Instance.Dispose();
            }
            catch (Exception ex)
            {
                var logger = new Logger();
                logger.Error("Error during shutdown: " + ex.ToString());
            }

            return Result.Succeeded;
        }
    }
}
