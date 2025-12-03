using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using System;

namespace revit_mcp_plugin.Core
{
    [Transaction(TransactionMode.Manual)]
    public class MCPServiceConnection : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                // 获取socket服务
                SocketService service = SocketService.Instance;

                if (service.IsRunning)
                {
                    service.Stop();
                    TaskDialog.Show("revitMCP", "Close Server");
                }
                else
                {
                    // Ensure the service is initialized, but only once.
                    if (!service.IsInitialized)
                    {
                        service.Initialize(commandData.Application);
                    }
                    service.Start();
                    TaskDialog.Show("revitMCP", "Open Server");
                }

                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                message = ex.Message;
                return Result.Failed;
            }
        }
    }
}
