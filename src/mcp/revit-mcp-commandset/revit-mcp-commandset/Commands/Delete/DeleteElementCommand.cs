//using Autodesk.Revit.UI;
//using Newtonsoft.Json.Linq;
//using RevitMCPCommandSet.Services;
//using RevitMCPSDK.API.Base;
//using RevitMCPSDK.API.Models;
//using RevitMCPSDK.Exceptions;

//namespace RevitMCPCommandSet.Commands.Delete
//{
//    public class DeleteElementCommand : ExternalEventCommandBase
//    {
//        private DeleteElementEventHandler _handler => (DeleteElementEventHandler)Handler;
//        public override string CommandName => "delete_element";
//        public DeleteElementCommand(UIApplication uiApp)
//            : base(new DeleteElementEventHandler(), uiApp)
//        {
//        }
//        public override object Execute(JObject parameters, string requestId)
//        {
//            try
//            {
//                // 解析数组参数
//                var elementIds = parameters?["elementIds"]?.ToObject<string[]>();
//                if (elementIds == null || elementIds.Length == 0)
//                {
//                    throw new CommandExecutionException(
//                        "元素ID列表不能为空",
//                        JsonRPCErrorCodes.InvalidParams);
//                }
//                // 设置要删除的元素ID数组
//                _handler.ElementIds = elementIds;
//                // 触发外部事件并等待完成
//                if (RaiseAndWaitForCompletion(15000))
//                {
//                    if (_handler.IsSuccess)
//                    {
//                        return CommandResult.CreateSuccess(new { deleted = true, count = _handler.DeletedCount });
//                    }
//                    else
//                    {
//                        throw new CommandExecutionException(
//                            "删除元素失败",
//                            JsonRPCErrorCodes.ElementDeletionFailed);
//                    }
//                }
//                else
//                {
//                    throw CreateTimeoutException(CommandName);
//                }
//            }
//            catch (CommandExecutionException)
//            {
//                throw;
//            }
//            catch (Exception ex)
//            {
//                throw new CommandExecutionException(
//                    $"删除元素失败: {ex.Message}",
//                    JsonRPCErrorCodes.InternalError);
//            }
//        }
//    }
//}
