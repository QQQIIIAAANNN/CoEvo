using Newtonsoft.Json.Linq;
using RevitMCPSDK.API.Interfaces;
using RevitMCPSDK.API.Models;
using RevitMCPSDK.API.Models.JsonRPC;
using RevitMCPSDK.Exceptions;
using System;

namespace revit_mcp_plugin.Core
{
    public class CommandExecutor
    {
        private readonly ICommandRegistry _commandRegistry;
        private readonly ILogger _logger;

        public CommandExecutor(ICommandRegistry commandRegistry, ILogger logger)
        {
            _commandRegistry = commandRegistry;
            _logger = logger;
        }

        public string ExecuteCommand(JsonRPCRequest request)
        {
            try
            {
                // 查找命令
                if (!_commandRegistry.TryGetCommand(request.Method, out var command))
                {
                    _logger.Warning("未找到命令: {0}", request.Method);
                    return CreateErrorResponse(request.Id,
                        JsonRPCErrorCodes.MethodNotFound,
                        $"未找到方法: '{request.Method}'");
                }

                _logger.Info("执行命令: {0}", request.Method);

                // 同步执行命令
                try
                {
                    object result = ExternalEventManager.Instance.ExecuteAction(uiApp =>
                    {
                        // This code now runs in the Revit API context
                        return command.Execute(request.GetParamsObject(), request.Id);
                    });

                    _logger.Info("命令 {0} 执行成功", request.Method);

                    return CreateSuccessResponse(request.Id, result);
                }
                catch (CommandExecutionException ex)
                {
                    _logger.Error("命令 {0} 执行失败: {1}", request.Method, ex.Message);
                    return CreateErrorResponse(request.Id,
                        ex.ErrorCode,
                        ex.Message,
                        ex.ErrorData);
                }
                catch (Exception ex)
                {
                    // The exception might be from the Task framework if the TCS was faulted.
                    // We should check for inner exceptions.
                    var actualEx = ex.InnerException ?? ex;
                    _logger.Error("命令 {0} 执行时发生异常: {1}", request.Method, actualEx.Message);
                    return CreateErrorResponse(request.Id,
                        JsonRPCErrorCodes.InternalError,
                        actualEx.Message);
                }
            }
            catch (Exception ex)
            {
                _logger.Error("执行命令处理过程中发生异常: {0}", ex.Message);
                return CreateErrorResponse(request.Id,
                    JsonRPCErrorCodes.InternalError,
                    $"内部错误: {ex.Message}");
            }
        }

        private string CreateSuccessResponse(string id, object result)
        {
            var response = new JsonRPCSuccessResponse
            {
                Id = id,
                Result = result is JToken jToken ? jToken : JToken.FromObject(result)
            };

            return response.ToJson();
        }

        private string CreateErrorResponse(string id, int code, string message, object data = null)
        {
            var response = new JsonRPCErrorResponse
            {
                Id = id,
                Error = new JsonRPCError
                {
                    Code = code,
                    Message = message,
                    Data = data != null ? JToken.FromObject(data) : null
                }
            };

            return response.ToJson();
        }
    }
}
