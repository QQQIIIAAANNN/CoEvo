using Autodesk.Revit.UI;
using Newtonsoft.Json.Linq;
using RevitMCPSDK.API.Base;
using RevitMCPCommandSet.Services.Generated; // 引用 Handler 命名空間
using System;

namespace RevitMCPCommandSet.Commands
{
    // 1. 繼承 ExternalEventCommandBase
    public class CreateRoomsAndTagsCommand : ExternalEventCommandBase
    {
        // 2. 宣告您的 Service Handler
        private CreateRoomsAndTagsEventHandler _handler => (CreateRoomsAndTagsEventHandler)Handler;

        // 3. 定義命令名稱 (必須與您在 AI/JSON 定義的 tool name 完全一致)
        public override string CommandName => "create_rooms_and_tags"; 

        // 4. 建構函數：將 Handler 實例傳給基底類別
        public CreateRoomsAndTagsCommand(UIApplication uiApp)
            : base(new CreateRoomsAndTagsEventHandler(), uiApp)
        {
        }

        // 5. 實作執行邏輯
        public override object Execute(JObject parameters, string requestId)
        {
            try
            {
                // 解析參數
                // 嘗試從參數中獲取 "roomName"
                string roomName = parameters.Value<string>("roomName");
                
                // 將參數傳給 Handler
                if (!string.IsNullOrEmpty(roomName))
                {
                    _handler.RoomName = roomName;
                }
                else
                {
                    _handler.RoomName = null; // 確保重置
                }

                // 解析座標參數
                double? x = null;
                double? y = null;
                
                JToken xToken = parameters["x"];
                JToken yToken = parameters["y"];

                if (xToken != null && xToken.Type != JTokenType.Null)
                {
                    x = xToken.Value<double>();
                }

                if (yToken != null && yToken.Type != JTokenType.Null)
                {
                    y = yToken.Value<double>();
                }

                _handler.X = x;
                _handler.Y = y;

                // C. 觸發外部事件並等待完成 (預設等待 60 秒)
                if (RaiseAndWaitForCompletion(60000)) // 給予較長的時間 (60s) 因為建立房間可能較耗時
                {
                    // D. 返回執行結果
                    return _handler.Result;
                }
                else
                {
                    throw new TimeoutException("建立房間與標籤操作逾時");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"執行 create_rooms_and_tags 失敗: {ex.Message}");
            }
        }
    }
}
