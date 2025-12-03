using Autodesk.Revit.UI;
using RevitMCPSDK.API.Interfaces;
using System;
using Newtonsoft.Json.Linq;

namespace RevitMCPCommandSet.Services.Generated
{
    public class SendCodeToRevitEventHandler : IExternalEventHandler, IWaitableExternalEventHandler
    {
        public object Result { get; private set; }
        public bool TaskCompleted { get; private set; }
        private readonly System.Threading.ManualResetEvent _resetEvent = new System.Threading.ManualResetEvent(false);
        private string _code;

        public void SetParameters(JObject parameters)
        {
            _code = parameters["code"]?.ToString();
        }

        public void Execute(UIApplication app)
        {
            try
            {
                var codeExecutor = new CodeExecutor();
                Result = codeExecutor.Execute(app, _code);
            }
            catch (Exception ex)
            {
                Result = $"Error: {ex.Message}\nStackTrace: {ex.StackTrace}";
            }
            finally
            {
                TaskCompleted = true;
                _resetEvent.Set();
            }
        }

        public string GetName()
        {
            return "send_code_to_revit";
        }

        public bool WaitForCompletion(int timeoutMilliseconds = 60000)
        {
            return _resetEvent.WaitOne(timeoutMilliseconds);
        }
    }
}
