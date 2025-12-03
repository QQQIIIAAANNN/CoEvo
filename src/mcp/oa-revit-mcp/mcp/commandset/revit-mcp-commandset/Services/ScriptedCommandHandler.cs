using Autodesk.Revit.UI;
using RevitMCPSDK.API.Interfaces;
using System;
using System.IO;
using Newtonsoft.Json.Linq;

namespace RevitMCPCommandSet.Services
{
    public class ScriptedCommandHandler : IExternalEventHandler, IWaitableExternalEventHandler
    {
        public object Result { get; private set; }
        public bool TaskCompleted { get; private set; }
        private readonly System.Threading.ManualResetEvent _resetEvent = new System.Threading.ManualResetEvent(false);
        private string _scriptPath;
        private JObject _parameters;

        public void SetParameters(JObject parameters)
        {
            _parameters = parameters;
        }

        public void SetScriptPath(string scriptPath)
        {
            _scriptPath = scriptPath;
        }

        public void Execute(UIApplication app)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(_scriptPath) || !File.Exists(_scriptPath))
                {
                    throw new InvalidOperationException($"Script file not found at '{_scriptPath}'.");
                }

                string code = File.ReadAllText(_scriptPath);

                // TODO: Find a way to pass parameters to the script.
                // For now, they are ignored.

                var codeExecutor = new CodeExecutor();
                Result = codeExecutor.Execute(app, code);
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
            return "scripted_command_handler";
        }

        public bool WaitForCompletion(int timeoutMilliseconds = 60000)
        {
            return _resetEvent.WaitOne(timeoutMilliseconds);
        }
    }
}
