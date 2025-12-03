using Autodesk.Revit.UI;
using Newtonsoft.Json.Linq;
using RevitMCPSDK.API.Interfaces;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RevitMCPCommandSet.Services;

namespace RevitMCPCommandSet.Commands
{
    public class GenericCommand : IRevitCommand, IRevitCommandInitializable
    {
        private static readonly Dictionary<string, Type> _eventHandlerRegistry = new Dictionary<string, Type>();
        private static JArray _commandDefinitions;
        private IExternalEventHandler _handler;
        private UIApplication _uiApp;

        public string CommandName { get; private set; }

        public GenericCommand() { }

        public GenericCommand(string commandName)
        {
            CommandName = commandName;
        }

        public void Initialize(UIApplication uiApp)
        {
            _uiApp = uiApp;
            LoadCommandDefinitions();

            var commandDef = _commandDefinitions.FirstOrDefault(c => c["commandName"].ToString() == CommandName) as JObject;
            if (commandDef == null)
            {
                throw new InvalidOperationException($"Command '{CommandName}' not found in command.json.");
            }

            string commandType = commandDef["type"]?.ToString() ?? "generated";

            if (commandType == "script")
            {
                var scriptedHandler = new ScriptedCommandHandler();
                string scriptPath = commandDef["scriptPath"].ToString();

                // Construct the full path relative to the addin directory
                string addinPath = Path.GetDirectoryName(typeof(GenericCommand).Assembly.Location);
                string fullScriptPath = Path.GetFullPath(Path.Combine(addinPath, @"..\..\", scriptPath));

                scriptedHandler.SetScriptPath(fullScriptPath);
                _handler = scriptedHandler;
            }
            else
            {
                if (_eventHandlerRegistry.TryGetValue(CommandName, out Type handlerType))
                {
                    _handler = (IExternalEventHandler)Activator.CreateInstance(handlerType);
                }
            }
        }

        private static void LoadCommandDefinitions()
        {
            if (_commandDefinitions == null)
            {
                string addinPath = Path.GetDirectoryName(typeof(GenericCommand).Assembly.Location);
                string commandJsonPath = Path.GetFullPath(Path.Combine(addinPath, @"Commands\RevitMCPCommandSet", "command.json"));

                if (!File.Exists(commandJsonPath))
                {
                    throw new FileNotFoundException("command.json not found.", commandJsonPath);
                }

                string json = File.ReadAllText(commandJsonPath);
                JObject commandSet = JObject.Parse(json);
                _commandDefinitions = (JArray)commandSet["commands"];
            }
        }

        public static void RegisterEventHandler(string commandName, Type eventHandlerType)
        {
            _eventHandlerRegistry[commandName] = eventHandlerType;
        }

        public object Execute(JObject parameters, string requestId)
        {
            if (_handler == null)
            {
                throw new InvalidOperationException($"No event handler registered or created for command '{CommandName}'.");
            }

            var setParametersMethod = _handler.GetType().GetMethod("SetParameters");
            if (setParametersMethod != null)
            {
                setParametersMethod.Invoke(_handler, new object[] { parameters });
            }

            var externalEvent = ExternalEvent.Create(_handler);
            externalEvent.Raise();

            if (_handler is IWaitableExternalEventHandler waitableHandler)
            {
                if (waitableHandler.WaitForCompletion(60000))
                {
                    var resultProperty = _handler.GetType().GetProperty("Result");
                    if (resultProperty != null)
                    {
                        return resultProperty.GetValue(_handler);
                    }
                    return null;
                }
                else
                {
                    throw new TimeoutException($"Command '{CommandName}' timed out.");
                }
            }

            return null;
        }
    }
}
