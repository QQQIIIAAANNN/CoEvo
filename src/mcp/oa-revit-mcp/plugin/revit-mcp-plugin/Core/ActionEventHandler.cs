using Autodesk.Revit.UI;
using System;
using System.Threading.Tasks;

namespace revit_mcp_plugin.Core
{
    /// <summary>
    /// A wrapper to hold the action to be executed and its completion source.
    /// </summary>
    public class ActionWrapper
    {
        public Func<UIApplication, object> Action { get; set; }
        public TaskCompletionSource<object> Tcs { get; set; }
    }

    /// <summary>
    /// An external event handler that executes a single action.
    /// Concurrency and thread safety are handled by the caller.
    /// </summary>
    public class ActionEventHandler : IExternalEventHandler
    {
        private ActionWrapper _action;

        public void SetAction(ActionWrapper action)
        {
            _action = action;
        }

        public void Execute(UIApplication app)
        {
            if (_action != null)
            {
                try
                {
                    var result = _action.Action(app);
                    _action.Tcs.SetResult(result);
                }
                catch (Exception ex)
                {
                    _action.Tcs.SetException(ex);
                }
                finally
                {
                    _action = null;
                }
            }
        }

        public string GetName()
        {
            return "ActionEventHandler";
        }
    }
}
