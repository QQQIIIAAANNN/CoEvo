using Autodesk.Revit.UI;
using RevitMCPSDK.API.Interfaces;
using RevitMCPSDK.API.Utils;
using revit_mcp_plugin.Configuration;
using revit_mcp_plugin.Utils;
using System;
using System.IO;
using System.Reflection;

namespace revit_mcp_plugin.Core
{
    /// <summary>
    /// 命令管理器，负责加载和管理命令
    /// </summary>
    public class CommandManager
    {
        private readonly ICommandRegistry _commandRegistry;
        private readonly ILogger _logger;
        private readonly ConfigurationManager _configManager;
        private readonly UIApplication _uiApplication;
        private readonly RevitVersionAdapter _versionAdapter;

        public CommandManager(
            ICommandRegistry commandRegistry,
            ILogger logger,
            ConfigurationManager configManager,
            UIApplication uiApplication)
        {
            _commandRegistry = commandRegistry;
            _logger = logger;
            _configManager = configManager;
            _uiApplication = uiApplication;
            _versionAdapter = new RevitVersionAdapter(_uiApplication.Application);
        }

        /// <summary>
        /// 加载配置文件中指定的所有命令
        /// </summary>
        public void LoadCommands()
        {
            _logger.Info("开始加载命令");
            string currentVersion = _versionAdapter.GetRevitVersion();
            _logger.Info("当前 Revit 版本: {0}", currentVersion);

            // 从配置加载外部命令
            foreach (var commandConfig in _configManager.Config.Commands)
            {
                try
                {
                    if (!commandConfig.Enabled)
                    {
                        _logger.Info("跳过禁用的命令: {0}", commandConfig.CommandName);
                        continue;
                    }

                    // 检查版本兼容性
                    if (commandConfig.SupportedRevitVersions != null &&
                        commandConfig.SupportedRevitVersions.Length > 0 &&
                        !_versionAdapter.IsVersionSupported(commandConfig.SupportedRevitVersions))
                    {
                        _logger.Warning("命令 {0} 不支持当前 Revit 版本 {1}，已跳过",
                            commandConfig.CommandName, currentVersion);
                        continue;
                    }

                    // 替换路径中的版本占位符
                    commandConfig.AssemblyPath = commandConfig.AssemblyPath.Contains("{VERSION}")
                        ? commandConfig.AssemblyPath.Replace("{VERSION}", currentVersion)
                        : commandConfig.AssemblyPath;

                    // 加载外部命令程序集
                    LoadCommandFromAssembly(commandConfig);
                }
                catch (Exception ex)
                {
                    _logger.Error("加载命令 {0} 失败: {1}", commandConfig.CommandName, ex.Message);
                }
            }

            _logger.Info("命令加载完成");
        }

        private void LoadCommandFromAssembly(CommandConfig config)
        {
            try
            {
                // 确定程序集路径
                string assemblyPath = config.AssemblyPath;
                if (!Path.IsPathRooted(assemblyPath))
                {
                    // 如果不是绝对路径，则相对于Commands目录
                    string baseDir = PathManager.GetCommandsDirectoryPath();
                    assemblyPath = Path.Combine(baseDir, assemblyPath);
                }

                if (!File.Exists(assemblyPath))
                {
                    _logger.Error("命令程序集不存在: {0}", assemblyPath);
                    return;
                }

                // 加载程序集
                Assembly assembly = Assembly.LoadFrom(assemblyPath);

                // 查找并注册所有 IExternalEventHandler
                foreach (Type type in assembly.GetTypes())
                {
                    if (typeof(IExternalEventHandler).IsAssignableFrom(type) && !type.IsInterface && !type.IsAbstract)
                    {
                        try
                        {
                            IExternalEventHandler handler = (IExternalEventHandler)Activator.CreateInstance(type);
                            string commandName = handler.GetName();
                            if (!string.IsNullOrEmpty(commandName))
                            {
                                // This is where the magic happens. We register the handler type with our generic command.
                                // Note: I will need to add a reference to the GenericCommand class, which is in a different project.
                                // I will assume for now that the reference is added.
                                RevitMCPCommandSet.Commands.GenericCommand.RegisterEventHandler(commandName, type);
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.Error("创建事件处理器实例失败 [{0}]: {1}", type.FullName, ex.Message);
                        }
                    }
                }

                // Now, create and register the generic command for the specific command config.
                var genericCommand = new RevitMCPCommandSet.Commands.GenericCommand(config.CommandName);
                genericCommand.Initialize(_uiApplication);
                _commandRegistry.RegisterCommand(genericCommand);
                _logger.Info("已注册通用命令: {0} (来自 {1})", config.CommandName, Path.GetFileName(assemblyPath));

            }
            catch (Exception ex)
            {
                _logger.Error("加载命令程序集失败: {0}", ex.Message);
            }
        }
    }
}
