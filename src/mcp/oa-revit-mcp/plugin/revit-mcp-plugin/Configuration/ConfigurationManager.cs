using Newtonsoft.Json;
using RevitMCPSDK.API.Interfaces;
using revit_mcp_plugin.Utils;
using System;
using System.IO;

namespace revit_mcp_plugin.Configuration
{
    public class ConfigurationManager
    {
        private readonly ILogger _logger;
        private readonly string _configPath;

        public FrameworkConfig Config { get; private set; }

        public ConfigurationManager(ILogger logger)
        {
            _logger = logger;

            // 配置文件路径
            _configPath = PathManager.GetCommandRegistryFilePath();
        }

        /// <summary>
        /// 加载配置
        /// </summary>
        public void LoadConfiguration()
        {
            try
            {
                if (File.Exists(_configPath))
                {
                    string json = File.ReadAllText(_configPath);
                    Config = JsonConvert.DeserializeObject<FrameworkConfig>(json);
                    _logger.Info("已加载配置文件: {0}", _configPath);
                }
                else
                {
                    _logger.Error("未找到配置文件");
                }
            }
            catch (Exception ex)
            {
                _logger.Error("加载配置文件失败: {0}", ex.Message);
            }

            // 记录加载时间
            _lastConfigLoadTime = DateTime.Now;
        }

        ///// <summary>
        ///// 重新加载配置
        ///// </summary>
        //public void RefreshConfiguration()
        //{
        //    LoadConfiguration();
        //    _logger.Info("配置已重新加载");
        //}

        //public bool HasConfigChanged()
        //{
        //    if (!File.Exists(_configPath))
        //        return false;

        //    DateTime lastWrite = File.GetLastWriteTime(_configPath);
        //    return lastWrite > _lastConfigLoadTime;
        //}

        private DateTime _lastConfigLoadTime;
    }
}
