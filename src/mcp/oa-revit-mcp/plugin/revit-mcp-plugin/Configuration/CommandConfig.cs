using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace revit_mcp_plugin.Configuration
{
    /// <summary>
    /// 命令配置类
    /// </summary>
    public class CommandConfig
    {
        /// <summary>
        /// 命令名称 - 对应IRevitCommand.CommandName
        /// </summary>
        [JsonProperty("commandName")]
        public string CommandName { get; set; }

        /// <summary>
        /// 程序集路径 - 包含此命令的DLL
        /// </summary>
        [JsonProperty("assemblyPath")]
        public string AssemblyPath { get; set; }

        /// <summary>
        /// 是否启用该命令
        /// </summary>
        [JsonProperty("enabled")]
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// 支持的Revit版本
        /// </summary>
        [JsonProperty("supportedRevitVersions")]
        public string[] SupportedRevitVersions { get; set; } = new string[0];

        /// <summary>
        /// 开发者信息
        /// </summary>
        [JsonProperty("developer")]
        public DeveloperInfo Developer { get; set; } = new DeveloperInfo();

        /// <summary>
        /// 命令描述
        /// </summary>
        [JsonProperty("description")]
        public string Description { get; set; } = "";
    }
}
