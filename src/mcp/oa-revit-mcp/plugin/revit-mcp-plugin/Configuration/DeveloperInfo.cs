using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace revit_mcp_plugin.Configuration
{
    /// <summary>
    /// 开发者信息
    /// </summary>
    public class DeveloperInfo
    {
        /// <summary>
        /// 开发者名称
        /// </summary>
        [JsonProperty("name")]
        public string Name { get; set; } = "";

        /// <summary>
        /// 开发者邮箱
        /// </summary>
        [JsonProperty("email")]
        public string Email { get; set; } = "";

        /// <summary>
        /// 开发者网站
        /// </summary>
        [JsonProperty("website")]
        public string Website { get; set; } = "";

        /// <summary>
        /// 开发者组织
        /// </summary>
        [JsonProperty("organization")]
        public string Organization { get; set; } = "";
    }
}
