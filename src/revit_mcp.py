# src/mcp_test.py
import os
import asyncio
import json
import traceback
import sys
import uuid
import base64
import time
import re
from collections import defaultdict

try:
    from PIL import Image as PILImage
except ImportError:
    print("錯誤：需要 Pillow 庫來處理圖像。請執行 pip install Pillow")
    PILImage = None # Indicate PIL is not available

if sys.platform.startswith("win"):
    # 強制使用 ProactorEventLoop，以支援 subprocess
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List, Union # Added Union
from contextlib import asynccontextmanager
import requests
import atexit
import platform


from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, add_messages

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

# =============================================================================
# API Key 動態管理 (Free/VIP 切換)
# =============================================================================
class APIKeyManager:
    """管理 Free 和 VIP API Key 的自動切換"""
    def __init__(self):
        self.using_vip = False
        self.vip_calls_remaining = 0
        self.last_error_time = 0
        self.consecutive_quota_errors = 0
        
    def should_use_vip(self) -> bool:
        """判斷是否應該使用 VIP Key"""
        return self.using_vip and self.vip_calls_remaining > 0
    
    def handle_quota_error(self) -> bool:
        """
        處理配額錯誤，切換到 VIP Key
        Returns: True if switched to VIP, False if no VIP available
        """
        if not os.getenv("GEMINI_API_KEY_VIP"):
            print("  !! ❌ 配額已滿且無 VIP Key (GEMINI_API_KEY_VIP)，無法繼續")
            return False
            
        self.consecutive_quota_errors += 1
        current_time = time.time()
        
        # 如果在短時間內連續遇到配額錯誤，增加 VIP 使用輪數
        if current_time - self.last_error_time < 60:  # 1分鐘內
            vip_rounds = min(10, 5 * self.consecutive_quota_errors)  # 最多10輪
        else:
            vip_rounds = 5
            self.consecutive_quota_errors = 1
        
        self.using_vip = True
        self.vip_calls_remaining = vip_rounds
        self.last_error_time = current_time
        
        print(f"  >> ⚠️  配額已滿，切換到 VIP Key，將執行 {vip_rounds} 輪後返回免費版")
        return True
    
    def decrement_vip_calls(self):
        """VIP 調用計數遞減"""
        if self.vip_calls_remaining > 0:
            self.vip_calls_remaining -= 1
            print(f"  >> 💎 VIP Key 剩餘調用: {self.vip_calls_remaining}")
            
            if self.vip_calls_remaining == 0:
                self.using_vip = False
                self.consecutive_quota_errors = 0
                print("  >> ✅ VIP 輪數用完，返回免費 Key")
    
    def get_current_key_type(self) -> str:
        """獲取當前 Key 類型描述"""
        return "VIP💎" if self.should_use_vip() else "Free🆓"

# 全局管理器實例
api_key_manager = APIKeyManager()

# --- LLM Setup ---
try:
    api_key = os.getenv("GEMINI_API_KEY")
    api_key_vip = os.getenv("GEMINI_API_KEY_VIP")

    if not api_key:
        print("錯誤：找不到 GEMINI_API_KEY 環境變數。")
        exit(1)
    else:
        # 1. 初始化免費版 LLM
        agent_llm_free = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.5,
            google_api_key=api_key
        )
        
        fast_llm_free = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.5,
            google_api_key=api_key
        )

        # 2. 初始化 VIP 版 LLM (如果有的話)
        agent_llm_vip = None
        fast_llm_vip = None
        if api_key_vip:
            agent_llm_vip = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0.5,
                google_api_key=api_key_vip
            )
            fast_llm_vip = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.5,
                google_api_key=api_key_vip
            )
            print(f"✅ VIP Agent LLM 初始化成功 (備用，遇到配額限制時自動切換)。")
        else:
            print("⚠️  未設置 GEMINI_API_KEY_VIP，遇到配額限制時將無法自動切換")
        
        # 預設主要 LLM 指向免費版
        agent_llm = agent_llm_free
        fast_llm = fast_llm_free
        
        print(f"Agent LLM ({agent_llm.model}) 初始化成功 (預設 Free)。")
        print(f"Fast LLM ({fast_llm.model}) 初始化成功 (預設 Free)。")

    utility_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    print("Utility LLM (OpenAI for Router) 初始化成功。")
except Exception as e:
    print(f"ERROR: 無法初始化 LLM。錯誤: {e}")
    traceback.print_exc()
    exit(1)

# --- MCP Server Configurations ---
MCP_CONFIGS = {
    "revit": {
        "command": "node",
        "args": ["D:\\MA system\\LangGraph\\src\\mcp\\revit-mcp\\build\\index.js"],
        "transport": "stdio",
    },
}

# --- 全局變數 ---
_loaded_mcp_tools: Dict[str, List[BaseTool]] = {}
_mcp_clients: Dict[str, MultiServerMCPClient] = {}
_mcp_init_lock = asyncio.Lock()

# =============================================================================
# 定義狀態 (State)
# =============================================================================
class MCPAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    initial_image_path: Optional[str]
    task_complete: bool = False
    # --- 用於存儲截圖/下載結果的字段 ---
    saved_image_path: Optional[str] # Stores the path returned by Revit
    saved_image_data_uri: Optional[str] # Stores the generated data URI
    # --- <<< 新增：連續文本響應計數器 >>> ---
    consecutive_llm_text_responses: int = 0 # Track consecutive non-tool/non-completion AI messages
    last_executed_node: Optional[str] = None # 記錄最後執行的節點名稱
    # --- 新增: 存儲CSV報告路徑 ---
    saved_csv_path: Optional[str] = None
    # --- 新增: LLM 調用延遲時間管理 (秒) ---
    rpm_delay: float = 12.5  # flash預設 6.5 秒，避免速率限制，如果有付費可以改為0.5  pro預設12

# =============================================================================
# 本地工具定義 (Local Tools)
# =============================================================================
@tool
def create_planned_data_summary_csv(data_rows: List[Dict[str, Union[str, float]]], total_area: float, bcr: Optional[float], far: Optional[float], filename: str = "planned_project_summary.csv") -> str:
    """
    根據「規劃好」的設計數據生成CSV摘要文件。
    此工具不與Revit互動；它只記錄計畫中提供的數據。
    在規劃階段結束時使用此工具，以創建設計意圖的摘要。

    Args:
        data_rows: 一個字典列表，每個字典代表一個空間。必須包含 'name' (str), 'area' (float), 'percentage' (float) 和 'floor' (str, 例如 "Floor 1") 鍵。
        total_area: 規劃的總樓地板面積 (float)。
        bcr: 規劃的建蔽率 (float, 百分比)。如果無則為空。
        far: 規劃的容積率 (float)。如果無則為空。
        filename: 輸出的CSV文件名。預設為 "planned_project_summary.csv"。

    Returns:
        一個確認成功和保存文件路徑的字串，以 [CSV_FILE_PATH]: 為前綴。
    """
    import csv
    import time
    from collections import defaultdict
    output_dir = r"D:\MA system\LangGraph\output\space"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base, ext = os.path.splitext(filename)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_filename = f"{base}_{timestamp}{ext}"
    file_path = os.path.join(output_dir, new_filename)
    headers = ["Space Name", "Area (sqm)", "Percentage (%)"]

    spaces_by_floor = defaultdict(list)
    for row in data_rows:
        floor = row.get('floor', 'Unassigned')
        spaces_by_floor[floor].append(row)
    
    try:
        with open(file_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.writer(csv_file)
            
            writer.writerow(["Project Summary (Based on Plan)"])
            writer.writerow(["Total Planned Floor Area (sqm)", round(total_area, 2)])
            writer.writerow(["Planned Building Coverage Ratio (%)", bcr if bcr is not None else "N/A"])
            writer.writerow(["Planned Floor Area Ratio", far if far is not None else "N/A"])
            writer.writerow([])
            
            writer.writerow(["Planned Space Details"])

            sorted_floors = sorted(spaces_by_floor.keys())
            for floor in sorted_floors:
                writer.writerow([])
                writer.writerow([f"--- {floor} ---"])
                writer.writerow(headers)
                for row in spaces_by_floor[floor]:
                    writer.writerow([
                        row.get('name', 'N/A'), 
                        round(row.get('area', 0.0), 2),
                        round(row.get('percentage', 0.0), 2)
                    ])
                
        return f"[CSV_FILE_PATH]:{file_path}"
    except Exception as e:
        return f"[ERROR] Failed to create planned summary table: {str(e)}"

# --- 新增: 本地工具列表 ---
LOCAL_TOOLS = [create_planned_data_summary_csv]
print(f"[INIT] LOCAL_TOOLS 已定義: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in LOCAL_TOOLS]}")

# =============================================================================
# 工具管理 (使用 print 替換 logging)
# =============================================================================
async def initialize_single_mcp(mcp_name: str) -> tuple[Optional[MultiServerMCPClient], List[BaseTool]]:
    """初始化單個 MCP 連接並獲取其工具 (使用 print)。"""
    print(f"--- [Lazy Init] 正在初始化 {mcp_name} MCP 連接 ---")
    config_item = MCP_CONFIGS.get(mcp_name)
    if not config_item:
        print(f"  !!! [Lazy Init] 錯誤: 在 MCP_CONFIGS 中找不到 {mcp_name} 的配置。")
        return None, []

    client = None
    tools = []
    try:
        # --- 命令和路徑檢查 (使用 print) ---
        command_path = config_item['command']
        # 檢查命令是否存在 (對 'python' 這類通用命令可能不適用)
        if command_path != "python" and not os.path.exists(command_path) and command_path != sys.executable:
            print(f"  !!! [Lazy Init] 警告: 命令路徑 '{command_path}' 不存在。")
        # 檢查 args 中的文件路徑 (如果有的話)
        for arg in config_item.get('args', []):
             # Check if it looks like a file path and doesn't exist
             if ('/' in arg or '\\' in arg) and not os.path.exists(arg):
                  print(f"  !!! [Lazy Init] 警告: 參數中的路徑 '{arg}' 不存在。")

        # print(f"  - [Lazy Init] 使用配置: {config_item}")
        print(f"  - [Lazy Init] 正在初始化 {mcp_name} Client...")
        try:
            single_server_config = {mcp_name: config_item}
            client = MultiServerMCPClient(single_server_config)
            print(f"  - [Lazy Init] {mcp_name} Client 初始化完成。")
        except Exception as client_init_e:
            print(f"  !!! [Lazy Init] 客戶端初始化錯誤: {client_init_e}")
            traceback.print_exc()
            return None, []

        # --- 連接和獲取工具 (使用 print) ---
        try:
            print(f"  - [Lazy Init] 正在啟動 {mcp_name} Client 連接 (__aenter__)...")
            await client.__aenter__()
            print(f"  - [Lazy Init] {mcp_name} Client 連接成功。")

            print(f"  - [Lazy Init] [開始] 正在從 {mcp_name} 獲取工具 (get_tools)...")
            try:
                tools = client.get_tools()
                print(f"  - [Lazy Init] [完成] 從 {mcp_name} 獲取工具完成。數量: {len(tools)}")
                if not tools:
                    print(f"  !!! [Lazy Init] 警告: {mcp_name} 返回了空的工具列表 !!!")
                else:
                    # --- 打印工具信息 (可選，保持開啟以供調試) ---
                    print(f"  --- 可用工具列表 ({mcp_name}) ---")
                    for i, tool in enumerate(tools):
                        tool_info = f"    工具 {i+1}: Name='{tool.name}'"
                        if hasattr(tool, 'description') and tool.description:
                             tool_info += f", Desc='{tool.description[:60]}...'"
                        print(tool_info)
                    print(f"  --- 工具列表結束 ({mcp_name}) ---")
            except Exception as tools_e:
                print(f"  !!! [Lazy Init] 獲取工具錯誤: {tools_e}")
                traceback.print_exc()
                tools = []
        except Exception as enter_e:
            print(f"  !!! [Lazy Init] 客戶端連接或獲取工具錯誤: {enter_e}")
            traceback.print_exc()
            if client:
                try:
                    print(f"  -- [Cleanup Attempt] 嘗試關閉失敗的 {mcp_name} client...")
                    await client.__aexit__(type(enter_e), enter_e, enter_e.__traceback__)
                    print(f"  -- [Cleanup Attempt] 關閉 {mcp_name} client 完成。")
                except Exception as exit_e:
                    print(f"  -- [Cleanup Attempt] 關閉 {mcp_name} client 時也發生錯誤: {exit_e}")
                    traceback.print_exc()
            client = None
            tools = []
        print(f"--- [Lazy Init] {mcp_name.capitalize()} 初始化流程完成 ---")
    except Exception as inner_e:
        print(f"!!!!! [Lazy Init] 錯誤: 在處理 {mcp_name} MCP 時發生外部錯誤 !!!!!")
        traceback.print_exc()
        client = None
        tools = []
    return client, tools

# --- shutdown_mcp_clients (使用 print) ---
async def shutdown_mcp_clients(clients_to_shutdown: Dict[str, MultiServerMCPClient]):
    print("\n--- [Cleanup] 正在關閉 MCP Client 連接 ---")
    if not clients_to_shutdown:
        print("  沒有需要關閉的客戶端。")
        return
    for name, client in clients_to_shutdown.items():
        try:
            print(f"  - 正在關閉 {name} Client (__aexit__)...")
            await client.__aexit__(None, None, None)
            print(f"  - {name} Client 已關閉")
        except Exception as close_e:
            print(f"錯誤: 關閉 {name} Client 時發生錯誤: {close_e}")
            traceback.print_exc()
    print("--- [Cleanup] 所有 MCP Client 已嘗試關閉 ---")

# --- _sync_cleanup (使用 print, 移除 _mcp_clients_initialized 檢查) ---
def _sync_cleanup():
    global _mcp_clients
    # 只檢查 _mcp_clients 是否有內容
    if _mcp_clients:
        print("--- [atexit] 檢測到需要清理 MCP 客戶端 ---")
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                loop.create_task(shutdown_mcp_clients(_mcp_clients))
                print("--- [atexit] 清理任務已創建 (循環運行中) ---")
            else:
                loop.run_until_complete(shutdown_mcp_clients(_mcp_clients))
                print("--- [atexit] 清理任務已同步執行 ---")
        except Exception as cleanup_err:
            print(f"--- [atexit] 執行異步清理時出錯: {cleanup_err} ---")
            traceback.print_exc()
        finally:
            _mcp_clients = {}
    else:
         print("--- [atexit] 無需清理 MCP 客戶端 ---")

atexit.register(_sync_cleanup)

# --- get_mcp_tools (使用 print) ---
async def get_mcp_tools(mcp_name: str) -> List[BaseTool]:
    global _loaded_mcp_tools, _mcp_clients
    if mcp_name in _loaded_mcp_tools:
        # print(f"--- [Lazy Load] 使用已緩存的 {mcp_name} MCP 工具 ---")
        return _loaded_mcp_tools[mcp_name]

    async with _mcp_init_lock:
        if mcp_name in _loaded_mcp_tools:
            # print(f"--- [Lazy Load] 使用已緩存的 {mcp_name} MCP 工具 (after lock) ---")
            return _loaded_mcp_tools[mcp_name]

        print(f"--- [Lazy Load] 觸發 {mcp_name} MCP 工具初始化 ---")
        client, tools = await initialize_single_mcp(mcp_name)

        _loaded_mcp_tools[mcp_name] = tools
        if client:
             _mcp_clients[mcp_name] = client

        print(f"--- [Lazy Load] {mcp_name} MCP 工具初始化完成並緩存 (找到 {len(tools)} 個工具) ---")
        return tools

# =============================================================================
# 提示詞定義 (修改 AGENT_EXECUTION_PROMPT 加入最終截圖指令)
# =============================================================================
# --- Revit BIM 執行提示 ---
REVIT_AGENT_EXECUTION_PROMPT = SystemMessage(content="""你是一個嚴格按計劃執行任務的助手，專門為 Revit BIM 環境生成指令。消息歷史中包含了用戶請求和一個分階段目標的計劃。
**你的核心任務：根據計劃，執行且僅執行下一個未完成的步驟。嚴禁重複已完成的步驟。**

**定位下一步驟的演算法 (必須嚴格遵守):**
1.  **檢查歷史紀錄:** 查看最近的幾條消息。你的主要依據是最後一條 `ToolMessage`。
2.  **匹配上次動作:** 將 `ToolMessage` 的結果與 `[目標階段計劃]:` 中的步驟進行比對，找出它對應的是計劃中的第幾個步驟。
3.  **確定下一步:** 緊接在上一步之後的那個步驟，就是你現在需要執行的**唯一目標**。例如，如果上一步是計劃的第 1 步，你現在就必須執行第 2 步。
4.  **初始情況:** 如果歷史紀錄中沒有 `ToolMessage` (代表這是計劃生成後的第一次執行)，則從計劃的第 1 步開始。
5.  **錯誤處理:** 如果 `ToolMessage` 指出上一步驟執行失敗，你的任務是分析錯誤原因，並嘗試**修正並重新執行同一個步驟**。
                                             
**執行規則:**                                                                       
1.  **要調用工具來執行動作，請必須生成 `tool_calls` 在首位的 AIMessage 以請求該工具調用**。**不要僅用文字描述你要調用哪個工具，而是實際生成工具調用指令。** 一次只生成一個工具調用請求。
2.  **嚴格的單步執行原則 (極度重要):**
    * **每次工具調用只能完成計劃中的一個階段目標**，絕不可嘗試在單次工具調用中完成多個步驟
    * **代碼長度限制:** 每次生成的 Revit 代碼應保持簡潔，通常不應超過 50-80 行。如果某個步驟需要更多代碼，請將其拆分為更小的子步驟
    * **專注當前目標:** 只生成完成當前階段目標所需的最少代碼，不要提前處理後續步驟
    * **範例:** 如果計劃中的步驟是"創建圖層結構"，則只創建圖層；如果步驟是"創建第一個拱形肋條"，則只創建一個肋條，不要同時創建多個或添加其他元素
4.  **Revit API 函數使用注意事項:**
    * **仔細查閱 Autodesk.Revit.DB 的正確函數名稱和參數**，避免使用不存在的函數
    * **注意 Transaction 管理**：所有修改文檔的操作都必須在 Transaction 中執行
    * **正確處理 ElementId 和參考**：確保使用有效的 ElementId 和參考對象
5.  **仔細參考工具描述或 Mcp 文檔確認函數用法與參數正確性，必須實際生成結構化的工具呼叫指令。**
6.  **工具使用優先順序 (極度重要):**
    * **優先使用專門的結構化工具**: 對於標準的 Revit 操作（如創建牆、樓板、門窗、樓層等），必須使用對應的專門工具（如 `create_line_based_element`, `create_surface_based_element`, `create_point_based_element` 等）
    * **send_code_to_revit 作為最終手段**: 只有在以下情況下才能使用：
        * 專門工具無法完成任務
        * 專門工具不存在對應的功能
    * **避免不必要的代碼執行**: 使用 `send_code_to_revit`前，應盡可能先使用結構化的專門工具來確保穩定性
7.  **send_code_to_revit 工具的 C# 代碼生成規則 (僅在必要時使用):**
    * **代碼結構:** 生成一個完整的 `RevitScript` 類別，其中包含 `Execute(Document doc)` 方法
    * **語法要求:**
        * **禁止使用字串插值**: 不要在代碼中使用 `$` 符號進行字串插值 (例如 `$"Text {var}"`)。必須使用 `string.Format("Text {0}", var)` 或字串連接 `+`。
        * 所有方法都必須有正確的返回類型 (不能省略 `void` 或具體類型)
        * 所有大括號 `{}` 必須正確匹配
        * 變數宣告必須在適當的作用域內
        * 使用 `using` 語句進行資源管理 (特別是 Transaction)
    * **Revit API 限制:**
        * **禁止修改唯讀屬性**: 例如 `Level.ProjectElevation` 是唯讀的。若要改變樓層高度，請使用 `ElementTransformUtils.MoveElement` 或設置相關參數。
        * **禁止嵌套 Transaction (重要)**: 
            * 代碼通常在一個主 Transaction 中執行。
            * **絕對不要**在輔助方法或循環內部再次調用 `new Transaction(doc, ...).Start()`，除非你使用了 `SubTransaction` 並且非常清楚其用法。
            * **錯誤範例**: 在 `FindOrCreateLevel` 方法中開啟新 Transaction，而該方法又被另一個 Transaction 塊調用。這會導致 "Starting new transaction is not permitted" 錯誤。
        * **樓層獲取策略**:
            * **不創建樓層**: 假設所有必要的樓層 (Level 1, Level 2, etc.) 都已存在於專案中。
            * **獲取樓層**: 使用 `FilteredElementCollector` 獲取現有樓層並根據名稱 (Name) 或高度 (Elevation) 進行匹配。
        * **樓板創建的正確方法**:
            * 使用 `Floor.Create(document, profile, floorTypeId, levelId)` 來創建樓板 (Revit 2022+)。
            * 對於舊版本或特定情況，才使用 `Document.Create.NewFloor`。
            * `profile` 必須是 `List<CurveLoop>` 類型。
    * **禁止事項:**
        * 不要在類別定義外寫可執行語句
        * 不要省略方法的返回類型
        * 避免語法錯誤，如缺少分號或括號
        * **禁止錯誤的單位轉換**：不要進行單位轉換，Revit API 期望英尺(Decimal Feet)單位。**嚴格保持數值為英尺。**
        * **禁止使用不存在的地界線創建方法**，必須使用正確的 Revit API 方法
    * **正確格式範例 (獲取樓層與創建樓板):**
        ```csharp
        using System;
        using System.Collections.Generic;
        using System.Linq; // 必須引用 System.Linq
        using Autodesk.Revit.DB;

        public class RevitScript
        {
            public void Execute(Document doc)
            {
                // 1. 獲取現有樓層 (不創建)
                Level level1 = new FilteredElementCollector(doc)
                    .OfClass(typeof(Level))
                    .Cast<Level>()
                    .FirstOrDefault(l => l.Name == "Level 1");

                if (level1 == null) 
                {
                    // 錯誤處理：如果找不到樓層，可以拋出異常或記錄日誌
                    throw new Exception("Level 1 not found in the project.");
                }

                // 2. 創建樓板 (Floor.Create for Revit 2022+)
                using (Transaction t = new Transaction(doc, "Create Floor"))
                {
                    t.Start();

                    // 創建樓板輪廓 (CurveLoop)
                    List<CurveLoop> profile = new List<CurveLoop>();
                    CurveLoop loop = new CurveLoop();
                    // 假設單位為英尺 (Feet)
                    loop.Append(Line.CreateBound(new XYZ(0, 0, 0), new XYZ(20, 0, 0)));
                    loop.Append(Line.CreateBound(new XYZ(20, 0, 0), new XYZ(20, 20, 0)));
                    loop.Append(Line.CreateBound(new XYZ(20, 20, 0), new XYZ(0, 20, 0)));
                    loop.Append(Line.CreateBound(new XYZ(0, 20, 0), new XYZ(0, 0, 0)));
                    profile.Add(loop);

                    // 獲取樓板類型
                    FloorType floorType = new FilteredElementCollector(doc)
                        .OfClass(typeof(FloorType))
                        .Cast<FloorType>()
                        .FirstOrDefault(ft => ft.Name == "Generic - 12\""); // 確保類型名稱正確

                    if (floorType != null)
                    {
                        // 創建樓板 (使用 Floor.Create)
                        Floor.Create(doc, profile, floorType.Id, level1.Id);
                    }

                    t.Commit();
                }
            }
        }
        ```
9.  **建築元件生成策略:**
    * **優先使用標準建築元件**：牆壁(Wall)、樓板(Floor)、門(Door)、窗(Window)等標準建築元件
    * **正確設置元件參數**：高度(偏移通常設為0)、樓層等參數必須正確設置，確保元件之間的連接和約束正確設置
    * **使用族群元件**：對於複雜元件，使用適當的族群類型

10. **房間與標籤創建 (重要說明):**
    * **`create_rooms_and_tags` 工具功能:** 此工具會**同時完成兩件事**：
        1. 在指定座標創建房間
        2. 自動在房間中心點放置標籤 (RoomTag)
    * **使用方式:** `create_rooms_and_tags(roomName="房間名稱", x=座標X, y=座標Y)`
    * **標籤會立即放置:** 無需額外調用任何工具來放置標籤，它會在房間創建後自動完成
    * **逐一調用:** 為每個房間分別調用一次，確保每個房間都有正確的名稱和標籤

11. **Revit 事務與上下文約束 (重要):**
    *   **Transaction 處理:** 當使用 `send_code_to_revit` 編寫 C# 代碼時，請注意代碼通常是在一個外部事件上下文中執行。
        *   **不要顯式開啟 Transaction**，除非你確定外層沒有開啟 Transaction。大部分簡單操作會被自動包裝，但如果代碼複雜，請先檢查 `Document.IsModifiable`。
        *   更好的做法是將你的代碼邏輯封裝在 `Transaction` 塊中，但使用 `using (Transaction t = new Transaction(doc, "Name"))` 時要做好異常處理。
    *   **唯讀狀態:** 避免在唯讀視圖或預覽模式下嘗試修改模型。

12. **樓層完成通知 (必須執行):**
    *   **何時執行:** 每當完成一個樓層的所有建模步驟（樓板、牆體、門、房間與標籤）後
    *   **如何執行:** 使用 `send_code_to_revit` 工具執行以下 C# 代碼：
        ```csharp
        using Autodesk.Revit.UI;
        public class RevitScript {
            public void Execute(Document doc) {
                TaskDialog.Show("樓層完成", "Level X 建模已完成，請檢查並切換到下一樓層視圖");
            }
        }
        ```
    *   **目的:** 通知用戶該樓層已完成，提醒切換到下一樓層的平面視圖
    *   **必要性:** 這是每個樓層建模循環的最後一個必要步驟，缺少此步驟用戶將不知道何時切換視圖

13. **最終步驟 (Revit):**
    *   對於 Revit 任務，每當完成一個方案就**必須**要完成建模工作。
    *   **僅當消息歷史清楚地表明計劃中的最後階段目標已成功執行**，你才能生成文本回復：`全部任務已完成` 以結束整個任務。

14. 如果當前階段目標不需要工具即可完成（例如，僅需總結信息），請生成說明性的自然語言回應。

15. 若遇工具錯誤，分析錯誤原因 (尤其是代碼執行錯誤)，**嘗試修正你的工具調用參數或生成的代碼**，然後再次請求工具調用。如果無法修正，請報告問題。

16. 規劃數據摘要報告 (空間規劃任務的必要首步):僅當**任務是關於**空間佈局規劃** (例如，量體配置等)，你**必須在第一個步驟**執行生成摘要報告。
                                             
**常規執行：對於計劃中的任何步驟，不要用自然語言解釋你要做什麼，直接生成包含 Tool Calls 結構的工具調用。**
**關鍵指令：不要用自然語言解釋你要做什麼，直接根據你用上述演算法定位到的下一步驟，生成包含 Tool Calls 結構的工具調用。**
**絕對指令：不要延續[目標階段計劃]生成 "任務完成" 或將任務完成當作一個步驟。當前一個訊息是[目標階段計劃]時直接進行工具調用，不要包含描述性文本！**
                                             
**可用工具清單:**
你能夠使用以下工具來完成計劃中的步驟。你必須使用這些工具，並嚴格按照其參數要求來生成工具調用。
{tool_descriptions}""")

# --- 住宅建模規範 (條件式載入) ---
RESIDENTIAL_MODELING_GUIDELINES = """
7.  **Revit 住宅建模規範 (僅適用於住宅建模任務，必須嚴格遵守):**

    * **Family & Type 映射 (強制直接使用以下名稱及尺寸):**
        * Exterior Envelope (建築周邊實牆): `Exterior - 1'`
        * Window/Glazing (窗戶/玻璃區域): `Exterior with lighting- 1'`
        * Unit Partition (單位分隔牆): `Generic - 8"`
        * Floor Slab (樓板及屋頂): `Generic - 12"`
        * Unit Entry Door (單位入口門): `Single-Flush` 尺寸為42" x 80"英制單位
        * Stairway/Core Door (樓梯/核心門): `Single-Flush"` 尺寸為36" x 84英制單位

    * **Room Naming 標準 (大小寫敏感，完全匹配):**
        * **重要:** 使用 `create_rooms_and_tags` 工具時，**必須**通過 `roomName` 參數傳入下列標準名稱或您規劃的特定名稱，以確保房間被正確標記。
        * **建議操作模式:** 為避免命名重複或錯誤，建議使用 `create_rooms_and_tags(roomName="...", x=..., y=...)` 模式，針對每個房間的中心點座標逐一創建並命名。
        * **Core & Circulation (核心及流通):**
          `Corridor & Elev. Lobby`, `Stairway`, `Elev. Shaft`, `Mec. Shaft`, `Elec. /MEP Room`, `Refuse room`
        * **Residential Units (住宅單位):**
          `0BR`, `1BR`, `2BR`, `3BR`, `4BR`
        * **Amenities & Others (設施及其他):**
          `Office`, `Retails`, `Facility space`, `Recreation space`, `Public restroom`, `Terrace`, `Balcony`, `Entry Lobby`

    * **建模規則與約束 (簡化版):**
        * **核心任務範圍:** 僅執行 **牆 (Walls)**、**版 (Floors)**、**門 (Doors)**、**房間與標籤 (Rooms & Tags)** 的建立。
        * **建模順序 (必須嚴格遵守 - 完成一層才做下一層):**
          1. **確認樓層:** 假設樓層 (Level 1, Level 2...) **已預先存在**。**不要**嘗試創建新樓層。
          2. **單層完整建置循環 (必須包含以下所有步驟才算完成該層):**
             * **Step A - 樓板 (Slab):** 創建該層結構樓板。**一樓樓板為基地大小。**
             * **Step B - 牆體 (Walls):** 創建外牆 (Exterior) 與內牆 (Interior)。配置盡量不要做滿基地。
             * **Step C - 門 (Doors):** 必須放置單元入口門與核心區門扇 (這是必要的，不可省略)。
             * **Step D - 房間與標籤 (Rooms & Tags):** 
               * **重要說明:** `create_rooms_and_tags` 工具會**同時創建房間並立即放置標籤**，無需分開操作
               * **逐一創建 (Point-by-Point):** 使用 `create_rooms_and_tags(roomName="具體名稱", x=座標, y=座標)` 逐一放置每個房間
               * **每次調用都會:**
                 1. 在指定座標創建房間
                 2. 將房間命名為 `roomName` 參數的值
                 3. 自動在房間中心點放置標籤
               * **禁止批量與預設:** 避免不帶座標的批量調用，以免所有房間被命名為相同名稱
               * **分批執行:** 確保每個功能空間都有獨立的調用指令
               * **錯誤處理:** 如果放置失敗，代表該位置的空間還沒完全封閉，需補充牆體
             * **Step E - 樓層完成通知 (必要步驟):**
               * 完成該層所有建模後，必須使用 `send_code_to_revit` 執行以下 C# 代碼來通知用戶：
               ```csharp
               using Autodesk.Revit.UI;
               public class RevitScript {
                   public void Execute(Document doc) {
                       TaskDialog.Show("樓層完成", "Level X 建模已完成，請檢查並切換到下一樓層視圖");
                   }
               }
               ```
               * 將 "Level X" 替換為實際完成的樓層名稱（如 "Level 1"、"Level 2" 等）
               確認都完成後才繼續做下一層。
          3. **重複上述循環:** 直到所有規劃樓層完成。
        * **樓層建模策略 (Representative Floors):**
          - **僅需建模代表性樓層**:
            - **1F, 2F:** 公設層 (Public/Amenity) 必須建模。
            - **Typical Floor (標準層):** 若 3F-10F 佈局相同，**僅需建模 3F** 作為代表。
            - **Setback Floor (退縮層):** 若 11F-20F 發生退縮，**僅需建模 11F** 作為代表。
            - 以此類推，僅建模佈局改變的第一個樓層。
        * **Core 建置規則 (重要):**
          - **分層建置:** 核心筒 (Core) 的牆體**必須**分層建立，**不可**一次從底層拉到頂層。
          - **約束:** 每層 Core 牆的底部約束為當前樓層，頂部約束為下一樓層 (或指定高度)。
        * **樓層高度參考 (假設現有):**
          - **Level 1:** 0'
          - **Level 2:** 20'
          - **Level 3+:** 30' (每層 +10')
        * **公設樓層:** 1、2 樓通常規劃為公設 (Lobby, Gym, etc.)。
        * **不建模項目:** 不要建模家具/設備、單元內佈局牆、實際樓梯族群、電梯族群；僅保留必要的空間體積(Rooms)和包圍牆
          - **禁止使用規範外族群**: 嚴格遵守下列 Wall/Floor/Door 類型，不得自行創造。
          - **鄰地側 (Lot Line):** 使用 `Exterior - 1'` (做滿，無開窗)。
          - **街道/後院側 (Street/Yard):** 使用 `Exterior with lighting- 1'` (適當退縮以利採光)。
          - **單位分隔:** 使用 `Generic - 8"` 分隔單位與走廊、單位與單位之間
        * **單位要求:** 每個樓層至少 2 種單位類型，單位深度 ≥ 15'。
        * **合理規劃 (Rational Planning):**
          - **Core 位置:** 應位於建築中央或適當位置以服務所有單元。Corridor建議5' 7"以上。
          - **長寬比:** 單元應保持合理的長寬比，避免過於狹長。
          - **採光:** 確保所有單元的主要居室都有面向街道或後院的採光面。靠內的部分建議預留空間作為單元採光面及後院。

    * **HPD guideline面積指南:**
        * **(Affordable Housing):**
        0BR: 400-550 SF, 1BR: 550-725 SF, 2BR: 725-950 SF, 3BR: 950-1,075 SF, 4BR: 1,075-1,175 SF
        * **(Luxury Housing):**
        0BR: 400-550 SF, 1BR: 550-725 SF, 2BR: 725-1,400 SF, 3BR: 950-2,100 SF, 4BR: 1,075-2,800 SF

    * **核心佈局要求:**
        * **大廳與出口數量 (Lobby & Exit Quantity):**
            - Occupant Load < 50: 1 Exit.
            - Occupant Load 50–500: 2 Exits (Standard).
            - Occupant Load > 500 or Height > 420’: 3 Exits.
        以下為每個樓層皆應包含的Core內容
        * **樓梯 (Stairway) 類型選擇:**
            - **Scissors Stairs (剪刀梯):** 當基地長寬比小於 1:1.5，或基地長寬任一邊小於 100' 時，建議使用剪刀梯以節省核心空間。
            - **U Shape Stairs (U型梯):** 當基地條件較為寬裕 (長寬比大於 1:1.5 且尺寸充足) 時，可使用標準 U 型梯。
            - **數量:** 至少 2 個 (除非面積很小則適用1個剪刀梯即可)；設置兩座梯時其距離需≥30′ OR ≥1/3 of building diagonal (whichever is less).
            - Width: ≥ 44" (inches).
            - Landing Depth: ≥ 48" (inches).
        * **電梯 (Elev. Shaft):**
            - Size (Standard): Min 7.5’ x 8.5’.
            - Quantity Rule of Thumb: ~1 lift per 6,000 SF floor plate (Residential).
        * **走廊 (Corridor & Elev. Lobby):** 走廊最小寬度 5'
        * **機電服務空間 (MEP/Service):** 包含 `Mec. Shaft`(電梯井), `Elec. /MEP Room`(機電房), `Refuse room`(垃圾間)

    * **核心位置與單元配置策略:**
        * **核心位置 (Core Location):**
            - 視基地長寬而定，可配置於**中央 (Center)** 或 **靠邊 (Side/Rear)**。
            - **優先考量:** 必須配合標準層單元配置，**確保所有居住單元 (Living Units) 都能靠外牆獲取自然採光**。
            - 若基地狹長，核心宜置於長向中央以減少走廊長度；若基地主要採光面受限，核心宜置於採光較差的一側 (如鄰地側)。
        
    * **場地與分區定義 (Site & Zoning Context):**
        * **地塊類型與策略 (Lot Types & Strategy):**
            - **Interior Lot:** 1 side facing road. Windows allowed on Street and Rear Yard.適用於正方形、長條形配置
            - **Corner Lot:** 2 sides facing road.適用於L型、T型、長條形配置
            - **Through Lot:** 2 opposite sides facing road.適用於H型、長條型配置
        * **採光與通風 (Light & Ventilation - Windows):**
            - **Street Facing:** Always allowed.
            - **Interior Lot Line:** Windows must open onto a Yard or Court.
        * **後院要求 (Yard Requirement):** for Interior Lot and Corner Lot.
            - Minimum clear dimension of 30’ is required for effective legal windows facing a lot line.
        * **量體 (Massing):**
            - **Contextual:** Continuous street-wall massing required.
            - **Setbacks:** Follow prompt specific instructions (e.g., 810FL @ 10’, 2040FL @ 10’).

    所有空間的數量比例、面積、特殊空間要求等會由用戶視需要提供。
"""



# --- Router Prompt (MODIFIED) ---
ROUTER_PROMPT = """你是一個智能路由代理。根據使用者的**初始請求文本**，判斷應將任務分配給哪個專業領域的代理。
目前可用的代理有：
- 'revit': 主要處理與 Revit 建築資訊模型相關的請求。

分析以下**初始使用者請求文本**，並決定最適合處理此請求的代理。所有建築和BIM相關的任務都應分配給revit代理。
你的回應必須是 'revit'。請只回應目標代理的名稱。

初始使用者請求文本：
"{user_request_text}"
"""

PLAN_PREFIX = "[目標階段計劃]:\n"

# --- Fallback Agent Prompt ---
FALLBACK_PROMPT = SystemMessage(content="""你是一個補救與驗證助手。主要助手可能已完成其步驟、卡住了，或聲稱任務已完成。
    你的任務是：
    1.  仔細分析消息歷史，特別是 `[目標階段計劃]:` 和最近幾條主要助手的回應。
    2.  **分析主要助手狀態**：
        *   如果主要助手的最後一條回應**不是工具調用**，而是描述性文本（例如 "正在執行階段 X..." 或類似的對話），這通常表示主要助手**卡住了**或者未能按預期生成工具調用。
    3.  **驗證完成狀態 (如果主要助手聲稱完成或歷史表明可能已到最後階段)**：
        *   查看 `[目標階段計劃]:`，識別出計劃中的**最後一個階段目標**。檢查最近的消息歷史，請獨立判斷這個**最後的階段目標是否已經成功執行完畢**。
    4.  **確定下一步**：
        *   如果根據上述驗證，計劃中的**最後一個階段目標確實已成功執行**，請**只輸出**文本消息：`[FALLBACK_CONFIRMED_COMPLETION]`。
        *   如果主要助手**卡住了**（如第 2 點所述），或者任務**未完成** (例如，最後的計劃步驟未完成，或者還有更早的計劃步驟未完成且你可以識別出來)，並且你可以根據計劃和歷史確定下一個**應該執行的階段目標**，請**生成執行該目標所需的 `tool_calls`**。直接輸出包含工具調用的 AIMessage。**優先嘗試從計劃中找到下一個應該執行的步驟並為其生成工具調用。**
        *   如果任務**未完成**，且你無法根據現有信息確定下一步、無法恢復流程（例如，無法識別計劃的最後一步，或無法判斷其是否完成，或無法為卡住的助手找到解決方案），請**只輸出**文本消息：`[FALLBACK_CANNOT_RECOVER]`。

   **關鍵：不要重複主要助手剛剛完成的步驟。專注於未完成的目標或驗證最終狀態。如果主要助手明顯卡在某個描述性文本而未生成工具調用，你的首要任務是根據計劃推斷並生成正確的工具調用。**
   
   消息歷史:
   {relevant_history}
   """)

# =============================================================================
# 輔助函數：執行工具
# =============================================================================
async def execute_tools(agent_action: AIMessage, selected_tools: List[BaseTool]) -> List[ToolMessage]:
    """執行 AI Message 中的工具調用，處理 Revit 工具返回，並確保 ToolMessage content 非空字串。"""
    tool_messages = []
    if not agent_action.tool_calls:
        return tool_messages
    name_to_tool_map = {tool.name: tool for tool in selected_tools}
    print(f"    準備執行 {len(agent_action.tool_calls)} 個工具調用...")
    print(f"    [DEBUG] 可用工具列表 ({len(selected_tools)} 個): {list(name_to_tool_map.keys())}")
    for tool_call in agent_action.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"      >> 調用工具: {tool_name} (ID: {tool_call_id})")

        tool_to_use = name_to_tool_map.get(tool_name)
        if not tool_to_use:
            error_msg = f"錯誤：找不到名為 '{tool_name}' 的工具。可用工具: {list(name_to_tool_map.keys())}"
            print(f"      !! {error_msg}")
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, name=tool_name))
            continue

        observation_str = f"[未成功執行工具 {tool_name}]"
        final_content = "UNEXPECTED_TOOL_EXECUTION_FAILURE"
        observation = None

        try:
            # --- 參數處理 (保持不變) ---
            if not isinstance(tool_args, dict):
                 try:
                     tool_args_dict = json.loads(str(tool_args)) if isinstance(tool_args, str) and str(tool_args).strip().startswith('{') else {"input": tool_args}
                 except json.JSONDecodeError:
                     tool_args_dict = {"input": tool_args}
            else:
                 tool_args_dict = tool_args

            # --- 調用工具 (ainvoke) ---
            print(f"        調用 {tool_name}.ainvoke...")
            observation = await tool_to_use.ainvoke(tool_args_dict, config=None)
            print(f"        {tool_name}.ainvoke 調用完成。觀察值類型: {type(observation).__name__}")

            # --- 轉換 observation 為字串 ---

            # --- 處理 Revit 工具返回 ---

            # --- 處理 bytes (保持不變) ---
            if isinstance(observation, bytes):
                try:
                    observation_str = observation.decode('utf-8', errors='replace')
                    print(f"      << 工具 '{tool_name}' 返回 bytes，已解碼。")
                except Exception as decode_err:
                    observation_str = f"[Error Decoding Bytes: {decode_err}]"
                    print(f"      !! 工具 '{tool_name}' 返回 bytes，解碼失敗: {decode_err}")
                final_content = observation_str if observation_str else "DECODED_EMPTY_STRING"

            # --- 處理 dict/list (排除 capture_viewport) ---
            elif isinstance(observation, (dict, list)):
                if isinstance(observation, list) and not observation:
                     error_msg = f"工具 '{tool_name}' 的 ainvoke 返回了空列表 `[]`。這可能表示 langchain-mcp-adapters 在處理工具響應時內部出錯，或者工具本身未按預期返回 (檢查工具實現)。"
                     print(f"      !! {error_msg}")
                     final_content = "ADAPTER_RETURNED_EMPTY_LIST"
                else:
                    try:
                        observation_str = json.dumps(observation, ensure_ascii=False, indent=2)
                        print(f"      << 工具 '{tool_name}' 返回普通 dict/list，已序列化為 JSON 字串。")
                    except TypeError as json_err:
                        observation_str = f"[Error JSON Serializing Result: {json_err}] 回退到 str(): {str(observation)}"
                        print(f"      !! 工具 '{tool_name}' 返回 dict/list，JSON 序列化失敗: {json_err}。回退到 str()")
                    except Exception as ser_err:
                        observation_str = f"[Error Serializing Result: {ser_err}]"
                        print(f"      !! 工具 '{tool_name}' 返回 dict/list，序列化時發生未知錯誤: {ser_err}")
                    final_content = observation_str

            # --- 處理其他類型 (保持不變) ---
            else:
                try:
                    temp_str = str(observation)
                    if temp_str == "[]":
                         print(f"      !! 工具 '{tool_name}' 返回值 string 化後為 '[]'，可能表示錯誤或空列表。原始類型: {type(observation).__name__}")
                         observation_str = "TOOL_RETURNED_EMPTY_LIST_STR"
                    elif temp_str == "":
                        observation_str = "EMPTY_TOOL_RESULT"
                        print(f"      << 工具 '{tool_name}' 返回空字串，已替換為佔位符。")
                    elif observation is None:
                        observation_str = "NONE_TOOL_RESULT"
                        print(f"      << 工具 '{tool_name}' 返回 None，已替換為佔位符。")
                    else:
                        observation_str = temp_str
                        print(f"      << 工具 '{tool_name}' 返回其他類型 ({type(observation).__name__})，已使用 str() 轉換。")
                except Exception as str_conv_err:
                     observation_str = f"[Error Converting Result to String: {str_conv_err}]"
                     print(f"      !! 工具 '{tool_name}' 返回其他類型，str() 轉換失敗: {str_conv_err}")
                final_content = observation_str

            # 最終防線 (保持不變)
            if not final_content:
                final_content = "FINAL_CONTENT_EMPTY"
                print(f"      !! 警告：最終 final_content 為空，使用最終佔位符。")

            tool_messages.append(ToolMessage(content=final_content, tool_call_id=tool_call_id, name=tool_name))

        except Exception as tool_exec_e:
            error_msg = f"錯誤：執行或處理工具 '{tool_name}' 時失敗: {tool_exec_e}"
            print(f"      !! {error_msg}")
            print(f"         調用時參數: {tool_args_dict}")
            if observation is not None:
                print(f"         ainvoke 返回的觀察值 (類型 {type(observation).__name__}): {repr(observation)[:500]}")
            traceback.print_exc()
            tool_messages.append(ToolMessage(content=str(error_msg), tool_call_id=tool_call_id, name=tool_name))

    return tool_messages


# =============================================================================
# 核心函數：調用 LLM 執行計劃步驟 (添加詳細打印)
# =============================================================================
async def call_llm_with_tools(
    messages: List[BaseMessage],
    selected_tools: List[BaseTool],
    execution_prompt: SystemMessage # <<< 新增參數
) -> AIMessage:
    """
    調用 agent_llm (Gemini) 根據消息歷史（含計劃）和可用工具來執行下一步。
    輸入消息應已包含多模態內容。
    會自動為 Revit 任務添加相關規範。
    """
    enhanced_prompt = execution_prompt

    # 始終為 Revit 執行提示添加建模規範
    # 假設此函數主要由 revit_mcp.py 中的 revit agent 使用
    
    print("  >> (Execution) 自動載入 RESIDENTIAL_MODELING_GUIDELINES 到執行提示...")
    original_content = execution_prompt.content
    # 在第6條規則後插入建模規範 (與 _is_residential_modeling_task 中的邏輯相同)
    insert_position = original_content.find("6.  **工具使用優先順序") # Updated find string based on previous prompt content
    
    # Fallback search strings if prompt was modified
    if insert_position == -1:
        insert_position = original_content.find("6.  **send_code_to_revit") # Older prompt version
    
    if insert_position != -1:
        enhanced_content = (
            original_content[:insert_position] +
            RESIDENTIAL_MODELING_GUIDELINES + "\n\n" + # Add newlines for separation
            original_content[insert_position:]
        )
        enhanced_prompt = SystemMessage(content=enhanced_content)
        print("  >> 建模規範已插入到執行提示中")
    else:
        print("  >> 警告：找不到插入點，直接附加建模規範到結尾")
        enhanced_content = original_content + "\n\n" + RESIDENTIAL_MODELING_GUIDELINES
        enhanced_prompt = SystemMessage(content=enhanced_content)

    try:
        # --- 動態選擇 LLM (根據 API Key Manager) ---
        global agent_llm, fast_llm, agent_llm_free, agent_llm_vip, fast_llm_free, fast_llm_vip
        
        if api_key_manager.should_use_vip():
            llm_to_use = agent_llm_vip if agent_llm_vip else agent_llm_free
            key_type = api_key_manager.get_current_key_type()
            print(f"  >> 使用 {key_type} LLM ({llm_to_use.model})，剩餘 {api_key_manager.vip_calls_remaining} 輪")
        else:
            llm_to_use = agent_llm_free
            key_type = api_key_manager.get_current_key_type()
            print(f"  >> 使用 {key_type} LLM ({llm_to_use.model}) 執行下一步")
        
    except Exception as e:
        print(f"Error selecting LLM: {e}")
        llm_to_use = agent_llm_free

    # 最多重試 4 次 (處理配額錯誤: Free -> VIP -> VIP Retry -> Fail)
    max_retries = 4
    for retry_count in range(max_retries):
        try:
            # --- 使用輔助函數獲取 Gemini 兼容的工具定義 ---
            if retry_count == 0:  # 只在第一次打印
                print("     正在準備 Gemini 兼容的工具定義列表...")
            gemini_compatible_tools = _prepare_gemini_compatible_tools(selected_tools)
            if retry_count == 0:
                print(f"     獲取了 {len(gemini_compatible_tools)} 個 Gemini 兼容的工具定義。")

            # --- 綁定工具到 LLM ---
            if retry_count == 0:
                print("     正在將 MCP 工具 (含手動定義) 綁定到 LLM...")
            llm_with_tools = llm_to_use.bind_tools(gemini_compatible_tools)
            if retry_count == 0:
                print("     MCP 工具綁定完成。")

            # --- 配置 Runnable 移除回調 ---
            if retry_count == 0:
                print("     正在配置 LLM runnable 以移除回調 (with_config)...")
            llm_configured_no_callbacks = llm_with_tools.with_config({"callbacks": None})
            if retry_count == 0:
                print("     LLM runnable 配置完成 (callbacks=None)。")

            # --- 準備調用消息 ---
            current_call_messages = [execution_prompt] + messages
            if retry_count == 0:
                print(f"     LLM 輸入消息數 (含執行提示): {len(current_call_messages)}")

            # --- 添加詳細打印 (檢查多模態消息格式) ---
            print("-" * 40)
            print(f">>> DEBUG: Messages Sent to LLM.ainvoke (Attempt {retry_count + 1}/{max_retries}):")
            for i, msg in enumerate(current_call_messages):
                print(f"  Message {i} ({type(msg).__name__}):")
                try:
                    # 使用更安全的方式獲取和打印內容
                    if isinstance(msg.content, str):
                        content_repr = repr(msg.content)
                    elif isinstance(msg.content, list):
                         # 對列表內容進行部分表示，避免過長
                         content_repr = "[" + ", ".join(repr(item)[:100] + ('...' if len(repr(item)) > 100 else '') for item in msg.content) + "]"
                    else:
                         content_repr = repr(msg.content)
                    print(f"    Content: {content_repr[:1000]}{'...' if len(content_repr) > 1000 else ''}")
                except Exception as repr_err:
                    print(f"    Content: [Error representing content: {repr_err}]")

                if isinstance(msg, AIMessage) and msg.tool_calls:
                    try:
                        tool_calls_repr = repr(msg.tool_calls)
                        print(f"    Tool Calls: {tool_calls_repr[:500]}{'...' if len(tool_calls_repr) > 500 else ''}")
                    except Exception as repr_err:
                        print(f"    Tool Calls: [Error representing tool_calls: {repr_err}]")
                elif isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id'):
                     print(f"    Tool Call ID: {msg.tool_call_id}")
            print("-" * 40)
            # --- 結束詳細打印 ---

            # --- 執行 LLM 調用 (使用配置後的 Runnable) ---
            if retry_count == 0:
                print(f"     正在調用配置後的 LLM.ainvoke (Model: {llm_to_use.model})...")
            else:
                print(f"     [Retry {retry_count}] 正在調用 LLM.ainvoke (Model: {llm_to_use.model})...")
            
            response = await llm_configured_no_callbacks.ainvoke(current_call_messages)
            
            # --- 成功調用，處理 VIP 計數器 ---
            if api_key_manager.should_use_vip():
                api_key_manager.decrement_vip_calls()
            
            if retry_count == 0:
                print(f"  << LLM 調用完成。")
            if isinstance(response, AIMessage) and response.tool_calls:
                 print(f"     LLM 請求調用 {len(response.tool_calls)} 個工具。")
            elif isinstance(response, AIMessage):
                 print(f"     LLM 返回內容: {response.content[:150]}...")
                 if "任務已完成" in response.content.lower():
                     print("     偵測到 '任務已完成'。")
            else:
                 print(f"     LLM 返回非預期類型: {type(response).__name__}")

            return response

        except Exception as e:
            str_e = str(e)
            is_quota_error = ("429" in str_e and ("quota" in str_e.lower() or "rate" in str_e.lower())) or \
                            ("Quota exceeded" in str_e) or \
                            ("You exceeded your current quota" in str_e)
            
            if is_quota_error and retry_count < max_retries - 1:
                print(f"  ⚠️  捕獲配額錯誤 (429 Quota Exceeded): {e}")
                # 嘗試切換到 VIP
                if api_key_manager.handle_quota_error():
                    # 檢查 agent_llm_vip 是否可用
                    if not agent_llm_vip:
                        # 嘗試動態初始化 VIP Agent
                        vip_key = os.getenv("GEMINI_API_KEY_VIP")
                        if vip_key:
                            print("  >> [Dynamic Init] 嘗試動態初始化 agent_llm_vip...")
                            try:
                                agent_llm_vip = ChatGoogleGenerativeAI(
                                    model="gemini-2.5-pro",
                                    temperature=0.5,
                                    google_api_key=vip_key
                                )
                                print("  >> [Dynamic Init] agent_llm_vip 初始化成功！")
                            except Exception as init_err:
                                print(f"  >> [Dynamic Init] agent_llm_vip 初始化失敗: {init_err}")
                        else:
                            print("  >> [Dynamic Init] 失敗: 環境變數 GEMINI_API_KEY_VIP 未設置")

                    # 切換 LLM
                    if agent_llm_vip:
                        llm_to_use = agent_llm_vip
                        print(f"  >> ✅ 成功切換到 VIP LLM ({llm_to_use.model})")
                    else:
                        llm_to_use = agent_llm_free
                        print(f"  >> ❌ 切換失敗: 無法獲取 VIP LLM 實例，將繼續使用 Free LLM 重試")

                    print(f"  >> 第 {retry_count + 1}/{max_retries} 次重試即將開始...")
                    await asyncio.sleep(2)  # 短暫等待
                    continue  # 重試
                else:
                    # 沒有 VIP Key，無法繼續
                    error_content = f"配額已滿且無 VIP Key 可用 (handle_quota_error returned False): {e}"
                    print(f"!! {error_content}")
                    return AIMessage(content=error_content)
            
            # 其他錯誤或已達最大重試次數
            print(f"!! 執行 LLM 調用 (call_llm_with_tools) 時發生錯誤: {e}")
            traceback.print_exc()
            
            error_content = f"執行 LLM 決策時發生錯誤: {e}"
            if isinstance(e, ValueError) and "Unexpected message with type" in str_e:
                 error_content = f"內部錯誤：調用 LLM 時消息順序或類型不匹配。錯誤: {e}"
            elif "Function and/or coroutine must be provided" in str_e or "bind_tools" in str_e.lower():
                 error_content = f"內部錯誤：綁定或調用工具時出錯。檢查工具定義或LLM兼容性。錯誤: {e}"
            elif "InvalidArgument: 400" in str_e:
                 reason = "未知原因"
                 if "missing field" in str_e:
                     reason = f"工具 Schema 無效 (即使手動修正後，仍可能存在問題或影響其他工具)"
                 elif "function declaration" in str_e:
                      reason = f"工具函數聲明格式錯誤"
                 elif "contents" in str_e:
                     reason = f"消息內容格式錯誤，可能多模態輸入未被正確處理"
                 error_content = f"內部錯誤：傳遞給 Gemini 的數據無效 ({reason})。錯誤: {e}"

            return AIMessage(content=error_content)
    
    # 如果所有重試都失敗
    return AIMessage(content="LLM 調用失敗：已達最大重試次數")


# --- NEW HELPER FUNCTION for preparing Gemini-compatible tools ---
def _fix_gemini_tool_schema(schema: Dict) -> Dict:
    """
    遞歸修復 Gemini API 工具 schema，確保數組類型參數有 items 字段。
    """
    if not isinstance(schema, dict):
        return schema

    # 複製 schema 以避免修改原始對象
    fixed_schema = schema.copy()

    # 檢查類型是否為 array 且缺少 items
    if fixed_schema.get("type") == "array" and "items" not in fixed_schema:
        # 為數組添加默認的 items 字段（假設元素是字符串類型）
        fixed_schema["items"] = {"type": "string"}
        print(f"     [Schema Fix] 添加缺失的 items 字段到數組參數，假設元素類型為 string")

    # 遞歸處理嵌套屬性
    for key, value in fixed_schema.items():
        if isinstance(value, dict):
            fixed_schema[key] = _fix_gemini_tool_schema(value)
        elif isinstance(value, list):
            fixed_schema[key] = [_fix_gemini_tool_schema(item) if isinstance(item, dict) else item for item in value]

    return fixed_schema

def _prepare_gemini_compatible_tools(mcp_tools: List[BaseTool]) -> List[Union[BaseTool, Dict]]:
    """
    為 Gemini LLM 準備工具列表，手動修正特定工具的 schema。
    """
    print("     [Helper] 準備 Gemini 兼容的工具定義列表...")
    tools_for_binding = []
    if not mcp_tools:
        print("     [Helper] 警告: 傳入的 mcp_tools 列表為空。")
        return []

    for tool_idx, tool in enumerate(mcp_tools):
        if not tool or not hasattr(tool, 'name'):
            print(f"     [Helper] 警告: 工具列表中發現無效工具對象: {tool}")
            continue

        try:
            # 嘗試將工具轉換為字典格式以便檢查和修復 schema
            tool_dict = None
            tool_name = getattr(tool, 'name', f'tool_{tool_idx}')

            # 調試：打印工具信息
            print(f"     [Helper] 處理工具: {tool_name}, 類型: {type(tool)}")

            if hasattr(tool, 'get_tool_definition') and callable(tool.get_tool_definition):
                tool_dict = tool.get_tool_definition()
                print(f"     [Helper] 使用 get_tool_definition() 獲取工具定義")
            elif hasattr(tool, 'tool_definition'):
                tool_dict = tool.tool_definition
                print(f"     [Helper] 使用 tool_definition 屬性獲取工具定義")
            else:
                print(f"     [Helper] 無法獲取工具定義，使用手動構造")
                # 如果無法獲取工具定義，嘗試手動構造
                tool_description = getattr(tool, 'description', '')

                # 獲取參數 schema
                parameters = None
                if hasattr(tool, 'args_schema') and tool.args_schema is not None:
                    print(f"     [Helper] 發現 args_schema，類型: {type(tool.args_schema)}")
                    # 如果是 Pydantic 模型，轉換為 schema 字典
                    try:
                        if isinstance(tool.args_schema, dict):
                             parameters = tool.args_schema
                             print(f"     [Helper] args_schema 是字典，直接使用")
                        elif hasattr(tool.args_schema, 'model_json_schema'):
                            # Pydantic v2
                            parameters = tool.args_schema.model_json_schema()
                            print(f"     [Helper] 使用 Pydantic v2 model_json_schema()")
                        elif hasattr(tool.args_schema, 'schema'):
                            # Pydantic v1
                            parameters = tool.args_schema.schema()
                            print(f"     [Helper] 使用 Pydantic v1 schema()")
                        else:
                            # 嘗試手動構造基本 schema
                            print(f"     [Helper] args_schema 沒有標準方法，使用默認 schema")
                            parameters = {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                    except Exception as schema_err:
                        print(f"     [Helper] 轉換 args_schema 時出錯: {schema_err}")
                        parameters = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                elif hasattr(tool, 'schema') and tool.schema is not None:
                    parameters = tool.schema
                    print(f"     [Helper] 使用 schema 屬性")
                else:
                    # 沒有參數的工具
                    print(f"     [Helper] 沒有找到參數 schema，使用默認")
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }

                tool_dict = {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": parameters
                }

            # 調試：檢查工具字典
            if tool_dict:
                print(f"     [Helper] 工具字典結構: name={tool_dict.get('name')}, has_parameters={'parameters' in tool_dict}")
                if 'parameters' in tool_dict:
                    print(f"     [Helper] 參數 schema 類型: {type(tool_dict['parameters'])}")
                    if isinstance(tool_dict['parameters'], dict):
                        print(f"     [Helper] 參數 schema 內容: {tool_dict['parameters']}")

            # 特殊處理特定工具的參數問題
            tool_name = getattr(tool, 'name', f'tool_{tool_idx}')
            if tool_name == 'send_code_to_revit':
                print(f"     [Helper] 修復 send_code_to_revit 的 parameters 參數")
                if tool_dict and 'parameters' in tool_dict and isinstance(tool_dict['parameters'], dict):
                    if 'properties' in tool_dict['parameters'] and 'parameters' in tool_dict['parameters']['properties']:
                        param_def = tool_dict['parameters']['properties']['parameters']
                        print(f"     [Helper] send_code_to_revit parameters 定義: {param_def}")
                        # 確保 parameters 是 array 類型
                        if isinstance(param_def, dict) and param_def.get('type') != 'array':
                            print(f"     [Helper] 修正 parameters 類型從 {param_def.get('type')} 到 array")
                            param_def['type'] = 'array'
                            param_def['items'] = {'type': 'string'}  # 默認元素類型

            elif tool_name == 'ai_element_filter':
                print(f"     [Helper] 修復 ai_element_filter 的 data 參數")
                if tool_dict and 'parameters' in tool_dict and isinstance(tool_dict['parameters'], dict):
                    if 'properties' in tool_dict['parameters'] and 'data' in tool_dict['parameters']['properties']:
                        data_def = tool_dict['parameters']['properties']['data']
                        print(f"     [Helper] ai_element_filter data 定義: {data_def}")
                        # 確保 data 是 object 類型且有屬性
                        if isinstance(data_def, dict):
                            if 'type' not in data_def:
                                data_def['type'] = 'object'
                            # 確保有 properties
                            if 'properties' not in data_def:
                                print(f"     [Helper] 添加缺失的 properties 到 data")
                                data_def['properties'] = {}
                            # 確保有 required 字段
                            if 'required' not in data_def:
                                data_def['required'] = []

            if tool_dict and isinstance(tool_dict, dict):
                # 確保 parameters 是字典格式
                if "parameters" not in tool_dict or tool_dict["parameters"] is None:
                    tool_dict["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }

                # 確保 parameters 是字典
                if not isinstance(tool_dict["parameters"], dict):
                    print(f"     [Helper] 警告: 工具 '{tool_dict.get('name', f'tool_{tool_idx}')}' 的 parameters 不是字典，將重置為默認值")
                    tool_dict["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }

                # 修復 schema 中的數組類型問題
                tool_dict["parameters"] = _fix_gemini_tool_schema(tool_dict["parameters"])

                tools_for_binding.append(tool_dict)
                print(f"     [Helper] 處理並修復了工具 '{tool_dict.get('name', f'tool_{tool_idx}')}' 的 schema")
            else:
                # 如果無法轉換，使用原始工具對象
                tools_for_binding.append(tool)
                print(f"     [Helper] 保留原始工具對象: {getattr(tool, 'name', f'tool_{tool_idx}')}")

        except Exception as e:
            print(f"     [Helper] 處理工具 '{getattr(tool, 'name', f'tool_{tool_idx}')}' 時出錯: {e}")
            # 保留原始工具對象作為後備
            tools_for_binding.append(tool)
    
    if not tools_for_binding and mcp_tools: # If all tools were invalid or some other issue
        print("     [Helper] 警告: 工具準備後列表為空，但原始列表非空。可能所有工具都無法處理。")
    elif not tools_for_binding and not mcp_tools:
        pass # Expected if input was empty
    else:
        print(f"     [Helper] 完成 Gemini 兼容工具準備，共 {len(tools_for_binding)} 個。")
    return tools_for_binding
# --- END NEW HELPER FUNCTION ---

# =============================================================================
# 圖節點 (Graph Nodes)
# =============================================================================

# --- Router Node ---
async def route_mcp_target(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """使用 utility_llm 判斷用戶初始請求文本應路由到哪個 MCP (revit)。"""
    print("--- 執行 MCP 路由節點 ---")

    # --- NEW: Check if target_mcp is already set in the state ---
    pre_set_target_mcp = state.get("target_mcp")
    valid_mcp_targets = ["revit"]
    if pre_set_target_mcp and pre_set_target_mcp in valid_mcp_targets:
        print(f"  檢測到已預設 target_mcp: '{pre_set_target_mcp}'。直接使用此目標，跳過 LLM 路由。")
        return {"target_mcp": pre_set_target_mcp, "last_executed_node": "router_skipped_due_to_preset"}
    # --- END NEW ---

    initial_request_text = state.get('initial_request', '')
    if not initial_request_text:
        print("錯誤：狀態中未找到 'initial_request' 且 target_mcp 未預設。默認為 revit。")
        # {{ edit_1 }}
        return {"target_mcp": "revit", "last_executed_node": "router_defaulted_revit_no_request"}
        # {{ end_edit_1 }}

    print(f"  根據初始請求文本路由: '{initial_request_text[:150]}...'")
    prompt = ROUTER_PROMPT.format(user_request_text=initial_request_text)
    try:
        response = await utility_llm.ainvoke([SystemMessage(content=prompt)], config=config)
        route_decision = response.content.strip().lower()
        print(f"  LLM 路由決定: {route_decision}")
        if route_decision in valid_mcp_targets: # Use the list here
            # {{ edit_2 }}
            return {"target_mcp": route_decision, "last_executed_node": "router_llm_decision"}
            # {{ end_edit_2 }}
        else:
            print(f"  警告: LLM 路由器的回應無法識別 ('{route_decision}')。預設為 revit。")
            # {{ edit_3 }}
            return {"target_mcp": "revit", "last_executed_node": "router_defaulted_revit_unknown_llm_response"}
            # {{ end_edit_3 }}
    except Exception as e:
        print(f"  路由 LLM 呼叫失敗: {e}")
        traceback.print_exc()
        # {{ edit_4 }}
        return {"target_mcp": "revit", "last_executed_node": "router_defaulted_revit_llm_exception"}
        # {{ end_edit_4 }}


# <<< 新增：訊息剪枝輔助函式 >>>
MAX_RECENT_INTERACTIONS_DEFAULT = 18
MAX_RECENT_INTERACTIONS_FORCING = 23

def _prune_messages_for_llm(full_messages: List[BaseMessage], max_recent_interactions: int = MAX_RECENT_INTERACTIONS_DEFAULT) -> List[BaseMessage]:
    if not full_messages:
        return []

    initial_human_message = None
    plan_ai_message = None

    # 找到初始的 HumanMessage (通常是列表中的第一個)
    if full_messages and isinstance(full_messages[0], HumanMessage): # Added check for full_messages not empty
        initial_human_message = full_messages[0]

    # 找到最新的計劃 AIMessage
    PLAN_PREFIX = "[目標階段計劃]:\n"
    for msg in reversed(full_messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX):
            plan_ai_message = msg
            break

    pruned_list = []
    added_message_ids = set() # 使用物件 id 來避免重複添加完全相同的訊息實例

    # 1. 添加初始 HumanMessage (如果存在)
    if initial_human_message:
        pruned_list.append(initial_human_message)
        added_message_ids.add(id(initial_human_message))

    # 2. 添加計劃 AIMessage (如果存在且與 initial_human_message 不同)
    if plan_ai_message and id(plan_ai_message) not in added_message_ids:
        # 確保計劃訊息不是列表中的第一個 HumanMessage (雖然不太可能，但以防萬一)
        if not (initial_human_message and id(plan_ai_message) == id(initial_human_message)):
            pruned_list.append(plan_ai_message)
            added_message_ids.add(id(plan_ai_message))

    # 3. 確定近期互動的候選訊息 (排除已添加的 initial_human_message 和 plan_ai_message)
    recent_interaction_candidates = []
    for msg in full_messages:
        if id(msg) not in added_message_ids:
            recent_interaction_candidates.append(msg)
    
    # 選取最後 N 條作為實際的近期互動訊息
    actual_recent_interactions = recent_interaction_candidates[-max_recent_interactions:]

    # 4. 將近期互動訊息添加到剪枝後的列表
    #    將 initial_human_message 和 plan_ai_message 放在前面，然後是 recent_interactions
    #    這裡的邏輯是重新構建 pruned_list，而不是在現有的 pruned_list 後追加
    final_pruned_list = []
    temp_added_ids = set()

    if initial_human_message:
        final_pruned_list.append(initial_human_message)
        temp_added_ids.add(id(initial_human_message))

    if plan_ai_message and id(plan_ai_message) not in temp_added_ids:
        final_pruned_list.append(plan_ai_message)
        temp_added_ids.add(id(plan_ai_message))
    
    for msg in actual_recent_interactions:
        if id(msg) not in temp_added_ids: # 避免再次添加 plan 或 initial human message 如果它們恰好在尾部
            final_pruned_list.append(msg)
            # temp_added_ids.add(id(msg)) # 不需要，因為是從尾部取的

    # --- 日誌記錄剪枝後的訊息 (可選，用於調試) ---
    # print(f"    原始訊息數量: {len(full_messages)}, 剪枝後訊息數量: {len(final_pruned_list)}")
    # pruned_message_summary = []
    # for i, m_obj in enumerate(final_pruned_list):
    #     m_content_str = ""
    #     if isinstance(m_obj.content, str):
    #         m_content_str = m_obj.content[:30].replace("\n", " ") + "..."
    #     elif isinstance(m_obj.content, list) and m_obj.content:
    #         first_item_content = m_obj.content[0]
    #         if isinstance(first_item_content, dict) and first_item_content.get("type") == "text":
    #             m_content_str = first_item_content.get("text", "")[:30] + "..."
    #         else:
    #             m_content_str = str(first_item_content)[:30] + "..."
    #     elif m_obj.content is None:
    #         m_content_str = "[None Content]"
    #     else:
    #         m_content_str = f"[{type(m_obj.content).__name__} Content]"
    #     pruned_message_summary.append(f"      {i}: {type(m_obj).__name__} - '{m_content_str}'")
    # print("    剪枝後訊息預覽:\n" + "\n".join(pruned_message_summary))
    # --- 結束日誌記錄 ---

    return final_pruned_list
# <<< 結束：訊息剪枝輔助函式 >>>

# =============================================================================
# Agent Nodes
# =============================================================================
async def agent_node_logic(state: MCPAgentState, config: RunnableConfig, mcp_name: str) -> Dict:
    """通用 Agent 節點邏輯：處理特定工具消息，規劃，或執行下一步。"""
    print(f"--- 執行 {mcp_name.upper()} Agent 節點 ---")
    
    current_messages = list(state['messages'])
    last_message = current_messages[-1] if current_messages else None
    current_consecutive_responses = state.get("consecutive_llm_text_responses", 0)
    current_revit_screenshot_counter = state.get("revit_screenshot_counter", 0)

    # --- 處理 Revit 工具的 ToolMessage 返回 ---
    CSV_PATH_PREFIX = "[CSV_FILE_PATH]:"
    IMAGE_PATH_PREFIX = "[IMAGE_FILE_PATH]:"

    if isinstance(last_message, ToolMessage):
        # Handle Local CSV Creation Tool - Save path to state
        if last_message.name == "create_planned_data_summary_csv":
            if last_message.content.startswith(CSV_PATH_PREFIX):
                csv_path = last_message.content[len(CSV_PATH_PREFIX):]
                print(f"  ✓ 計劃數據CSV報告已生成於: {csv_path}")
                print(f"  → CSV 是計劃的第 1 步，已完成")
                print(f"  → 將路徑保存到狀態，然後讓 LLM 處理 ToolMessage 並執行第 2 步")
                # 保存 CSV 路徑到狀態，但不提前返回
                # 讓後續的 call_llm_with_tools 正常處理這個 ToolMessage
                state["saved_csv_path"] = csv_path
                # CSV 工具處理完畢，不需要處理圖像，直接跳到 LLM 調用
                # 注意：下面的圖像處理代碼應該被跳過
        elif last_message.content.startswith(IMAGE_PATH_PREFIX):
            # Handle screenshot/image tools
            uuid_image_path = last_message.content[len(IMAGE_PATH_PREFIX):]
            print(f"    原始文件路徑 (UUID based): {uuid_image_path}")

            new_image_path_for_state = uuid_image_path # Default to original if rename fails
            data_uri_for_state = None
            # {{ edit_2 }}
            # --- MODIFIED: Renaming logic for Revit screenshots ---
            if mcp_name == "revit":
                current_revit_screenshot_counter += 1 # Increment counter from state

                # Sanitize initial_request for use in filename (take first 20 chars, replace spaces, keep alphanum and underscore)
                req_str_part = state.get('initial_request', 'RevitTask')
                sanitized_req_prefix = "".join(filter(lambda x: x.isalnum() or x == '_', req_str_part.replace(" ", "_")[:20]))

                original_extension = os.path.splitext(uuid_image_path)[1]
                new_filename = f"{sanitized_req_prefix}_Shot-{current_revit_screenshot_counter}{original_extension}"

                try:
                    if os.path.exists(uuid_image_path):
                        new_renamed_path = os.path.join(os.path.dirname(uuid_image_path), new_filename)
                        os.rename(uuid_image_path, new_renamed_path)
                        new_image_path_for_state = new_renamed_path # Use renamed path
                        print(f"    文件已重命名為: {new_renamed_path}")
                    else:
                        print(f"  !! 錯誤：capture_viewport 返回的原始文件路徑不存在: {uuid_image_path}。無法重命名。")
                        # new_image_path_for_state remains uuid_image_path, which is problematic if it doesn't exist.
                        # Consider how to handle this error - perhaps return an error message.
                        # For now, it will proceed and likely fail to generate URI / be found later.
                except Exception as rename_err:
                    print(f"  !! 重命名文件 '{uuid_image_path}' 至 '{new_filename}' 時出錯: {rename_err}")
                    # new_image_path_for_state remains uuid_image_path
            # --- END MODIFICATION ---
            # {{ end_edit_2 }}

            try:
                if not os.path.exists(new_image_path_for_state):
                    print(f"  !! 錯誤：處理後的圖像文件路徑不存在: {new_image_path_for_state}")
                    # {{ edit_3 }}
                    return {
                          "messages": [AIMessage(content=f"截圖文件未找到: {new_image_path_for_state}。")],
                          "saved_image_path": None, "saved_image_data_uri": None,
                          "task_complete": False,
                          "consecutive_llm_text_responses": 0,
                          "revit_screenshot_counter": current_revit_screenshot_counter # Return updated counter
                          # {{ end_edit_3 }}
                      }
                with open(new_image_path_for_state, "rb") as f: image_bytes = f.read()
                base64_data = base64.b64encode(image_bytes).decode('utf-8')
                mime_type = "image/png"
                ext = os.path.splitext(new_image_path_for_state)[1].lower()
                if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
                data_uri_for_state = f"data:{mime_type};base64,{base64_data}"
                # {{ edit_4 }}
                return {
                     "messages": [AIMessage(content=f"已成功截取畫面並保存至 {new_image_path_for_state}。")],
                     "saved_image_path": new_image_path_for_state,
                     "saved_image_data_uri": data_uri_for_state,
                     "task_complete": False,
                     "consecutive_llm_text_responses": 0,
                     "revit_screenshot_counter": current_revit_screenshot_counter # Return updated counter
                     # {{ end_edit_4 }}
                }
            except Exception as img_proc_err:
                print(f"  !! 處理截圖文件 '{new_image_path_for_state}' 或編碼時出錯: {img_proc_err}")
                # {{ edit_5 }}
                return {
                     "messages": [AIMessage(content=f"處理截圖文件 '{new_image_path_for_state}' 時失敗: {img_proc_err}。")],
                     "task_complete": False,
                     "consecutive_llm_text_responses": 0,
                     "revit_screenshot_counter": current_revit_screenshot_counter # Return updated counter
                     # {{ end_edit_5 }}
                 }
        elif last_message.content.startswith("[Error: Viewport Capture Failed]:"): 
                error_msg = last_message.content 
                print(f"  檢測到 capture_viewport 工具返回錯誤: {error_msg}")
                # {{ edit_6 }}
                return {"messages": [AIMessage(content=f"任務因截圖錯誤而中止: {error_msg}")], "task_complete": True, "consecutive_llm_text_responses": 0, "revit_screenshot_counter": current_revit_screenshot_counter} 
                # {{ end_edit_6 }}


            # Add more elif blocks here if other tools return specific results needing processing

    # --- 如果不是處理特定工具返回，則執行正常規劃/執行邏輯 ---
    try:
        # ... (Planning/Execution logic starts here) ...
        initial_image_path = state.get('initial_image_path')
        has_input_image = initial_image_path and os.path.exists(initial_image_path)
        if has_input_image: print(f"  檢測到初始圖片輸入: {initial_image_path}")
        else: print("  未檢測到有效初始圖片輸入。")

        if not current_messages or not isinstance(current_messages[0], HumanMessage):
             print("!! 錯誤：狀態 'messages' 為空或第一個消息不是 HumanMessage。")
             return {"messages": [AIMessage(content="內部錯誤：缺少有效的初始用戶請求消息。")]}
        initial_user_message_obj = current_messages[0]
        initial_user_text = ""
        if isinstance(initial_user_message_obj.content, str): initial_user_text = initial_user_message_obj.content
        elif isinstance(initial_user_message_obj.content, list):
            for item in initial_user_message_obj.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    initial_user_text = item.get("text", ""); break
        if not initial_user_text:
            print("!! 錯誤：無法從初始 HumanMessage 提取文本內容。")
            return {"messages": [AIMessage(content="內部錯誤：無法解析初始用戶請求文本。")]}
        print(f"  使用初始文本 '{initial_user_text[:100]}...' 作為基礎。")

        # PLAN_PREFIX = "[目標階段計劃]:\n" # <<< 移除此處的局部定義 >>>
        plan_exists = any(
            isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX)
            for msg in current_messages
        )

        # ========================
        # === PLANNING PHASE ===
        # ========================
        if not plan_exists:
            print(f"  檢測到無計劃，進入規劃階段...")
            # --- 獲取工具用於規劃提示 ---
            mcp_tools = await get_mcp_tools(mcp_name)
            print(f"  獲取了 {len(mcp_tools)} 個 {mcp_name} MCP 工具 (用於規劃提示)。")
            if not mcp_tools: print(f"  警告：未找到 {mcp_name} 工具！")

            # --- 新增: 將本地工具加入列表 ---
            all_available_tools = mcp_tools + LOCAL_TOOLS
            print(f"  提供給規劃師的工具總數: {len(all_available_tools)} (MCP: {len(mcp_tools)}, Local: {len(LOCAL_TOOLS)})")

            # --- 選擇規劃提示 ---
            active_planning_prompt_content = ""
            if mcp_name == "revit":
                active_planning_prompt_content = """你是一位優秀的任務規劃助理，專門為 Revit BIM 任務制定計劃。
            基於使用者提供的文字請求、可選的圖像以及下方列出的可用工具，生成一個清晰的、**分階段目標**的計劃。

            **重要要求：**
            1.  **工具使用策略 (優先順序):**
                * **優先使用結構化工具**: 在規劃階段，應優先考慮使用專門的 Revit 工具（如牆壁創建使用(create_line_based_element)、樓板創建(create_surface_based_element)、門窗放置(create_point_based_element)等）。只有在工具無法滿足需求時，才考慮使用 `send_code_to_revit` 進行自定義編程。
                * **send_code_to_revit 作為最後手段**: 這個工具僅用於複雜的自定義邏輯、批次處理或條件操作。對於標準的建築元件創建，應使用對應的專門工具以確保安全性和一致性。

            2.  **細緻的分步規劃 (極度重要):**
                * 將任務拆解為**非常細緻的小步驟**，每個步驟應該是一個可以獨立完成的原子操作
                * **代碼複雜度考量:** 每個步驟應該對應不超過 50-80 行的代碼量
                * 如果某個操作涉及多個子物件或重複動作，應該規劃為多個獨立步驟

            3.  **量化與具體化:** 對於建築元件操作 (Revit)，每個階段目標**必須**包含盡可能多的**具體數值、尺寸、高度、樓層、元件類型、材料、數量、距離、方向、或清晰的建築關係描述**。

            4.  **邏輯順序:** 確保階段目標按邏輯順序排列，後續步驟依賴於先前步驟的結果。

            5.  **建築座標系統意識 (Revit - 極度重要):**
                *   **確立基準方位:** 在進行任何建築設計時，**第一步必須是確立一個清晰的建築座標系統和方向基準**。明確定義建築的「北」方與其他「東、西、南」對應的方向，並在後續所有步驟中嚴格遵守此基準。
                *   **樓層與高度意識:** 正確設置建築樓層(Level)和高度參數，確保元件放置在正確的樓層上。
                *   **邊界意識:** 如果任務提供了基地邊界，**必須**將處理基地邊界作為優先步驟。
                    *   a. 規劃創建或識別代表基地邊界的線條或區域。
                    *   b. 在規劃放置任何建築元件之前，**必須**先驗證其預計位置**完全位於**已定義的基地邊界內部。

            6.  **標準 Revit 建模原則 (Revit Native):**
                *   **禁止創建工作集 (Worksets)**:不要規劃創建新的工作集。標準流程是直接在當前活動工作集或視圖中創建元件。
                *   **使用標準類別**: 規劃時應直接使用 Revit 的標準類別 (Categories) 如 `OST_Walls`, `OST_Doors`, `OST_Floors` 等來組織元件。
                *   **功能空間分離**: 每個功能空間（如1BR、2BR)應由其實際的圍護結構（牆、樓板）定義。不需要為每個房間創建抽象的容器或群組。
                *   **房型規劃**: 只需要規劃總共幾間各BR類型的房間（如1BR、2BR、3BR等）以及它們的整體佈局位置和朝向。不需要規劃房間內部的詳細LDK佈局。

            7.  **多樓層處理 (Revit):**
                *   計劃應清晰地標示每個樓層的開始和結束。每個樓層都要注意當前牆的基準約束和頂部約束是正確的。
                *   **樓層完成通知 (必須規劃):**
                    *   每完成一個樓層的所有建模工作後，**必須**規劃一個步驟來顯示通知對話框
                    *   使用 `send_code_to_revit` 執行簡單的 TaskDialog 代碼來通知用戶該樓層已完成
                    *   這樣用戶才知道何時該切換視圖到下一樓層
                *   **房間與標籤工具說明:**
                    *   `create_rooms_and_tags` 工具會同時創建房間並自動放置標籤，無需額外步驟
                    *   每次調用時指定 `roomName`、`x`、`y` 參數即可完成房間命名和標籤放置

            8.  **圖像參考規劃 (若有提供圖像):**
                *   在生成具體的建模計劃之前，**必須**先進行詳細的"圖像分析與解讀"階段。
                *   規劃時應基於：觀察到的主要建築元件組成和它們之間的**建築關係**（例如，相鄰牆、共享牆、樓板連接）；估計主要元件之間的尺寸、高度距離關係；主次要元件的空間關係；主要的立面特徵；門窗等開口位置。
                *   **必須**將上述圖像分析得出的觀察結果，轉化為後續 Revit 建模步驟中的具體參數和元件類型選擇。**需特別注意建築元件的位置關係；高度和樓層設置；元件類型的選擇，以構成符合圖片目標的建築設計。**
                *   **如果任務是參考圖片進行建築設計規劃，要在主要建築元件的關係下發展詳細的元件配置和空間佈局。不需要建立精確的細部裝飾。**

            9.  **基地邊界與單元配置規劃 (Revit - 極度重要):**
                *   **嚴格遵守基地邊界**: 所有規劃的建築單元和結構**必須完全位於**給定的基地邊界內。如果任務提供了街道及後院朝向信息，配置時必須將此納入首要考量。
                *   **創建邊界**: 當要求在基地內建置單元布局時，應該要先**創建當前專案中的地界線範圍**並以此為建模邊界**。如果無法創建邊界也應該要注意建模不能超出基地範圍。
                *   **單位及測量標準 (Imperial Units):**
                    *   **單位**: 所有長度單位必須使用**英尺 (feet)** 及 **分數英寸 (fractional inches)**。
                    *   **面積**: 所有面積單位必須使用 **平方英尺 (SF)**。
                *   **單元配置原則**:
                    1.  **公設樓層 (新增)**: 通常 1、2 樓為規劃公設的樓層，包含 Core 之外的 Entry Lobby、Community room、gym、pool & spa 等。
                    2.  **矩形優先**: 單元形狀應盡量以乾淨的矩形為優先，並加以組合成L、U、T、長型、方型等。
                    3.  **最小尺寸**: 單元的長或寬最小尺寸應大於 **15英尺 (15')**。
                    4.  **外牆優先**: 每個單元**只需要規劃外牆**，**不需要**規劃內部的隔間牆佈局。重點在於單元的整體量體和位置。
                    5.  **入口動線**: 所有單元的入口應規劃在**靠近核心筒 (Core) 的走廊與電梯廳 (Corridor & Elev. Lobby)** 的位置，以確保動線效率。
                    6.  **採光朝向**: 單元的主要採光面（開窗面）應朝向**街道**或**後院**方向。在規劃單元起始位置與朝向時，必須立即對應這兩個方向。
                    7.  **窗戶與牆類型 (關鍵):**
                        *   **不放置實際窗戶元件**: 不需要使用 `OST_Windows` 或放置窗戶族群。
                        *   **透明/開窗區域**: 使用牆類型 `Exterior with lighting - 1'` 來表示有開窗或玻璃的牆面區域。
                        *   **實牆區域**: 使用牆類型 `Exterior - 1'` 來表示最外圍的實體牆面。
                *   **步驟要求**: 規劃的第一步必須包含「分析基地邊界與朝向」，隨後才是「規劃單元配置」。

            10. **目標狀態:** 計劃應側重於**每個階段要達成的目標狀態**，說明該階段完成後場景應有的變化。
                *   **最後一個計劃應包含"全部任務已完成"時的相關行動，引導實際執行時的處理。**

            11.  **規劃數據摘要報告 (空間規劃任務的必要首步):**
                *   **僅當**任務是關於**空間佈局規劃** (例如，單元配置等)，你**必須**將生成摘要報告作為計劃的**第一個步驟**。
                *   **此步驟基於你即將制定的後續建模步驟，先行總結和報告規劃的量化數據。如果是要求分析已有的方案，則應該要先分析再進行數據摘要整理。**
                *   **規劃的第一步應如下：**
                    1.  **預先匯總:** 在腦中構思好所有建模步驟後，審查你計劃要創建的所有空間（如客廳、臥室等）的名稱、**所屬樓層**和具體尺寸/面積。
                    2.  **計算匯總數據:** 基於這些規劃數值，計算出總面積、每個空間的面積佔比，以及建蔽率(BCR)和容積率(FAR)（如果適用）。
                    3.  **規劃首個工具調用:** 將匯總好的數據（`data_rows` - 其中每個空間字典需包含 `name`, `area`, `percentage` **和 `floor`**，`total_area`, `bcr`, `far`）作為參數，將對 `create_planned_data_summary_csv` 工具的調用規劃為整個計劃的**第 1 步**。
                    4.  **後續步驟:** 在此報告步驟之後，再依次列出所有實際的 Revit 模型建構步驟。

            **revit提醒:目前單位是英制 (Imperial: Feet/Inches, SF)，符合建築設計標準。** **絕對禁止轉換單位**: 保持所有數值為英制 (Feet)。這個計劃應側重於**每個階段要達成的目標狀態並包含細節**，而不是具體的工具使用細節。將任務分解成符合邏輯順序及細節的多個階段目標。直接輸出這個階段性目標計劃，不要额外的開場白或解釋。
            可用工具如下 ({mcp_name}):
            {tool_descriptions}"""
            else: # Fallback
                tool_descriptions_for_fallback_str = "\n".join([f"- {tool.name}: {tool.description}" for tool in mcp_tools])
                active_planning_prompt_content = f"請為使用 {mcp_name} 的任務制定計劃。可用工具：\n{tool_descriptions_for_fallback_str}"

            # --- 格式化規劃提示 (For Revit BIM tasks) ---
            planning_system_content_final = active_planning_prompt_content
            if mcp_name == "revit":
                tool_descriptions_for_prompt = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_available_tools])
                planning_system_content_final = active_planning_prompt_content.format(
                    mcp_name=mcp_name,
                    tool_descriptions=tool_descriptions_for_prompt
                )
                # --- Always append Residential Guidelines for Revit Planning ---
                print("    [Planning] Appending RESIDENTIAL_MODELING_GUIDELINES to planning prompt.")
                planning_system_content_final += "\n\n" + RESIDENTIAL_MODELING_GUIDELINES
            # Note: No formatting needed for Fallback as prompts are already complete strings

            planning_system_message = SystemMessage(content=planning_system_content_final)
            print(f"    為 {mcp_name} 構造了規劃 SystemMessage")

            # --- 構造規劃 HumanMessage ---
            planning_human_content = [{"type": "text", "text": initial_user_text}]
            if has_input_image:
                try:
                    with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    # Determine mime type properly if possible, default to png
                    mime_type="image/png"
                    file_ext = os.path.splitext(initial_image_path)[1].lower()
                    if file_ext in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
                    elif file_ext == '.gif': mime_type = 'image/gif'
                    elif file_ext == '.webp': mime_type = 'image/webp'

                    planning_human_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
                    })
                    print("    已將初始圖片添加到規劃 HumanMessage 中。")
                except Exception as img_read_err:
                    print(f"    !! 無法讀取或編碼初始圖片: {img_read_err}")
                    # Fallback to text only if image fails
                    planning_human_content = [{"type": "text", "text": initial_user_text}]

            # Ensure content is always a list for multi-modal models
            if not isinstance(planning_human_content, list):
                 planning_human_content = [{"type": "text", "text": str(planning_human_content)}] # Should not happen with above logic, but safe fallback

            planning_human_message_user_input = HumanMessage(content=planning_human_content)

            # --- 調用 LLM 進行規劃 ---
            print(f"     正在調用 LLM ({agent_llm.model}) 進行規劃...")
            plan_message = None
            try:
                # Use the main agent LLM for planning
                planning_llm_no_callbacks = agent_llm.with_config({"callbacks": None})
                planning_response = await planning_llm_no_callbacks.ainvoke(
                    [planning_system_message, planning_human_message_user_input]
                )

                if isinstance(planning_response, AIMessage) and planning_response.content:
                    # Prepend the prefix to identify it as a plan
                    plan_content = PLAN_PREFIX + planning_response.content.strip()
                    plan_message = AIMessage(content=plan_content)
                    print(f"  生成階段目標計劃:\n------\n{plan_content[:500]}...\n------")
                else:
                    # Handle cases where planning LLM failed or returned unexpected format
                    error_msg = "LLM 未能生成有效計劃。"
                    if isinstance(planning_response, AIMessage) and not planning_response.content:
                         error_msg += " (回應內容為空)"
                    elif not isinstance(planning_response, AIMessage):
                         error_msg += f" (返回類型為 {type(planning_response).__name__})"
                    print(f"  !! {error_msg}")
                    plan_message = AIMessage(content=f"無法為您的請求制定計劃。({error_msg})") # Provide some error info

            except Exception as planning_err:
                 error_msg = f"調用規劃 LLM 時發生錯誤: {planning_err}"
                 print(f"  !! {error_msg}")
                 traceback.print_exc()
                 plan_message = AIMessage(content=error_msg) # Return the error message
            finally:
                rpm_delay = state.get("rpm_delay", 6.5)
                print(f"     規劃 LLM 調用結束，等待 {rpm_delay} 秒...")
                await asyncio.sleep(rpm_delay)
                print("     等待結束。")

            # --- *** 規劃完成後直接返回，觸發 should_continue *** ---
            # Return the plan message (or error message if planning failed)
            # Reset counter as this node completed its current task (planning)
            return {"messages": [plan_message] if plan_message else [], "consecutive_llm_text_responses": 0, "last_executed_node": f"{mcp_name}_agent"}

        # ==========================
        # === EXECUTION PHASE ===
        # ==========================
        else:
            print(f"  檢測到已有計劃，進入執行階段...")
            # --- 獲取 MCP 工具 ---
            mcp_tools = await get_mcp_tools(mcp_name)
            print(f"  獲取了 {len(mcp_tools)} 個 {mcp_name} MCP 工具 (用於執行)。")
            if not mcp_tools: print(f"  警告：執行階段未找到 {mcp_name} 工具！")

            # --- 組合所有可用工具 ---
            all_tools_for_execution = mcp_tools + LOCAL_TOOLS

            # --- 選擇執行提示 ---
            active_execution_prompt_template = None # Use template now
            if mcp_name == "revit":
                # Use the globally defined REVIT_AGENT_EXECUTION_PROMPT
                active_execution_prompt_template = REVIT_AGENT_EXECUTION_PROMPT
            else: # Only Revit is supported
                print(f"  警告：執行階段找不到為 {mcp_name} 定義的特定執行提示，將使用 Revit 後備提示。")
                active_execution_prompt_template = REVIT_AGENT_EXECUTION_PROMPT

            if not active_execution_prompt_template:
                 # Safety check
                 print(f"  !! 嚴重錯誤：未能為 {mcp_name} 確定有效的執行提示！")
                 return {"messages": [AIMessage(content=f"內部錯誤：無法為 {mcp_name} 加載執行指令。")], "consecutive_llm_text_responses": 0, "last_executed_node": f"{mcp_name}_agent_error"}

            # --- NEW: Format execution prompt with tools for relevant agents ---
            active_execution_prompt = None
            if "{tool_descriptions}" in active_execution_prompt_template.content:
                tool_descriptions_for_exec = "\n".join([f"- {tool.name}: {tool.description}" for tool in all_tools_for_execution])
                try:
                    formatted_content = active_execution_prompt_template.content.format(tool_descriptions=tool_descriptions_for_exec)
                    active_execution_prompt = SystemMessage(content=formatted_content)
                    print(f"  >> 成功格式化執行提示，長度: {len(formatted_content)}")
                except Exception as format_error:
                    print(f"  >> 格式化錯誤: {format_error}")
                    print(f"  >> 原始內容片段: {active_execution_prompt_template.content[:200]}...")
                    # 作為後備，直接使用未格式化的內容
                    active_execution_prompt = SystemMessage(content=active_execution_prompt_template.content.replace("{tool_descriptions}", tool_descriptions_for_exec))
            else:
                # For prompts that don't need tool formatting
                active_execution_prompt = active_execution_prompt_template
            # --- END NEW ---

            # --- 判斷是否為計劃生成後首次執行 ---
            is_first_execution_after_plan = False
            # 如果 plan_exists (我們在執行分支) 且最後一條消息是有效的計劃消息,
            # 這意味著我們剛從規劃階段過渡到執行階段的第一步。
            if plan_exists and isinstance(last_message, AIMessage) and \
               isinstance(last_message.content, str) and \
               last_message.content.strip().startswith(PLAN_PREFIX):
                
                # 再次確認這不是一個包含 PLAN_PREFIX 的錯誤消息
                is_actual_plan_msg = "無法為您的請求制定計劃" not in last_message.content and \
                                     "調用規劃 LLM 時發生錯誤" not in last_message.content
                if is_actual_plan_msg:
                    is_first_execution_after_plan = True
                    print("    檢測到這是計劃生成後的第一個執行調用 (最後一條消息是有效的計劃)。")

            # --- 準備執行階段的消息 ---
            messages_for_execution = current_messages
            # Ensure the first HumanMessage includes the image if provided and not already multi-modal
            if has_input_image and isinstance(messages_for_execution[0], HumanMessage) and not isinstance(messages_for_execution[0].content, list):
                # ... (修正 HumanMessage 以包含圖片的邏輯不變) ...
                 print("   修正執行階段的初始 HumanMessage 以包含圖片...")
                 try:
                     # Re-read image and create multi-modal content
                     with open(initial_image_path, "rb") as img_file: img_bytes = img_file.read()
                     img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                     mime_type="image/png" # Re-detect or use default
                     file_ext = os.path.splitext(initial_image_path)[1].lower()
                     if file_ext in ['.jpg', '.jpeg']: mime_type = 'image/jpeg'
                     elif file_ext == '.gif': mime_type = 'image/gif'
                     elif file_ext == '.webp': mime_type = 'image/webp'

                     initial_human_content = [
                         {"type": "text", "text": initial_user_text}, # Use the extracted text
                         {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}}
                     ]
                     messages_for_execution[0] = HumanMessage(content=initial_human_content)
                 except Exception as img_read_err:
                     print(f"   !! 無法讀取或編碼初始圖片用於執行階段: {img_read_err}")
                     # Proceed with text-only if image fails during execution prep


            # --- 調用 LLM 執行下一步 ---
            execution_response = None
            try:
                # --- PRUNE MESSAGES ---
                pruned_messages_for_llm = messages_for_execution # 預設不剪枝

                if mcp_name == "revit":
                    max_interactions_for_revit_pruning = MAX_RECENT_INTERACTIONS_DEFAULT
                    if is_first_execution_after_plan:
                        max_interactions_for_revit_pruning = 2
                        print(f"    為 Revit 首次執行調用，設定 max_interactions_for_pruning={max_interactions_for_revit_pruning} (保留初始請求、計劃和少量近期互動)。")
                    else:
                        print(f"    為 Revit 非首次執行調用，使用預設歷史記錄交互數量: {max_interactions_for_revit_pruning}")

                    print(f"  Revit: 準備執行 LLM 調用，原始待處理消息數: {len(messages_for_execution)}")
                    pruned_messages_for_llm = _prune_messages_for_llm(messages_for_execution, max_interactions_for_revit_pruning)
                else: # 對於其他 MCP
                    print(f"  {mcp_name.upper()}: 不執行訊息剪枝。原始待處理消息數: {len(messages_for_execution)}")
                    # pruned_messages_for_llm 已設為 messages_for_execution (不剪枝)

                print(f"  剪枝後/處理後傳遞給 LLM 的消息數: {len(pruned_messages_for_llm)}")
                
                execution_response = await call_llm_with_tools(pruned_messages_for_llm, all_tools_for_execution, active_execution_prompt)

            finally:
                rpm_delay = state.get("rpm_delay", 6.5)
                print(f"     執行 LLM 調用結束，等待 {rpm_delay} 秒...")
                await asyncio.sleep(rpm_delay)
                print("     等待結束。")

            # --- 更新連續空響應計數器 ---
            new_consecutive_responses = 0 # Reset by default
            if isinstance(execution_response, AIMessage):
                has_tool_calls = hasattr(execution_response, 'tool_calls') and execution_response.tool_calls
                has_content = execution_response.content is not None and execution_response.content.strip() != ""
                if has_tool_calls:
                    new_consecutive_responses = 0 # Corrected Indentation
                    print(f"  LLM 返回 {len(execution_response.tool_calls)} 個工具調用，重置連續文本響應計數器為 0。")
                elif has_content:
                    # Includes error messages, completion messages, etc.
                    new_consecutive_responses = 0 # Corrected Indentation
                    print(f"  LLM 返回帶有內容的文本消息 ('{execution_response.content[:50]}...')，重置連續文本響應計數器為 0。")
                else: # No tool calls, no content (empty string or None)
                    new_consecutive_responses = current_consecutive_responses + 1 # Corrected Indentation
                    print(f"  LLM 返回空內容且無工具調用，遞增連續文本響應計數器為 {new_consecutive_responses}。")
            else: # Not an AIMessage (e.g., internal error in call_llm_with_tools returned something else)
                new_consecutive_responses = 0
                print(f"  最終返回非 AIMessage 類型 ({type(execution_response).__name__})，重置連續文本響應計數器為 0。") # Corrected Indentation

            # --- 檢查計數器閾值 ---
            task_complete_due_to_counter = False
            messages_to_return = [] # Initialize list for messages to add to state this turn
            if new_consecutive_responses >= 3:
                print(f"  已連續收到 {new_consecutive_responses} 次無效響應，將標記任務完成。") # Corrected Indentation
                task_complete_due_to_counter = True
                error_msg = f"[系統錯誤：連續 {new_consecutive_responses} 次未能生成有效工具調用或完成消息，任務強制終止。]" # Corrected Indentation
                # Append the problematic response if it exists and isn't the error message itself
                if execution_response and (not isinstance(execution_response, AIMessage) or execution_response.content != error_msg): # Corrected Indentation
                    messages_to_return.append(execution_response) # Corrected Indentation
                messages_to_return.append(AIMessage(content=error_msg)) # Add the termination message
            elif execution_response: # If counter not exceeded, add the valid response from LLM # Corrected Indentation
                messages_to_return.append(execution_response) # Corrected Indentation

            # --- 返回執行結果 ---
            return_dict = {
                "messages": messages_to_return,
                "consecutive_llm_text_responses": new_consecutive_responses,
                "last_executed_node": f"{mcp_name}_agent", # 更新執行的節點名
                "revit_screenshot_counter": current_revit_screenshot_counter # Pass back updated counter
            }
            if task_complete_due_to_counter:
                return_dict["task_complete"] = True # Mark task complete if counter triggered

            return return_dict

    except Exception as e:
        print(f"!! 執行 {mcp_name.upper()} Agent 節點時發生外部錯誤: {e}")
        traceback.print_exc()
        # Return error message and reset counter
        # {{ edit_2 }}
        return {"messages": [AIMessage(content=f"執行 {mcp_name} Agent 時發生外部錯誤: {e}")], "consecutive_llm_text_responses": 0, "last_executed_node": f"{mcp_name}_agent_error", "revit_screenshot_counter": current_revit_screenshot_counter}
        # {{ end_edit_2 }}

# --- 具體的 Agent Nodes (添加 OSM) ---
async def call_revit_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    return await agent_node_logic(state, config, "revit")



# --- Tool Executor Node (保持不變) ---
async def agent_tool_executor(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """執行 Agent 請求的工具調用 - 專門用於 Revit MCP。"""
    print("--- 執行 Agent 工具節點 (Revit) ---")
    messages = state['messages']
    last_message = messages[-1] if messages else None

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("  最後消息沒有工具調用，跳過。")
        return {"last_executed_node": "agent_tool_executor_skipped"}

    # 直接使用 "revit"，不需要從狀態中讀取
    mcp_name = "revit"
    print(f"  目標 MCP: {mcp_name}")
    
    try:
        mcp_tools = await get_mcp_tools(mcp_name)
        all_tools_for_execution = mcp_tools + LOCAL_TOOLS
        print(f"  使用 {len(all_tools_for_execution)} 個總工具 ({mcp_name} MCP: {len(mcp_tools)}, Local: {len(LOCAL_TOOLS)})。")
        tool_messages = await execute_tools(last_message, all_tools_for_execution)
        print(f"  工具執行完成，返回 {len(tool_messages)} 個 ToolMessage。")
        return {"messages": tool_messages, "last_executed_node": "agent_tool_executor"}
    except Exception as e:
        print(f"!! 執行 Agent 工具節點時發生錯誤: {e}")
        traceback.print_exc()
        error_msg = f"執行工具時出錯: {e}"
        error_tool_messages = [ ToolMessage(content=error_msg, tool_call_id=tc.get("id"), name=tc.get("name", "unknown_tool")) for tc in last_message.tool_calls ]
        return {"messages": error_tool_messages, "last_executed_node": "agent_tool_executor_error"}

# --- Fallback Agent Node ---
async def call_fallback_agent(state: MCPAgentState, config: RunnableConfig) -> Dict:
    """調用補救 LLM 嘗試恢復流程 - 專門用於 Revit MCP。"""
    print("--- 執行 Fallback Agent 節點 (Revit) ---")
    current_messages = state['messages']

    # 直接使用 "revit"，不需要從狀態中讀取
    mcp_name = "revit"

    # 提取相關歷史記錄用於提示
    plan_message = next((msg for msg in reversed(current_messages) if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip().startswith(PLAN_PREFIX)), None)
    plan_content_for_prompt = ""
    if plan_message and isinstance(plan_message.content, str):
        # MODIFIED: Use the full plan content for the prompt
        plan_content_for_prompt = plan_message.content.strip()
        print(f"  提取到完整計劃內容 (用於 Fallback Prompt): {plan_content_for_prompt[:500]}...")

    # 只取最近幾條消息 + 計劃 (計劃已單獨處理，這裡只取非計劃的近期消息)
    # MODIFIED: _prune_messages_for_llm now only gets recent *non-plan* messages if plan is found
    # Or, we can build the history string more explicitly. Let's build it explicitly for clarity.
    
    history_items = []
    # Add the initial human message if available (usually the first)
    if current_messages and isinstance(current_messages[0], HumanMessage):
        first_human_msg = current_messages[0]
        # Summarize the initial human message if it's the one with the image content list
        if isinstance(first_human_msg.content, list):
            text_part = ""
            for item in first_human_msg.content:
                 if isinstance(item, dict) and item.get("type") == "text":
                      text_part = item.get("text", "")
                      break
            history_items.append(f"初始用戶請求 (HumanMessage): {text_part[:300]}...") # Summarize initial request text
        else:
             history_items.append(f"初始用戶請求 (HumanMessage): {str(first_human_msg.content)[:300]}...") # Summarize initial request string

    # Add the plan message's full content (already extracted above)
    if plan_content_for_prompt:
         history_items.append(f"\n---\n完整目標階段計劃 (AIMessage):\n{plan_content_for_prompt}\n---")


    # Add recent messages (excluding the initial human message and the plan message if they are at the end)
    # Let's grab the last N messages, but skip the first if it's the initial human, and skip the last if it's the plan message itself.
    messages_for_recent_history = current_messages[1:] # Skip the first message assuming it's the initial Human
    if messages_for_recent_history and plan_message and id(messages_for_recent_history[-1]) == id(plan_message):
         messages_for_recent_history = messages_for_recent_history[:-1] # Skip the plan message if it's the last

    # Get the last few relevant messages (e.g., last 5-7 interactions)
    max_recent = 7 # Limit recent history to avoid overwhelming the LLM
    recent_messages_to_summarize = messages_for_recent_history[-max_recent:]


    for msg in recent_messages_to_summarize:
        msg_summary = f"{type(msg).__name__}: "
        if isinstance(msg.content, str):
            msg_summary += f"{msg.content[:500]}..." if len(msg.content) > 500 else msg.content
        elif isinstance(msg.content, list):
            # Summarize list content (e.g., tool message with file path)
            summary_parts = []
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    summary_parts.append(item.get("text", "")[:100] + "...")
                elif isinstance(item, str): # Handle ToolMessage content which might be JSON string or simple string
                    summary_parts.append(item[:100] + "...")
                else:
                    summary_parts.append(f"[{type(item).__name__} content]")
            msg_summary += " | ".join(summary_parts)
        elif hasattr(msg, 'tool_calls') and msg.tool_calls: # Check for tool_calls attribute
            # Summarize tool calls
            tool_call_summaries = []
            for tc in msg.tool_calls:
                 tool_call_summaries.append(f"ToolCall(name={tc.get('name', 'N/A')}, args={str(tc.get('args', {}))[:100]}...)")
            msg_summary += f"ToolCalls: {'; '.join(tool_call_summaries)}"

        history_items.append(msg_summary)

    # Join the history items into a single string for the prompt
    relevant_history_str = "\n".join(history_items)


    prompt_content = FALLBACK_PROMPT.content.format(relevant_history=relevant_history_str)
    fallback_system_message = SystemMessage(content=prompt_content)
    print(f"  Fallback Agent Prompt (Partial Preview):\n{prompt_content[:1000]}...") # Print a longer preview

    original_fallback_response = None
    fallback_response_to_return = None
    mcp_tools_raw = [] # Define outside try for access in parsing block

    try:
        # 獲取工具以供綁定（補救 LLM 也需要知道可用工具）
        mcp_tools_raw = await get_mcp_tools(mcp_name) # Assign to mcp_tools_raw
        if not mcp_tools_raw:
             print(f"  !! Fallback Agent 警告：未找到 {mcp_name} 工具！")
        
        # --- 使用輔助函數準備 Gemini 兼容的工具 ---
        gemini_compatible_fallback_tools = _prepare_gemini_compatible_tools(mcp_tools_raw)

        # 使用 agent_llm (Gemini) 或 fast_llm (如果已定義)
        fallback_llm = fast_llm if 'fast_llm' in globals() else agent_llm
        llm_with_tools = fallback_llm.bind_tools(gemini_compatible_fallback_tools) # Bind corrected tools
        llm_configured = llm_with_tools.with_config({"callbacks": None})

        messages_for_llm_invoke = [fallback_system_message]
        # Add a neutral HumanMessage to ensure the 'contents' field is not empty
        # when the SystemMessage is potentially moved to 'system_instruction' by LangChain.
        # This message also serves as a conversational turn for the LLM to respond to.
        # Using "." is a common minimal prompt to trigger a response based on system instructions.
        messages_for_llm_invoke.append(HumanMessage(content="."))

        # original_fallback_response = await llm_configured.ainvoke([fallback_system_message]) # OLD
        original_fallback_response = await llm_configured.ainvoke(messages_for_llm_invoke) # NEW
        print(f"  Fallback Agent 原始響應: {original_fallback_response}")
        
        fallback_response_to_return = original_fallback_response # Default


        # --- Reinstated: Process fallback_response to extract tool_calls from content if necessary ---
        if isinstance(original_fallback_response, AIMessage) and \
           isinstance(original_fallback_response.content, str) and \
           not original_fallback_response.tool_calls: 
            
            content_str = original_fallback_response.content.strip()
            is_potential_json_tool_call = False
            if (content_str.startswith('{') and content_str.endswith('}')):
                 is_potential_json_tool_call = True
            elif content_str.startswith('```json'):
                 match = re.match(r'^```json\s*(\{.*?\})\s*```$', content_str, re.DOTALL | re.IGNORECASE)
                 if match:
                     content_str = match.group(1).strip()
                     is_potential_json_tool_call = True
                 else:
                     if "tool_calls" in content_str and ("recipient_name" in content_str or "name" in content_str) : # Added "name"
                          cleaned_md_json_str = re.sub(r'^```(?:json)?\s*|\s*```$', '', original_fallback_response.content.strip(), flags=re.IGNORECASE)
                          if cleaned_md_json_str.strip().startswith('{'):
                              content_str = cleaned_md_json_str.strip()
                              is_potential_json_tool_call = True
            
            if is_potential_json_tool_call:
                try:
                    parsed_json = json.loads(content_str)
                    if isinstance(parsed_json, dict) and "tool_calls" in parsed_json and isinstance(parsed_json["tool_calls"], list):
                        processed_tool_calls = []
                        for tc_orig in parsed_json["tool_calls"]:
                            if isinstance(tc_orig, dict):
                                tc = tc_orig.copy() 
                                tool_name_to_set = None
                                tool_args_to_set = tc.get("parameters", tc.get("args", {}))
                                raw_name = tc.get("recipient_name", tc.get("name"))

                                if raw_name:
                                    func_name_part = raw_name
                                    if raw_name.startswith("functions."):
                                        func_name_part = raw_name.split("functions.", 1)[1]
                                    
                                    found_tool_match = False
                                    # Use mcp_tools_raw which contains the original BaseTool objects
                                    for t_obj in mcp_tools_raw: 
                                        if t_obj.name == func_name_part: 
                                            tool_name_to_set = t_obj.name
                                            found_tool_match = True
                                            break
                                        if t_obj.name.endswith(f"_{func_name_part}"): 
                                            tool_name_to_set = t_obj.name
                                            found_tool_match = True
                                            break
                                    if not found_tool_match:
                                         print(f"  Fallback Agent: Could not reliably map name '{raw_name}' to a known tool. Using '{func_name_part}'.")
                                         tool_name_to_set = func_name_part
                                else:
                                    print(f"  Fallback Agent: Tool call missing 'recipient_name' or 'name': {tc_orig}")
                                    continue 
                                
                                new_tc_entry = {
                                    "name": tool_name_to_set,
                                    "args": tool_args_to_set,
                                    "id": tc.get("id", str(uuid.uuid4()))
                                }
                                processed_tool_calls.append(new_tc_entry)
                            
                        if processed_tool_calls:
                             placeholder_content = "[Fallback agent initiated tool call via content parsing.]"
                             fallback_response_to_return = AIMessage(
                                 content=placeholder_content, 
                                 tool_calls=processed_tool_calls,
                                 id=original_fallback_response.id if original_fallback_response else str(uuid.uuid4()), 
                                 additional_kwargs=original_fallback_response.additional_kwargs if original_fallback_response else {},
                                 response_metadata=original_fallback_response.response_metadata if original_fallback_response else {},
                                 # tool_call_chunks should be fine as None/default if not streaming
                             )
                             print(f"  Fallback Agent: Reconstructed AIMessage with tool_calls attribute: {fallback_response_to_return.tool_calls} and content: '{placeholder_content}'")
                        else:
                            print("  Fallback Agent: Parsed JSON from content, but 'tool_calls' list was empty or malformed after processing.")
                    # else:
                        # print(f"  Fallback Agent: Content was JSON, but not in expected tool_calls format. Parsed: {json.dumps(parsed_json, indent=2)}")


                except json.JSONDecodeError:
                    print(f"  Fallback Agent: Content looked like JSON for tool call but failed to parse: {content_str[:200]}...")
                except Exception as e_proc:
                    print(f"  Fallback Agent: Error processing content for tool_calls: {e_proc} on content {content_str[:200]}")
        # --- END Reinstated Parsing ---


    except Exception as e:
        print(f"!! Fallback Agent 調用 LLM 或解析時發生錯誤: {e}") # Modified error message
        traceback.print_exc()
        # Ensure fallback_response_to_return is an AIMessage
        if not isinstance(fallback_response_to_return, AIMessage):
            fallback_response_to_return = AIMessage(content=f"[FALLBACK_LLM_ERROR_OR_PARSING] {e}")
        else: # If it was already an AIMessage (e.g. from LLM and parsing failed later), append error
            fallback_response_to_return.content += f" [Error during post-processing: {e}]"

    finally:
        # 短暫等待，避免速率限制
        rpm_delay = state.get("rpm_delay", 6.5)
        await asyncio.sleep(rpm_delay / 2) # Shorter delay for fallback
        print("     Fallback Agent 等待結束。")

    return {"messages": [fallback_response_to_return] if fallback_response_to_return else [], "last_executed_node": "fallback_agent"}

# =============================================================================
# Conditional Edge Logic (修改 should_continue 處理 task_complete)
# =============================================================================
def should_continue(state: MCPAgentState) -> str:
    """確定是否繼續處理請求、調用工具、調用補救或結束 - 專門用於 Revit MCP。"""
    print("--- 判斷是否繼續 ---")
    messages = state['messages']
    last_message = messages[-1] if messages else None
    last_node = state.get("last_executed_node")

    # 直接使用 "revit"，不需要從狀態中讀取
    mcp_name = "revit"

    # --- 優先檢查 task_complete 標誌 (通常由 agent_node_logic 中的工具結果或連續錯誤觸發) ---
    if state.get("task_complete"):
        print(f"  檢測到 task_complete 標誌 (可能來自工具或連續錯誤) -> end")
        return END

    if not last_message:
        print("  消息列表為空 -> end")
        return END

    # --- 檢查 AI 是否請求工具調用 (來自任何 Agent，包括 Fallback) ---
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"  AI請求工具 ({len(last_message.tool_calls)}個從 {last_node}) -> agent_tool_executor")
        return "agent_tool_executor" 

    # --- 處理計劃生成 (通常由 primary agent 在沒有計劃時觸發) ---
    if isinstance(last_message, AIMessage) and isinstance(last_message.content, str) and last_message.content.strip().startswith(PLAN_PREFIX):
        is_actual_plan = "無法為您的請求制定計劃" not in last_message.content and "調用規劃 LLM 時發生錯誤" not in last_message.content
        if is_actual_plan:
            if last_node and (last_node.endswith("_agent") or last_node.endswith("_planner")):
                 print(f"  最後消息是新生成的計劃 (來自 {last_node}) -> 返回 {mcp_name}_agent 執行第一步")
                 return f"{mcp_name}_agent"
            else: 
                 print(f"  !! 錯誤: 計劃意外來自非 Agent 節點 ({last_node}) -> end (異常)")
                 return END
        else: 
             print(f"  最後消息是計劃生成錯誤 ('{last_message.content[:50]}...') -> end") 
             return END

    # --- 檢查是否為工具執行結果 (ToolMessage) ---
    if isinstance(last_message, ToolMessage):
        print(f"  最後消息是 ToolMessage (來自工具 '{last_message.name}') -> 返回 {mcp_name}_agent 處理結果")
        return f"{mcp_name}_agent"

    # --- 處理 AIMessage (非計劃，且沒有 tool_calls) ---
    if isinstance(last_message, AIMessage):
        # 確保 content_str 是實際的字串，如果 content 為 None，則預設為空字串以便安全處理
        raw_content = last_message.content
        content_str = str(raw_content).lower() if raw_content is not None else ""

        # --- 處理 Fallback Agent 的輸出 (沒有 tool_calls attribute) ---
        if last_node == "fallback_agent":
            fallback_end_keywords = [
                "[fallback_cannot_recover]", "[fallback_error]", "[fallback_llm_error]",
                "[fallback_llm_error_or_parsing]", "[fallback_confirmed_completion]",
            ]
            if any(keyword in content_str for keyword in fallback_end_keywords):
                if "[fallback_confirmed_completion]" in content_str:
                    print(f"  檢測到 Fallback Agent 確認任務成功完成 ('{content_str[:50]}...') -> end")
                else:
                    print(f"  檢測到 Fallback Agent 明確的失敗/無法恢復消息 ('{content_str[:50]}...') -> end")
                return END
            else:
                print(f"  !! 錯誤: Fallback Agent ({last_node}) 輸出非工具/非明確結束信號的 AIMessage ('{content_str[:50]}...') -> end (異常)")
                return END

        # --- 處理來自 主要 Agent / Planner 的 AIMessage ---
        if last_node and (last_node.endswith("_agent") or last_node.endswith("_planner")):
            # 1. 檢查主要 Agent/Planner 的完成關鍵字
            primary_agent_completion_keywords = [ "全部任務已完成", "任務完成" ]
            if any(keyword in content_str for keyword in primary_agent_completion_keywords):
                print(f"  檢測到主要 Agent/Planner ({last_node}) 的完成消息 ('{content_str[:50]}...'). 路由到 fallback_agent 進行驗證。")
                return "fallback_agent"

            # 2. 檢查主要 Agent/Planner 的內容是否為空
            #    (沒有 tool_calls 的情況已在最前面處理)
            if not content_str.strip(): # 如果內容為空或僅包含空白字符
                print(f"  來自主要 Agent/Planner ({last_node}) 的 AIMessage 內容為空或僅空白。路由到 fallback_agent。")
                return "fallback_agent"
            
            # 3. 如果內容非空且不是完成關鍵字，則是主要 Agent/Planner 的中間步驟文本。
            #    路由回主要 Agent 繼續其自身邏輯。
            print(f"  來自主要 Agent/Planner ({last_node}) 的中間文本 AIMessage ('{content_str[:50]}...'). 路由回 {mcp_name}_agent。")
            return f"{mcp_name}_agent"

        # --- 處理來自 agent_tool_executor 的 AIMessage ---
        # (這通常是在 agent_node_logic 處理 ToolMessage 後生成的文本消息，
        #  例如 "screenshot saved at X")
        if last_node == "agent_tool_executor":
            # 這類消息是資訊性的。主要 Agent 需要看到它們才能繼續執行計劃。
            # 如果這裡的消息為空，也應該路由到 fallback。
            if not content_str.strip(): # 如果內容為空或僅包含空白字符
                print(f"  來自 agent_tool_executor 的 AIMessage 內容為空或僅空白。路由到 fallback_agent。")
                return "fallback_agent"

            print(f"  來自 agent_tool_executor 的 AIMessage (工具結果處理後的信息) ('{content_str[:50]}...'). 路由回 {mcp_name}_agent。")
            return f"{mcp_name}_agent"

        # --- 其他 AIMessage 的捕獲 ---
        # (例如，來自未知節點，或以上邏輯未能覆蓋的情況)
        print(f"  來自節點 '{last_node}' 的無法分類的 AIMessage (無工具、非計劃) ('{content_str[:50]}...'). 路由到 fallback_agent。")
        return "fallback_agent"

    # --- 其他意外情況 ---
    elif isinstance(last_message, HumanMessage):
        print("  在流程中意外出現 HumanMessage (非初始請求) -> end (異常)")
        return END
    else:
        print(f"  未知的最後消息類型 ({type(last_message).__name__}) 或無法處理的狀態 -> end")
        return END

# =============================================================================
# 建立和編譯 LangGraph (添加 OSM 節點和邊)
# =============================================================================
workflow = StateGraph(MCPAgentState)
workflow.add_node("revit_agent", call_revit_agent)
workflow.add_node("agent_tool_executor", agent_tool_executor)
# --- 新增 Fallback Node ---
workflow.add_node("fallback_agent", call_fallback_agent)

workflow.set_entry_point("revit_agent")  # 直接進入 revit_agent

# --- Primary Agent Edges ---
# 由於 should_continue 的邏輯已修改，主要 agent 不再直接連接到 END。
# 它會請求工具 (agent_tool_executor)，處理工具結果後返回自身，或者如果卡住/聲稱完成，
# should_continue 會將它路由到 fallback_agent。

workflow.add_conditional_edges(
    "revit_agent",
    should_continue,
    {
        "agent_tool_executor": "agent_tool_executor",
        "revit_agent": "revit_agent",
        "fallback_agent": "fallback_agent",
        END: END
    }
)


# --- Fallback Agent Edges ---
workflow.add_conditional_edges(
    "fallback_agent",
    should_continue, # Reuse the same logic
    {
        "agent_tool_executor": "agent_tool_executor", # Fallback succeeded in generating tool call
        "revit_agent": "revit_agent",
        # For now, this setup relies on FALLBACK_PROMPT guiding it to either tool_call or [FALLBACK_CANNOT_RECOVER]
        "fallback_agent": "fallback_agent", # Allows fallback to re-evaluate if it produces text instead of tools/end
        END: END # If should_continue detects explicit fallback failure or other critical errors
    }
)


# --- Tool Executor Edges ---
# After tools are executed, should_continue will route to the correct primary agent
# (revit_agent) based on the target_mcp in the state,
# or to fallback_agent if the primary agent then gets stuck.
workflow.add_conditional_edges(
   "agent_tool_executor",
   should_continue, # should_continue correctly routes ToolMessages back to the target_mcp_agent
   {
       "revit_agent": "revit_agent",
       "fallback_agent": "fallback_agent", # This path is less likely if ToolMessage logic in should_continue is robust
                                        # as ToolMessages should go to primary agents.
                                        # However, if a primary agent immediately yields to fallback after a tool, this covers it.
       END: END # If should_continue determines an end condition after tool execution (e.g. task_complete set by tool)
   }
)

graph = workflow.compile().with_config({"recursion_limit": 1000})
# --- 修改 Graph Name ---
graph.name = "Revit_BIM_Agent_V1" # 專用於 Revit BIM 建模
print(f"LangGraph 編譯完成: {graph.name}")







