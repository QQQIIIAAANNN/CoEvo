# PM 節點架構說明

## 📋 概述

PM (Project Manager) 節點現在是整個 LangGraph 工作流的**中央控制器**，負責：
1. **初始規劃**：決定要執行哪些代理
2. **迭代控制**：決定是否繼續下一輪或結束
3. **最終評估**：整合所有評估結果並生成報告

## 🔄 工作流架構

```
START 
  ↓
PM (初始規劃 - 決定本輪要執行哪些節點)
  ↓
question_summary* → analyze_site* → designThinking 
  ↓                                     ↑
GateCheck1 ←───────────────────────────┘
  ↓
shell_prompt → image_render → GateCheck2
  ↓                               ↑
future_scenario ←─────────────────┘
  ↓
generate_3D → deep_evaluation
  ↓
PM (評估後決策：繼續/結束)
  ↓             ↓
  ├─ CONTINUE → question_summary* (下一輪，節點自動檢查是否執行)
  └─ END → 生成最終報告並結束

* 標記星號的節點會檢查 PM 計劃，第二輪起可能自動跳過
```

## 🎯 PM 節點的三種運作模式

### 1. 初始規劃階段（輪次 0）
- **觸發條件**：`current_round == 0`
- **行為**：PM 決定第一輪要執行的代理
  ```python
  pm_plan = {
      "enabled_agents": {
          "question_summary": True,    # ✅ 首輪必須分析用戶需求
          "analyze_site": True/False,  # ⚙️ 根據 config.run_site_analysis 決定
          "designThinking": True,      # ✅ 首輪必須生成設計
          "GateCheck1": True,
          "shell_prompt": True,
          "image_render": True,
          "GateCheck2": True,
          "future_scenario": True,
          "generate_3D": True,
          "deep_evaluation": True
      },
      "workflow_mode": "adaptive",
      "current_round": 0
  }
  ```
- **輸出**：`evaluation_status = "CONTINUE"`，流程開始執行

### 2. 第二輪及以後（智能跳過）
- **觸發條件**：`current_round >= 1`
- **智能決策邏輯**：
  ```python
  pm_plan = {
      "enabled_agents": {
          "question_summary": False,   # ⏭️ 第二輪起不再需要
          "analyze_site": False,       # ⏭️ 第二輪起不再需要
          "designThinking": True,      # ✅ 每輪都需要根據改進建議重新設計
          "GateCheck1": True,
          "shell_prompt": True,
          "image_render": True,
          "GateCheck2": True,
          "future_scenario": True,
          "generate_3D": True,
          "deep_evaluation": True
      }
  }
  ```
- **效果**：`question_summary` 和 `analyze_site` 被跳過，直接使用第一輪的結果

### 3. 評估後決策與最終評估
- **觸發條件**：每輪 `deep_evaluation` 完成後
- **行為**：
  - **未達最大輪次**：生成下一輪計劃 → `evaluation_status = "CONTINUE"`
  - **已達最大輪次**：
    - 整合所有輪次的評估結果
    - 調用 LLM 生成總結報告
    - 儲存至 `state["final_evaluation"]`
    - 設置 `pm_phase = "completed"`, `evaluation_status = "END"`
    - 流程結束

## 🔧 如何使用

### 基本執行
```python
from design_T_graph import graph, initial_state

# 執行工作流
result = graph.invoke(
    initial_state,
    config={
        "configurable": {
            "max_evaluation_rounds": 3,  # PM 會根據此參數決定迭代次數
            "run_site_analysis": True,   # PM 會讀取此配置
            # ... 其他配置
        }
    }
)

# 取得結果
print(result["final_evaluation"])  # PM 生成的最終報告
print(result["pm_plan"])           # PM 的執行計劃
```

### 自訂 PM 行為
你可以在 `ProjectManagerTask.run()` 中修改決策邏輯：

```python
# 在 design_T_graph.py 的 ProjectManagerTask 類中

# 範例：根據評分決定是否提前結束
if is_after_evaluation:
    eval_counts = self.state.get("evaluation_count", [])
    latest_score = list(eval_counts[-1].values())[0] if eval_counts else 0
    
    # 如果分數超過 90 分，提前結束
    if latest_score > 90:
        print("🎉 分數達標，提前結束！")
        should_continue = False
    else:
        should_continue = current_round < max_rounds
```

## 📊 State 結構變化

### 新增欄位
```python
{
    "pm_plan": {
        "enabled_agents": {...},
        "workflow_mode": "full/quick/minimal",
        "iteration_strategy": "continue/end"
    },
    "pm_phase": "initial/execution/completed",
    "evaluation_status": "CONTINUE/END"
}
```

## 🔍 執行流程範例

### 第一輪（輪次 0）
```
PM 啟動 → 生成計劃（所有節點 enabled）
  ↓
✅ question_summary（執行）→ ✅ analyze_site（執行）→ ✅ designThinking（執行）
  ↓
✅ GateCheck1 → ✅ shell_prompt → ✅ image_render → ✅ GateCheck2
  ↓
✅ future_scenario → ✅ generate_3D → ✅ deep_evaluation
  ↓
PM 評估（current_round: 0 → 1，未達最大輪次）→ CONTINUE
```

### 第二輪（輪次 1）
```
PM 啟動 → 生成計劃（智能跳過前兩個節點）
  ↓
⏭️ question_summary（跳過，使用第一輪結果）→ ⏭️ analyze_site（跳過）→ ✅ designThinking（執行新設計）
  ↓
✅ GateCheck1 → ✅ shell_prompt → ✅ image_render → ✅ GateCheck2
  ↓
✅ future_scenario → ✅ generate_3D → ✅ deep_evaluation
  ↓
PM 評估（current_round: 1 → 2，未達最大輪次）→ CONTINUE
```

### 第三輪及最終評估
```
PM 啟動 → 生成計劃
  ↓
... （同第二輪流程）...
  ↓
PM 評估（current_round: 2 → 3，達到最大輪次 3）
  ↓
生成最終評估報告 → END
```

## 💡 優勢

1. ✅ **不破壞原有流程**：所有節點保持原有功能
2. ✅ **智能跳過**：第二輪起自動跳過不必要的分析節點，節省時間和成本
3. ✅ **動態控制**：PM 可以根據輪次和狀態智能決定執行策略
4. ✅ **易於擴展**：在 PM 的 `_decide_agents_for_round` 方法中添加決策邏輯
5. ✅ **統一管理**：初始規劃、迭代控制、最終評估都在 PM 節點
6. ✅ **配置靈活**：可透過 `config` 影響 PM 的決策（如 `run_site_analysis`）

## 🚀 未來擴展方向

### 1. ✅ 代理智能跳過（已實現）
節點現在會檢查 PM 計劃並自動跳過：
```python
def run(self, state, config):
    # PM 計劃檢查
    if not should_execute_node(state, "question_summary"):
        return {
            "設計目標x設計需求x方案偏好": state.get("設計目標x設計需求x方案偏好", []),
            "design_summary": state.get("design_summary", "")
        }
    
    # 正常執行邏輯...
```

**擴展建議**：為更多節點添加檢查（目前僅 `question_summary` 和 `analyze_site` 實現）

### 2. 基於評估結果的智能決策
```python
def _decide_agents_for_round(self, current_round, config, state):
    if current_round >= 1:
        # 獲取上一輪評分
        eval_counts = state.get("evaluation_count", [])
        latest_score = list(eval_counts[-1].values())[0] if eval_counts else 0
        
        # 如果分數很高，可以跳過某些生成步驟
        if latest_score > 85:
            return {
                "question_summary": False,
                "analyze_site": False,
                "designThinking": True,
                "future_scenario": False,  # 跳過未來情境
                "generate_3D": False,      # 跳過3D生成
                "deep_evaluation": True
            }
```

### 3. 多模式執行（可透過 Config 配置）
在 `T_config.py` 中添加：
```python
class GraphOverallConfig(BaseModel):
    workflow_mode: Literal["full", "quick", "minimal"] = Field(
        default="full",
        title="工作流模式",
        description="full: 所有節點; quick: 跳過3D; minimal: 僅核心設計流程"
    )
```

PM 根據模式調整：
```python
if config.workflow_mode == "minimal":
    return {
        "question_summary": True,
        "analyze_site": False,
        "designThinking": True,
        "GateCheck1": True,
        "shell_prompt": True,
        "image_render": True,
        "GateCheck2": False,
        "future_scenario": False,
        "generate_3D": False,
        "deep_evaluation": False
    }
```

## 📝 注意事項

1. **初次執行**：PM 會自動進入初始規劃階段，生成第一輪的執行計劃
2. **智能跳過**：第二輪起 `question_summary` 和 `analyze_site` 自動跳過，節省 API 調用成本
3. **迭代控制**：由 `max_evaluation_rounds` 配置控制最大輪次
4. **狀態傳遞**：PM 的決策透過 `state["pm_plan"]` 傳遞給所有節點
5. **節點檢查**：每個節點透過 `should_execute_node()` 檢查是否執行
6. **擴展性**：在 `ProjectManagerTask._decide_agents_for_round()` 中修改決策邏輯

## 🎨 視覺化工作流

你可以使用 LangGraph 的視覺化功能查看新的流程圖：

```python
from IPython.display import Image, display

# 顯示工作流圖
display(Image(graph.get_graph().draw_mermaid_png()))
```

---

## 📈 更新日誌

### v2.0 (2025-10-03)
- ✨ **智能跳過功能**：第二輪起自動跳過不必要的分析節點
- ✨ **動態決策邏輯**：PM 根據輪次智能決定執行哪些代理
- ✨ **節點檢查機制**：透過 `should_execute_node()` 實現真正的跳過
- 🔄 **流程優化**：PM 不再直接連接到 `designThinking`，而是 `question_summary`
- 📊 **成本節省**：第二輪起節省約 20-30% 的 API 調用成本

### v1.0 (2025-10-03)
- 🎯 初始 PM 節點架構
- 📋 基本的流程規劃功能
- 🔄 迭代控制邏輯

---

**當前版本**: 2.0  
**最後更新**: 2025-10-03  
**作者**: MA System LangGraph Team

