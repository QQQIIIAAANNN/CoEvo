"""
PM 節點智能架構使用範例
展示如何使用新的 PM 節點來動態控制工作流執行
"""

from design_T_graph import graph, initial_state
from langchain_core.messages import HumanMessage

# =============================================================================
# 範例 1: 基本執行（3 輪迭代）
# =============================================================================
def example_basic_execution():
    """
    基本執行範例：
    - 第一輪：完整執行所有節點
    - 第二輪：自動跳過 question_summary 和 analyze_site
    - 第三輪：同第二輪
    - 達到最大輪次後生成最終評估
    """
    print("=" * 60)
    print("範例 1: 基本執行（3 輪迭代）")
    print("=" * 60)
    
    # 準備輸入
    user_input_state = initial_state.copy()
    user_input_state["設計目標x設計需求x方案偏好"] = [
        HumanMessage(content="""
        請設計一個位於台北市大安森林公園旁的木構造涼亭。
        設計需求：
        1. 使用數位製造工法
        2. 強調循環經濟和永續性
        3. 外殼需要有創意的曲面設計
        4. 能夠展現參數化設計的美感
        """)
    ]
    
    # 配置
    config = {
        "configurable": {
            "max_evaluation_rounds": 3,  # PM 會控制最多 3 輪
            "run_site_analysis": True,   # 第一輪執行，第二輪起 PM 自動跳過
            "case_scenario_image_count": 3,
            "llm_output_language": "繁體中文"
        }
    }
    
    # 執行
    result = graph.invoke(user_input_state, config=config)
    
    # 查看結果
    print("\n🎯 執行完成！")
    print(f"📊 總輪次: {result.get('current_round', 0)}")
    print(f"📋 PM 計劃: {result.get('pm_plan', {})}")
    print(f"📝 最終評估:\n{result.get('final_evaluation', '未生成')[:200]}...")


# =============================================================================
# 範例 2: 跳過基地分析（透過配置）
# =============================================================================
def example_skip_site_analysis():
    """
    演示如何透過配置跳過基地分析
    PM 會尊重配置，在第一輪就不執行 analyze_site
    """
    print("\n" + "=" * 60)
    print("範例 2: 跳過基地分析")
    print("=" * 60)
    
    user_input_state = initial_state.copy()
    user_input_state["設計目標x設計需求x方案偏好"] = [
        HumanMessage(content="設計一個現代木構造涼亭，強調永續性。")
    ]
    
    config = {
        "configurable": {
            "max_evaluation_rounds": 2,
            "run_site_analysis": False,  # ❌ PM 在所有輪次都不執行基地分析
            "llm_output_language": "繁體中文"
        }
    }
    
    result = graph.invoke(user_input_state, config=config)
    
    print(f"\n✅ PM 第一輪計劃:")
    pm_plan = result.get("pm_plan", {})
    enabled_agents = pm_plan.get("enabled_agents", {})
    for agent, enabled in enabled_agents.items():
        status = "✅" if enabled else "⏭️ (SKIP)"
        print(f"  {status} {agent}")


# =============================================================================
# 範例 3: 查看 PM 的決策過程
# =============================================================================
def example_pm_decision_process():
    """
    演示如何追蹤 PM 的決策過程
    """
    print("\n" + "=" * 60)
    print("範例 3: PM 決策過程追蹤")
    print("=" * 60)
    
    user_input_state = initial_state.copy()
    user_input_state["設計目標x設計需求x方案偏好"] = [
        HumanMessage(content="設計一個木構造涼亭。")
    ]
    
    config = {
        "configurable": {
            "max_evaluation_rounds": 2,
            "llm_output_language": "繁體中文"
        }
    }
    
    # 使用 stream 來追蹤每個節點的執行
    print("\n📊 執行流程追蹤:\n")
    for event in graph.stream(user_input_state, config=config):
        for node_name, node_output in event.items():
            if node_name == "pm":
                pm_plan = node_output.get("pm_plan", {})
                eval_status = node_output.get("evaluation_status", "")
                
                print(f"\n🎯 PM 節點執行:")
                print(f"   狀態: {eval_status}")
                
                if pm_plan:
                    enabled = pm_plan.get("enabled_agents", {})
                    current_round = pm_plan.get("current_round", "?")
                    print(f"   輪次: {current_round}")
                    print(f"   啟用節點數: {sum(enabled.values())}/{len(enabled)}")
            elif node_name in ["question_summary", "analyze_site"]:
                print(f"  ↪️  {node_name} 執行")


# =============================================================================
# 範例 4: 自訂 PM 決策邏輯（需修改 design_T_graph.py）
# =============================================================================
def example_custom_pm_logic():
    """
    展示如何自訂 PM 的決策邏輯
    
    修改 ProjectManagerTask._decide_agents_for_round() 方法：
    
    ```python
    def _decide_agents_for_round(self, current_round, config, state):
        # 第一輪
        if current_round == 0:
            return {
                "question_summary": True,
                "analyze_site": config.run_site_analysis,
                "designThinking": True,
                # ... 其他節點
            }
        
        # 第二輪及以後：根據評估結果決定
        else:
            eval_counts = state.get("evaluation_count", [])
            latest_score = list(eval_counts[-1].values())[0] if eval_counts else 0
            
            # 如果分數很高，跳過某些生成步驟
            if latest_score > 85:
                return {
                    "question_summary": False,
                    "analyze_site": False,
                    "designThinking": True,
                    "future_scenario": False,  # 跳過！
                    "generate_3D": False,      # 跳過！
                    "deep_evaluation": True
                }
            else:
                # 分數不夠高，完整執行（除了前兩個分析節點）
                return {
                    "question_summary": False,
                    "analyze_site": False,
                    "designThinking": True,
                    "future_scenario": True,
                    "generate_3D": True,
                    "deep_evaluation": True
                }
    ```
    """
    print("\n" + "=" * 60)
    print("範例 4: 自訂 PM 決策邏輯")
    print("=" * 60)
    print("\n請查看函數的 docstring 以了解如何修改 PM 的決策邏輯。")
    print("這樣可以根據評估結果動態調整執行策略，進一步優化成本和效率。")


# =============================================================================
# 主程式
# =============================================================================
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         PM 節點智能架構使用範例                              ║
    ║                                                              ║
    ║  展示如何使用 PM 節點動態控制工作流執行                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 執行範例（注意：實際執行需要完整的環境配置和 API 金鑰）
    # example_basic_execution()           # 範例 1
    # example_skip_site_analysis()        # 範例 2
    # example_pm_decision_process()       # 範例 3
    example_custom_pm_logic()             # 範例 4
    
    print("\n" + "=" * 60)
    print("✅ 範例演示完成！")
    print("=" * 60)
    print("""
    📚 更多資訊請參考 PM_NODE_README.md
    
    關鍵優勢：
    ✅ 智能跳過：第二輪起自動跳過不必要的分析，節省成本
    ✅ 動態決策：PM 根據輪次和狀態智能決定執行策略
    ✅ 易於擴展：修改 _decide_agents_for_round() 即可自訂邏輯
    """)









