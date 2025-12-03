# 🧠 CoEvo：智慧多代理建築設計系統

👉 [📘 系統詳細介紹：CoEvo 技術手冊.pdf](./CoEvo技術手冊.pdf)

👉 🎬 **系統展示影片**  

[![CoEvo 系統展示影片](https://img.youtube.com/vi/-EVTmppnLtw/0.jpg)](https://youtu.be/-EVTmppnLtw)

---

## LangGraph 自動化建築設計流程系統介紹

這是一個基於 LangGraph 開發的智慧多代理 (Multi-Agent) 建築設計系統，模擬一個由 AI 組成的建築設計團隊，自動化地執行從概念發想、資料搜集、方案設計、多方案比較、視覺化到最終評估的完整流程。

---

## 專案總覽

本框架利用大型語言模型 (LLM) 和生成式 AI 技術，將複雜的建築設計任務分解為一系列由專門化 AI 代理 (Agent) 執行的子任務。透過動態的工作流規劃與管理，系統能夠應對多樣的設計需求，並在過程中進行自我評估與修正，最終產出多樣化且高品質的設計成果。

本系統主要有兩個 graph 工作流，分別為：

- `General_Arch_graph`: 一個層級化的建築設計工作流，最通用且適合所有任務。
- `design_T_graph`: 一個順序化的建築設計工作流，已經編譯好固定的任務流程，僅適合測試用途、特定目標的建築設計探索，或作為個人定義特定設計探索目標的參考範本。

---

## 其餘依賴安裝項目介紹

要完整運行本系統功能，除 `.env` 中配置模型 API 外，還需安裝以下本地工具：

### 🔧 ComfyUI 安裝（圖像生成引擎）

請依照以下專案頁面安裝：
- https://github.com/if-ai/ComfyUI-IF_Trellis


### 🗺️ RHINO MCP 叮嚀
rhino前置作業：
- 需要先設定虛擬環境 rhino_mcp 使用python=3.10
- 接者使用"pip install uv" 安裝uv
- 再到資料夾LangGraph\src\mcp\rhino-mcp中使用指令"uv pip install -e ."安裝依賴(本專案專屬，已新增工具)

rhino使用辦法：
1. 確認gemini API Key有放進.env內；
2. 打開rhino_mcp修改第68行的command路徑位置為你的 rhino_mcp 虛擬環境python.exe ；
3. 打開rhino進行以下操作開啟MCP:
- Click on the "Tools" menu
- Select "Python Script" -> "Run.."
- Navigate to and select rhino_mcp_client.py
- Navigate to and select grasshopper_mcp_client (如果有用GH)
位置通常在LangGraph\src\mcp\rhino-mcp\rhino_mcp內。
4. 先確認MCP有沒有安裝成功，可以用claude抓看看，OK了就能正常使用。

其他事項：
目前只支援gemini，如果想換模型要修改rhino_mcp 45-63行的LLM set。
如果有付費可以調快運作速度，將rpm_delay設為0或0.5。

### 🗺️ MCP 相關原始專案（街廓資料與建模工具）

MCP 工具包含多個模組：
- https://github.com/jagan-shanmugam/open-streetmap-mcp  
  > ※ OSM 功能需註冊 [Canvasflare API](https://canvasflare.com)，並將金鑰填入 `.env` 中。
- https://github.com/terryso/mcp-pinterest  
- https://github.com/SerjoschDuering/rhino-mcp  
---

## 🌍 English Version

### LangGraph-Based Automated Architectural Design Framework

This project is a multi-agent architectural design system powered by LangGraph. It simulates a collaborative AI design team to automate the complete workflow — from concept generation, data collection, design iterations, multi-option evaluation, visualization, to final assessment.

---

### Project Overview

The framework decomposes complex architectural design tasks into modular subtasks executed by specialized AI agents. By dynamically orchestrating workflows, the system supports various design objectives while enabling continuous self-assessment and refinement. It generates diverse and high-quality architectural proposals through collaborative generative processes.

There are two primary design graphs:

- `General_Arch_graph`: A hierarchical, generalized workflow for most design scenarios.
- `design_T_graph`: A sequential, pre-compiled workflow for specific-use cases, testing, or as a template for fixed design targets.

---

### Installation of Required Tools

To fully activate the system, in addition to configuring the model API key in `.env`, you need to install the following tools locally:

#### ComfyUI (for image generation)
- https://github.com/if-ai/ComfyUI-IF_Trellis

#### MCP Tools (for street block & massing data)
- https://github.com/jagan-shanmugam/open-streetmap-mcp  
  > Requires a [Canvasflare API](https://canvasflare.com) key in `.env`.
- https://github.com/terryso/mcp-pinterest  
- https://github.com/SerjoschDuering/rhino-mcp  

---
## 🤝 貢獻指南

歡迎提交 Issues 和 Pull Requests！

1. Fork 此專案
2. 創建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交變更 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

✨ 若需技術支援或共建合作，歡迎聯繫或提 Issue 🙌
**⭐ 如果這個專案對您有幫助，請給個 Star！**

> 開發者：NCKU IA Lab QQQIIIAAANNN| 適合：LangGraph 進階用戶
