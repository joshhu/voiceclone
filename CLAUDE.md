# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

VoiceClone 是基於 Alibaba Qwen3-TTS 的本地聲音複製與語音合成系統，提供三種模式：Voice Design（自然語言設計聲音）、Voice Clone（上傳參考音訊複製聲音）、CustomVoice TTS（9 種預設角色語音合成）。支援中/英/日/韓等 10 種語言。

## 開發環境

- Python 3.12+，使用 **uv** 管理依賴（勿使用 pip 或 base 環境）
- GPU: CUDA 自動檢測，已在 RTX 3080 Ti (12GB VRAM) 測試

```bash
# 安裝依賴
uv sync

# Web UI（推薦）
uv run python app.py        # http://localhost:7860

# CLI
uv run python clone.py custom "你好" --speaker Vivian
uv run python clone.py clone "你好" --ref-audio ref_audio/sample.wav --ref-text "參考文字"
```

## 架構

兩個主要檔案，無測試：

- **app.py** — Gradio Web UI，包含模型管理、ASR、三種生成模式、UI 建構
- **clone.py** — CLI 工具，提供 `custom` 和 `clone` 兩個子命令

### GPU 記憶體管理（核心設計）

同時只保留一個模型在 VRAM，透過全域變數追蹤：
- `_current_model` / `_current_model_key`: 當前 TTS 模型
- `_whisper_model`: ASR 模型（與 TTS 互斥使用 VRAM）

`load_model()` 切換模型時會先釋放前一個並呼叫 `torch.cuda.empty_cache()`。`transcribe_audio()` 會先卸載 TTS 模型再載入 Whisper。

### 模型對應

| 模式 | HuggingFace 模型 | 大小 |
|------|------------------|------|
| Voice Design | Qwen3-TTS-1.7B-VoiceDesign | 僅 1.7B |
| Voice Clone | Qwen3-TTS-{0.6B,1.7B}-Base | 兩者皆可 |
| CustomVoice TTS | Qwen3-TTS-{0.6B,1.7B}-CustomVoice | 兩者皆可 |

### 預設角色

Vivian、Serena、Uncle_fu、Dylan、Eric、Ryan、Aiden、Ono_anna、Sohee

### 輸出

所有合成音訊存至 `outputs/`，命名格式 `{prefix}_{timestamp}.wav`（如 `clone_20260224_205848.wav`）。
