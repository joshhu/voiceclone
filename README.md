# VoiceClone - Qwen3-TTS 聲音複製與語音合成系統

基於 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的本地聲音複製與語音合成系統，提供 Gradio 網頁介面和 CLI 命令列工具。支援 CUDA GPU 加速，充分利用 BF16、TF32、Flash Attention 等技術提升推論速度。

## 功能

| 模式 | 說明 | 模型 |
|------|------|------|
| **Voice Design** | 用自然語言描述設計全新聲音 | 1.7B |
| **Voice Clone** | 上傳 5~15 秒參考音訊，複製該聲音 | 0.6B / 1.7B |
| **CustomVoice TTS** | 9 種預設角色 + 情緒/風格控制 | 0.6B / 1.7B |

- 支援 10 種語言：中文、英文、日文、韓文、德文、法文、俄文、葡萄牙文、西班牙文、義大利文
- Voice Clone 上傳音訊後，自動使用 Whisper 辨識參考文字
- 多模型同時快取：VRAM 充足時可同時載入多個模型，無需切換等待

## GPU 加速特性

- **CUDA 自動偵測**：有 GPU 自動啟用 CUDA，無 GPU 自動退回 CPU 模式
- **BF16 推論**：使用 bfloat16 精度，大幅降低 VRAM 用量與加速運算
- **TF32 矩陣加速**：Ampere 架構以上 GPU 自動啟用 TF32
- **cuDNN Benchmark**：自動選擇最快的卷積演算法
- **SDPA / Flash Attention 2**：自動偵測並使用最佳 Attention 實作
- **`torch.inference_mode()`**：推論時關閉梯度追蹤，減少記憶體開銷

## 硬體需求

| 模型 | VRAM 需求 | 建議 GPU |
|------|-----------|----------|
| 0.6B | ~2-4 GB | RTX 3060 (12GB) 以上 |
| 1.7B | ~4-6 GB | RTX 3080 Ti (12GB) 以上 |
| 多模型同時載入 | ~10-12 GB | RTX 4090 (24GB) 以上 |

> 已在 NVIDIA GB10 (128GB VRAM) 上測試通過。

## 快速開始

### 環境建置

需要 [uv](https://docs.astral.sh/uv/) 和 Python 3.12+：

```bash
git clone https://github.com/joshhu/voiceclone.git
cd voiceclone
uv sync
```

首次執行會自動從 Hugging Face 下載模型。

### Web UI（推薦）

```bash
uv run python app.py
```

開啟瀏覽器 http://localhost:7860

### CLI 命令列

```bash
# 預設角色語音合成
uv run python clone.py custom "你好，世界！" --speaker Vivian

# 加上情緒指令
uv run python clone.py custom "你好！" --speaker Serena --instruct "用開心的語氣"

# 聲音複製
uv run python clone.py clone "這是複製的聲音" --ref-audio ref_audio/sample.wav --ref-text "參考音訊文字"

# 使用 0.6B 小模型
uv run python clone.py custom "Hello!" --speaker Ryan --small
```

## 預設角色

| 角色 | 說明 | 語言 |
|------|------|------|
| Vivian | 明亮、略帶鋒利感的年輕女聲 | 中文 |
| Serena | 溫暖、溫柔的年輕女聲 | 中文 |
| Uncle_fu | 低沉醇厚的成熟男聲 | 中文 |
| Dylan | 清亮的年輕男聲 | 北京方言 |
| Eric | 活潑帶沙啞的男聲 | 四川方言 |
| Ryan | 富有節奏感的動感男聲 | 英文 |
| Aiden | 陽光的美式男聲 | 英文 |
| Ono_anna | 俏皮活潑的女聲 | 日文 |
| Sohee | 情感豐富溫暖的女聲 | 韓文 |

## 專案結構

```
voiceclone/
├── app.py           # Gradio 網頁介面（GPU 加速版）
├── clone.py         # CLI 命令列工具
├── outputs/         # 合成音訊輸出目錄
├── ref_audio/       # 參考音訊目錄（聲音複製用）
├── pyproject.toml   # 專案設定與依賴
└── uv.lock          # 鎖定依賴版本
```

## 致謝

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team
- [OpenAI Whisper](https://github.com/openai/whisper) 語音辨識
- [Gradio](https://gradio.app/) 網頁介面

## 授權

本專案基於 [Apache 2.0](LICENSE) 授權。模型授權請參考 [Qwen3-TTS 授權](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)。
