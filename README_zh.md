# ComfyUI-TypedDecision

[English](README.md) · [日本語](README_ja.md) · 中文

一组自定义节点：让多模态 LLM 不生成文本，只做判定（Jev 风格的 typed decision）。针对图像、视频和文本提出一个类型明确的问题，即可得到答案及经过校准的概率。

- **noul**：是 / 否
- **choice**：从给定选项中选出一项
- **score**：在有序等级中给出位置

推理直接复用 ComfyUI 自带的 LLM 推理引擎（模型实现、加载器、显存管理），运行方式与原生节点几乎无异，也无需安装额外依赖。

## 支持的模型

| 模型 | 发布日期 | 基础模型 | int8 大小 | 图像 | 视频 |
| --- | --- | --- | --- | --- | --- |
| [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) | 2026-09-24 | Qwen3.5-4B + LoRA | 5.76 GB + 490 MB | 最多 2 张 | × |
| [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 2026-09-30 | Qwen3.5-9B | 12.2 GB | 不限 | ○ |
| [d3-flash](https://huggingface.co/vllm-sr/d3-flash) | 2026-10-10 | Qwen3.5-9B | 9.88 GB | 不限 | ○ |
| [d3-mini](https://huggingface.co/vllm-sr/d3-mini) | 2026-10-10 | Qwen3.5-4B | 5.52 GB | 不限 | ○ |
| [d3-nano](https://huggingface.co/vllm-sr/d3-nano) | 2026-10-10 | Qwen3.5-2B | 3.06 GB | 不限 | ○ |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) | 2026-10-10 | Qwen3.5-0.8B | 1.21 GB | 不限 | ○ |

详细介绍见博客：[日本語](https://comfyui.nomadoor.net/ja/notes/typed-decision/) · [English](https://comfyui.nomadoor.net/en/notes/typed-decision/) · [中文](https://comfyui.nomadoor.net/zh/notes/typed-decision/)

## 安装

在 ComfyUI Manager 中搜索 **ComfyUI-TypedDecision** 并安装。

## 模型

### imajev

需要 Qwen3.5-4B 和 imajev-4b 文件夹。

| | int8 ConvRot | bf16 |
| --- | --- | --- |
| Qwen3.5-4B | [qwen3.5_4b_int8_convrot](https://huggingface.co/nomadoor/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_int8_convrot.safetensors) (5.76 GB) | [qwen3.5_4b_bf16](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_bf16.safetensors) (9.32 GB) |

从 [mohit67890/imajev-4b](https://huggingface.co/mohit67890/imajev-4b/tree/main) 下载以下文件（490 MB），放入 `imajev-4b` 文件夹。直接下载整个仓库也可以。

```text
📂ComfyUI/
└── 📂models/
    ├── 📂text_encoders/
    │   └── qwen3.5_4b_int8_convrot.safetensors
    └── 📂typed_decision/
        └── 📂imajev-4b/
            ├── adapter_config.json
            ├── adapter_model.safetensors
            ├── calibration-rot4-modality.json
            ├── decision_readout.json
            └── decision_readout.safetensors
```

### d3 / Clef-flash

每个文件都包含完整模型（含判定头），将要使用的文件放入 `text_encoders` 即可。

| | int8 ConvRot | bf16 |
| --- | --- | --- |
| d3-flash | [d3_flash_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_flash_int8_convrot.safetensors) (9.88 GB) | [d3_flash_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_flash_bf16.safetensors) (16.8 GB) |
| d3-mini | [d3_mini_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_mini_int8_convrot.safetensors) (5.52 GB) | [d3_mini_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_mini_bf16.safetensors) (9.08 GB) |
| d3-nano | [d3_nano_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_nano_int8_convrot.safetensors) (3.06 GB) | [d3_nano_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_nano_bf16.safetensors) (4.43 GB) |
| d3-lite | [d3_lite_int8_convrot](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_lite_int8_convrot.safetensors) (1.21 GB) | [d3_lite_bf16](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_lite_bf16.safetensors) (1.71 GB) |
| Clef-flash | [clef_flash_int8_convrot](https://huggingface.co/nomadoor/clef/blob/main/text_encoders/clef_flash_int8_convrot.safetensors) (12.2 GB) | [clef_flash_bf16](https://huggingface.co/nomadoor/clef/blob/main/text_encoders/clef_flash_bf16.safetensors) (19.1 GB) |

```text
📂ComfyUI/
└── 📂models/
    └── 📂text_encoders/
        ├── d3_mini_int8_convrot.safetensors
        └── clef_flash_int8_convrot.safetensors
```

## 节点

### Load Typed Decision Model

在 `backbone` 中选择模型，再在 `adapter` 中选择与之对应的一项。

| 模型 | `backbone` | `adapter` |
| --- | --- | --- |
| imajev | Qwen3.5 | `imajev-4b` |
| d3 | d3 的文件 | `d3` |
| Clef-flash | Clef-flash 的文件 | `clef` |

选择 `adapter` 后，会显示该模型专用的设置。

- `debias`（imajev、d3）：以不同的选项顺序多次判定并取平均，速度会变慢。
  - imajev：适用于所有 mode，最多 4 种顺序
  - d3：仅适用于 choice，额外按倒序判定一次
- `fps`（d3、Clef-flash）：每秒读取的视频帧数（默认为 2）
- `calibration`（imajev）：应用温度校准

### Typed Decision

对一个问题进行判定。问题和选项请用英文书写。

| 输入 | 内容 |
| --- | --- |
| `mode` | `noul` / `choice` / `score` |
| `instructions` | 问题 |
| `criteria` | choice 的选项、score 的等级（每行一个） |
| `criteria_true` / `criteria_false` | noul 中判定为是 / 否的条件（可选） |
| `images` | 要判定的图像。imajev 输入 2 张时，第 1 张为参考图，第 2 张为判定对象 |
| `videos` | 要判定的视频（d3、Clef-flash） |
| `state` | 作为判定前提的事实，可用文本或 JSON（可选） |
| `threshold` | `pass` 的阈值 |

| 输出 | 内容 |
| --- | --- |
| `value` | noul：是的概率<br>choice：所选选项的概率<br>score：等级的期望值（第 1 行 = 0） |
| `label` | 答案（不受 `threshold` 影响） |
| `index` | noul：是 = 1<br>choice、score：答案的行号（从 0 开始） |
| `pass` | `value` >= `threshold` 且未弃权时为 true |
| `abstained` | 仅凭图像和 state 无法作答时为 true（仅限 imajev） |
| `result` | 完整答案（JSON） |

#### 关于视频

- 按加载器的 `fps` 读取整段视频，所有帧在一次判定中处理。例如 1 分钟的视频按 2 fps 读取，共 120 帧。
- d3 处理长视频时速度会急剧下降（d3-mini int8、12 GB 显存下，20 秒的视频耗时 4 秒，70 秒的视频耗时 5 分钟）。处理长视频时请调低 `fps`。
- Clef-flash 的像素上限由同一视频的所有帧分摊，因此视频越长，每帧读取的尺寸越小。
- 实际读取的帧数与帧间隔会写入 `result`。

### Typed Decision Moment Search

在视频中查找满足条件的时间区间（例：“A dog is visible.”）。适用于看单帧即可判断的条件，无法判断运动方向之类的动作。

| 输入 | 内容 |
| --- | --- |
| `instructions`、`criteria_true`、`criteria_false`、`state` | 与 Typed Decision 的 noul 相同 |
| `images` | 参考图像（可选），用于在视频中寻找图中的人或物 |
| `video` | 要搜索的视频 |
| `window` | 第一轮合并判定的帧数（默认 8，即 4 秒） |
| `threshold` | 帧判定为通过所需的概率 |

| 输出 | 内容 |
| --- | --- |
| `intervals` | 区间的 JSON：`start`、`end`（秒）、`start_frame`、`end_frame`（原视频帧号，不含 end）、`score`（区间内最高概率） |
| `videos` | 按区间裁剪出的视频。未找到任何区间时，后续的视频节点不会执行 |

#### 工作原理

以 2 fps 读取视频，分三步判定：

1. 每 `window` 帧合并判定一次，未通过的块整体跳过
2. 对通过的块逐帧判定
3. 在通过帧与未通过帧之间，逐个检查原视频的帧，确定区间边界

#### 注意

- 使用参考图像时，问题应针对视频本身。“The video shows the same person as the image.” 是合适的写法；而 “The person in the image is visible.” 对参考图像本身也成立，会导致所有帧都通过。
- `window` 越大速度越快，但也越容易漏掉短暂的事件（设为 16 时，d3-mini 和 d3-flash 可能漏掉持续 0.5 秒的事件）。imajev 最大为 2，使用参考图像时为 1。
- 区间完全取决于模型的判定。物体有一半移出画面时，可能已被判为“不在”，请根据想要计入的情形来写问题。
- 持续不到 0.5 秒的事件可能恰好落在两个采样帧之间而被漏掉。

## 示例工作流

- [noul](example_workflows/typed_decision_noul.json)：图像中是否有很多人
- [choice](example_workflows/typed_decision_choice.json)：画面的主体是什么
- [score](example_workflows/typed_decision_score.json)：照片作为训练数据的可用程度如何
- [两张图像](example_workflows/typed_decision_noul_two_images.json)：两张图像中是否为同一个人
- [数据集分拣](example_workflows/typed_decision_noul_dataset_sort.json)：按照片中是否有人，以不同的文件名保存
- [根据提示词选分辨率](example_workflows/typed_decision_choice_resolution_from_prompt_krea2.json)：根据提示词选择宽高比，再用 Krea 2 生成
- [Moment Search](example_workflows/typed_decision_moment_search_d3_mini.json)：用 d3-mini 找出参考图像中的人物出现的区间，拼接后保存

## 致谢

- [imajev](https://github.com/mohit67890/imajev) 和 [imajev-4b](https://huggingface.co/mohit67890/imajev-4b)（mohit67890）：模型、提示词格式、评分
- [d3 (Decision 3.0)](https://huggingface.co/vllm-sr/d3-flash)（vLLM Semantic Router 团队）：模型、提示词格式、readout
- [Clef-flash](https://huggingface.co/Cloudflare/clef-flash)（Cloudflare）：模型、提示词格式、joint schema head
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B)（Qwen 团队），ComfyUI 版由 [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen3.5) 重新打包
