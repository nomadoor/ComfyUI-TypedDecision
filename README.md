# ComfyUI-TypedDecision

English · [日本語](README_ja.md) · [中文](README_zh.md)

Custom nodes that let a multimodal LLM make a decision instead of writing text (Jev-style typed decisions). Ask a typed question about images, videos and text, and get the answer with calibrated probabilities.

- **noul**: true or false
- **choice**: one of your options
- **score**: a position on your ordered levels

Inference runs on ComfyUI's own LLM engine (model implementations, loaders, memory management), so it works almost natively. No extra dependencies.

## Supported models

| Model | Released | Base | int8 size | Images | Video |
| --- | --- | --- | --- | --- | --- |
| [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) | 2026-09-24 | Qwen3.5-4B + LoRA | 5.76 GB + 490 MB | up to 2 | × |
| [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) | 2026-09-30 | Qwen3.5-9B | 12.2 GB | any number | ○ |
| [d3-flash](https://huggingface.co/vllm-sr/d3-flash) | 2026-10-10 | Qwen3.5-9B | 9.88 GB | any number | ○ |
| [d3-mini](https://huggingface.co/vllm-sr/d3-mini) | 2026-10-10 | Qwen3.5-4B | 5.52 GB | any number | ○ |
| [d3-nano](https://huggingface.co/vllm-sr/d3-nano) | 2026-10-10 | Qwen3.5-2B | 3.06 GB | any number | ○ |
| [d3-lite](https://huggingface.co/vllm-sr/d3-lite) | 2026-10-10 | Qwen3.5-0.8B | 1.21 GB | any number | ○ |

More on the blog: [日本語](https://comfyui.nomadoor.net/ja/notes/typed-decision/) · [English](https://comfyui.nomadoor.net/en/notes/typed-decision/) · [中文](https://comfyui.nomadoor.net/zh/notes/typed-decision/)

## Installation

Search for **ComfyUI-TypedDecision** in ComfyUI Manager and install it.

## Models

### imajev

Uses Qwen3.5-4B and the imajev-4b folder.

| | int8 ConvRot | bf16 |
| --- | --- | --- |
| Qwen3.5-4B | [qwen3.5_4b_int8_convrot](https://huggingface.co/nomadoor/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_int8_convrot.safetensors) (5.76 GB) | [qwen3.5_4b_bf16](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_bf16.safetensors) (9.32 GB) |

Put the files below from [mohit67890/imajev-4b](https://huggingface.co/mohit67890/imajev-4b/tree/main) (490 MB) in an `imajev-4b` folder. Downloading the whole repository also works.

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

One file holds the whole model, decision head included. Put the ones you use in `text_encoders`.

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

## Nodes

### Load Typed Decision Model

Pick the model as `backbone`, then the matching `adapter`.

| Model | `backbone` | `adapter` |
| --- | --- | --- |
| imajev | Qwen3.5 | `imajev-4b` |
| d3 | a d3 file | `d3` |
| Clef-flash | a Clef-flash file | `clef` |

The `adapter` shows that model's settings.

- `debias` (imajev, d3): judge with the options in other orders and average. Slower.
  - imajev: every mode, up to 4 orders
  - d3: choice only, also in reverse order
- `fps` (d3, Clef-flash): video frames read per second (default 2)
- `calibration` (imajev): apply temperature calibration

### Typed Decision

Judges one question. Write questions and options in English.

| Input | Meaning |
| --- | --- |
| `mode` | `noul` / `choice` / `score` |
| `instructions` | The question |
| `criteria` | choice options or score levels, one per line |
| `criteria_true` / `criteria_false` | When noul answers yes / no (optional) |
| `images` | Images to judge. With 2 on imajev, the first is the reference and the second is judged |
| `videos` | Videos to judge (d3, Clef-flash) |
| `state` | Facts to judge against, as text or JSON (optional) |
| `threshold` | Threshold for `pass` |

| Output | Meaning |
| --- | --- |
| `value` | noul: probability of yes<br>choice: probability of the chosen option<br>score: expected level (first line = 0) |
| `label` | The answer (not affected by `threshold`) |
| `index` | noul: yes = 1<br>choice, score: line number of the answer, from 0 |
| `pass` | true when `value` >= `threshold` and the model did not abstain |
| `abstained` | true when the images and state are not enough to answer (imajev only) |
| `result` | Full answer as JSON |

#### Videos

- The whole video is read at the loader's `fps` and all frames are judged at once. A minute at 2 fps is 120 frames.
- d3 slows down sharply on long videos (d3-mini int8, 12 GB: 20 s took 4 s, 70 s took 5 min). Lower `fps` for long videos.
- Clef-flash shares one pixel budget across a video, so longer videos are read smaller.
- `result` lists the frames read and their spacing.

### Typed Decision Moment Search

Finds the intervals of a video where a condition holds (e.g. "A dog is visible."). Meant for conditions one frame shows; motion such as direction cannot be judged.

| Input | Meaning |
| --- | --- |
| `instructions`, `criteria_true`, `criteria_false`, `state` | As noul in Typed Decision |
| `images` | Reference images (optional), to find the person or thing in them |
| `video` | The video to search |
| `window` | Frames judged together at first (default 8 = 4 seconds) |
| `threshold` | Probability for a frame to pass |

| Output | Meaning |
| --- | --- |
| `intervals` | JSON of the intervals: `start`, `end` (seconds), `start_frame`, `end_frame` (source frames, end exclusive), `score` (highest probability in the interval) |
| `videos` | The video trimmed to each interval. When nothing is found, the video nodes after it are skipped |

#### How it works

Reads the video at 2 fps and judges in three steps.

1. Judge `window` frames at a time and skip blocks that fail
2. Judge each frame of the remaining blocks
3. Check the source frames between passing and failing frames to set the edges

#### Notes

- With reference images, ask about the video. "The video shows the same person as the image." works; "The person in the image is visible." is true of the reference itself, so every frame passes.
- A larger `window` is faster but misses short events more easily (at 16, d3-mini and d3-flash can drop a half-second event). imajev takes up to 2, or 1 with a reference image.
- Intervals follow the model's answers. Something half out of frame may already count as gone, so write the question for what you want to count.
- Events shorter than half a second can fall between the 2 fps frames.

## Sample workflows

- [noul](example_workflows/typed_decision_noul.json): are many people in the image?
- [choice](example_workflows/typed_decision_choice.json): what is the main subject?
- [score](example_workflows/typed_decision_score.json): how usable is the photo for training?
- [two images](example_workflows/typed_decision_noul_two_images.json): do two images show the same person?
- [dataset sort](example_workflows/typed_decision_noul_dataset_sort.json): save photos with and without people under different names
- [resolution from prompt](example_workflows/typed_decision_choice_resolution_from_prompt_krea2.json): pick the aspect ratio from the prompt, then generate with Krea 2
- [Moment Search](example_workflows/typed_decision_moment_search_d3_mini.json): find where the person in the reference image appears with d3-mini, join and save the intervals

## Credits

- [imajev](https://github.com/mohit67890/imajev) and [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) by mohit67890: model, prompt format and scoring
- [d3 (Decision 3.0)](https://huggingface.co/vllm-sr/d3-flash) by the vLLM Semantic Router team: models, prompt format and readout
- [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) by Cloudflare: model, prompt format and joint schema head
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team, repackaged for ComfyUI by [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen3.5)
