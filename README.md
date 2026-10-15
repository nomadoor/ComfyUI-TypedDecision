# ComfyUI-TypedDecision

Custom nodes for Jev-style typed decisions: a multimodal LLM does not write text, it only makes a decision. You ask one typed question about images and text, and get the answer with calibrated probabilities:

- **noul**: true or false
- **choice**: one of your options
- **score**: a position on your ordered levels

Several local alternatives to Jev exist. ComfyUI already has its own multimodal LLM inference engine, so this pack reuses it. Supported models: [imajev](https://github.com/mohit67890/imajev), [d3 (Decision 3.0)](https://huggingface.co/vllm-sr/d3-flash) and [Clef-flash](https://huggingface.co/Cloudflare/clef-flash).

More details on the blog: [日本語](https://comfyui.nomadoor.net/ja/notes/typed-decision/) · [English](https://comfyui.nomadoor.net/en/notes/typed-decision/) · [中文](https://comfyui.nomadoor.net/zh/notes/typed-decision/)

## Installation

Search for **ComfyUI-TypedDecision** in ComfyUI Manager and install it.

## Models

### imajev

- text_encoders
  - [qwen3.5_4b_int8_convrot.safetensors](https://huggingface.co/nomadoor/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_int8_convrot.safetensors) (5.76 GB)
  - or the original [qwen3.5_4b_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_4b_bf16.safetensors) (9.32 GB)
- typed_decision
  - [mohit67890/imajev-4b](https://huggingface.co/mohit67890/imajev-4b/tree/main) (490 MB for the files below)
    - Put these files in an `imajev-4b` folder. Downloading the whole repository also works.

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

### d3

d3 is a full fine-tune, so it does not use plain Qwen3.5. Each file below is the whole model for one size, decision head included; download one and put it in `text_encoders`.

- [d3_flash_int8_convrot.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_flash_int8_convrot.safetensors) (9.88 GB) or [d3_flash_bf16.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_flash_bf16.safetensors) (16.8 GB)
- [d3_mini_int8_convrot.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_mini_int8_convrot.safetensors) (5.52 GB) or [d3_mini_bf16.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_mini_bf16.safetensors) (9.08 GB)
- [d3_nano_int8_convrot.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_nano_int8_convrot.safetensors) (3.06 GB) or [d3_nano_bf16.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_nano_bf16.safetensors) (4.43 GB)
- [d3_lite_int8_convrot.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_lite_int8_convrot.safetensors) (1.21 GB) or [d3_lite_bf16.safetensors](https://huggingface.co/nomadoor/d3/blob/main/text_encoders/d3_lite_bf16.safetensors) (1.71 GB)

### Clef-flash

Clef-flash is a full fine-tune of Qwen3.5-9B with a small joint schema head. Each file below is the whole model, head included; download one and put it in `text_encoders`.

- [clef_flash_int8_convrot.safetensors](https://huggingface.co/nomadoor/clef/blob/main/text_encoders/clef_flash_int8_convrot.safetensors) (12.2 GB) or [clef_flash_bf16.safetensors](https://huggingface.co/nomadoor/clef/blob/main/text_encoders/clef_flash_bf16.safetensors) (19.1 GB)

## Nodes

**Load Typed Decision Model**: for imajev, pick Qwen3.5 as `backbone` and imajev-4b as `adapter`. For d3 and Clef-flash, pick the model file as `backbone` and `d3` or `clef` as `adapter`. The model's own settings appear under `adapter`:

- `debias`: average over several option orders (slower). d3 also scores choice options in reverse order.
- `fps` (d3, Clef-flash): video frames read per second, over the whole video (default 2, as both read videos).
- `calibration` (imajev): apply the model's temperature calibration

**Typed Decision**: answers one question. Write questions and options in English.

- `mode`: `noul`, `choice` or `score`
- `instructions` / `criteria`: the question, and the options or levels, one per line. noul takes optional `criteria_true` / `criteria_false` for when to answer yes or no.
- `images`: imajev takes up to 2 (with two, the first is the reference and the second is the one to judge); d3 and Clef-flash take any number.
- `videos` (d3, Clef-flash): any number of videos, read at the loader's `fps` over the whole video. All the frames go into one request: a minute of video at 2 fps is 120 frames, and long videos get slow quickly on d3, so lower `fps` for them. Clef-flash shares one pixel budget among a video's frames, as its release does, so a long video is read at a smaller size instead. Only the frames read are kept in memory. The `result` JSON lists how many frames were read and how far apart they are.
- `state`: optional facts to judge against, as text or JSON

| Output | Meaning |
| --- | --- |
| `value` | noul: probability of true<br>choice: probability of the chosen option<br>score: expected level (0 = first line) |
| `label` | The answer. Not affected by `threshold`. |
| `index` | noul: yes = 1<br>choice, score: line number of the answer, from 0 |
| `pass` | `value` >= `threshold` and the model did not abstain |
| `abstained` | The images and state were not enough to answer (imajev; always false for d3 and Clef-flash) |
| `result` | Full answer as JSON |

**Typed Decision Moment Search**: finds when in a video something one frame can show holds, such as "A dog is visible." It reads the whole video at 2 fps (the rate d3 is trained on) and judges blocks of `window` frames; a block passes once one of its frames shows it, so a block that fails is skipped whole. Blocks pass at half the `threshold`, since a model can score a block with one matching frame below it. It then judges each frame of the blocks that pass, and the source frames between each passing frame and the failing one next to it, so the edges fall on source frames. A gap of one 2 fps frame splits an interval.

- `instructions`, `criteria_true`, `criteria_false`, `state`: as noul in Typed Decision.
- `images` (optional): reference images every judgment sees before the video, to search for the person or thing in them. Ask about the video, such as "The video shows the same person as the image."; "The person in the image is visible." is true of the reference itself, so every frame passes. imajev takes 2 images in all, so with one reference `window` is 1. The references are read within the frames' pixel budget (about 200,000 px), as every judgment carries them.
- `window`: frames judged together at first (default 8, 4 seconds). A larger window skips more at once, but a block with one matching frame scores lower: at 16, d3-mini and d3-flash can drop a half-second event with its block. imajev takes at most 2.
- The intervals follow the model's answer on single frames: an object half out of the frame may already count as gone. Write the question for what you want to count.
- Motion, such as which way something moves, is not searched for: a window passes on part of it, and how much depends on the video, so its edges cannot be found.
- Something shown for less than half a second can fall between the 2 fps frames and be missed.
- d3 and Clef-flash see each frame twice, as a group of its own: d3 barely sees the second frame of a pair. Frames are read at d3's video frame size (about 200,000 px a frame).
- d3-flash does not fit next to other programs on a 12 GB GPU, and each forward then streams its weights: a 30 s video took about 3 minutes, against 43 s on d3-mini and 15 s on d3-lite.

| Output | Meaning |
| --- | --- |
| `intervals` | JSON list of `start`, `end` (seconds), `start_frame`, `end_frame` (source frames, end exclusive) and `score` (highest probability of its 2 fps frames) |
| `videos` | The video trimmed to each interval. When nothing is found, the video nodes after it are skipped. |

## Sample workflows

- [noul](example_workflows/typed_decision_noul.json): are many people in the image?
- [choice](example_workflows/typed_decision_choice.json): what is the main subject?
- [score](example_workflows/typed_decision_score.json): how usable is the photo for training?
- [two images](example_workflows/typed_decision_noul_two_images.json): do two images show the same person?
- [dataset sort](example_workflows/typed_decision_noul_dataset_sort.json): save photos with and without people under different names
- [resolution from prompt](example_workflows/typed_decision_choice_resolution_from_prompt_krea2.json): pick the aspect ratio from the prompt, then generate with Krea 2

## Credits

- [imajev](https://github.com/mohit67890/imajev) and [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) by mohit67890: the model, prompt format and scoring this pack reproduces
- [d3 (Decision 3.0)](https://huggingface.co/vllm-sr/d3-flash) by the vLLM Semantic Router team: the d3 models, prompt format and readout this pack reproduces
- [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) by Cloudflare: the model, prompt format and joint schema head this pack reproduces
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team, repackaged for ComfyUI by [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen3.5)
