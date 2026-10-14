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
- `fps` (d3): video frames read per second, over the whole video (default 2, as d3 reads videos).
- `calibration` (imajev): apply the model's temperature calibration

**Typed Decision**: answers one question. Write questions and options in English.

- `mode`: `noul`, `choice` or `score`
- `instructions` / `criteria`: the question, and the options or levels, one per line. noul takes optional `criteria_true` / `criteria_false` for when to answer yes or no.
- `images`: imajev takes up to 2 (with two, the first is the reference and the second is the one to judge); d3 and Clef-flash take any number.
- `videos` (d3): any number of videos, read at the loader's `fps` over the whole video. All the frames go into one request: a minute of video at 2 fps is 120 frames, and long videos get slow quickly, so lower `fps` for them. Only the frames read are kept in memory. The `result` JSON lists how many frames were read and how far apart they are.
- `state`: optional facts to judge against, as text or JSON

| Output | Meaning |
| --- | --- |
| `value` | noul: probability of true<br>choice: probability of the chosen option<br>score: expected level (0 = first line) |
| `label` | The answer. Not affected by `threshold`. |
| `index` | noul: yes = 1<br>choice, score: line number of the answer, from 0 |
| `pass` | `value` >= `threshold` and the model did not abstain |
| `abstained` | The images and state were not enough to answer (imajev; always false for d3 and Clef-flash) |
| `result` | Full answer as JSON |

**Typed Decision Region Search**: finds where in an image a decision holds. It is a search over decisions, not an object detector. It judges the whole image; when the tile's `value` >= the threshold (abstaining counts as no), it splits the tile into four and judges each, down to `max_depth`.

- `mode`, `state`: as in Typed Decision. Each tile is one decision on one image. With choice, `value` is the probability of the most likely option.
- `threshold_top` / `threshold_bottom`: the threshold for the whole image and at `max_depth`, linear in between. Keep the top lenient so small hits are not cut off early. With score, the value is the expected level, so set both on that scale.
- `context_margin`: each tile is judged on a crop widened by this fraction of the tile size on every side.
- A batch is searched image by image. The tiles of one level are judged together.
- Small hits such as a mosaic or a broken hand need d3-flash or d3-mini; d3-lite is too weak for them.

| Output | Meaning |
| --- | --- |
| `mask` | Positive tiles at `max_depth` (or too small to split), filled. A tile that was split counts only through its four tiles. |
| `bboxes` | Per image, one box around each group of touching positive tiles; `score` is the highest `value` in the group |
| `heatmap` | The `value` of the deepest tile judged at each pixel (white = 1; score: the last level) |

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
