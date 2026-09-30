# ComfyUI-TypedDecision

Custom nodes for Jev-style typed decisions: a multimodal LLM does not write text, it only makes a decision. You ask one typed question about images and text, and get the answer with calibrated probabilities:

- **noul**: true or false
- **choice**: one of your options
- **score**: a position on your ordered levels

Several local alternatives to Jev exist. ComfyUI already has its own multimodal LLM inference engine, so this pack reuses it and starts with [imajev](https://github.com/mohit67890/imajev).

More details on the blog: [日本語](https://comfyui.nomadoor.net/ja/notes/typed-decision/) · [English](https://comfyui.nomadoor.net/en/notes/typed-decision/) · [中文](https://comfyui.nomadoor.net/zh/notes/typed-decision/)

## Installation

Search for **ComfyUI-TypedDecision** in ComfyUI Manager and install it.

## Models

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

## Nodes

**Load Typed Decision Model**: pick Qwen3.5 as `backbone` and imajev-4b as `adapter`.

**Typed Decision**: answers one question. Write questions and options in English.

- `mode`: `noul`, `choice` or `score`
- `instructions` / `criteria`: the question, and the options or levels, one per line. noul takes optional `criteria_true` / `criteria_false` for when to answer yes or no.
- `images`: up to 2 images. With two, the first is the reference and the second is the one to judge.
- `state`: optional facts to judge against, as text or JSON
- `debias`: average over several option orders (slower)

| Output | Meaning |
| --- | --- |
| `value` | noul: probability of true<br>choice: probability of the chosen option<br>score: expected level (0 = first line) |
| `label` | The answer. Not affected by `threshold`. |
| `index` | noul: yes = 1<br>choice, score: line number of the answer, from 0 |
| `pass` | `value` >= `threshold` and the model did not abstain |
| `abstained` | The images and state were not enough to answer |
| `result` | Full answer as JSON |

## Sample workflows

- [noul](example_workflows/typed_decision_noul.json): are many people in the image?
- [choice](example_workflows/typed_decision_choice.json): what is the main subject?
- [score](example_workflows/typed_decision_score.json): how usable is the photo for training?
- [two images](example_workflows/typed_decision_noul_two_images.json): do two images show the same person?
- [dataset sort](example_workflows/typed_decision_noul_dataset_sort.json): save photos with and without people under different names
- [resolution from prompt](example_workflows/typed_decision_choice_resolution_from_prompt_krea2.json): pick the aspect ratio from the prompt, then generate with Krea 2

## Credits

- [imajev](https://github.com/mohit67890/imajev) and [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) by mohit67890: the model, prompt format and scoring this pack reproduces
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team, repackaged for ComfyUI by [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen3.5)
