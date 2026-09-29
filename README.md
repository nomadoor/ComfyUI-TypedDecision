# ComfyUI-TypedDecision

Typed decisions in ComfyUI: one question about images and text, answered with calibrated probabilities (noul / choice / score). Supports typed-decision models that run on ComfyUI's own text-encoder engine; currently [imajev](https://github.com/mohit67890/imajev).

## Models

| Folder | File |
| --- | --- |
| `models/text_encoders` | `qwen3.5_4b_bf16.safetensors` from [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5), or its int8 / fp8 version from [nomadoor/Qwen3.5](https://huggingface.co/nomadoor/Qwen3.5) |
| `models/typed_decision/imajev-4b` | The [mohit67890/imajev-4b](https://huggingface.co/mohit67890/imajev-4b) repository as downloaded |

## Nodes

**Load Typed Decision Model** — pick Qwen3.5 as `backbone` and imajev-4b as `adapter`.

**Typed Decision** — answers one question. Write questions and options in English.

- `mode`: `noul` (true / false), `choice` (one of the options), `score` (ordered levels)
- `instructions` / `criteria`: the question, and the options or levels, one per line
- `images`: 0-2 images. With two, the first is the reference and the second is the one to judge.
- `state`: optional facts to judge against, as text or JSON
- `debias`: average over several option orders (slower)

| Output | Meaning |
| --- | --- |
| `value` | noul: probability of true / choice: probability of the chosen option / score: expected level (0 = first line) |
| `label` | The answer |
| `index` | noul: yes = 1 / choice, score: line number of the answer, from 0 |
| `pass` | `value` >= `threshold` and the model did not abstain |
| `abstained` | The images and state were not enough to answer |
| `result` | Full answer as JSON, in the official API shape |
