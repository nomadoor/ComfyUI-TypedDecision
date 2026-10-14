# AGENTS.md

This repository is a ComfyUI custom node pack (V3 schema, `comfy_api.latest`).
Follow the engineering rules in the parent ComfyUI `AGENTS.md` (../../AGENTS.md) where they apply to node code.

## Scope

This pack runs typed-decision models (noul / choice / score with calibrated probabilities) on ComfyUI's text-encoder inference engine: Core's model implementations, loaders and memory management. A new model family is in scope when Core implements its backbone architecture and computes every backbone layer with its own modules. Adding it means a family module next to `imajev.py`: prompt, readout or decision head, calibration, and at most a short loop over Core's blocks when the family needs a different attention mask. Model-specific settings go on the loader under the adapter option; a new family adds no nodes. Beyond Typed Decision, the pack has search nodes built only from typed decisions (Region Search); they call the families' `decide_many` and add no model code. Out of scope: backbones Core does not implement, families that change Core's layer math, and remote APIs.

## Implementation constraints

Every family has `decide_many(requests)`; `decide` is `decide_many` with one request. Rows (requests and their presentation orders) share a forward only when their media layout matches (`imajev.batches`), because Core gives a whole batch the multimodal positions of one row; rows share a forward up to 8,192 padded tokens, and a longer row runs alone. Batched and one-by-one results agree to 1e-6. Each family gives Core a `memory_estimation_function` (activations of the padded batch), and `token_budget` cuts batches to fit next to the fully loaded weights: on d3-flash int8 (12 GB GPU) a 50-row forward with ~6 GB of weights offloaded took 376 s, the same rows in fitting batches about 16 s. A model that cannot fit at all keeps the 8,192-token batches (Clef-flash int8: 16 tiles in 4.6 s batched, 11.9 s one row at a time). Memory other programs hold on the GPU is not counted.

The nodes reproduce the official imajev torch path (`mohit67890/imajev` @ 523436b8, adapter @ c9e5f132): answers and abstentions match on the reference cases, probabilities within 0.02 and unknown probability within 0.04 (the largest gaps come from one borderline abstention case; int8 ConvRot and fp8 backbones reach 0.07 / 0.04 there and stay under 0.01 elsewhere). Keep these when changing code:

- The prompt text and chat boundary are ported verbatim (`imajev.build_prompt`, `render`); one token of drift changes results. Core's Qwen chat template is not used.
- Images are resized in the node to the Qwen processor's final grid (400k px LANCZOS, then smart_resize with min 65,536 px, bicubic + antialias) so Core's own Qwen resize is a no-op. Core's resize differs and shifts tokens and probabilities on small images.
- The decision reads the final-normed hidden state of the last token. `clip_layer` with a negative index is reset by `set_clip_options`, so the node selects `[num_layers]` as a list.
- PEFT LoRA keys map to `text_encoders.transformer.model.*` with `lora_alpha` supplied as `.alpha`; merged LoRA (the default Core path) matches the unmerged official path, also on the fp8 backbone.
- Presentation orders (debias) run as one batch within the token budget, with right padding; Core does not pad batch items itself.

d3 reproduces the vllm-sr d3 runtime (`d3-code-readout-v1`; flash @ f6075bf, mini @ 0e78bbd, nano @ 9d9b937, lite @ 3c49f5c). Core computes in fp32 with bf16 weights, so the reference is the official run in fp32. Against it, answers match on the reference cases (text, JSON, empty state, one and two images) and probabilities stay within 0.012 on bf16 and 0.021 on int8 ConvRot (flash, nano and lite measured; mini against bf16 only, within 0.012). Against the official bf16 run, rounding moves a few image cases by up to 0.061 (nano) and flips one borderline score on nano. Keep these when changing code:

- The prompt text is ported verbatim (`d3.render`): system prompt, state as `describe` renders it (`(empty)` when blank), noul listed false then true, score levels as bare options.
- The full-attention layers are non-causal. `D3Model.hidden` runs Core's blocks with a padding-only mask; Core's own encode is causal and moves probabilities by up to 0.45.
- Images are resized in the node to the d3 processor's grid (smart_resize, max 1,638,400 px, bicubic + antialias); the patches match the official processor to 1e-7.
- debias is the runtime's `permutation_average`: choice options also scored in reverse order, probabilities averaged.
- Videos: frames at the loader's `fps` (default 2 as the runtime; at least 4), picked by index over the whole clip as the runtime does, each resized like its video processor (200,704 px a frame), two frames a group after a `<t seconds>` timestamp (index / fps before averaging, as the processor rounds), each video wrapped in an outer vision_start / vision_end as the runtime's processor leaves it. Token ids and positions match the runtime exactly. Each group's features come from Core's vision encoder and take the positions of a one-frame image; the timestamp text is part of the batch layout. Deliberate deviation: the runtime caps a video at 32 frames, which skips short events between samples; here a long video is thinned only by a lower `fps` the user sets, and the result JSON reports the frames read. All frames go into one request, so the cost grows with the clip: on d3-mini int8 (12 GB, 1 GB used elsewhere) 20 s took 4 s, 70 s 5 min and 180 s 22 min, because the activations of one long request push weights out of VRAM. `frames.py` decodes a file once from its trim window and keeps only the picked frames (exact count by demuxing packets; 8-bit RGB like the runtime's OpenCV), each frame resized as it is kept, so the decode buffer does not grow with the clip (the kept frames take about 0.55 MB each); cropped files and in-memory component videos (VideoFromComponents and other VideoInput types) go through `get_components`. The attention mask is `[B, 1, 1, L]`, never `[L, L]`, and is passed even without padding: without one, a 32k-token video needed 70 GiB in attention. Against the official fp32 run fed the same frames, answers match on 5 video questions; probabilities within 0.005 (flash bf16) and 0.045 (lite, one noul).
- The readout and `decision_config.json` travel inside the converted backbone file (`d3_readout.weight`, metadata `d3_decision_config`, files on `nomadoor/d3`), because each size's head only fits the backbone it was trained with. The loader offers a fixed `d3` adapter instead of a folder. imajev keeps its folder because its backbone is plain Qwen3.5.

Clef reproduces Cloudflare's `joint_schema_model.py` (clef-flash @ 8b2e5fd), one question per record with the field ID `decision`. Against the official fp32 run, answers match on the 16 reference questions (text, JSON, empty state, one and two images) and probabilities stay within 0.003 on bf16 and 0.015 on int8 ConvRot. Keep these when changing code:

- The prompt is tokenized piece by piece exactly as `encode_record` does (`clef.encode`); tokenizing the joined text gives different ids. Choice options are listed sorted by id and mapped back; blank state (and a typed `{}`) renders as an empty string. The state is cut to the release's 16,384-token limit; where the release refuses (question, options and images alone over the limit), the node drops the state and runs.
- The head reads the final-normed hidden states of every token (Core's causal encode, `clip_layer([num_layers])`) and the `lm_head` rows of each option's tokens. `lm_head` stays bf16 in the int8 file: an int8 ConvRot `lm_head` gives a max diff of 0.0151 against official fp32 (0.0147 with bf16) but would need a quantized row gather. The head (118M parameters) keeps the release's torch modules (`MultiheadAttention`, `TransformerDecoderLayer`) for exact parity and runs in fp32 under its own ModelPatcher, fully loaded.
- Images are resized like the release processor (smart_resize, bicubic + antialias) but capped at 1,638,400 px as d3, not the release's 16,777,216 px; larger images are read smaller on purpose.
- The head (`clef_head.*`) and `joint_head_config.json` (metadata `clef_head_config`) travel inside the converted backbone file on `nomadoor/clef`; the loader offers a fixed `clef` adapter.

## Verification

From the ComfyUI root (the pack's `__init__.py` uses relative imports, so pytest must use importlib mode):

```bash
venv/bin/python -m pytest custom_nodes/ComfyUI-TypedDecision/tests -q -p no:cacheprovider --import-mode=importlib
```

## Agent skills

### Language
- Project prose language: Japanese
- Keep AI-only instructions, schemas, identifiers, template headings, tool keywords and canonical terms in English.
  Use Japanese for human-reviewed content: Issues, commit messages, PRs, ADRs.
- `README.md` is English. Translations may be added as `README_ja.md` / `README_zh.md`. Keep READMEs minimal; details live in the owner's blog.

### Issue tracker: GitHub
- GitHub Issues are the source of truth for specifications and tickets; infer the repository from the Git remote and use `gh`.
- Present specifications and ticket sets for approval before creating them. Prefer native issue dependencies; link tickets to their parent spec.
- Close tickets via `Closes #N` in the completing PR; close a parent spec only when the whole spec is done.

### Labels
- Ready for an implementation agent: `ready-for-agent` (apply only after publication approval).

### Domain docs
- Single context: root `CONTEXT.md` and `docs/adr/`, created lazily through domain modeling. Surface ADR conflicts instead of overriding them.
- Use canonical glossary vocabulary in issues, specs, tests and code.

### Workflow boundaries
1. Create a work branch immediately before the first approved repository change (after `GO` if Phase 1 changes nothing).
2. Apply `prior-art` before specifying or implementing new behavior.
3. New features and behavior changes need a spec; clear small fixes, refactors, docs and mechanical config may skip it.
4. Use tickets when work spans sessions or agents or exceeds one context window.
5. Domain-doc approval authorizes only that glossary/ADR edit. Spec and ticket publication each need explicit approval.
6. `GO` authorizes implementation, tests, internal review and safe fixes — not commit.
7. TDD for new behavior and bug fixes (exceptions: docs, comments, mechanical changes, generated files, external config).
8. Review the full uncommitted tree against Standards and Spec. Behavior-changing commits need a separate read-only AI review
   (requirements source, spec/tickets, domain docs, diff, verification). Re-review after fixing P0/P1. P0 and P1 block commit.
9. Commit needs explicit approval of the commit packet. Push and PR creation need separate approval; default to a Draft PR.
10. Mark a Draft PR ready only after CI passes and the user approves. Merge is out of scope.
11. Keep the file count minimal: do not add docs, folders or helper files unless the change requires them.
