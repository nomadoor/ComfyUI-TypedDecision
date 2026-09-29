# AGENTS.md

This repository is a ComfyUI custom node pack (V3 schema, `comfy_api.latest`).
Follow the engineering rules in the parent ComfyUI `AGENTS.md` (../../AGENTS.md) where they apply to node code.

## Scope

This pack runs typed-decision models (noul / choice / score with calibrated probabilities) that ComfyUI's text-encoder inference engine can run: Core's model implementations, loaders and memory management. A new model family is in scope when Core implements its backbone architecture; adding it means a family module next to `imajev.py` (prompt, readout, calibration), not new nodes or inputs. Out of scope: backbones Core does not implement, models that need their own inference code, and remote APIs.

## Implementation constraints

The nodes reproduce the official imajev torch path (`mohit67890/imajev` @ 523436b8, adapter @ c9e5f132): answers and abstentions match on the reference cases, probabilities within 0.02 and unknown probability within 0.04 (the largest gaps come from one borderline abstention case; int8 ConvRot and fp8 backbones reach 0.07 / 0.04 there and stay under 0.01 elsewhere). Keep these when changing code:

- The prompt text and chat boundary are ported verbatim (`imajev.build_prompt`, `render`); one token of drift changes results. Core's Qwen chat template is not used.
- Images are resized in the node to the Qwen processor's final grid (400k px LANCZOS, then smart_resize with min 65,536 px, bicubic + antialias) so Core's own Qwen resize is a no-op. Core's resize differs and shifts tokens and probabilities on small images.
- The decision reads the final-normed hidden state of the last token. `clip_layer` with a negative index is reset by `set_clip_options`, so the node selects `[num_layers]` as a list.
- PEFT LoRA keys map to `text_encoders.transformer.model.*` with `lora_alpha` supplied as `.alpha`; merged LoRA (the default Core path) matches the unmerged official path, also on the fp8 backbone.
- Presentation orders (debias) run as one batch with right padding; Core does not pad batch items itself.

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
