import json
import os

import folder_paths
from comfy_api.latest import io

from . import clef, d3, imajev
from .imajev import Question

MAX_STATE_BYTES = 131072

TypedDecisionModel = io.Custom("TYPED_DECISION_MODEL")


def parse_state(text):
    text = text.strip()
    if not text:
        return {}
    try:
        state = json.loads(text)
    except json.JSONDecodeError:
        state = text
    if not isinstance(state, (dict, list)):
        state = text
    try:
        size = len(json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    except ValueError:
        raise ValueError("state: NaN and Infinity are not valid JSON.")
    if size > MAX_STATE_BYTES:
        raise ValueError(f"state exceeds {MAX_STATE_BYTES // 1024} KB.")
    return state


def parse_options(text):
    options = []
    for line in text.splitlines():
        label, _, description = line.strip().partition(": ")
        if label:
            options.append((label.strip(), description.strip() or None))
    return options


def parse_levels(text):
    return list(enumerate(line.strip() for line in text.splitlines() if line.strip()))


def question_from_mode(mode):
    kind = mode["mode"]
    for field in ("instructions", "criteria"):
        if field in mode and not mode[field].strip():
            raise ValueError(f"{kind}: {field} is empty. The grey text in the box is only an example; type your own.")
    if kind == "noul":
        return Question("noul", mode["instructions"].strip(),
                        [(True, mode.get("criteria_true", "").strip() or None), (False, mode.get("criteria_false", "").strip() or None)])
    if kind == "choice":
        question = Question("choice", mode["instructions"].strip(), parse_options(mode["criteria"]))
        if len(question.answers) < 2:
            raise ValueError(f"choice needs at least 2 criteria, one per line. Got {len(question.answers)}: {mode['criteria'].strip()!r}")
        if len({label for label, _ in question.answers}) != len(question.answers):
            raise ValueError("choice criteria must be unique.")
        return question
    question = Question("score", mode["instructions"].strip(), parse_levels(mode["criteria"]))
    if len(question.answers) < 2:
        raise ValueError(f"score needs at least 2 criteria (levels), one per line, lowest first. Got {len(question.answers)}.")
    return question


def check_limits(model, question, n_images, n_videos):
    if n_images > model.max_images:
        raise ValueError(f"This model accepts at most {model.max_images} images; got {n_images}.")
    if n_videos > model.max_videos:
        raise ValueError(f"This model does not accept videos; got {n_videos}.")
    if question.kind == "choice" and len(question.answers) > model.max_options:
        raise ValueError(f"This model accepts at most {model.max_options} options; got {len(question.answers)}.")
    if question.kind == "score" and len(question.answers) > model.max_levels:
        raise ValueError(f"This model accepts at most {model.max_levels} levels; got {len(question.answers)}.")


def answer(model, question, decision, threshold, notes=None):
    """Typed Decision outputs from model.decide, in the model family's official answer shape, plus `notes` in the result.

    Models without unknown report unknown as None and never abstain."""
    scores, unknown, abstained, calibration_version = decision
    total = sum(scores)
    known = [p / total for p in scores] if total > 0 else [1 / len(scores)] * len(scores)
    common = {} if unknown is None else {"unknown_probability": unknown, "abstained": abstained}
    if calibration_version is not None:
        common["calibration_version"] = calibration_version
    if question.kind == "noul":
        value = scores[0] + 0.5 * (unknown or 0.0)
        index = 1 if scores[0] >= scores[1] else 0
        label = "yes" if index else "no"
        result = {"type": "noul", "noul": value, **common}
    else:
        index = max(range(len(known)), key=known.__getitem__)
        confidence = model.confidence(known, unknown)
        if question.kind == "choice":
            value, label = known[index], question.answers[index][0]
            result = {"type": "choice", "choice": label, "probabilities": {v: p for (v, _), p in zip(question.answers, known)},
                      "confidence": confidence, **common}
        else:
            value, label = sum(i * p for i, p in enumerate(known)), question.answers[index][1]
            result = {"type": "score", "score": value, "legend": {str(v): d for v, d in question.answers},
                      "probabilities": {str(v): p for (v, _), p in zip(question.answers, known)}, "confidence": confidence, **common}
    return value, label, index, value >= threshold and not abstained, abstained, json.dumps({**result, **(notes or {})}, ensure_ascii=False)


def decision_folders():
    return sorted({os.path.dirname(f) for f in folder_paths.get_filename_list("typed_decision") if os.path.basename(f) == imajev.MARKER} - {"d3", "clef"})


def adapter_options():
    """imajev folders in models/typed_decision, then d3 and clef, whose heads are inside their backbone files."""
    fps = lambda: io.Float.Input("fps", default=2.0, min=0.01, max=30.0, step=0.1, tooltip="Video frames read per second, over the whole video. "
                                 "d3 and Clef-flash read videos at 2. All frames go into one request, so long videos are slow; lower it to 1 or 0.5 to "
                                 "make them lighter. The result JSON tells how many frames were read.")
    d3_settings = [io.Boolean.Input("debias", default=False, tooltip="Also score choice options in reverse order and average. Slower."), fps()]
    return ([io.DynamicCombo.Option(folder, imajev_settings()) for folder in decision_folders()]
            + [io.DynamicCombo.Option("d3", d3_settings), io.DynamicCombo.Option("clef", [fps()])])


def imajev_settings():
    return [
        io.Boolean.Input("debias", default=False, tooltip="Average over up to 4 option orders to cancel position bias. Slower, and changes little in practice."),
        io.Boolean.Input("calibration", default=True, tooltip="Apply the model's temperature calibration. Changes probabilities, never the answer."),
    ]


def noul_inputs():
    return [
        io.String.Input("instructions", multiline=True, default="",
                        placeholder="instructions: a claim to judge true or false, or a yes/no question.\ne.g. Many people are in the image."),
        io.String.Input("criteria_true", multiline=True, default="", optional=True,
                        placeholder="criteria_true (optional): when the answer is yes.\ne.g. five or more people are visible"),
        io.String.Input("criteria_false", multiline=True, default="", optional=True,
                        placeholder="criteria_false (optional): when the answer is no.\ne.g. four or fewer people are visible"),
    ]


def mode_input():
    return io.DynamicCombo.Input("mode", options=[
        io.DynamicCombo.Option("noul", noul_inputs()),
        io.DynamicCombo.Option("choice", [
            io.String.Input("instructions", multiline=True, default="",
                            placeholder="instructions: the question to answer.\ne.g. What is the main subject of the image?"),
            io.String.Input("criteria", multiline=True, default="",
                            placeholder="criteria: the options, one per line. Add ': description' to explain one.\ne.g.\nperson\nanimal\nvehicle\nfood\nlandscape: scenery with no clear main subject"),
        ]),
        io.DynamicCombo.Option("score", [
            io.String.Input("instructions", multiline=True, default="",
                            placeholder="instructions: the question to rate.\ne.g. How usable is this photo for a training dataset?"),
            io.String.Input("criteria", multiline=True, default="",
                            placeholder="criteria: the levels, one per line, lowest first.\ne.g.\nunusable: blurry, dark or badly cropped\npoor\nacceptable\ngood\nexcellent: sharp, well lit, subject fills the frame"),
        ]),
    ], tooltip="noul: true or false. choice: pick one option. score: rate on ordered levels. Write in English.")


def images_input(tooltip="Images looked at together, in slot order. imajev: first = reference, second = the one to judge."):
    return io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(input=io.Image.Input("image"), prefix="image_", min=0),
                             tooltip=tooltip)


def in_slots(items):
    """The connected values of an Autogrow input, in slot order."""
    return [value for _, value in sorted((items or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[1])) if value is not None]


def state_input():
    return io.String.Input("state", multiline=True, default="",
                           placeholder="state: the facts to judge against (optional). Plain text or JSON.\ne.g. full body shot of a knight standing in a castle hall",
                           tooltip="With JSON, instructions can name its fields in `backticks`, e.g. `listing.color`.")


class LoadTypedDecisionModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadTypedDecisionModel",
            display_name="Load Typed Decision Model",
            category="typed decision",
            description="Loads a typed-decision model: an LLM backbone plus the decision head trained with it (imajev: a folder; d3 and clef: inside the backbone file).",
            inputs=[
                io.Combo.Input("backbone", options=folder_paths.get_filename_list("text_encoders"),
                               tooltip="The LLM in models/text_encoders: Qwen3.5 for imajev-4b, a d3_*.safetensors file from nomadoor/d3 for d3, a clef_*.safetensors file from nomadoor/clef for clef."),
                io.DynamicCombo.Input("adapter", options=adapter_options(),
                                      tooltip="imajev: the model folder in models/typed_decision, e.g. imajev-4b (the Hugging Face repository as downloaded). "
                                              "d3 / clef: pick d3 or clef; the backbone file carries the rest."),
            ],
            outputs=[TypedDecisionModel.Output(display_name="typed_decision_model")],
        )

    @classmethod
    def execute(cls, backbone, adapter):
        name = adapter["adapter"]
        backbone = folder_paths.get_full_path_or_raise("text_encoders", backbone)
        if name == "d3":
            return io.NodeOutput(d3.D3Model(backbone, adapter["debias"], adapter["fps"]))
        if name == "clef":
            return io.NodeOutput(clef.ClefModel(backbone, adapter["fps"]))
        folder = os.path.dirname(folder_paths.get_full_path_or_raise("typed_decision", os.path.join(name, imajev.MARKER)))
        if not imajev.is_imajev(folder):
            raise ValueError(f"{name}: unsupported typed decision model format (expected adapter_config.json, adapter_model.safetensors and decision_readout.*).")
        return io.NodeOutput(imajev.ImajevModel(backbone, folder, adapter["debias"], adapter["calibration"]))


class TypedDecision(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TypedDecision",
            display_name="Typed Decision",
            category="typed decision",
            description="Answers one typed question about the images and state with calibrated probabilities.",
            inputs=[
                mode_input(),
                TypedDecisionModel.Input("model"),
                images_input(),
                io.Autogrow.Input("videos", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Video.Input("video"), prefix="video_", min=0),
                    tooltip="Videos looked at together with the images, in slot order (d3, Clef-flash). Read at the loader's fps over the whole video."),
                state_input(),
                io.Float.Input("threshold", default=0.5, min=0.0, max=10.0, step=0.01, tooltip="pass is true when value >= threshold and the model did not abstain."),
            ],
            outputs=[
                io.Float.Output(display_name="value", tooltip="noul: probability the claim is true (0-1). choice: probability of the chosen option (0-1). "
                                "score: expected level, 0 = first line (not a probability)."),
                io.String.Output(display_name="label", tooltip="The answer: yes / no, the chosen option, or the most likely level. Not affected by threshold."),
                io.Int.Output(display_name="index", tooltip="noul: yes = 1, no = 0. choice / score: line number of the answer, from 0."),
                io.Boolean.Output(display_name="pass", tooltip="True when value >= threshold and the model did not abstain. Use it to branch."),
                io.Boolean.Output(display_name="abstained", tooltip="True when the images and state are not enough to answer, e.g. to send the case to a person. Always false for models without unknown (d3)."),
                io.String.Output(display_name="result", tooltip="Full answer as JSON, in the model's official answer shape: every probability and confidence (imajev also unknown probability)."),
            ],
        )

    @classmethod
    def execute(cls, mode, model, state, threshold, images=None, videos=None):
        frames = [batch[i] for batch in in_slots(images) for i in range(batch.shape[0])]
        clips = in_slots(videos)
        question = question_from_mode(mode)
        check_limits(model, question, len(frames), len(clips))
        notes = None
        if clips:
            plans = [model.video_plan(video) for video in clips]
            notes = {"videos": [{"frames": len(indices), "interval": (indices[-1] - indices[0]) / fps / max(1, len(indices) - 1)} for indices, fps in plans]}
        return io.NodeOutput(*answer(model, question, model.decide(frames, clips, parse_state(state), question), threshold, notes))
