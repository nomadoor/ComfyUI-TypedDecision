import json
import os

import folder_paths
from comfy_api.latest import io

from .imajev import MARKER, ImajevModel, Question, is_imajev

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
    if isinstance(state, list):
        state = {"state": state}
    elif not isinstance(state, dict):
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
        return Question("noul", mode["instructions"].strip(), [(True, None), (False, None)])
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


def check_limits(model, question, n_images):
    if n_images > model.max_images:
        raise ValueError(f"This model accepts at most {model.max_images} images; got {n_images}.")
    if question.kind == "choice" and len(question.answers) > model.max_options:
        raise ValueError(f"This model accepts at most {model.max_options} options; got {len(question.answers)}.")
    if question.kind == "score" and len(question.answers) > model.max_levels:
        raise ValueError(f"This model accepts at most {model.max_levels} levels; got {len(question.answers)}.")


def answer(question, scores, selected, threshold, calibration_version):
    """Typed Decision outputs from probabilities over answers + unknown (last), in the official Jev answer shape."""
    unknown = scores[-1]
    known = scores[:-1]
    total = sum(known)
    known = [p / total for p in known] if total > 0 else [1 / len(known)] * len(known)
    abstained = selected == len(scores) - 1
    common = {"unknown_probability": unknown, "abstained": abstained}
    if calibration_version is not None:
        common["calibration_version"] = calibration_version
    if question.kind == "noul":
        value = scores[0] + 0.5 * unknown
        index = 1 if scores[0] >= scores[1] else 0
        label = "yes" if index else "no"
        result = {"type": "noul", "noul": value, **common}
    else:
        index = max(range(len(known)), key=known.__getitem__)
        confidence = max(0.0, (len(known) * known[index] - 1) / (len(known) - 1)) * (1 - unknown)  # Jev's concentration
        if question.kind == "choice":
            value, label = known[index], question.answers[index][0]
            result = {"type": "choice", "choice": label, "probabilities": {v: p for (v, _), p in zip(question.answers, known)},
                      "confidence": confidence, **common}
        else:
            value, label = sum(i * p for i, p in enumerate(known)), question.answers[index][1]
            result = {"type": "score", "score": value, "legend": {str(v): d for v, d in question.answers},
                      "probabilities": {str(v): p for (v, _), p in zip(question.answers, known)}, "confidence": confidence, **common}
    return value, label, index, value >= threshold and not abstained, abstained, json.dumps(result, ensure_ascii=False)


def decision_folders():
    return sorted({os.path.dirname(f) for f in folder_paths.get_filename_list("typed_decision") if os.path.basename(f) == MARKER})


class LoadTypedDecisionModel(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadTypedDecisionModel",
            display_name="Load Typed Decision Model",
            category="typed decision",
            description="Loads a typed-decision model: an LLM backbone plus the decision adapter trained on it.",
            inputs=[
                io.Combo.Input("backbone", options=folder_paths.get_filename_list("text_encoders"),
                               tooltip="The LLM in models/text_encoders, e.g. qwen3.5_4b_bf16.safetensors for imajev-4b."),
                io.Combo.Input("adapter", options=decision_folders(),
                               tooltip="The model folder in models/typed_decision, e.g. imajev-4b (the Hugging Face repository as downloaded)."),
            ],
            outputs=[TypedDecisionModel.Output(display_name="typed_decision_model")],
        )

    @classmethod
    def execute(cls, backbone, adapter):
        folder = os.path.dirname(folder_paths.get_full_path_or_raise("typed_decision", os.path.join(adapter, MARKER)))
        if not is_imajev(folder):
            raise ValueError(f"{adapter}: unsupported typed decision model format (expected adapter_config.json, adapter_model.safetensors and decision_readout.*).")
        return io.NodeOutput(ImajevModel(folder_paths.get_full_path_or_raise("text_encoders", backbone), folder))


class TypedDecision(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TypedDecision",
            display_name="Typed Decision",
            category="typed decision",
            description="Answers one typed question about the images and state with calibrated probabilities.",
            inputs=[
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option("noul", [
                        io.String.Input("instructions", multiline=True, default="",
                                        placeholder="instructions: a claim to judge true or false, or a yes/no question.\ne.g. The image matches the prompt."),
                    ]),
                    io.DynamicCombo.Option("choice", [
                        io.String.Input("instructions", multiline=True, default="",
                                        placeholder="instructions: the question to answer.\ne.g. Which part of the prompt is missing from the image?"),
                        io.String.Input("criteria", multiline=True, default="",
                                        placeholder="criteria: the options, one per line. Add ': description' to explain one.\ne.g.\nthe red car\nthe beach\nthe sunset\nnothing is missing"),
                    ]),
                    io.DynamicCombo.Option("score", [
                        io.String.Input("instructions", multiline=True, default="",
                                        placeholder="instructions: the question to rate.\ne.g. How closely does the image follow the prompt?"),
                        io.String.Input("criteria", multiline=True, default="",
                                        placeholder="criteria: the levels, one per line, lowest first.\ne.g.\nnot at all\npartly\nmostly\nfully"),
                    ]),
                ], tooltip="noul: true or false. choice: pick one option. score: rate on ordered levels. Write in English."),
                TypedDecisionModel.Input("model"),
                io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("image"), prefix="image_", min=0),
                    tooltip="Images looked at together, in slot order. imajev: first = reference, second = the one to judge."),
                io.String.Input("state", multiline=True, default="",
                                placeholder="state: the facts to judge against (optional). Plain text or JSON.\ne.g. Prompt: a red sports car parked on a beach at sunset.",
                                tooltip="With JSON, instructions can name its fields in `backticks`, e.g. `listing.color`."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=10.0, step=0.01, tooltip="pass is true when value >= threshold and the model did not abstain."),
                io.Boolean.Input("debias", default=False, tooltip="Average over up to 4 option orders to cancel position bias. Slower, and changes little in practice."),
                io.Boolean.Input("calibration", default=True, tooltip="Apply the model's temperature calibration. Changes probabilities, never the answer."),
            ],
            outputs=[
                io.Float.Output(display_name="value", tooltip="noul: probability the claim is true (0-1). choice: probability of the chosen option (0-1). "
                                "score: expected level, 0 = first line (not a probability)."),
                io.String.Output(display_name="label", tooltip="The answer: yes / no, the chosen option, or the most likely level."),
                io.Int.Output(display_name="index", tooltip="noul: yes = 1, no = 0. choice / score: line number of the answer, from 0."),
                io.Boolean.Output(display_name="pass", tooltip="True when value >= threshold and the model did not abstain. Use it to branch."),
                io.Boolean.Output(display_name="abstained", tooltip="True when the images and state are not enough to answer, e.g. to send the case to a person."),
                io.String.Output(display_name="result", tooltip="Full answer as JSON, in the official Jev shape: every probability, unknown probability, confidence."),
            ],
        )

    @classmethod
    def execute(cls, mode, model, state, threshold, debias, calibration, images=None):
        slots = sorted((images or {}).items(), key=lambda item: int(item[0].rsplit("_", 1)[1]))
        frames = [batch[i] for _, batch in slots if batch is not None for i in range(batch.shape[0])]
        question = question_from_mode(mode)
        check_limits(model, question, len(frames))
        scores, selected, version = model.decide(frames, parse_state(state), question, debias, calibration)
        return io.NodeOutput(*answer(question, scores, selected, threshold, version))
