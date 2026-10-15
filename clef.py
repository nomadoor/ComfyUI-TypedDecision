"""Clef: Qwen3.5 fine-tune + joint schema head, reproducing Cloudflare's joint_schema_model.py (clef-flash)."""
import json
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.text_encoders.qwen35

from .d3 import VIDEO_FPS, hidden, read_video, video_plan
from .imajev import batches, fit_grid, image_tokens, media_layout, memory_estimation, token_budget

HEAD = "clef_head."  # head weights stored in the converted backbone file next to the Core keys
CONFIG = "clef_head_config"  # safetensors metadata: the release's joint_head_config.json
QUESTION_ID = "decision"  # the node asks one question; the official API names each one
SYSTEM_PROMPT = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."
NOUL_TEXTS = {True: "The proposition is true or the answer is yes.", False: "The proposition is false or the answer is no."}
TYPES = {"noul": 0, "choice": 1, "score": 2}
MAX_LENGTH = 16384  # the release's systemone limit; the state is cut to fit
MAX_PIXELS = 1_638_400  # as d3; the release allows 16,777,216 px (about 16k tokens an image), so larger images differ from it
VIDEO_MAX_PIXELS = 25_165_824  # over all frames of a video, as the release's video processor
VIDEO_MIN_PIXELS = 4_096


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def options(question):
    """(answer index, option id, description) in the order the release lists them: choice options sorted by id."""
    if question.kind == "noul":
        return [(i, "true" if v else "false", d or NOUL_TEXTS[v]) for i, (v, d) in enumerate(question.answers)]
    if question.kind == "choice":
        return sorted(((i, v, d) for i, (v, d) in enumerate(question.answers)), key=lambda o: o[1])
    return [(i, str(v), d) for i, (v, d) in enumerate(question.answers)]


def encode(tokenizer, state, question, images, videos=()):
    """-> (Core tokens, (question span, option spans), answer index of each option, token ids of each option). `videos` holds
    (frames [T, 3, H, W], group timestamps) of each video.

    The release tokenizes the prompt piece by piece; spans count each image and video group as its expanded tokens. Its
    processor replaces a video's placeholder with a timestamp and a vision block for each group of two frames."""
    ids = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    schema = ids("\n\nSCHEMA FIELDS:\n") + ids(f"\nFIELD 1\nID: {QUESTION_ID}\nTYPE: {question.kind}\nINSTRUCTION: ")
    start = len(schema)
    schema += ids(render(question.text))
    question_span = (start, len(schema))
    schema += ids("\nALLOWED OPTIONS:\n")
    option_spans, order = [], []
    for number, (answer, option_id, description) in enumerate(options(question), 1):
        schema += ids(f"OPTION {number}: ")
        start = len(schema)
        schema += ids(render({"option_id": option_id} if description is None else {"option_id": option_id, "description": description}))
        option_spans.append((start, len(schema)))
        order.append(answer)
        schema += ids("\n")
    schema += ids("END FIELD\n")
    prefix = ids(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    start, end = tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|vision_end|>"])
    for image in images:
        prefix += [start, {"type": "image", "data": image, "original_type": "image"}, end]
    for frames, times in videos:
        for g, t in enumerate(times):
            prefix += ids(f"<{t:.1f} seconds>") + [start, {"type": "video", "frames": frames, "group": g, "tokens": frames.shape[2] * frames.shape[3] // 1024}, end]
    if images or videos:
        prefix += ids("\n")
    suffix = ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    expanded = len(prefix) + sum(image_tokens(image) - 1 for image in images) + sum(t["tokens"] - 1 for t in prefix if isinstance(t, dict) and t["type"] == "video")
    fixed = expanded + len(schema) + len(suffix)
    state_ids = ids(render(state if state != {} else ""))[:max(0, MAX_LENGTH - fixed)]  # the release refuses when fixed > limit
    shift = expanded + len(state_ids)
    move = lambda span: (span[0] + shift, span[1] + shift)
    return (prefix + state_ids + schema + suffix, (move(question_span), [move(s) for s in option_spans]), order,
            [schema[slice(*span)] for span in option_spans])


def restore(order, probabilities):
    """Probabilities in presented option order -> question answer order."""
    scores = [0.0] * len(order)
    for answer, p in zip(order, probabilities):
        scores[answer] = p
    return scores


def video_size(width, height, n_frames):
    """(width, height) of each frame of an n_frames video after the release's video processor: a pixel budget over the
    whole video (two frames a temporal patch), on a 32-pixel grid."""
    if height < 32 or width < 32:
        raise ValueError(f"Video frames must be at least 32 pixels on each side; got {width} x {height}.")
    h, w = round(height / 32) * 32, round(width / 32) * 32
    t = math.ceil(n_frames / 2) * 2
    if t * h * w > VIDEO_MAX_PIXELS:
        beta = math.sqrt(n_frames * height * width / VIDEO_MAX_PIXELS)
        h, w = max(32, math.floor(height / beta / 32) * 32), max(32, math.floor(width / beta / 32) * 32)
    elif t * h * w < VIDEO_MIN_PIXELS:
        beta = math.sqrt(VIDEO_MIN_PIXELS / (n_frames * height * width))
        h, w = math.ceil(height * beta / 32) * 32, math.ceil(width * beta / 32) * 32
    return w, h


def read_head(path):
    """-> (head state dict, head config) from a converted Clef backbone file."""
    with safe_open(path, "pt") as f:
        metadata = f.metadata() or {}
        keys = [k for k in f.keys() if k.startswith(HEAD)]
        if CONFIG not in metadata or not keys:
            raise ValueError(f"{os.path.basename(path)} is not a Clef backbone. Pick a clef_*.safetensors file from huggingface.co/nomadoor/clef.")
        return {k[len(HEAD):]: f.get_tensor(k) for k in keys}, json.loads(metadata[CONFIG])


def prepare_image(image):
    """[H, W, C] float -> [1, H', W', 3] at the Clef processor's grid, capped at MAX_PIXELS, so Core's own resize is a no-op."""
    u8 = (image[..., :3].float().cpu() * 255).round().clamp(0, 255).to(torch.uint8)
    return fit_grid(u8, MAX_PIXELS)


class EvidenceRoutingLayer(torch.nn.Module):
    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        # Identity slots keep the release's Sequential indices (dropout there) for the state dict.
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(torch.nn.Linear(width, feedforward), torch.nn.GELU(), torch.nn.Identity(),
                                               torch.nn.Linear(feedforward, width))

    def forward(self, queries, memory):
        memory = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), memory, memory, need_weights=False)[0]
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    """The release's joint schema head for one record with one question."""

    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        linear = lambda: torch.nn.Linear(hidden_size, width, bias=False)
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection, self.question_projection, self.option_question_projection = linear(), linear(), linear()
        self.global_projection, self.option_context_projection, self.option_lexical_projection = linear(), linear(), linear()
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList([EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)])
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList([torch.nn.TransformerDecoderLayer(width, heads, feedforward, dropout=0.0, activation="gelu",
                                                                            batch_first=True, norm_first=True) for _ in range(layers)])
        self.field_norm = torch.nn.LayerNorm(width)
        self.option_norm = torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(torch.nn.Linear(width * 4, width), torch.nn.GELU(), torch.nn.Identity(), torch.nn.Linear(width, 1))
        self.prior_logit_scale = torch.nn.Parameter(torch.empty(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.empty(()))
        self.residual_gate = torch.nn.Parameter(torch.empty(()))

    def forward(self, hidden, question_span, option_spans, kind, lexical):
        """hidden [L, hidden]; lexical [options, hidden] (mean output-embedding rows of each option) -> option logits."""
        hidden = self.hidden_norm(hidden)
        memory = self.memory_projection(hidden).unsqueeze(0)
        last = hidden[-1]
        question = hidden[slice(*question_span)].mean(dim=0)
        contexts = torch.stack([hidden[slice(*span)].mean(dim=0) for span in option_spans])
        options = (self.option_context_projection(contexts) + self.option_lexical_projection(lexical)
                   + self.option_question_projection(question).unsqueeze(0)).unsqueeze(0)
        for layer in self.evidence_layers:
            options = layer(options, memory)
        options = options[0]
        field = self.question_projection(question)
        weights = torch.softmax(options @ field / math.sqrt(options.shape[-1]), dim=0)
        field = (field + self.option_summary_norm((weights.unsqueeze(-1) * options).sum(dim=0)) + self.global_projection(last)
                 + self.type_embedding.weight[TYPES[kind]]).view(1, 1, -1)
        for layer in self.layers:
            field = layer(field, memory)
        field = self.field_norm(field[0, 0])
        anchor = F.normalize(question + last, dim=-1)
        prior = self.prior_logit_scale.clamp(max=math.log(100.0)).exp() * (F.normalize(lexical, dim=-1) @ anchor)
        options = self.option_norm(options)
        field = field.unsqueeze(0).expand_as(options)
        residual = self.residual_scorer(torch.cat([field, options, field * options, (field - options).abs()], dim=-1)).squeeze(-1)
        joint = self.joint_logit_scale.clamp(max=math.log(100.0)).exp() * F.cosine_similarity(field, options, dim=-1) + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


class ClefModel:
    max_images = math.inf
    max_videos = math.inf
    max_options = math.inf
    max_levels = math.inf

    def __init__(self, backbone_path, fps=VIDEO_FPS):
        self.fps = fps
        weights, config = read_head(backbone_path)
        head = JointSchemaHead(**config)
        head.load_state_dict(weights)
        self.clip = comfy.sd.load_clip(ckpt_paths=[backbone_path])
        if not isinstance(self.clip.cond_stage_model, comfy.text_encoders.qwen35.Qwen35TEModel):
            raise ValueError("The Clef backbone did not load as Qwen3.5.")
        self.name = self.clip.cond_stage_model.clip_name
        self.encoder = getattr(self.clip.cond_stage_model, self.name)
        self.clip.cond_stage_model.memory_estimation_function = memory_estimation(self.encoder.transformer.model.config.hidden_size)
        self.pad = self.encoder.special_tokens["pad"]
        self.tokenizer = getattr(self.clip.tokenizer, self.name).tokenizer
        self.head = comfy.model_patcher.ModelPatcher(head.float(), load_device=self.clip.patcher.load_device,
                                                     offload_device=self.clip.patcher.offload_device)

    @staticmethod
    def confidence(known, unknown):
        """The probability of the chosen answer, as the release reports it."""
        return max(known)

    def video_plan(self, video):
        """(source frame indices, fps) this model reads from a VIDEO."""
        return video_plan(video, self.fps)

    def prepare_video(self, video):
        return read_video(video, self.fps, video_size)

    def decide(self, images, videos, state, question):
        """-> (probabilities over answers, unknown probability (None), abstained (False), calibration version (None))."""
        return self.decide_many([(images, videos, state, question)])[0]

    def decide_many(self, requests):
        """decide for each (images, videos, state, question). Requests of the same media layout share backbone forwards,
        which run Core's blocks causally as Core's encode does; Core's encode takes no videos."""
        rows = []  # (images, videos, question, tokens, spans, order, option ids, length)
        for images, videos, state, question in requests:
            images, videos = [prepare_image(image) for image in images], [self.prepare_video(video) for video in videos]
            tokens, spans, order, option_ids = encode(self.tokenizer, state, question, images, videos)
            if self.pad in tokens:  # Core masks everything after the pad token
                raise ValueError("The text must not contain <|endoftext|>.")
            rows.append((images, videos, question, tokens, spans, order, option_ids,
                         sum(1 if not isinstance(t, dict) else t["tokens"] if t["type"] == "video" else image_tokens(t["data"]) for t in tokens)))
        # Positions follow the media and the timestamp text between video groups, so both are part of the layout.
        layouts = [(media_layout(row[0]), tuple((tuple(v.shape), tuple(f"{t:.1f}" for t in times)) for v, times in row[1])) for row in rows]
        states = [None] * len(rows)
        budget = token_budget(self.clip.patcher, self.encoder.transformer.model.config.hidden_size)
        for batch in batches(layouts, [row[-1] for row in rows], budget):
            longest = max(len(rows[i][3]) for i in batch)
            cond = hidden(self.clip, self.name, [rows[i][3] + [self.pad] * (longest - len(rows[i][3])) for i in batch], causal=True)
            for k, i in enumerate(batch):  # a copy of the record's own tokens, so the padded batch output is freed
                states[i] = cond[k, :rows[i][-1]].to(self.clip.patcher.offload_device, copy=True)
        # The head runs after every backbone forward: both rarely fit in VRAM together, so it loads once.
        lm_head = self.encoder.transformer.model.lm_head.weight
        comfy.model_management.load_models_gpu([self.head], force_full_load=True)
        device = self.head.load_device
        out = []
        for (_, _, question, _, (question_span, option_spans), order, option_ids, _), h in zip(rows, states):
            lexical = torch.stack([lm_head[ids].float().mean(dim=0) for ids in option_ids])
            logits = self.head.model(h.to(device), question_span, option_spans, question.kind, lexical.to(device))
            out.append((restore(order, torch.softmax(logits, dim=-1).tolist()), None, False, None))
        return out
