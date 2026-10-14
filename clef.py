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

from .imajev import fit_grid, image_tokens

HEAD = "clef_head."  # head weights stored in the converted backbone file next to the Core keys
CONFIG = "clef_head_config"  # safetensors metadata: the release's joint_head_config.json
QUESTION_ID = "decision"  # the node asks one question; the official API names each one
SYSTEM_PROMPT = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options."
NOUL_TEXTS = {True: "The proposition is true or the answer is yes.", False: "The proposition is false or the answer is no."}
TYPES = {"noul": 0, "choice": 1, "score": 2}
MAX_LENGTH = 16384  # the release's systemone limit; the state is cut to fit
MAX_PIXELS = 1_638_400  # as d3; the release allows 16,777,216 px (about 16k tokens an image), so larger images differ from it


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def options(question):
    """(answer index, option id, description) in the order the release lists them: choice options sorted by id."""
    if question.kind == "noul":
        return [(i, "true" if v else "false", d or NOUL_TEXTS[v]) for i, (v, d) in enumerate(question.answers)]
    if question.kind == "choice":
        return sorted(((i, v, d) for i, (v, d) in enumerate(question.answers)), key=lambda o: o[1])
    return [(i, str(v), d) for i, (v, d) in enumerate(question.answers)]


def encode(tokenizer, state, question, images):
    """-> (Core tokens, (question span, option spans), answer index of each option, token ids of each option).

    The release tokenizes the prompt piece by piece; spans count each image as its expanded tokens."""
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
    if images:
        start, end = tokenizer.convert_tokens_to_ids(["<|vision_start|>", "<|vision_end|>"])
        for image in images:
            prefix += [start, {"type": "image", "data": image, "original_type": "image"}, end]
        prefix += ids("\n")
    suffix = ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    expanded = len(prefix) + sum(image_tokens(image) - 1 for image in images)
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
    max_options = math.inf
    max_levels = math.inf

    def __init__(self, backbone_path):
        weights, config = read_head(backbone_path)
        head = JointSchemaHead(**config)
        head.load_state_dict(weights)
        self.clip = comfy.sd.load_clip(ckpt_paths=[backbone_path])
        if not isinstance(self.clip.cond_stage_model, comfy.text_encoders.qwen35.Qwen35TEModel):
            raise ValueError("The Clef backbone did not load as Qwen3.5.")
        self.name = self.clip.cond_stage_model.clip_name
        self.encoder = getattr(self.clip.cond_stage_model, self.name)
        self.clip.clip_layer([self.encoder.num_layers])  # the layer after the last block: the final-normed hidden state
        self.pad = self.encoder.special_tokens["pad"]
        self.tokenizer = getattr(self.clip.tokenizer, self.name).tokenizer
        self.head = comfy.model_patcher.ModelPatcher(head.float(), load_device=self.clip.patcher.load_device,
                                                     offload_device=self.clip.patcher.offload_device)

    @staticmethod
    def confidence(known, unknown):
        """The probability of the chosen answer, as the release reports it."""
        return max(known)

    def decide(self, images, state, question):
        """-> (probabilities over answers, unknown probability (None), abstained (False), calibration version (None))."""
        images = [prepare_image(image) for image in images]
        tokens, (question_span, option_spans), order, option_ids = encode(self.tokenizer, state, question, images)
        if self.pad in tokens:  # Core masks everything after the pad token
            raise ValueError("The text must not contain <|endoftext|>.")
        cond = self.clip.encode_from_tokens({self.name: [[(t, 1.0) for t in tokens]]})
        hidden = cond.reshape(-1, cond.shape[-1])
        lm_head = self.encoder.transformer.model.lm_head.weight
        lexical = torch.stack([lm_head[ids].float().mean(dim=0) for ids in option_ids])
        comfy.model_management.load_models_gpu([self.head], force_full_load=True)
        device = self.head.load_device
        logits = self.head.model(hidden.to(device), question_span, option_spans, question.kind, lexical.to(device))
        return restore(order, torch.softmax(logits, dim=-1).tolist()), None, False, None
