"""Region Search: a quadtree of typed decisions over one image.

A tile is judged together with the crops above it, each with a red box around the next: a tile alone loses the context
that makes a hand part of a person."""
from typing import NamedTuple

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import io

from .nodes import TypedDecisionModel, answer, check_limits, noul_inputs, parse_state, question_from_mode, state_input

ZOOM = "Each image is a close-up of the red box in the image before it. Judge the last image only; what is partly in it counts: "
VIEW_SIDE = 448  # the long side the crops above a tile are shrunk to
CHUNK = 32  # tiles whose images are built and judged at a time


class Tile(NamedTuple):
    depth: int
    row: int
    col: int
    value: float
    positive: bool


def threshold_at(depth, max_depth, top, bottom):
    return top if max_depth == 0 else top + (bottom - top) * depth / max_depth


def tile_box(depth, row, col, width, height):
    """(x0, y0, x1, y1) of a tile on the 2**depth grid."""
    n = 2 ** depth
    return col * width // n, row * height // n, (col + 1) * width // n, (row + 1) * height // n


def context_box(box, margin, width, height):
    """The tile widened by margin on every side; at an image edge it shifts inward instead of shrinking, so the tiles of
    a level keep about one crop size and mostly share forwards."""
    def widen(a, b, size):
        d = round((b - a) * margin)
        length = min(size, b - a + 2 * d)
        start = min(max(0, a - d), size - length)
        return start, start + length
    (x0, x1), (y0, y1) = widen(box[0], box[2], width), widen(box[1], box[3], height)
    return x0, y0, x1, y1


def search(judge, width, height, max_depth, top, bottom, margin):
    """Judges the whole image, then the four tiles of every positive tile down to max_depth, one level per judge call.

    judge(chains, threshold) -> [(value, positive)] in the same order; a chain is the crop boxes from the whole image down
    to the tile."""
    tiles, frontier = [], [(0, 0, [])]
    for depth in range(max_depth + 1):
        if not frontier:
            break
        boxes = [tile_box(depth, row, col, width, height) for row, col, _ in frontier]
        chains = [above + [context_box(box, margin, width, height)] for (_, _, above), box in zip(frontier, boxes)]
        results = judge(chains, threshold_at(depth, max_depth, top, bottom))
        children = []
        for (row, col, _), box, chain, (value, positive) in zip(frontier, boxes, chains, results):
            tiles.append(Tile(depth, row, col, value, positive))
            if positive and min(box[2] - box[0], box[3] - box[1]) >= 2:
                children += [(2 * row + r, 2 * col + c, chain) for r in (0, 1) for c in (0, 1)]
        frontier = children
    return tiles


def views(frame, chain):
    """The images a tile is judged on: each crop above it with a red box around the next crop, shrunk to VIEW_SIDE, then
    the tile as it is."""
    out = []
    for (x0, y0, x1, y1), (a, b, c, d) in zip(chain, chain[1:]):
        scale = min(VIEW_SIDE / max(y1 - y0, x1 - x0), 1.0)
        if scale < 1:
            size = (round((y1 - y0) * scale), round((x1 - x0) * scale))
            crop = F.interpolate(frame[y0:y1, x0:x1].movedim(-1, 0)[None], size=size, mode="bilinear", antialias=True)[0].movedim(0, -1)
        else:
            crop = frame[y0:y1, x0:x1].clone()
        a, b, c, d = (round((v - o) * scale) for v, o in ((a, x0), (b, y0), (c, x0), (d, y0)))
        red = torch.tensor([1.0, 0.0, 0.0], dtype=crop.dtype, device=crop.device)
        crop[b:b + 2, a:c] = red
        crop[d - 2:d, a:c] = red
        crop[b:d, a:a + 2] = red
        crop[b:d, c - 2:c] = red
        out.append(crop)
    x0, y0, x1, y1 = chain[-1]
    return out + [frame[y0:y1, x0:x1]]


def leaves(tiles):
    """Positive tiles that were not split: at max_depth or too small. A split tile is overruled by its four, judged at a stricter threshold."""
    judged = {(t.depth, t.row, t.col) for t in tiles}
    return [t for t in tiles if t.positive and (t.depth + 1, 2 * t.row, 2 * t.col) not in judged]


def boxes(found, width, height):
    """One box per 4-neighbour group of leaves; a leaf above the deepest level (too small to split) counts as a block of cells on its grid."""
    if not found:
        return []
    depth = max(t.depth for t in found)
    cells = {}
    for t in found:
        k = 2 ** (depth - t.depth)
        for r in range(t.row * k, (t.row + 1) * k):
            for c in range(t.col * k, (t.col + 1) * k):
                cells[(r, c)] = t.value
    out = []
    for start in sorted(cells):
        if start not in cells:
            continue
        group, stack = [], [start]
        score = cells.pop(start)
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in cells:
                    score = max(score, cells.pop(cell))
                    stack.append(cell)
        x0, y0, _, _ = tile_box(depth, min(r for r, _ in group), min(c for _, c in group), width, height)
        _, _, x1, y1 = tile_box(depth, max(r for r, _ in group), max(c for _, c in group), width, height)
        out.append({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "score": score})
    return out


def mask(found, width, height):
    out = torch.zeros(height, width, device=comfy.model_management.intermediate_device())
    for t in found:
        x0, y0, x1, y1 = tile_box(t.depth, t.row, t.col, width, height)
        out[y0:y1, x0:x1] = 1.0
    return out


def heatmap(tiles, width, height):
    """Each pixel shows the value of the deepest tile judged there, as gray."""
    out = torch.zeros(height, width, device=comfy.model_management.intermediate_device())
    for t in sorted(tiles, key=lambda t: t.depth):
        x0, y0, x1, y1 = tile_box(t.depth, t.row, t.col, width, height)
        out[y0:y1, x0:x1] = t.value
    return out.unsqueeze(-1).expand(-1, -1, 3)


class RegionSearch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TypedDecisionRegionSearch",
            display_name="Typed Decision Region Search",
            category="typed decision",
            description="Finds the regions of an image where a yes/no decision holds: judges the whole image, then splits every positive tile into four and judges those, "
                        "down to max_depth. Each tile is judged with the crops above it, so a fragment is seen in its context.",
            inputs=[
                *noul_inputs(),
                TypedDecisionModel.Input("model"),
                io.Image.Input("image", tooltip="Each image of a batch is searched on its own."),
                state_input(),
                io.Int.Input("max_depth", default=4, min=0, max=6, tooltip="0 judges only the whole image. Each level splits positive tiles into four; "
                             "depth 4 resolves a 16 x 16 grid. Deeper levels split objects into fragments."),
                io.Float.Input("threshold_top", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="Threshold for the whole image. Keep it lenient so a small hit is not cut off early."),
                io.Float.Input("threshold_bottom", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Threshold at max_depth. Depths in between move linearly from threshold_top."),
                io.Float.Input("context_margin", default=0.1, min=0.0, max=2.0, step=0.05,
                               tooltip="Each tile is judged on a crop widened by this fraction of the tile size on every side, so objects cut by a tile edge are still seen."),
            ],
            outputs=[
                io.Mask.Output(display_name="mask", tooltip="Positive tiles at max_depth (or too small to split), filled."),
                io.BoundingBox.Output(display_name="bboxes", tooltip="One box per group of touching positive tiles, per image. score is the highest value in the group."),
                io.Image.Output(display_name="heatmap", tooltip="The value of the deepest tile judged at each pixel, black = 0, white = 1."),
            ],
        )

    @classmethod
    def execute(cls, model, image, instructions, state, max_depth, threshold_top, threshold_bottom, context_margin, criteria_true="", criteria_false=""):
        question = question_from_mode({"mode": "noul", "instructions": instructions, "criteria_true": criteria_true, "criteria_false": criteria_false})
        zoomed = question._replace(text=ZOOM + question.text)  # the whole image alone has nothing to zoom from
        check_limits(model, question, 1, 0)
        state = parse_state(state)
        _, height, width, _ = image.shape
        masks, bboxes, heatmaps = [], [], []
        for frame in image:
            def judge(chains, threshold):
                # imajev takes two images: the whole image and the tile
                fitted = [chain if len(chain) <= model.max_images else [chain[0], chain[-1]] for chain in chains]
                out = []
                for c in range(0, len(fitted), CHUNK):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    requests = [(views(frame, chain), [], state, zoomed if len(chain) > 1 else question) for chain in fitted[c:c + CHUNK]]
                    for (_, _, _, asked), decision in zip(requests, model.decide_many(requests)):
                        value, _, _, passed, _, _ = answer(model, asked, decision, threshold)
                        out.append((value, passed))
                return out

            tiles = search(judge, width, height, max_depth, threshold_top, threshold_bottom, context_margin)
            found = leaves(tiles)
            masks.append(mask(found, width, height))
            bboxes.append(boxes(found, width, height))
            heatmaps.append(heatmap(tiles, width, height))
        return io.NodeOutput(torch.stack(masks), bboxes, torch.stack(heatmaps))
