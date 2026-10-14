"""Interval Search: where in a video a yes/no decision holds, to the source frame.

Stage 1 judges windows of frames read at 2 fps (the rate d3 reads videos at) over the whole video. Stage 2 finds each edge
of a run of passing windows by sliding a window one source frame at a time, by bisection, between a passing and a failing
window. A window passes once one of its frames shows what is asked, so questions are about what one frame shows; motion,
whose edges depend on how much of it a window holds, is not searched for."""
import json
import math

import numpy as np

import comfy.model_management
import comfy.utils
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker

from . import d3, frames
from .nodes import TypedDecisionModel, answer, check_limits, noul_inputs, parse_state, question_from_mode, state_input

CHUNK = 128  # frames per decide_many call; the frames of one chunk are held in float at a time


def sample(total, video_fps):
    """Source frames read at 2 per second, evenly over the whole video."""
    count = min(max(int(total / video_fps * d3.VIDEO_FPS), 1), total)
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def windows(n, size):
    """(start, end) of each window over n samples, half a window apart; the last window ends on the last sample."""
    if n <= size:
        return [(0, n)]
    starts = list(range(0, n - size + 1, max(size // 2, 1)))
    if starts[-1] != n - size:
        starts.append(n - size)
    return [(s, s + size) for s in starts]


def runs(passed):
    """(first, last) window of each run of passing windows. Windows overlap, so a run covers its samples without a gap."""
    out = []
    for i, ok in enumerate(passed):
        if ok and out and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        elif ok:
            out.append((i, i))
    return out


def searches(spans, first, last, indices, total):
    """The start and the end search of the run of passing windows first..last: (frames of the window that slides, shift
    where it passes, shift where it fails, from_end).

    The start slides the window before the run, which failed, towards the first passing window; the end slides the last
    passing window towards the window after it. At an end of the video the window slides out of it, where it sees nothing."""
    if first:
        before = indices[spans[first - 1][0]:spans[first - 1][1]]
        start = (before, indices[spans[first][1] - 1] - before[-1], 0, False)  # the samples are uneven: line up the far frames
    else:
        seen = indices[spans[0][0]:spans[0][1]]
        start = (seen, 0, -seen[-1] - 1, False)
    seen = indices[spans[last][0]:spans[last][1]]
    end = (seen, 0, indices[spans[last + 1][0]] - seen[0] if last + 1 < len(spans) else total - seen[0], True)
    return [start, end]


def slide(wanted, total, judge):
    """The shift, by bisection, where each search's window last passes next to where it fails. judge([frames, ...]) ->
    passes of those windows; frames outside the video are left out, and a window with none fails."""
    bounds = [[ok, bad] for _, ok, bad, _ in wanted]
    while open_ := [i for i, (ok, bad) in enumerate(bounds) if abs(ok - bad) > 1]:
        shifts = [sum(bounds[i]) // 2 for i in open_]
        shifted = [[f + d for f in wanted[i][0] if 0 <= f + d < total] for i, d in zip(open_, shifts)]
        kept = [w for w in shifted if w]
        judged = iter(judge(kept) if kept else [])
        for i, d, w in zip(open_, shifts, shifted):
            bounds[i][0 if w and next(judged) else 1] = d
    return [ok for ok, _ in bounds]


def edge(search, shift):
    """The source frame a search found (from_end: the end, after the last frame): where the far frame of the window meets
    what is asked."""
    seen, _, _, from_end = search
    return seen[0] + shift + 1 if from_end else seen[-1] + shift


def decide(model, question, state, threshold, clips, fps, bar):
    """(passes, values) of each clip ([T, H, W, 3] uint8, `fps` frames a second). d3 sees a clip as a video, the others as images."""
    passed, values = [], []
    step = max(CHUNK // max(len(clip) for clip in clips), 1)
    for c in range(0, len(clips), step):
        comfy.model_management.throw_exception_if_processing_interrupted()
        if model.max_videos > 0:
            requests = [([], [frames.Clip(clip, fps)], state, question) for clip in clips[c:c + step]]
        else:
            requests = [(list(clip.float() / 255.0), [], state, question) for clip in clips[c:c + step]]
        for decision in model.decide_many(requests):
            value, _, _, ok, _, _ = answer(model, question, decision, threshold)
            values.append(value)
            passed.append(ok)
        bar.update(len(requests))
    return passed, values


class IntervalSearch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TypedDecisionIntervalSearch",
            display_name="Typed Decision Interval Search",
            category="typed decision",
            description="Finds the intervals of a video where a yes/no decision holds, to the source frame. It judges windows of frames read at 2 fps over the "
                        "whole video, then slides the window one source frame at a time to find each edge.",
            inputs=[
                *noul_inputs(),
                TypedDecisionModel.Input("model"),
                io.Video.Input("video"),
                state_input(),
                io.Int.Input("window", default=8, min=1, max=64, tooltip="Frames judged together, read at 2 fps (8: 4 seconds). A larger window takes fewer "
                             "judgments for a long video. imajev takes at most 2."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="A window passes when the probability of yes reaches this."),
            ],
            outputs=[
                io.String.Output(display_name="intervals", tooltip="JSON list of {start, end (seconds), start_frame, end_frame (end exclusive), score}."),
                io.Video.Output(display_name="videos", is_output_list=True, tooltip="The video trimmed to each interval."),
            ],
        )

    @classmethod
    def execute(cls, model, video, instructions, state, window, threshold, criteria_true="", criteria_false=""):
        question = question_from_mode({"mode": "noul", "instructions": instructions, "criteria_true": criteria_true, "criteria_false": criteria_false})
        check_limits(model, question, 0 if model.max_videos > 0 else window, 1 if model.max_videos > 0 else 0)
        state = parse_state(state)
        total = frames.frame_count(video)
        if total == 0:
            raise ValueError("The video has no frames in its trim window.")
        video_fps = float(video.get_frame_rate())
        resize = lambda frame: d3.resize_frame(frame, 2 * window)  # the size d3 gives a window, which it reads each frame of twice; Clef sees this size too

        indices = sample(total, video_fps)
        spans = windows(len(indices), window)
        u8 = frames.read_frames(video, indices, resize)
        rate = video_fps * (len(indices) - 1) / (indices[-1] - indices[0]) if len(indices) > 1 else video_fps
        bar = comfy.utils.ProgressBar(len(spans))
        passed, values = decide(model, question, state, threshold, [u8[s:e] for s, e in spans], rate, bar)
        found = runs(passed)

        def judge(windows):  # each round reads only the frames it judges, so memory does not grow with the video
            near = sorted({f for w in windows for f in w})
            row = {f: i for i, f in enumerate(near)}
            source = frames.read_frames(video, near, resize)
            return decide(model, question, state, threshold, [source[[row[f] for f in w]] for w in windows], rate, bar)[0]

        wanted = [search for first, last in found for search in searches(spans, first, last, indices, total)]
        bar.update_absolute(bar.current, len(spans) + sum(math.ceil(math.log2(abs(ok - bad))) for _, ok, bad, _ in wanted))  # one judgment a bisection step
        bounds = [edge(search, shift) for search, shift in zip(wanted, slide(wanted, total, judge))]
        bar.update_absolute(bar.total)  # the total counted every bisection step; a search can end in fewer
        intervals = []
        for (first, last), start, end in zip(found, bounds[::2], bounds[1::2]):
            end = max(end, start + 1)  # an event under half a second can end before it starts
            intervals.append({"start": start / video_fps, "end": end / video_fps, "start_frame": start, "end_frame": end, "score": max(values[first:last + 1])})
        # a microsecond short: VideoFromComponents drops a duration that rounds past its end. More would lose the last frame
        # of a file whose trim start falls between frames.
        clips = [video.as_trimmed(f["start"], f["end"] - f["start"] - 1e-6, strict_duration=False) for f in intervals]
        return io.NodeOutput(json.dumps(intervals), clips or [ExecutionBlocker(None)])  # nothing found: video nodes after it are skipped
