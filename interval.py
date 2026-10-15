"""Interval Search: where in a video a yes/no decision holds, to the source frame.

The video is read at 2 fps (the rate d3 reads videos at) and cut into blocks of `window` samples. A block passes once one
of its frames shows what is asked, so a block that fails holds nothing and is dropped whole; the samples of a block that
passes are judged one by one, and the samples that pass make the intervals. Each edge is then found among the source frames
between a failing and a passing sample. The three steps keep the forwards few, since a model that does not fit in VRAM
streams its weights for each. Questions are about what one frame shows; motion is not searched for."""
import json

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


def blocks(n, size):
    """(start, end) of each block of size samples over n samples, without overlap."""
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def positives(n, size, judge, threshold):
    """{sample: value} of the samples that pass alone. judge([(start, end), ...], threshold) -> [(passed, value)] of those
    sample ranges; it is called for the blocks, then for the samples of the blocks that pass.

    A block only drops what holds nothing, so it passes at half the threshold: a block with one positive sample in it can
    score below the threshold (d3-flash 0.38 at 8 samples), and a block passed in vain costs only its samples' judgments."""
    ranges = blocks(n, size)
    if size == 1:  # the blocks are the samples
        return {a: value for (a, _), (passed, value) in zip(ranges, judge(ranges, threshold)) if passed}
    samples = [(i, i + 1) for (a, b), (passed, _) in zip(ranges, judge(ranges, threshold / 2)) if passed for i in range(a, b)]
    return {a: value for (a, _), (passed, value) in zip(samples, judge(samples, threshold) if samples else []) if passed}


def runs(samples):
    """(first, last) of each run of neighbouring samples, from sorted sample numbers."""
    out = []
    for i in samples:
        if out and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def edges(searches, judge):
    """For each (passing sample's frame, failing sample's frame), the last frame of the run of passing frames joined to the
    passing one, judging every frame between in one call; judge([frame, ...]) -> passes. A failing frame outside the
    video (-1 or total) is never judged."""
    between = [list(range(ok - 1, bad, -1)) if bad < ok else list(range(ok + 1, bad)) for ok, bad in searches]  # nearest first
    wanted = sorted({f for frames_ in between for f in frames_})
    passed = dict(zip(wanted, judge(wanted) if wanted else []))
    out = []
    for (ok, _), frames_ in zip(searches, between):
        for f in frames_:
            if not passed[f]:
                break
            ok = f
        out.append(ok)
    return out


def decide(model, question, state, threshold, clips, fps, advance):
    """(passes, values) of each clip ([T, H, W, 3] uint8, `fps` frames a second). Models that take videos see a clip as one, imajev as images.
    advance(n) is called as each n clips are judged."""
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
        advance(len(requests))
    return passed, values


class IntervalSearch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TypedDecisionIntervalSearch",
            display_name="Typed Decision Interval Search",
            category="typed decision",
            description="Finds the intervals of a video where a yes/no decision holds, to the source frame. It reads the video at 2 fps, judges blocks of frames, "
                        "then each frame of the blocks that pass, then the source frames around each edge.",
            inputs=[
                *noul_inputs(),
                TypedDecisionModel.Input("model"),
                io.Video.Input("video"),
                state_input(),
                io.Int.Input("window", default=8, min=1, max=64, tooltip="Frames judged together at first, read at 2 fps (8: 4 seconds). The frames of a block "
                             "that passes are judged one by one. A larger window can drop a half-second event with its block. imajev takes at most 2."),
                io.Float.Input("threshold", default=0.5, min=0.0, max=1.0, step=0.01, tooltip="A frame passes when the probability of yes reaches this; a block of frames, at half of it."),
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
        u8 = frames.read_frames(video, indices, resize)
        rate = video_fps * (len(indices) - 1) / (indices[-1] - indices[0]) if len(indices) > 1 else video_fps
        # a third of the bar for each step: the blocks, their samples, the frames around the edges; the judgments of a step
        # are known only after the step before
        bar = comfy.utils.ProgressBar(300)
        step = [0]

        def judge(clips, fps, threshold):
            done = [0]

            def advance(n):
                done[0] += n
                bar.update_absolute(100 * step[0] + 100 * done[0] // len(clips))
            out = decide(model, question, state, threshold, clips, fps, advance)
            step[0] += 1
            return out

        def judge_samples(ranges, threshold):
            passed, values = judge([u8[a:b] for a, b in ranges], rate, threshold)
            return list(zip(passed, values))

        def judge_frames(wanted):  # only the frames around the edges are read again
            step[0] = 2  # with window 1 there is no step for the samples
            source = frames.read_frames(video, wanted, resize)
            return judge([source[i:i + 1] for i in range(len(wanted))], video_fps, threshold)[0]

        found = positives(len(indices), window, judge_samples, threshold)
        groups = runs(sorted(found))
        # each run starts after the failing sample before it and ends before the one after it, or at an end of the video
        searches = [search for first, last in groups for search in ((indices[first], indices[first - 1] if first else -1),
                                                                     (indices[last], indices[last + 1] if last + 1 < len(indices) else total))]
        bounds = edges(searches, judge_frames)
        bar.update_absolute(300)
        intervals = [{"start": start / video_fps, "end": (end + 1) / video_fps, "start_frame": start, "end_frame": end + 1,
                      "score": max(found[i] for i in range(first, last + 1))} for (first, last), start, end in zip(groups, bounds[::2], bounds[1::2])]
        # a microsecond short: VideoFromComponents drops a duration that rounds past its end. More would lose the last frame
        # of a file whose trim start falls between frames.
        clips = [video.as_trimmed(f["start"], f["end"] - f["start"] - 1e-6, strict_duration=False) for f in intervals]
        return io.NodeOutput(json.dumps(intervals), clips or [ExecutionBlocker(None)])  # nothing found: video nodes after it are skipped
