"""Read chosen frames of a VIDEO without decoding the whole clip into memory.

A file is decoded once from the start of its trim window, and only the requested frames (by index, as the d3 runtime
picks them) are kept. Other videos, and files with a crop that cannot be read back, go through get_components."""
from typing import NamedTuple

import av
import numpy as np
import torch

from comfy_api.latest import InputImpl


class Clip(NamedTuple):
    """Frames already read ([T, H, W, 3] uint8), `fps` per second. A model sees each of them and does not thin them again."""
    frames: torch.Tensor
    fps: float


def streamed(video):
    """True when the frames can be read from the file directly: a file with no crop."""
    if not isinstance(video, InputImpl.VideoFromFile):
        return False
    with av.open(video.get_stream_source()) as container:
        stream = container.streams.video[0]
        return video.get_dimensions() == (stream.width, stream.height)


def window(container, video):
    """(first video stream, start pts, end pts or None) of the video's trim window, as Core decodes it."""
    stream = container.streams.video[0]
    start, duration = video.get_active_trim_window()
    start_pts = int(start / stream.time_base)
    if start_pts:
        container.seek(start_pts, stream=stream)
    return stream, start_pts, int((start + duration) / stream.time_base) if duration else None


def frame_count(video):
    """Frames in the video. A file's packets are counted without decoding (a crop does not change the count); container
    metadata can be off by one."""
    if not isinstance(video, InputImpl.VideoFromFile):
        return video.get_components().images.shape[0]
    with av.open(video.get_stream_source()) as container:
        stream, start_pts, end_pts = window(container, video)
        return sum(1 for packet in container.demux(stream)
                   if packet.pts is not None and packet.pts >= start_pts and (end_pts is None or packet.pts < end_pts))


def decoded(container, video):
    """The frames of the trim window in order. Like Core's decoder, a packet that fails to decode is skipped."""
    stream, start_pts, end_pts = window(container, video)
    stream.thread_type = "AUTO"
    for packet in container.demux(stream):
        try:
            for frame in packet.decode():
                if frame.pts < start_pts:
                    continue
                if end_pts is not None and frame.pts >= end_pts:
                    return
                yield frame
        except av.error.InvalidDataError:
            continue


def read_frames(video, indices, resize=lambda frame: frame):
    """[len(indices), H, W, 3] uint8 frames at the given indices, in order, each passed through `resize` ([H, W, 3] uint8 ->
    the same) as it is kept. An index past the last decodable frame repeats that frame."""
    if not streamed(video):
        images = video.get_components().images[indices][..., :3]
        return torch.stack([resize(frame) for frame in (images.float() * 255).round().clamp(0, 255).to(torch.uint8)])
    wanted, last, found = set(indices), max(indices), {}
    with av.open(video.get_stream_source()) as container:
        for index, frame in enumerate(decoded(container, video)):
            if index > last:
                break
            if index in wanted:
                image = frame.to_ndarray(format="rgb24")
                quadrant = int(round(frame.rotation // 90)) % 4 if frame.rotation else 0  # as Core's decoder rotates
                found[index] = resize(torch.from_numpy(np.ascontiguousarray(np.rot90(image, k=quadrant, axes=(0, 1)))))
    if not found:
        raise ValueError("No frame of the video could be decoded.")
    earlier = lambda index: max([i for i in found if i <= index] or [min(found)])
    return torch.stack([found[earlier(index)] for index in indices])
