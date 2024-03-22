import os

import av
import megfile
import numpy as np

from omni.utils.image_utils import ImageType, ImageTypeStr, load_image


def get_average_frame_inds(frames_length, num_frames=1) -> list[int]:
    if frames_length < num_frames:
        frame_inds = list(range(frames_length)) + [frames_length - 1 for _ in range(num_frames - frames_length)]
        return frame_inds
    else:
        frame_inds = np.linspace(0, frames_length - 1, num_frames, endpoint=True, dtype=int)
        frame_inds = np.clip(frame_inds, 0, frames_length - 1)
        return frame_inds


def load_video(
    video: str | os.PathLike,
    num_frames: int,
    output_type: ImageTypeStr = "pil",
    return_info: bool = False,
) -> list[ImageType] | tuple[list[ImageType], dict]:
    """
    Loads `video` to a list of PIL Image, NumPy array or PyTorch tensor.

    Args:
        video (str): The path to video.
        output_type (ImageTypeStr, optional): The type of the output image. Defaults to "pil".
            The current version supports "pil", "np", "pt".
        return_info (bool, optional): Whether to return the video information. Defaults to False.

    Returns:
        ImageType: The loaded image in the given type.
    """
    if isinstance(video, str) or isinstance(video, os.PathLike):
        if video.startswith("s3"):
            video = megfile.smart_open(video, mode="rb")
            video.seek(0)
        elif os.path.isfile(video):
            pass
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `s3+[profile]://`, and `{video}` is not a valid path."
            )
    else:
        raise ValueError(f"`video` must be a path, got `{type(video)}`")

    container = av.open(video, metadata_errors="ignore")
    stream = next(s for s in container.streams if s.type == "video")

    video_info = dict(
        frames=stream.frames,
        fps=float(stream.average_rate),
        duration=stream.duration * stream.time_base if stream.duration is not None else None,
        width=stream.width,
        height=stream.height,
    )

    frame_inds = get_average_frame_inds(stream.frames, num_frames)

    pt = 0
    images = []
    for frame in container.decode(stream):
        if frame.index > frame_inds[-1] or pt >= len(frame_inds):
            break
        if frame.index == frame_inds[pt]:
            images.append(load_image(frame.to_image(), output_type=output_type))
            pt += 1
    assert len(images) == num_frames, f"Expected {num_frames} frames, but got {len(images)} frames."

    if hasattr(video, "close"):
        video.close()
    container.close()

    if return_info:
        return images, video_info
    return images
