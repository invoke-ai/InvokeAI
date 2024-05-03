from pathlib import Path
from typing import Dict

import numpy as np
import torch
from controlnet_aux.util import resize_image
from PIL import Image

from invokeai.backend.image_util.dw_openpose.utils import NDArrayInt, draw_bodypose, draw_facepose, draw_handpose
from invokeai.backend.image_util.dw_openpose.wholebody import Wholebody

DWPOSE_MODELS = {
    "yolox_l.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true",
    "dw-ll_ucoco_384.onnx": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true",
}


def draw_pose(
    pose: Dict[str, NDArrayInt | Dict[str, NDArrayInt]],
    H: int,
    W: int,
    draw_face: bool = True,
    draw_body: bool = True,
    draw_hands: bool = True,
    resolution: int = 512,
) -> Image.Image:
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]

    assert isinstance(bodies, dict)
    candidate = bodies["candidate"]

    assert isinstance(bodies, dict)
    subset = bodies["subset"]

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = draw_bodypose(canvas, candidate, subset)

    if draw_hands:
        assert isinstance(hands, np.ndarray)
        canvas = draw_handpose(canvas, hands)

    if draw_face:
        assert isinstance(hands, np.ndarray)
        canvas = draw_facepose(canvas, faces)  # type: ignore

    dwpose_image: Image.Image = resize_image(
        canvas,
        resolution,
    )
    dwpose_image = Image.fromarray(dwpose_image)

    return dwpose_image


class DWOpenposeDetector:
    """
    Code from the original implementation of the DW Openpose Detector.
    Credits: https://github.com/IDEA-Research/DWPose
    """

    def __init__(self, onnx_det: Path, onnx_pose: Path) -> None:
        self.pose_estimation = Wholebody(onnx_det=onnx_det, onnx_pose=onnx_pose)

    def __call__(
        self,
        image: Image.Image,
        draw_face: bool = False,
        draw_body: bool = True,
        draw_hands: bool = False,
        resolution: int = 512,
    ) -> Image.Image:
        np_image = np.array(image)
        H, W, C = np_image.shape

        with torch.no_grad():
            candidate, subset = self.pose_estimation(np_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            score = subset[:, :18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            candidate[un_visible] = -1

            # foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            bodies = {"candidate": body, "subset": score}
            pose = {"bodies": bodies, "hands": hands, "faces": faces}

            return draw_pose(
                pose, H, W, draw_face=draw_face, draw_hands=draw_hands, draw_body=draw_body, resolution=resolution
            )


__all__ = ["DWPOSE_MODELS", "DWOpenposeDetector"]
