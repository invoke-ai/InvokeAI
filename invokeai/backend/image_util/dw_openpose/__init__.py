from pathlib import Path
from typing import Dict

import huggingface_hub
import numpy as np
import onnxruntime as ort
import torch
from controlnet_aux.util import resize_image
from PIL import Image

from invokeai.backend.image_util.dw_openpose.onnxdet import inference_detector
from invokeai.backend.image_util.dw_openpose.onnxpose import inference_pose
from invokeai.backend.image_util.dw_openpose.utils import NDArrayInt, draw_bodypose, draw_facepose, draw_handpose
from invokeai.backend.image_util.dw_openpose.wholebody import Wholebody
from invokeai.backend.image_util.util import np_to_pil
from invokeai.backend.util.devices import TorchDevice

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


class DWOpenposeDetector2:
    """
    Code from the original implementation of the DW Openpose Detector.
    Credits: https://github.com/IDEA-Research/DWPose

    This implementation is similar to DWOpenposeDetector, with some alterations to allow the onnx models to be loaded
    and managed by the model manager.
    """

    hf_repo_id = "yzd-v/DWPose"
    hf_filename_onnx_det = "yolox_l.onnx"
    hf_filename_onnx_pose = "dw-ll_ucoco_384.onnx"

    @classmethod
    def get_model_url_det(cls) -> str:
        """Returns the URL for the detection model."""
        return huggingface_hub.hf_hub_url(cls.hf_repo_id, cls.hf_filename_onnx_det)

    @classmethod
    def get_model_url_pose(cls) -> str:
        """Returns the URL for the pose model."""
        return huggingface_hub.hf_hub_url(cls.hf_repo_id, cls.hf_filename_onnx_pose)

    @staticmethod
    def create_onnx_inference_session(model_path: Path) -> ort.InferenceSession:
        """Creates an ONNX Inference Session for the given model path, using the appropriate execution provider based on
        the device type."""

        device = TorchDevice.choose_torch_device()
        providers = ["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        return ort.InferenceSession(path_or_bytes=model_path, providers=providers)

    def __init__(self, session_det: ort.InferenceSession, session_pose: ort.InferenceSession):
        self.session_det = session_det
        self.session_pose = session_pose

    def pose_estimation(self, np_image: np.ndarray):
        """Does the pose estimation on the given image and returns the keypoints and scores."""

        det_result = inference_detector(self.session_det, np_image)
        keypoints, scores = inference_pose(self.session_pose, det_result, np_image)

        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]

        return keypoints, scores

    def run(
        self,
        image: Image.Image,
        draw_face: bool = False,
        draw_body: bool = True,
        draw_hands: bool = False,
    ) -> Image.Image:
        """Detects the pose in the given image and returns an solid black image with pose drawn on top, suitable for
        use with a ControlNet."""

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

            return DWOpenposeDetector2.draw_pose(
                pose, H, W, draw_face=draw_face, draw_hands=draw_hands, draw_body=draw_body
            )

    @staticmethod
    def draw_pose(
        pose: Dict[str, NDArrayInt | Dict[str, NDArrayInt]],
        H: int,
        W: int,
        draw_face: bool = True,
        draw_body: bool = True,
        draw_hands: bool = True,
    ) -> Image.Image:
        """Draws the pose on a black image and returns it as a PIL Image."""

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

        dwpose_image = np_to_pil(canvas)

        return dwpose_image
