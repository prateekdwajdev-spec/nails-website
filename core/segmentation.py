import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry


class NailSegmenter:
    """MobileSAM segmenter using combined Bounding Box and Point Prompts

    to strictly isolate individual fingernail beds.
    """

    def __init__(
        self,
        model_type: str = "vit_t",
        checkpoint_path: str = "weights/mobile_sam.pt",
        device: str = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        mobile_sam.to(device=self.device)
        mobile_sam.eval()
        self.predictor = SamPredictor(mobile_sam)
        self.img_height = 0
        self.img_width = 0

    def set_image(self, image: np.ndarray):
        """Prepares image for SAM inference and records spatial bounds."""
        self.img_height, self.img_width = image.shape[:2]
        self.predictor.set_image(image)

    def compute_nail_prompts(
        self, fingertip_pt: np.ndarray, dip_joint_pt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates point prompts AND a localized bounding box around the nail bed."""
        vec = dip_joint_pt - fingertip_pt
        vec_length = np.linalg.norm(vec)
        if vec_length == 0:
            vec_length = 1e-6
        unit_vec = vec / vec_length
        perp_vec = np.array([-unit_vec[1], unit_vec[0]])

        # 1. Target center of nail plate (~20% down towards DIP joint)
        nail_center = fingertip_pt + (0.20 * vec)

        # 2. Negative prompts on lateral skin walls and cuticle line
        lateral_offset = max(5, int(vec_length * 0.16))
        left_skin = nail_center + (lateral_offset * perp_vec)
        right_skin = nail_center - (lateral_offset * perp_vec)
        cuticle_skin = nail_center + (0.28 * vec)

        point_coords = np.array(
            [nail_center, left_skin, right_skin, cuticle_skin], dtype=np.float32
        )
        point_labels = np.array([1, 0, 0, 0], dtype=np.int32)

        # 3. Tightened Bounding Box (Max 30% down finger vector to prevent cuticle bleeding)
        box_half_width = max(12, int(vec_length * 0.28))
        box_top_margin = max(8, int(vec_length * 0.10))
        box_bottom_margin = max(12, int(vec_length * 0.30))

        x_coords = [
            fingertip_pt[0],
            nail_center[0] - box_half_width,
            nail_center[0] + box_half_width,
        ]
        y_coords = [
            fingertip_pt[1] - box_top_margin,
            nail_center[1] + box_bottom_margin,
        ]

        x_min = max(0, int(min(x_coords)))
        y_min = max(0, int(min(y_coords)))
        x_max = min(self.img_width, int(max(x_coords)))
        y_max = min(self.img_height, int(max(y_coords)))

        box_coords = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return point_coords, point_labels, box_coords

    def segment_nail(
        self, fingertip_pt: np.ndarray, dip_joint_pt: np.ndarray
    ) -> np.ndarray:
        """Runs MobileSAM prediction using point and box prompts to isolate the nail mask."""
        point_coords, point_labels, box_coords = self.compute_nail_prompts(
            fingertip_pt, dip_joint_pt
        )

        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords[None, :],
            multimask_output=True,
        )

        # Pick non-empty mask with smallest area (finest sub-part = nail bed)
        valid_masks = []
        for mask in masks:
            area = np.sum(mask)
            if area > 20:
                valid_masks.append((area, mask))

        if valid_masks:
            valid_masks.sort(key=lambda item: item[0])
            best_mask = valid_masks[0][1]
        else:
            best_mask = masks[0]

        return (best_mask * 255).astype(np.uint8)


_global_segmenter = None


def segment_nail_region(
    image: np.ndarray,
    finger_or_tip,
    dip_joint_pt=None,
    segmenter: NailSegmenter = None,
) -> np.ndarray:
    """Wrapper function accepting Finger dataclasses or explicit raw coordinates."""
    global _global_segmenter
    if segmenter is None:
        if _global_segmenter is None:
            _global_segmenter = NailSegmenter()
        segmenter = _global_segmenter

    # Extract tip and DIP joint coordinates
    if hasattr(finger_or_tip, "tip_pt") and hasattr(finger_or_tip, "dip_pt"):
        tip_pt = finger_or_tip.tip_pt
        dip_pt = finger_or_tip.dip_pt
    elif dip_joint_pt is not None:
        tip_pt = finger_or_tip
        dip_pt = dip_joint_pt
    elif hasattr(finger_or_tip, "x") and hasattr(finger_or_tip, "y"):
        tip_pt = (finger_or_tip.x, finger_or_tip.y)
        if hasattr(finger_or_tip, "dip_x") and hasattr(finger_or_tip, "dip_y"):
            dip_pt = (finger_or_tip.dip_x, finger_or_tip.dip_y)
        else:
            dip_pt = (finger_or_tip.x, finger_or_tip.y + 40.0)
    else:
        raise ValueError(f"Cannot extract landmark points from {finger_or_tip}")

    segmenter.set_image(image)
    return segmenter.segment_nail(
        np.array(tip_pt, dtype=np.float32), np.array(dip_pt, dtype=np.float32)
    )


def segment_all_nails(
    image: np.ndarray, landmark_dict: dict, segmenter: NailSegmenter = None
) -> dict:
    """Pipeline helper function to segment all 5 nails on a hand."""
    if segmenter is None:
        segmenter = NailSegmenter()

    segmenter.set_image(image)
    nail_masks = {}

    for finger_name, pts in landmark_dict.items():
        if isinstance(pts, (tuple, list)) and len(pts) == 2:
            tip_pt, dip_pt = pts[0], pts[1]
        else:
            tip_pt, dip_pt = pts.tip_pt, pts.dip_pt

        mask = segmenter.segment_nail(
            np.array(tip_pt, dtype=np.float32), np.array(dip_pt, dtype=np.float32)
        )
        nail_masks[finger_name] = mask

    return nail_masks