from __future__ import annotations
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def _load_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _crop_with_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return image_rgb[y:y + h, x:x + w]


def _encode_image(model, preprocess, image_rgb: np.ndarray, device: str) -> torch.Tensor:
    image_pil = Image.fromarray(image_rgb)
    image_input = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image_input).float()


def _encode_text(model, prompts: List[str], device: str) -> torch.Tensor:
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        return model.encode_text(tokens).float()  # (N, D)


def _colored_segmentation(image_shape: Tuple[int, int, int], masks: List[dict]) -> np.ndarray:
    h, w = image_shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        color = [(i * 37) % 256, (i * 59) % 256, (i * 83) % 256]
        vis[seg] = color
    return vis


def _circularity_score(mask: np.ndarray) -> float:
    """Return [0..1], 1 is perfect circle (using 4πA / P^2)."""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    per = float(cv2.arcLength(c, True) + 1e-6)
    circ = (4.0 * np.pi * area) / (per * per)
    # clamp to [0,1]
    return float(max(0.0, min(1.0, circ)))


def create_semantic_mask(
    video_path: str,
    output_folder: str,
    reference_images_dir: str,
    sam_checkpoint: str,
    model_type: str = "vit_h",
    device: str = "cuda",
    use_center_priority: bool = False,
    target_class: Optional[str] = None,
    disallow_classes: Optional[List[str]] = None,
    enforce_circularity: bool = False,
) -> Optional[str]:
    """
    Generate a segmentation mask from the first frame using SAM.
    Segment selection:
      - If center priority: choose segment closest to center.
      - Else: CLIP scoring with (a) image references, (b) optional text bias toward `target_class`,
              and (c) optional negative class penalty via text prompts.
      - Optional geometric prior to prefer circular shapes (for balls).

    Saves:
      - *_all_segments.png
      - *_best_mask.png
      - *_best_visualization.png

    Returns: path to *_best_mask.png or None on failure.
    """
    os.makedirs(output_folder, exist_ok=True)

    frame_rgb = _load_first_frame(video_path)
    H, W = frame_rgb.shape[:2]
    stem = os.path.splitext(os.path.basename(video_path))[0]

    mask_path = os.path.join(output_folder, f"{stem}_best_mask.png")
    vis_path = os.path.join(output_folder, f"{stem}_best_visualization.png")

    if os.path.isfile(mask_path):
        print('Maks already created')
        return mask_path

    # --- SAM ---
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.92,
        min_mask_region_area=2000,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    print("Generating SAM segments ...")
    masks = mask_gen.generate(frame_rgb)
    all_vis = _colored_segmentation(frame_rgb.shape, masks)
    cv2.imwrite(os.path.join(output_folder, f"{stem}_all_segments.png"),
                cv2.cvtColor(all_vis, cv2.COLOR_RGB2BGR))

    if not masks:
        print("No SAM segments found.")
        return None

    # --- CLIP ---
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Image reference embeddings
    ref_embeds: list[torch.Tensor] = []
    if not use_center_priority:
        if not os.path.isdir(reference_images_dir):
            print(f"Reference images directory not found: {reference_images_dir}")
            return None
        for fname in sorted(os.listdir(reference_images_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                path = os.path.join(reference_images_dir, fname)
                img = np.array(Image.open(path).convert("RGB"))
                ref_embeds.append(_encode_image(clip_model, preprocess, img, device))
        if not ref_embeds:
            print(f"No reference images found in: {reference_images_dir}")
            return None

    # Text guidance
    text_pos = _encode_text(clip_model, [target_class], device) if target_class else None
    text_neg = _encode_text(clip_model, disallow_classes, device) if disallow_classes else None

    # --- rank segments ---
    cx, cy = W / 2.0, H / 2.0
    ranked = []

    with torch.no_grad():
        for i, m in enumerate(masks):
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            if xs.size == 0:
                continue

            # Base crop embedding
            crop = _crop_with_mask(frame_rgb, seg)
            emb = _encode_image(clip_model, preprocess, crop, device)  # (1, D)

            score = 0.0

            # (a) image refs (max cosine similarity)
            if ref_embeds and not use_center_priority:
                sim_img = torch.stack([
                    torch.nn.functional.cosine_similarity(emb, r, dim=1)[0] for r in ref_embeds
                ]).max().item()
                score += 1.0 * float(sim_img)

            # (b) text positive bias
            if text_pos is not None:
                sim_pos = torch.nn.functional.cosine_similarity(emb, text_pos, dim=1)[0].item()
                score += 0.7 * float(sim_pos)

            # (c) text negative penalty (use the worst offender)
            if text_neg is not None:
                sim_neg = torch.nn.functional.cosine_similarity(emb, text_neg, dim=1)  # (K,)
                penalty = sim_neg.max().item()
                score -= 0.7 * float(penalty)

            # (d) geometric prior: prefer near-circular (gym ball-like)
            if enforce_circularity:
                circ = _circularity_score(seg.astype(np.uint8))
                area = float(seg.sum())
                area_norm = area / float(H * W)
                score += 0.5 * circ + 0.2 * area_norm  # mild bias

            # (e) optional center proximity (very small tie-breaker)
            centroid = (float(xs.mean()), float(ys.mean()))
            dist = np.hypot(centroid[0] - cx, centroid[1] - cy)
            score += -0.0005 * dist  # tiny nudge toward center

            ranked.append({"idx": i, "mask": seg, "score": score})

    if not ranked:
        print("No valid segments after scoring.")
        return None

    ranked.sort(key=lambda d: d["score"], reverse=True)
    best = ranked[0]["mask"].astype(bool)

    # Save outputs
    mask_out = (best.astype(np.uint8) * 255)
    overlay = frame_rgb.copy()
    overlay[best] = [0, 255, 0]
    vis = (0.7 * frame_rgb + 0.3 * overlay).astype(np.uint8)

    cv2.imwrite(mask_path, mask_out)
    cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print(f"Selected segment saved -> {mask_path}")
    return mask_path
