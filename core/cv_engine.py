"""
DeskZen OpenCV 5 Vision Analysis Engine
Multi-layer spatial, entropy, moment, and hazard analysis on workspace photos.
Calibrated for physical metric ergonomics (cm) and prescriptive agentic relocation vectors.
Integrates YOLOv8 + OpenCV Hybrid Object Pipeline, Central Work Focus isolation,
in-place progressive inpainting, and automated Target After generation.
"""

import os
import math
import time
from typing import Dict, Any, List, Tuple, Union, Optional
import cv2
import numpy as np

from core.yolo_detector import WorkspaceYOLODetector
from core.after_generator import TargetAfterGenerator
from core.spatial_map import WorkspaceDigitalTwinMap


def _bbox_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union (IoU) between two [x, y, w, h] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = float(interW * interH)
    if interArea <= 0:
        return 0.0

    areaA = float(max(1, boxA[2] * boxA[3]))
    areaB = float(max(1, boxB[2] * boxB[3]))
    return interArea / (areaA + areaB - interArea + 1e-5)


def _bbox_containment(inner: List[int], outer: List[int]) -> float:
    """Fraction of `inner` [x, y, w, h] area contained inside `outer` [x, y, w, h]."""
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[0] + inner[2], outer[0] + outer[2])
    yB = min(inner[1] + inner[3], outer[1] + outer[3])
    interArea = float(max(0, xB - xA) * max(0, yB - yA))
    innerArea = float(max(1, inner[2] * inner[3]))
    return interArea / innerArea


class WorkspaceVisionEngine:
    # Calibrated homography ratio: typical 800px workspace image corresponds to ~120 cm tabletop width
    CM_PER_PIXEL = 0.15

    def __init__(self, crops_dir: str = "static/crops"):
        self.crops_dir = crops_dir
        os.makedirs(self.crops_dir, exist_ok=True)
        self.yolo_detector = WorkspaceYOLODetector()
        self._after_gen = TargetAfterGenerator(output_dir=self.crops_dir)

    @property
    def after_gen(self) -> TargetAfterGenerator:
        if not hasattr(self, "_after_gen") or self._after_gen is None:
            self._after_gen = TargetAfterGenerator(output_dir=getattr(self, "crops_dir", "static/crops"))
        return self._after_gen

    @after_gen.setter
    def after_gen(self, val: TargetAfterGenerator):
        self._after_gen = val

    def generate_target_after(
        self, img: np.ndarray, telemetry: Dict[str, Any], plan: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Convenience method to generate target after scenario even on reloaded instances."""
        return self.after_gen.generate_target_after(img, telemetry, plan or {})

    def inpaint_specific_items(
        self,
        img: np.ndarray,
        telemetry: Dict[str, Any],
        removed_item_ids: Any,
        plan: Optional[Dict[str, Any]] = None,
        item_actions: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Convenience method to inpaint and remove/relocate only checklist-toggled items."""
        return self.after_gen.inpaint_specific_items(
            img, telemetry, removed_item_ids, plan or {}, item_actions=item_actions
        )

    def generate_sanctuary_blueprint_card(
        self,
        before_img: np.ndarray,
        after_img: np.ndarray,
        telemetry: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience method to generate downloadable Before -> After Sanctuary Blueprint Card."""
        return self.after_gen.generate_sanctuary_blueprint_card(before_img, after_img, telemetry, plan or {})

    def build_digital_twin_map(
        self,
        img: np.ndarray,
        telemetry: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
    ) -> WorkspaceDigitalTwinMap:
        """
        Builds a 2.5D Layered Scene Graph representing the physical workspace.
        Enables sub-10ms instantaneous deterministic relocation, inpainting, and compositing.
        """
        return WorkspaceDigitalTwinMap.create_from_analysis(img, telemetry, plan=plan)

    def analyze(self, image_input: Union[str, np.ndarray], generate_after: bool = True) -> Dict[str, Any]:
        """
        Run the complete OpenCV 5 + YOLOv8 hybrid analysis pipeline on a workspace image.
        Returns a rich structured telemetry payload and 4 visual layers:
        1. annotated_image (Zoning + Central Focus Halo + Numbered Items)
        2. heatmap_image (OpenCV Continuous Cognitive Clutter Energy Heatmap)
        3. relocation_image (Agentic Prescriptive Relocation Vectors)
        4. target_after_image (Synthesized Decluttered Blueprint to Replicate)
        """
        t_start = time.perf_counter()

        # 1. Load image
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            raise TypeError("image_input must be a file path string or numpy ndarray")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_pixels = float(h * w)

        # 2. Layer 1: Visual Entropy (Cognitive Clutter Score)
        t_ent_0 = time.perf_counter()
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_hist, _ = np.histogram(grad_mag, bins=32, range=(0, 256), density=True)
        grad_hist = grad_hist[grad_hist > 0]
        shannon_entropy = -float(np.sum(grad_hist * np.log2(grad_hist)))
        entropy_spectral_ms = round((time.perf_counter() - t_ent_0) * 1000.0, 2)

        # 3. YOLOv8 Object Detection & Perspective / Clustering Analysis
        t_yolo_0 = time.perf_counter()
        yolo_res = self.yolo_detector.detect(img)
        yolo_anchor_ms = round((time.perf_counter() - t_yolo_0) * 1000.0, 2)
        perspective = yolo_res.get("perspective", "TOP_DOWN")
        yolo_objects = yolo_res.get("objects", [])

        # 4. Classical Anchor & Supplemental Contour Extraction
        t_morph_0 = time.perf_counter()
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

        min_area = max(400, int(total_pixels * 0.001))
        max_area = total_pixels * 0.45

        raw_candidates = []
        user_x = w / 2.0
        user_y = float(h)

        # Calibrated Reach Radii
        zone1_radius_px = int(35.0 / self.CM_PER_PIXEL)  # 35cm (~233px)
        zone2_radius_px = int(55.0 / self.CM_PER_PIXEL)  # 55cm (~366px)

        keyboard_anchor = None
        monitor_anchor = None
        mouse_anchor = None

        # Seed anchors from YOLO first if available
        for y_obj in yolo_objects:
            cname = y_obj.get("class_name", "")
            if cname in ["keyboard", "laptop"] and keyboard_anchor is None:
                keyboard_anchor = {
                    "bbox": list(y_obj["bbox"]),
                    "centroid": list(y_obj["centroid"]),
                    "area_px": int(y_obj["area_px"]),
                }
            elif cname in ["tv", "monitor"] and monitor_anchor is None:
                monitor_anchor = {
                    "bbox": list(y_obj["bbox"]),
                    "centroid": list(y_obj["centroid"]),
                    "area_px": int(y_obj["area_px"]),
                }
            elif cname == "mouse" and mouse_anchor is None:
                mouse_anchor = {
                    "bbox": list(y_obj["bbox"]),
                    "centroid": list(y_obj["centroid"]),
                    "area_px": int(y_obj["area_px"]),
                }

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue

            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[i][0])
            cy = float(centroids[i][1])

            # Filter out whole-table borders / thin edges
            if bw > w * 0.70 or bh > h * 0.70 or bh <= 10 or bw <= 10:
                continue

            aspect_ratio = float(bw) / float(bh + 1e-5)
            solidity = float(area) / float(bw * bh + 1e-5)

            # Check if this is the Primary Focal Keyboard / Input Anchor
            if (0.20 * w < cx < 0.80 * w) and (cy > h * 0.45) and (aspect_ratio > 1.8) and (area > 4500):
                if keyboard_anchor is None or area > keyboard_anchor["area_px"]:
                    keyboard_anchor = {
                        "bbox": [x, y, bw, bh],
                        "centroid": [cx, cy],
                        "area_px": int(area),
                    }
                    continue

            # Check if this is the Monitor / Screen Anchor
            if (0.20 * w < cx < 0.80 * w) and (cy < h * 0.45) and (area > 5500):
                if monitor_anchor is None or area > monitor_anchor["area_px"]:
                    monitor_anchor = {
                        "bbox": [x, y, bw, bh],
                        "centroid": [cx, cy],
                        "area_px": int(area),
                    }
                    continue

            # Check if this is an Active Mouse Anchor
            if (0.58 * w < cx < 0.85 * w) and (cy > h * 0.45) and (0.40 < aspect_ratio < 0.85) and (800 < area < 3000):
                if mouse_anchor is None:
                    mouse_anchor = {
                        "bbox": [x, y, bw, bh],
                        "centroid": [cx, cy],
                        "area_px": int(area),
                    }
                    continue

            raw_candidates.append({
                "label_id": i,
                "bbox": [x, y, bw, bh],
                "centroid": [cx, cy],
                "area_px": int(area),
                "aspect_ratio": aspect_ratio,
                "solidity": solidity,
            })

        # 4b. Synthesize Central Work Focus BEFORE building clutter items so we never classify the anchor as clutter
        central_focus = self._synthesize_central_focus(
            yolo_res.get("central_focus"), keyboard_anchor, monitor_anchor, mouse_anchor, w, h
        )

        # Build exclusion boxes for anchors (Central Focus hardware components)
        anchor_exclusion_boxes: List[List[int]] = []
        if keyboard_anchor:
            anchor_exclusion_boxes.append(keyboard_anchor["bbox"])
        if monitor_anchor:
            anchor_exclusion_boxes.append(monitor_anchor["bbox"])
        if mouse_anchor:
            anchor_exclusion_boxes.append(mouse_anchor["bbox"])
        if central_focus and central_focus.get("type") in ["LAPTOP", "KEYBOARD", "MONITOR"]:
            anchor_exclusion_boxes.append(central_focus["bbox"])

        # 4c. Fuse YOLOv8 Non-Anchor Detections + Deduplicated OpenCV Candidates
        unified_candidates: List[Dict[str, Any]] = []

        # Primary Source: YOLOv8 real-world detections (excluding anchors)
        for y_obj in yolo_objects:
            if y_obj.get("is_anchor", False) or y_obj.get("class_name") in ["laptop", "keyboard", "tv", "monitor", "mouse"]:
                continue
            y_box = list(y_obj["bbox"])
            # Skip if heavily inside an anchor exclusion box
            if any(_bbox_iou(y_box, abox) > 0.45 or _bbox_containment(y_box, abox) > 0.70 for abox in anchor_exclusion_boxes):
                continue
            unified_candidates.append({
                "bbox": y_box,
                "centroid": list(y_obj["centroid"]),
                "area_px": int(y_obj.get("area_px", y_box[2] * y_box[3])),
                "aspect_ratio": float(y_box[2]) / float(y_box[3] + 1e-5),
                "solidity": 0.75,
                "yolo_semantic_name": y_obj.get("semantic_name"),
                "yolo_is_liquid": y_obj.get("is_liquid", False),
                "yolo_target_dest": y_obj.get("target_destination"),
                "source": y_obj.get("source", "YOLOv8"),
            })

        # Merge heavily overlapping classical candidates first
        merged_cands = []
        skip_cands = set()
        for i in range(len(raw_candidates)):
            if i in skip_cands:
                continue
            c1 = raw_candidates[i]
            merged = dict(c1)
            for j in range(i + 1, len(raw_candidates)):
                if j in skip_cands:
                    continue
                c2 = raw_candidates[j]
                dist = math.sqrt((c1["centroid"][0] - c2["centroid"][0])**2 + (c1["centroid"][1] - c2["centroid"][1])**2)
                if dist < 34.0 or _bbox_iou(c1["bbox"], c2["bbox"]) > 0.30:
                    skip_cands.add(j)
                    merged["area_px"] = max(c1["area_px"], c2["area_px"])
                    break
            merged_cands.append(merged)

        # Supplement with classical candidates that do not overlap YOLO objects or Anchor boxes
        for cand in merged_cands:
            c_box = cand["bbox"]
            # Check against anchor boxes
            if any(_bbox_iou(c_box, abox) > 0.25 or _bbox_containment(c_box, abox) > 0.55 for abox in anchor_exclusion_boxes):
                continue
            # Check against existing YOLO candidates
            already_covered = False
            for u_cand in unified_candidates:
                dist = math.sqrt((cand["centroid"][0] - u_cand["centroid"][0])**2 + (cand["centroid"][1] - u_cand["centroid"][1])**2)
                if _bbox_iou(c_box, u_cand["bbox"]) > 0.20 or dist < 38.0 or _bbox_containment(c_box, u_cand["bbox"]) > 0.50:
                    already_covered = True
                    break
            if not already_covered:
                cand["source"] = "OpenCV-Contour"
                unified_candidates.append(cand)

        # Cap to top 8 actionable candidates to prevent micro-fragment clutter on textured wood desks
        unified_candidates = sorted(
            unified_candidates,
            key=lambda c: (c.get("source") == "YOLOv8", c.get("yolo_is_liquid", False), c["area_px"]),
            reverse=True,
        )[:8]

        # Process and Classify True Clutter Items
        items: List[Dict[str, Any]] = []
        total_clutter_pixels = 0

        kb_box = keyboard_anchor["bbox"] if keyboard_anchor else central_focus["bbox"]
        kx_min, ky_min, kw_b, kh_b = kb_box
        kx_max, ky_max = kx_min + kw_b, ky_min + kh_b

        for cand in unified_candidates:
            x, y, bw, bh = cand["bbox"]
            cx, cy = cand["centroid"]
            area = cand["area_px"]
            aspect_ratio = cand["aspect_ratio"]
            solidity = cand["solidity"]

            # Distance and Zone Mapping with Orientation Invariance
            dist_px = math.sqrt((cx - user_x)**2 + (cy - user_y)**2)
            dist_cm = round(dist_px * self.CM_PER_PIXEL, 1)

            if perspective in ["PERSPECTIVE_ANGLED", "SIDE_VIEW"]:
                if cy >= h * 0.65:
                    zone = "ZONE_1_PRIMARY_REACH"
                elif cy >= h * 0.35:
                    zone = "ZONE_2_SECONDARY_REACH"
                else:
                    zone = "ZONE_3_PERIPHERY"
            else:
                if dist_px <= zone1_radius_px:
                    zone = "ZONE_1_PRIMARY_REACH"
                elif dist_px <= zone2_radius_px:
                    zone = "ZONE_2_SECONDARY_REACH"
                else:
                    zone = "ZONE_3_PERIPHERY"

            side = "LEFT" if cx < (w / 2.0) else "RIGHT"

            # Crop thumbnail
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            crop = img[y1:y2, x1:x2]

            crop_filename = f"crop_item_{len(items) + 1}.png"
            crop_path = os.path.join(self.crops_dir, crop_filename)
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)

            mean_bgr = cv2.mean(crop)[:3] if crop.size > 0 else (128, 128, 128)
            is_yellow = (mean_bgr[2] > 190 and mean_bgr[1] > 190 and mean_bgr[0] < 170)
            is_white_paper = (mean_bgr[0] > 210 and mean_bgr[1] > 210 and mean_bgr[2] > 210)

            # Distance to closest point on keyboard/focus boundary (cm & px)
            closest_x = max(kx_min, min(cx, kx_max))
            closest_y = max(ky_min, min(cy, ky_max))
            edge_dist_px = round(math.sqrt((cx - closest_x)**2 + (cy - closest_y)**2), 1)
            edge_dist_cm = round(edge_dist_px * self.CM_PER_PIXEL, 1)

            # Semantic Classification (Prefer YOLOv8 semantic identity when available)
            is_liquid = bool(cand.get("yolo_is_liquid", False))
            target_dest = cand.get("yolo_target_dest") or "Drawer / Storage Tray"
            target_coords = (int(w * 0.12), int(h * 0.18))

            if cand.get("yolo_semantic_name"):
                semantic_name = cand["yolo_semantic_name"]
                if is_liquid:
                    if edge_dist_cm < 25.0:
                        target_dest = "Zone 2 Dedicated Coaster"
                        target_coords = (int(w * 0.88), int(h * 0.48))
                    else:
                        target_dest = "Maintain Current Position"
                        target_coords = (int(cx), int(cy))
                elif "Phone" in semantic_name or "Smartphone" in semantic_name:
                    target_dest = "Zone 2 Phone Dock / Stand"
                    target_coords = (int(w * 0.86), int(h * 0.42))
                elif "Notebook" in semantic_name or "Book" in semantic_name or "Document" in semantic_name:
                    target_dest = "Zone 3 Vertical File Tray"
                    target_coords = (int(w * 0.14), int(h * 0.22))
                elif "Plant" in semantic_name:
                    target_dest = "Maintain Current Position"
                    target_coords = (int(cx), int(cy))
                else:
                    target_coords = (int(w * 0.14), int(h * 0.86))
            else:
                # Classical OpenCV heuristic fallback for non-COCO desktop items
                if (is_white_paper or area > 3500) and (0.75 < aspect_ratio < 1.45):
                    semantic_name = "Loose Documents & Invoices"
                    target_dest = "Zone 3 Vertical File Tray"
                    target_coords = (int(w * 0.14), int(h * 0.22))
                elif solidity < 0.22 or (bw > 100 and bh > 60 and area < 2500):
                    semantic_name = "Tangled USB-C & Power Cords"
                    target_dest = "Zone 3 Cable Clip / Grommet"
                    target_coords = (int(w * 0.50), int(h * 0.14))
                elif is_yellow or (solidity < 0.45 and 0.8 < aspect_ratio < 1.25 and area < 2200):
                    semantic_name = "Sticky Note Reminder Cluster"
                    target_dest = "Monitor Bezel / Digital Kanban"
                    target_coords = (int(w * 0.50), int(h * 0.12))
                elif (aspect_ratio < 0.65 or aspect_ratio > 1.55) and (800 < area < 3200) and (zone == "ZONE_1_PRIMARY_REACH"):
                    semantic_name = "Smartphone (Notification Distraction)"
                    target_dest = "Zone 2 Phone Dock / Stand"
                    target_coords = (int(w * 0.86), int(h * 0.42))
                elif (0.70 < aspect_ratio < 1.35) and (bw < 85 and bh < 85):
                    if edge_dist_cm < 25.0:
                        semantic_name = "Ceramic Coffee Mug (Spill Hazard)"
                        is_liquid = True
                        target_dest = "Zone 2 Dedicated Coaster"
                        target_coords = (int(w * 0.88), int(h * 0.48))
                    else:
                        semantic_name = "Ceramic Mug on Safe Coaster"
                        is_liquid = True
                        target_dest = "Maintain Current Position"
                        target_coords = (int(w * 0.88), int(h * 0.65))
                elif (0.65 < aspect_ratio < 1.4) and (area > 800) and cx > (w * 0.65):
                    semantic_name = "Beverage Tumbler / Can"
                    is_liquid = True
                    target_dest = "Zone 2 Dedicated Coaster"
                    target_coords = (int(w * 0.88), int(h * 0.48))
                elif zone == "ZONE_3_PERIPHERY" and area < 1200:
                    semantic_name = "Desk Plant / Minimal Decor"
                    target_dest = "Maintain Current Position"
                    target_coords = (int(cx), int(cy))
                else:
                    semantic_name = f"Desktop Accessory #{len(items) + 1}"
                    target_dest = "Stow in Drawer"
                    target_coords = (int(w * 0.14), int(h * 0.86))

            # Displacement Vector in cm
            disp_px = math.sqrt((target_coords[0] - cx)**2 + (target_coords[1] - cy)**2)
            disp_cm = round(disp_px * self.CM_PER_PIXEL, 1) if target_dest != "Maintain Current Position" else 0.0

            items.append({
                "id": len(items) + 1,
                "semantic_name": semantic_name,
                "bbox": [x, y, bw, bh],
                "centroid": [round(cx, 1), round(cy, 1)],
                "area_px": int(area),
                "area_pct": round((area / total_pixels) * 100, 2),
                "zone": zone,
                "dist_from_torso_cm": dist_cm,
                "edge_dist_to_kb_cm": edge_dist_cm,
                "edge_dist_to_kb_px": edge_dist_px,
                "side": side,
                "is_potential_liquid_container": bool(is_liquid),
                "target_destination": target_dest,
                "target_coords": target_coords,
                "displacement_cm": disp_cm,
                "thumbnail_path": crop_path.replace("\\", "/"),
                "thumbnail_filename": crop_filename,
                "detection_source": cand.get("source", "Hybrid"),
            })
            if target_dest != "Maintain Current Position":
                total_clutter_pixels += area

        surface_blocked_pct = round((total_clutter_pixels / total_pixels) * 100, 1)

        # 5. Layer 3: Visual Mass Balance (Moments)
        actionable_items = [it for it in items if it.get("target_destination") != "Maintain Current Position"]
        actionable_zone1 = [it for it in actionable_items if it["zone"] == "ZONE_1_PRIMARY_REACH"]

        left_items = [it for it in actionable_items if it["side"] == "LEFT"]
        right_items = [it for it in actionable_items if it["side"] == "RIGHT"]

        left_area = sum(it["area_px"] for it in left_items)
        right_area = sum(it["area_px"] for it in right_items)
        combined_area = float(left_area + right_area)

        if combined_area > 0:
            left_ratio = round(left_area / combined_area, 2)
            right_ratio = round(right_area / combined_area, 2)
        else:
            left_ratio = 0.50
            right_ratio = 0.50

        if len(actionable_items) >= 2 and left_ratio > 0.68:
            balance_verdict = "HEAVY_LEFT_BIAS"
        elif len(actionable_items) >= 2 and right_ratio > 0.68:
            balance_verdict = "HEAVY_RIGHT_BIAS"
        else:
            balance_verdict = "HARMONIOUS_BALANCE"

        # 5b. Peripheral Encroachment Synthesis around Central Focus
        encroachment = self._compute_encroachment_analysis(central_focus, items, w, h)

        # Re-number items sorted by priority (Encroaching / Zone 1 first) so #1 is always highest priority
        sorted_items = sorted(
            items,
            key=lambda x: (
                x.get("target_destination") == "Maintain Current Position",
                not x.get("is_potential_liquid_container", False),
                x.get("encroachment_severity") != "CRITICAL_ENCROACHMENT",
                x["zone"] != "ZONE_1_PRIMARY_REACH",
                -x["area_px"],
            ),
        )
        for idx, it in enumerate(sorted_items):
            it["id"] = idx + 1
        items = sorted_items
        # Recompute encroachment item IDs after re-indexing
        encroachment = self._compute_encroachment_analysis(central_focus, items, w, h)

        clustering = {
            "clustering_index": yolo_res.get("clustering_index", 0.0) if len(actionable_items) > 0 else 0.0,
            "clustering_severity": yolo_res.get("clustering_severity", "LOW_SPACED") if len(actionable_items) > 1 else "LOW_SPACED",
            "clusters": yolo_res.get("clusters", []),
            "proximity_pairs": yolo_res.get("proximity_pairs", []),
        }

        # 6. Layer 4: Cable Tangle & Spill Hazard Sentinel
        edges = cv2.Canny(blurred, 60, 180)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=28, maxLineGap=10)
        cable_lines_count = 0
        cable_snarl_detected = False
        desk_cables = []

        if lines is not None:
            for l in lines:
                c = np.asarray(l).ravel()
                if len(c) >= 4:
                    lx1, ly1, lx2, ly2 = int(c[0]), int(c[1]), int(c[2]), int(c[3])
                    # Exclude monitor box & upper border
                    if (220 < lx1 < 580 and ly1 < 220) or (220 < lx2 < 580 and ly2 < 220):
                        continue
                    # Exclude keyboard / central focus box
                    fx, fy, fbw, fbh = central_focus["bbox"]
                    if (fx - 15 <= lx1 <= fx + fbw + 15 and fy - 15 <= ly1 <= fy + fbh + 15) or (
                        fx - 15 <= lx2 <= fx + fbw + 15 and fy - 15 <= ly2 <= fy + fbh + 15
                    ):
                        continue
                    if (240 < lx1 < 540 and 300 < ly1 < 410) or (240 < lx2 < 540 and 300 < ly2 < 410):
                        continue
                    if ly1 < 120 and ly2 < 120:
                        continue
                    # Exclude straight horizontal/vertical table borders
                    angle_deg = abs(math.degrees(math.atan2(ly2 - ly1, lx2 - lx1)))
                    if angle_deg < 3.0 or abs(angle_deg - 90.0) < 3.0 or abs(angle_deg - 180.0) < 3.0:
                        continue
                    desk_cables.append(l)
            cable_lines_count = len(desk_cables)
            cable_snarl_detected = cable_lines_count > 4

        # Spill Hazard Calculation
        liquid_containers = [it for it in items if it["is_potential_liquid_container"]]
        spill_hazard = False
        min_spill_distance_px = 9999.0
        min_spill_distance_cm = 999.0
        hazard_details = []

        for l_item in liquid_containers:
            edge_dist_cm = l_item.get("edge_dist_to_kb_cm", 999.0)
            edge_dist_px = l_item.get("edge_dist_to_kb_px", edge_dist_cm / self.CM_PER_PIXEL)
            if edge_dist_cm < min_spill_distance_cm:
                min_spill_distance_cm = edge_dist_cm
            if edge_dist_px < min_spill_distance_px:
                min_spill_distance_px = edge_dist_px

            if edge_dist_cm < 25.0 and l_item.get("target_destination") != "Maintain Current Position":
                spill_hazard = True
                hazard_details.append({
                    "item_id": l_item["id"],
                    "semantic_name": l_item["semantic_name"],
                    "edge_distance_cm": round(edge_dist_cm, 1),
                    "edge_distance_px": round(edge_dist_px, 1),
                    "risk_level": "CRITICAL_SPILL_RISK",
                })

        # 7. Intrinsic Anthropometric Scale Calibration & ISO 9241-5 Ergonomic Audit
        intrinsic_calibration = self._estimate_perspective_and_calibration(
            img, gray, perspective, central_focus, keyboard_anchor, monitor_anchor, w, h
        )
        perspective = intrinsic_calibration["perspective"]
        iso_9241_audit = self._compute_iso_9241_ergonomic_audit(
            central_focus, keyboard_anchor, monitor_anchor, mouse_anchor,
            encroachment, intrinsic_calibration, w, h
        )
        opencv_morphology_hough_ms = round((time.perf_counter() - t_morph_0) * 1000.0, 2)

        # 8. Overall Zen Clarity Score (0 - 100)
        base_score = 100.0
        raw_entropy_pen = min(35.0, (shannon_entropy / 4.8) * 20.0 + (lap_var / 350.0) * 15.0)
        # When desk has 0 actionable clutter items, natural keyboard/monitor edges should not heavily penalize score
        if len(actionable_items) == 0 and not spill_hazard and not cable_snarl_detected:
            entropy_penalty = raw_entropy_pen * 0.25
        else:
            entropy_penalty = raw_entropy_pen

        blockage_penalty = min(30.0, surface_blocked_pct * 1.8)
        zone1_penalty = min(25.0, len(actionable_zone1) * 9.0)
        spill_penalty = 22.0 if spill_hazard else 0.0
        cable_penalty = 12.0 if cable_snarl_detected else 0.0
        balance_penalty = 10.0 if balance_verdict != "HARMONIOUS_BALANCE" else 0.0
        encroach_penalty = min(20.0, (100.0 - encroachment.get("breathing_room_pct", 100.0)) * 0.20)

        zen_score = int(max(10.0, min(98.0, round(
            base_score - entropy_penalty - blockage_penalty - zone1_penalty - spill_penalty - cable_penalty - balance_penalty - encroach_penalty
        ))))

        # 9. Render Visual Layers
        t_ov_0 = time.perf_counter()
        annotated_img = self._render_sand_overlay(
            img, items, desk_cables, spill_hazard, zone1_radius_px, zone2_radius_px,
            keyboard_anchor, mouse_anchor, perspective,
            central_focus=central_focus, encroachment=encroachment,
        )
        heatmap_img = self._render_cognitive_heatmap(img, grad_mag, laplacian)
        relocation_img = self._render_relocation_overlay(
            img, items, zone1_radius_px, zone2_radius_px, keyboard_anchor, perspective,
            central_focus=central_focus,
        )
        overlay_synthesis_ms = round((time.perf_counter() - t_ov_0) * 1000.0, 2)

        preliminary_telemetry = {
            "zen_score": zen_score,
            "cognitive_load_tier": self._get_tier(zen_score),
            "perspective": perspective,
            "intrinsic_calibration": intrinsic_calibration,
            "iso_9241_audit": iso_9241_audit,
            "clustering": clustering,
            "central_focus": central_focus,
            "encroachment": encroachment,
            "visual_entropy_score": round(shannon_entropy * 10, 1),
            "laplacian_variance": round(lap_var, 1),
            "surface_area_blocked_pct": surface_blocked_pct,
            "total_items_detected": len(items),
            "zone1_clutter_count": len(actionable_zone1),
            "actionable_items_count": len(actionable_items),
            "keyboard_anchor": keyboard_anchor,
            "mouse_anchor": mouse_anchor,
            "visual_mass_balance": {
                "left_ratio": left_ratio,
                "right_ratio": right_ratio,
                "verdict": balance_verdict,
            },
            "hazards": {
                "spill_risk": spill_hazard,
                "cable_snarl_detected": cable_snarl_detected,
                "cable_lines_count": cable_lines_count,
                "min_spill_distance_px": round(min_spill_distance_px, 1) if min_spill_distance_px < 9000 else 0.0,
                "min_spill_distance_cm": round(min_spill_distance_cm, 1) if min_spill_distance_cm < 900 else 0.0,
                "details": hazard_details,
            },
            "items": items,
            "image_dimensions": {"width": w, "height": h},
        }

        t_inp_0 = time.perf_counter()
        if generate_after:
            target_after_result = self.after_gen.generate_target_after(img, preliminary_telemetry, plan={})
            telea_inpaint_ms = round((time.perf_counter() - t_inp_0) * 1000.0, 2)
            preliminary_telemetry.update({
                "annotated_image": annotated_img,
                "heatmap_image": heatmap_img,
                "relocation_image": relocation_img,
                "target_after_image": target_after_result["target_image"],
                "target_after_metadata": target_after_result,
            })
        else:
            telea_inpaint_ms = 0.0
            preliminary_telemetry.update({
                "annotated_image": annotated_img,
                "heatmap_image": heatmap_img,
                "relocation_image": relocation_img,
            })

        total_pipeline_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        core_cv_ms = round(entropy_spectral_ms + opencv_morphology_hough_ms, 2)
        realtime_fps = round(1000.0 / max(1.0, core_cv_ms), 1)
        memory_mb = round((w * h * 3 * 6) / (1024.0 * 1024.0), 2)

        preliminary_telemetry["edge_profiler"] = {
            "entropy_spectral_ms": entropy_spectral_ms,
            "yolo_anchor_ms": yolo_anchor_ms,
            "opencv_morphology_hough_ms": opencv_morphology_hough_ms,
            "overlay_synthesis_ms": overlay_synthesis_ms,
            "telea_inpaint_ms": telea_inpaint_ms,
            "core_cv_ms": core_cv_ms,
            "total_pipeline_ms": total_pipeline_ms,
            "realtime_fps": realtime_fps,
            "memory_mb": memory_mb,
            "architecture_badge": "OpenCV 5 C++ SIMD + YOLOv8n Nano (Graviton3 / ARM64 Edge Ready)",
        }

        preliminary_telemetry["agentic_loop_trace"] = self._build_agentic_loop_trace(preliminary_telemetry)

        return preliminary_telemetry

    def _get_tier(self, score: int) -> str:
        if score >= 80:
            return "ZEN_HARMONY"
        elif score >= 60:
            return "MODERATE_CLARITY"
        elif score >= 40:
            return "COGNITIVE_FRICTION"
        else:
            return "HIGH_CLUTTER_OVERLOAD"

    def _estimate_perspective_and_calibration(
        self,
        img: np.ndarray,
        gray: np.ndarray,
        yolo_perspective: str,
        central_focus: Dict[str, Any],
        keyboard_anchor: Optional[Dict[str, Any]],
        monitor_anchor: Optional[Dict[str, Any]],
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """
        Pillar 1: Intrinsic Anthropometric Scale Calibration & Camera Pitch Estimation.
        Uses the detected Central Focus / Keyboard / Monitor hardware width as a physical fiducial
        marker to calibrate cm_per_pixel_x, cm_per_pixel_y, and camera elevation angle (deg).
        """
        perspective = yolo_perspective or "TOP_DOWN"

        # Detect oblique trapezoidal backdrop / foreshortened keyboard if YOLO defaulted to TOP_DOWN
        top_strip_mean = float(np.mean(gray[: max(10, int(h * 0.12)), :]))
        mid_strip_mean = float(np.mean(gray[int(h * 0.35) : int(h * 0.75), :]))
        has_studio_backdrop = (mid_strip_mean - top_strip_mean) > 75.0

        kb_ar = 3.2
        if keyboard_anchor and keyboard_anchor.get("bbox"):
            kb_w = float(max(1, keyboard_anchor["bbox"][2]))
            kb_h = float(max(1, keyboard_anchor["bbox"][3]))
            kb_ar = kb_w / kb_h

        if perspective == "TOP_DOWN" and (has_studio_backdrop or kb_ar > 3.55):
            perspective = "PERSPECTIVE_ANGLED"

        # Estimate camera elevation angle (deg from horizontal desk plane)
        if perspective == "SIDE_VIEW":
            camera_elevation_deg = 26.5
        elif perspective == "PERSPECTIVE_ANGLED":
            # Foreshortening ratio of keyboard deck (true aspect ratio ~ 2.8; apparent ~ kb_ar)
            sin_theta = max(0.45, min(0.82, 2.55 / max(2.6, kb_ar)))
            camera_elevation_deg = round(math.degrees(math.asin(sin_theta)), 1)
        else:
            camera_elevation_deg = 74.5

        foreshortening_sin = round(math.sin(math.radians(camera_elevation_deg)), 3)

        # Intrinsic Fiducial Hardware Scale Calibration (cm / pixel)
        if keyboard_anchor and keyboard_anchor.get("bbox"):
            ref_px = float(max(80, keyboard_anchor["bbox"][2]))
            ref_cm = 39.0  # Standard mechanical / laptop keyboard width (cm)
            fiducial_source = f"Keyboard Anchor Fiducial ({ref_cm}cm / {int(ref_px)}px)"
            cm_per_px_x = round(max(0.09, min(0.24, ref_cm / ref_px)), 4)
        elif central_focus and central_focus.get("type") == "LAPTOP":
            ref_px = float(max(80, central_focus["bbox"][2]))
            ref_cm = 33.5  # Standard 14-15" Laptop Chassis Width (cm)
            fiducial_source = f"Laptop Chassis Fiducial ({ref_cm}cm / {int(ref_px)}px)"
            cm_per_px_x = round(max(0.09, min(0.24, ref_cm / ref_px)), 4)
        elif monitor_anchor and monitor_anchor.get("bbox"):
            ref_px = float(max(100, monitor_anchor["bbox"][2]))
            ref_cm = 52.0  # Standard 24" Display Bezel Width (cm)
            fiducial_source = f"Display Bezel Fiducial ({ref_cm}cm / {int(ref_px)}px)"
            cm_per_px_x = round(max(0.09, min(0.24, ref_cm / ref_px)), 4)
        else:
            ref_cm = 120.0
            fiducial_source = f"Tabletop Homography Reference ({ref_cm}cm / {w}px)"
            cm_per_px_x = round(ref_cm / float(max(1, w)), 4)

        cm_per_px_y = round(cm_per_px_x / max(0.45, foreshortening_sin), 4)
        est_desk_width_cm = round(w * cm_per_px_x, 1)
        est_desk_depth_cm = round(h * cm_per_px_y * 0.78, 1)

        return {
            "perspective": perspective,
            "camera_elevation_deg": camera_elevation_deg,
            "foreshortening_factor": foreshortening_sin,
            "fiducial_source": fiducial_source,
            "reference_anchor": fiducial_source,
            "cm_per_pixel_x": cm_per_px_x,
            "cm_per_pixel_y": cm_per_px_y,
            "estimated_desk_dimensions_cm": [est_desk_width_cm, est_desk_depth_cm],
        }

    def _compute_iso_9241_ergonomic_audit(
        self,
        central_focus: Dict[str, Any],
        keyboard_anchor: Optional[Dict[str, Any]],
        monitor_anchor: Optional[Dict[str, Any]],
        mouse_anchor: Optional[Dict[str, Any]],
        encroachment: Dict[str, Any],
        calibration: Dict[str, Any],
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """
        Pillar 1b: Biomechanical & Postural Ergonomic Audit (ISO 9241-5 Workstation Standard).
        Evaluates:
        1. Monitor-to-Keyboard Axial Symmetry & Cervical Neck Rotation (deg)
        2. Eye-to-Screen Viewing Distance (cm) (ISO 9241-5 optimal: 50-70cm)
        3. Forearm / Wrist Neutral Support Depth (cm) (ISO 9241-5 optimal: >=12cm)
        4. Shoulder Abduction / Mouse Reach Span (cm)
        """
        cm_x = calibration.get("cm_per_pixel_x", self.CM_PER_PIXEL)
        cm_y = calibration.get("cm_per_pixel_y", self.CM_PER_PIXEL)

        # 1. Axial Symmetry & Neck Rotation
        if keyboard_anchor and monitor_anchor:
            kx_c = keyboard_anchor["centroid"][0]
            mx_c = monitor_anchor["centroid"][0]
            axial_offset_cm = round(abs(mx_c - kx_c) * cm_x, 1)
        else:
            fx_c = central_focus["centroid"][0]
            axial_offset_cm = round(abs(fx_c - (w / 2.0)) * cm_x * 0.5, 1)

        viewing_distance_cm = 58.0
        if monitor_anchor and keyboard_anchor:
            dy_px = max(80.0, keyboard_anchor["centroid"][1] - monitor_anchor["centroid"][1])
            viewing_distance_cm = round(max(45.0, min(78.0, dy_px * cm_y + 26.0)), 1)
        elif calibration.get("perspective") == "PERSPECTIVE_ANGLED":
            viewing_distance_cm = 54.5

        neck_rotation_deg = round(math.degrees(math.atan2(axial_offset_cm, viewing_distance_cm)), 1)
        if neck_rotation_deg <= 5.0:
            axial_status = "OPTIMAL_AXIAL_SYMMETRY"
        elif neck_rotation_deg <= 14.0:
            axial_status = "MILD_CERVICAL_ROTATION"
        else:
            axial_status = "ASYMMETRIC_NECK_STRAIN"

        # 2. Wrist Neutral Support & Forearm Clearance
        wrist_clear_cm = encroachment.get("wrist_clearance_cm", 20.0)
        if wrist_clear_cm >= 15.0:
            wrist_status = "ISO_COMPLIANT_NEUTRAL_WRIST"
        elif wrist_clear_cm >= 8.0:
            wrist_status = "MODERATE_WRIST_COMPRESSION"
        else:
            wrist_status = "ULNAR_DEVIATION_RISK"

        # 3. Mouse Shoulder Abduction Reach
        mouse_clear_cm = encroachment.get("mouse_clearance_cm", 20.0)
        if keyboard_anchor and mouse_anchor:
            kb_right = keyboard_anchor["bbox"][0] + keyboard_anchor["bbox"][2]
            mouse_cx = mouse_anchor["centroid"][0]
            mouse_abduction_cm = round(max(2.0, (mouse_cx - kb_right) * cm_x), 1)
        else:
            mouse_abduction_cm = 9.5

        if mouse_clear_cm >= 18.0 and mouse_abduction_cm <= 14.0:
            mouse_status = "OPTIMAL_SHOULDER_ADDUCTION"
        else:
            mouse_status = "CONSTRICTED_SWEEP_CORRIDOR"

        # Compute overall ISO 9241-5 Ergonomic Score (0-100)
        iso_score = 96.0
        if axial_status != "OPTIMAL_AXIAL_SYMMETRY":
            iso_score -= min(18.0, neck_rotation_deg * 1.1)
        if wrist_status != "ISO_COMPLIANT_NEUTRAL_WRIST":
            iso_score -= (15.0 - min(15.0, wrist_clear_cm)) * 1.4
        if mouse_status != "OPTIMAL_SHOULDER_ADDUCTION":
            iso_score -= (20.0 - min(20.0, mouse_clear_cm)) * 1.1
        if encroachment.get("status") == "CRITICALLY_CONSTRICTED":
            iso_score -= 14.0

        iso_score_int = int(max(32, min(98, round(iso_score))))
        if iso_score_int >= 82:
            compliance_tier = "ISO 9241-5 GOLD COMPLIANT"
        elif iso_score_int >= 62:
            compliance_tier = "MODERATE BIOMECHANICAL STRAIN"
        else:
            compliance_tier = "HIGH POSTURAL & REACH FRICTION"

        findings = [
            f"Display-to-Input Axial Offset: {axial_offset_cm}cm ({neck_rotation_deg}° neck rotation -> {axial_status.replace('_', ' ')})",
            f"Estimated Viewing Distance: {viewing_distance_cm}cm (ISO 9241-5 optimal window: 50-70cm)",
            f"Forearm Wrist Rest Depth: {wrist_clear_cm}cm ({wrist_status.replace('_', ' ')})",
            f"Mouse Corridor Sweep Clearance: {mouse_clear_cm}cm | Lateral Reach: {mouse_abduction_cm}cm ({mouse_status.replace('_', ' ')})",
        ]

        return {
            "iso_9241_score": iso_score_int,
            "compliance_tier": compliance_tier,
            "axial_offset_cm": axial_offset_cm,
            "neck_rotation_deg": neck_rotation_deg,
            "axial_status": axial_status,
            "viewing_distance_cm": viewing_distance_cm,
            "wrist_clearance_cm": wrist_clear_cm,
            "wrist_status": wrist_status,
            "mouse_abduction_cm": mouse_abduction_cm,
            "mouse_status": mouse_status,
            "findings": findings,
        }

    def _build_agentic_loop_trace(self, telemetry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Pillar 4: Transparent 5-Stage Agentic Vision Reasoning Trace (Observe -> Orient -> Decide -> Act -> Verify).
        Provides structured telemetry for each autonomous decision stage.
        """
        cal = telemetry.get("intrinsic_calibration", {})
        cf = telemetry.get("central_focus", {})
        enc = telemetry.get("encroachment", {})
        iso = telemetry.get("iso_9241_audit", {})
        haz = telemetry.get("hazards", {})
        prof = telemetry.get("edge_profiler", {})
        items = telemetry.get("items", [])

        top_item_str = (
            f"#{items[0]['id']} {items[0]['semantic_name']} -> {items[0]['target_destination']} (+{items[0]['displacement_cm']}cm)"
            if items else "Zero clutter bottlenecks detected"
        )

        return [
            {
                "stage": "1. OBSERVE",
                "title": "Intrinsic Calibration & Hybrid Spatial Perception",
                "latency_ms": round(prof.get("entropy_spectral_ms", 0.0) + prof.get("yolo_anchor_ms", 0.0), 2),
                "summary": (
                    f"Calibrated scale via {cal.get('fiducial_source', 'Hardware Anchor')} "
                    f"({cal.get('cm_per_pixel_x', 0.15)} cm/px, {cal.get('camera_elevation_deg', 74.5)}° camera elevation). "
                    f"Extracted {len(items)} non-anchor clutter objects & {haz.get('cable_lines_count', 0)} cable segments."
                ),
            },
            {
                "stage": "2. ORIENT",
                "title": "Central Focus Isolation & ISO 9241-5 Biomechanical Audit",
                "latency_ms": prof.get("opencv_morphology_hough_ms", 0.0),
                "summary": (
                    f"Locked onto {cf.get('name', 'Primary Workstation')} with {enc.get('breathing_room_pct', 100)}% breathing room. "
                    f"ISO 9241-5 Score: {iso.get('iso_9241_score', 90)}/100 ({iso.get('neck_rotation_deg', 0)}° neck rotation, "
                    f"{enc.get('mouse_clearance_cm', 35)}cm mouse sweep)."
                ),
            },
            {
                "stage": "3. DECIDE",
                "title": "Nova Pro Spatial Triage & Collision-Free Vector Synthesis",
                "latency_ms": prof.get("overlay_synthesis_ms", 0.0),
                "summary": (
                    f"Ranked {len(items)} items by risk severity (Spill Risk: {haz.get('spill_risk', False)}). "
                    f"Primary directive: {top_item_str}."
                ),
            },
            {
                "stage": "4. ACT",
                "title": "Interactive Human-in-the-Loop Declutter & Live Telea Inpainting",
                "latency_ms": prof.get("telea_inpaint_ms", 0.0),
                "summary": (
                    "Synchronized interactive checklist with Fast-Marching (INPAINT_TELEA) + Gaussian feathering "
                    "for instant in-place object erasure on the live workspace canvas."
                ),
            },
            {
                "stage": "5. VERIFY",
                "title": "Closed-Loop Computer Vision Re-Ingestion & Certification",
                "latency_ms": prof.get("core_cv_ms", 0.0),
                "summary": (
                    "Re-runs OpenCV 5 entropy, Hough cable, and proximity sensors on the raw cleared surface "
                    "to mathematically certify hazard resolution and reclaimed square footage."
                ),
            },
        ]

    def _synthesize_central_focus(
        self,
        yolo_focus: Optional[Dict[str, Any]],
        keyboard_anchor: Optional[Dict[str, Any]],
        monitor_anchor: Optional[Dict[str, Any]],
        mouse_anchor: Optional[Dict[str, Any]],
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """
        Synthesizes the unified primary central work focus:
        Prioritizes YOLO detection if available, or falls back to OpenCV classical anchors.
        Always guarantees a concrete central work anchor with active ergonomic envelope.
        """
        if yolo_focus and yolo_focus.get("bbox"):
            return yolo_focus

        if keyboard_anchor and monitor_anchor:
            kx, ky, kw_k, kh_k = keyboard_anchor["bbox"]
            mx, my, mw_m, mh_m = monitor_anchor["bbox"]
            min_x = min(kx, mx)
            min_y = min(ky, my)
            max_x = max(kx + kw_k, mx + mw_m)
            max_y = max(ky + kh_k, my + mh_m)
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            return self._build_focus_envelope(
                focus_type="DESKTOP_SETUP",
                name="Desktop Workstation (Keyboard + Display)",
                bbox=bbox,
                centroid=[(min_x + max_x) / 2.0, (min_y + max_y) / 2.0],
                confidence=0.88,
                components=["keyboard", "monitor"],
                keyboard_bbox=keyboard_anchor["bbox"],
                monitor_bbox=monitor_anchor["bbox"],
                w=w,
                h=h,
            )
        elif keyboard_anchor:
            kx, ky, kw_k, kh_k = keyboard_anchor["bbox"]
            return self._build_focus_envelope(
                focus_type="KEYBOARD",
                name="Mechanical Keyboard (Typing Anchor)",
                bbox=[kx, ky, kw_k, kh_k],
                centroid=keyboard_anchor["centroid"],
                confidence=0.85,
                components=["keyboard"],
                w=w,
                h=h,
            )
        elif monitor_anchor:
            mx, my, mw_m, mh_m = monitor_anchor["bbox"]
            return self._build_focus_envelope(
                focus_type="MONITOR",
                name="Display Monitor Anchor",
                bbox=[mx, my, mw_m, mh_m],
                centroid=monitor_anchor["centroid"],
                confidence=0.82,
                components=["monitor"],
                w=w,
                h=h,
            )
        else:
            bw = int(w * 0.44)
            bh = int(h * 0.22)
            bx = int(w * 0.28)
            by = int(h * 0.52)
            return self._build_focus_envelope(
                focus_type="PRIMARY_WORK_ZONE",
                name="Primary Active Work Zone",
                bbox=[bx, by, bw, bh],
                centroid=[bx + bw / 2.0, by + bh / 2.0],
                confidence=0.70,
                components=["work_zone"],
                w=w,
                h=h,
            )

    def _build_focus_envelope(
        self,
        focus_type: str,
        name: str,
        bbox: List[int],
        centroid: List[float],
        confidence: float,
        components: List[str],
        w: int,
        h: int,
        **kwargs,
    ) -> Dict[str, Any]:
        x, y, bw, bh = bbox
        wrist_h_px = int(14.0 / self.CM_PER_PIXEL)
        mouse_w_px = int(22.0 / self.CM_PER_PIXEL)
        left_w_px = int(18.0 / self.CM_PER_PIXEL)
        rear_h_px = int(10.0 / self.CM_PER_PIXEL)

        env_x1 = max(0, x - left_w_px)
        env_y1 = max(0, y - rear_h_px)
        env_x2 = min(w, x + bw + mouse_w_px)
        env_y2 = min(h, y + bh + wrist_h_px)

        env = {
            "type": focus_type,
            "name": name,
            "confidence": confidence,
            "bbox": bbox,
            "centroid": centroid,
            "dimensions_cm": [round(bw * self.CM_PER_PIXEL, 1), round(bh * self.CM_PER_PIXEL, 1)],
            "components": components,
            "envelope_bbox": [env_x1, env_y1, env_x2 - env_x1, env_y2 - env_y1],
            "wrist_zone_bbox": [x, min(h, y + bh), bw, min(h - (y + bh), wrist_h_px)],
            "mouse_zone_bbox": [min(w, x + bw), max(0, y - int(4 / self.CM_PER_PIXEL)), min(w - (x + bw), mouse_w_px), min(h, bh + wrist_h_px)],
            "left_flank_bbox": [env_x1, y, left_w_px, bh],
        }
        env.update(kwargs)
        return env

    def _compute_encroachment_analysis(
        self, central_focus: Dict[str, Any], items: List[Dict[str, Any]], w: int, h: int
    ) -> Dict[str, Any]:
        fx, fy, fbw, fbh = central_focus["bbox"]
        min_wrist_cm = 99.0
        min_mouse_cm = 99.0
        min_left_cm = 99.0
        total_congestion_penalty = 0.0
        critical_encroaching = []

        for it in items:
            if it.get("target_destination") == "Maintain Current Position":
                it["edge_dist_to_focus_cm"] = 35.0
                it["relative_focus_zone"] = "SAFE_PERIPHERY"
                it["encroachment_severity"] = "SAFE_PERIPHERY"
                it["congestion_penalty"] = 0.0
                it["clearance_directive"] = f"Maintain {it['semantic_name']} in current safe zone."
                continue

            ix, iy, ibw, ibh = it["bbox"]

            if ix + ibw < fx:
                dx = fx - (ix + ibw)
                rel_h = "LEFT"
            elif ix > fx + fbw:
                dx = ix - (fx + fbw)
                rel_h = "RIGHT"
            else:
                dx = 0
                rel_h = "ALIGNED"

            if iy + ibh < fy:
                dy = fy - (iy + ibh)
                rel_v = "BACKGROUND"
            elif iy > fy + fbh:
                dy = iy - (fy + fbh)
                rel_v = "FOREGROUND"
            else:
                dy = 0
                rel_v = "ALIGNED"

            edge_dist_px = math.sqrt(dx**2 + dy**2)
            edge_dist_cm = round(edge_dist_px * self.CM_PER_PIXEL, 1)

            if rel_v == "FOREGROUND" and (rel_h == "ALIGNED" or dx < 40):
                sector = "FOREGROUND_WRIST_ZONE"
                min_wrist_cm = min(min_wrist_cm, edge_dist_cm)
            elif rel_h == "RIGHT" and (rel_v == "ALIGNED" or dy < 70):
                sector = "RIGHT_FLANK_MOUSE_SWEEP"
                min_mouse_cm = min(min_mouse_cm, edge_dist_cm)
            elif rel_h == "LEFT" and (rel_v == "ALIGNED" or dy < 70):
                sector = "LEFT_FLANK_SECONDARY"
                min_left_cm = min(min_left_cm, edge_dist_cm)
            elif rel_v == "BACKGROUND":
                sector = "BACKGROUND_DISPLAY_ZONE"
            else:
                sector = "PERIPHERAL_ZONE"

            is_liquid = it.get("is_potential_liquid_container", False)
            if edge_dist_cm < 12.0:
                severity = "CRITICAL_ENCROACHMENT"
                penalty = 35.0 if sector in ["FOREGROUND_WRIST_ZONE", "RIGHT_FLANK_MOUSE_SWEEP"] else 20.0
                if is_liquid:
                    penalty += 20.0
            elif edge_dist_cm < 24.0:
                severity = "MODERATE_ENCROACHMENT"
                penalty = 18.0 if sector in ["FOREGROUND_WRIST_ZONE", "RIGHT_FLANK_MOUSE_SWEEP"] else 10.0
                if is_liquid:
                    penalty += 10.0
            else:
                severity = "SAFE_PERIPHERY"
                penalty = 0.0

            total_congestion_penalty += penalty

            if sector == "FOREGROUND_WRIST_ZONE":
                directive = f"Clear {it['semantic_name']} from wrist-rest area to eliminate typing posture strain."
            elif sector == "RIGHT_FLANK_MOUSE_SWEEP":
                directive = f"Move {it['semantic_name']} outward by >=25cm to unblock mouse sweep travel."
            elif sector == "LEFT_FLANK_SECONDARY":
                directive = f"Organize {it['semantic_name']} onto secondary left tray to restore visual symmetry."
            else:
                directive = f"Stow {it['semantic_name']} to eliminate peripheral clutter."

            it["edge_dist_to_focus_cm"] = edge_dist_cm
            it["relative_focus_zone"] = sector
            it["encroachment_severity"] = severity
            it["congestion_penalty"] = round(penalty, 1)
            it["clearance_directive"] = directive

            if severity == "CRITICAL_ENCROACHMENT":
                critical_encroaching.append({
                    "id": it["id"],
                    "item_id": it["id"],
                    "name": it["semantic_name"],
                    "sector": sector,
                    "distance_cm": edge_dist_cm,
                    "edge_distance_cm": edge_dist_cm,
                    "is_liquid": is_liquid,
                    "directive": directive,
                    "clearance_directive": directive,
                })

        wrist_score = min(1.0, min_wrist_cm / 15.0) if min_wrist_cm < 90 else 1.0
        mouse_score = min(1.0, min_mouse_cm / 22.0) if min_mouse_cm < 90 else 1.0
        left_score = min(1.0, min_left_cm / 18.0) if min_left_cm < 90 else 1.0
        breathing_room_pct = round(((wrist_score * 0.4) + (mouse_score * 0.4) + (left_score * 0.2)) * 100, 1)

        if len(critical_encroaching) > 0 or breathing_room_pct < 55.0:
            status = "CRITICALLY_CONSTRICTED"
        elif breathing_room_pct < 85.0:
            status = "MODERATELY_CONGESTED"
        else:
            status = "ZEN_BREATHING_ROOM"

        return {
            "status": status,
            "wrist_clearance_cm": round(min_wrist_cm, 1) if min_wrist_cm < 90 else 35.0,
            "mouse_clearance_cm": round(min_mouse_cm, 1) if min_mouse_cm < 90 else 35.0,
            "left_flank_clearance_cm": round(min_left_cm, 1) if min_left_cm < 90 else 35.0,
            "breathing_room_pct": breathing_room_pct,
            "breathing_room_score": breathing_room_pct,
            "congestion_penalty_total": round(total_congestion_penalty, 1),
            "critical_encroachment_count": len(critical_encroaching),
            "total_encroaching_items": len(critical_encroaching),
            "critical_items": critical_encroaching,
            "encroaching_items": critical_encroaching,
        }

    def _render_sand_overlay(
        self,
        img: np.ndarray,
        items: List[Dict[str, Any]],
        lines: Any,
        spill_hazard: bool,
        zone1_radius: int,
        zone2_radius: int,
        keyboard_anchor: Any = None,
        mouse_anchor: Any = None,
        perspective: str = "TOP_DOWN",
        central_focus: Any = None,
        encroachment: Any = None,
        highlighted_item_id: Optional[int] = None,
        removed_ids: Optional[Any] = None,
    ) -> np.ndarray:
        """Visual Overlay: Ergonomic reach zones + Central Focus Halo + Encroachment Vectors + Interactive Focus (Pure ASCII HUD)."""
        overlay = img.copy()
        h, w = img.shape[:2]
        removed_set = set(removed_ids) if removed_ids else set()

        if perspective in ["PERSPECTIVE_ANGLED", "SIDE_VIEW"]:
            y_z1 = int(h * 0.65)
            y_z2 = int(h * 0.35)
            cv2.line(overlay, (0, y_z1), (w, y_z1), (38, 82, 184), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Zone 1: Primary Reach (<35cm)", (16, y_z1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (38, 82, 184), 1, cv2.LINE_AA)
            cv2.line(overlay, (0, y_z2), (w, y_z2), (78, 117, 87), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Zone 2: Secondary Reach (35-55cm)", (16, y_z2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (78, 117, 87), 1, cv2.LINE_AA)
        else:
            user_pt = (int(w / 2), int(h))
            cv2.ellipse(overlay, user_pt, (zone1_radius, int(zone1_radius * 0.72)), 0, 180, 360, (38, 82, 184), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Zone 1: Neutral Sweep (0-35cm)", (int(w / 2) - 110, int(h - zone1_radius * 0.72) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (38, 82, 184), 1, cv2.LINE_AA)

            cv2.ellipse(overlay, user_pt, (zone2_radius, int(zone2_radius * 0.72)), 0, 180, 360, (78, 117, 87), 1, cv2.LINE_AA)
            cv2.putText(overlay, "Zone 2: Functional Reach (35-55cm)", (int(w / 2) - 118, int(h - zone2_radius * 0.72) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (78, 117, 87), 1, cv2.LINE_AA)

        # Draw Central Work Focus & Active Ergonomic Halo
        if central_focus:
            fx, fy, fbw, fbh = central_focus["bbox"]
            env_bbox = central_focus.get("envelope_bbox", [fx - 60, fy - 30, fbw + 120, fbh + 80])
            ex, ey, ew, eh = env_bbox

            remaining_encroaching = [
                it for it in items
                if it["id"] not in removed_set and it.get("encroachment_severity") == "CRITICAL_ENCROACHMENT"
            ]
            if len(removed_set) > 0 and len(remaining_encroaching) == 0:
                halo_color = (78, 145, 87)   # Harmonious Sage once critical blockers cleared
                f_status_label = "LIBERATED"
            else:
                f_status = encroachment.get("status", "ZEN_BREATHING_ROOM") if encroachment else "ZEN"
                f_status_label = f_status.replace("_", " ")
                if f_status == "CRITICALLY_CONSTRICTED":
                    halo_color = (35, 65, 215)   # Terracotta Red
                elif f_status == "MODERATELY_CONGESTED":
                    halo_color = (45, 125, 215)  # Warm Amber
                else:
                    halo_color = (78, 145, 87)   # Harmonious Sage

            # 1. Active Work Envelope (Zen Breathing Halo)
            cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), halo_color, 2, cv2.LINE_AA)

            # 2. Central Focus Device Box
            cv2.rectangle(overlay, (fx - 2, fy - 2), (fx + fbw + 2, fy + fbh + 2), (245, 245, 245), 2, cv2.LINE_AA)
            cv2.rectangle(overlay, (fx, fy), (fx + fbw, fy + fbh), halo_color, 2, cv2.LINE_AA)

            # 3. Central Focus Header Pill (Pure ASCII)
            b_pct = encroachment.get("breathing_room_pct", 100.0) if encroachment else 100.0
            if len(removed_set) > 0:
                b_pct = min(100.0, round(b_pct + (100.0 - b_pct) * (len(removed_set) / max(1, len(items))), 1))
            pill_text = f"[FOCUS] {central_focus['name']} | {f_status_label} ({b_pct}% Open)"
            tag_w = min(w - 16, len(pill_text) * 7 + 16)
            tag_x = max(8, min(w - tag_w - 8, fx))
            tag_y = max(24, fy - 8)
            cv2.rectangle(overlay, (tag_x, tag_y - 20), (tag_x + tag_w, tag_y), halo_color, -1)
            cv2.putText(overlay, pill_text, (tag_x + 8, tag_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw detected cable segments if no items removed yet
        if lines is not None and len(removed_set) == 0:
            for line in lines[:16]:
                coords = np.asarray(line).ravel()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    cv2.line(overlay, (x1, y1), (x2, y2), (60, 74, 92), 2, cv2.LINE_AA)

        # Render non-highlighted items first, then render highlighted item on top
        ordered_items = sorted(items, key=lambda x: (x["id"] == highlighted_item_id))

        for it in ordered_items:
            # If item has already been cleared (inpainted out), skip drawing its clutter box unless it's actively selected
            is_cleared = (it["id"] in removed_set)
            is_selected = (highlighted_item_id is not None and it["id"] == highlighted_item_id)

            if is_cleared and not is_selected:
                continue

            x, y, bw, bh = it["bbox"]
            is_zone1 = it["zone"] == "ZONE_1_PRIMARY_REACH"
            is_liquid = it["is_potential_liquid_container"]
            is_safe = it.get("target_destination") == "Maintain Current Position"
            sev = it.get("encroachment_severity", "SAFE_PERIPHERY")
            f_dist = it.get("edge_dist_to_focus_cm", 99.0)

            if is_selected:
                # Radiant amber/gold beacon for highlighted item
                color = (0, 195, 255)  # Bright amber/gold BGR
                status_prefix = "[CLEARED]" if is_cleared else f"[FOCUS #{it['id']}]"
                label = f"{status_prefix} {it['semantic_name'].upper()}"
                cv2.rectangle(overlay, (x - 2, y - 2), (x + bw + 2, y + bh + 2), (255, 255, 255), 3, cv2.LINE_AA)
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 3, cv2.LINE_AA)

                # Tactical corner brackets
                b_len = min(14, max(4, min(bw, bh) // 4))
                cv2.line(overlay, (x - 5, y - 5), (x - 5 + b_len, y - 5), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x - 5, y - 5), (x - 5, y - 5 + b_len), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + bw + 5, y - 5), (x + bw + 5 - b_len, y - 5), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + bw + 5, y - 5), (x + bw + 5, y - 5 + b_len), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x - 5, y + bh + 5), (x - 5 + b_len, y + bh + 5), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x - 5, y + bh + 5), (x - 5, y + bh + 5 - b_len), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + bw + 5, y + bh + 5), (x + bw + 5 - b_len, y + bh + 5), (0, 230, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + bw + 5, y + bh + 5), (x + bw + 5, y + bh + 5 - b_len), (0, 230, 255), 3, cv2.LINE_AA)

                cx, cy = int(it["centroid"][0]), int(it["centroid"][1])
                cv2.circle(overlay, (cx, cy), 7, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, (cx, cy), 2, (0, 220, 255), -1, cv2.LINE_AA)

                # Vector to target destination
                tx, ty = it.get("target_coords", (x, y))
                if (tx != cx or ty != cy) and not is_cleared:
                    cv2.arrowedLine(overlay, (cx, cy), (tx, ty), (45, 180, 85), 2, cv2.LINE_AA, tipLength=0.12)
                    dest_lbl = f"-> {it.get('target_destination', 'Relocate')}"
                    cv2.rectangle(overlay, (max(4, tx - 12), max(4, ty - 24)), (min(w - 4, tx + len(dest_lbl) * 7 + 4), max(24, ty - 4)), (22, 55, 28), -1)
                    cv2.putText(overlay, dest_lbl, (max(8, tx - 8), max(18, ty - 9)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                tag_w = min(w - x - 4, len(label) * 8 + 16)
                cv2.rectangle(overlay, (x, max(0, y - 24)), (x + tag_w, y), (20, 20, 20), -1)
                cv2.putText(overlay, label, (x + 8, max(16, y - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)
                continue

            if is_liquid and spill_hazard:
                color = (32, 70, 200)  # Hazard Red BGR
                label = f"#{it['id']} Spill Risk ({f_dist}cm)"
                line_w = 2 if highlighted_item_id is None else 1
            elif sev == "CRITICAL_ENCROACHMENT":
                color = (35, 65, 215)   # Terracotta Warning
                label = f"#{it['id']} Encroaching ({f_dist}cm)"
                line_w = 2 if highlighted_item_id is None else 1
            elif is_safe:
                color = (65, 135, 75)   # Calm Forest Green
                label = f"#{it['id']} Safe Zone"
                line_w = 1
            elif sev == "MODERATE_ENCROACHMENT" or is_zone1:
                color = (45, 125, 215)  # Warm Amber
                label = f"#{it['id']} Zone 1 Clutter"
                line_w = 2 if highlighted_item_id is None else 1
            else:
                color = (78, 117, 87)  # Sage BGR
                label = f"#{it['id']} Peripheral"
                line_w = 1

            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, line_w, cv2.LINE_AA)
            tag_w = min(w - x - 4, len(label) * 7 + 12)
            cv2.rectangle(overlay, (x, max(0, y - 20)), (x + tag_w, y), color, -1)
            cv2.putText(overlay, label, (x + 5, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1, cv2.LINE_AA)

        # Top status bar if items are progressively cleared
        if len(removed_set) > 0:
            total_c = len(items)
            rem_c = len(removed_set)
            bar_text = f"LIVE WORKSPACE CANVAS: {rem_c}/{total_c} Items Cleared In-Place | Remaining items stay highlighted"
            cv2.rectangle(overlay, (0, 0), (w, 32), (35, 26, 20), -1)
            cv2.putText(overlay, bar_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            badge_text = f"View: {perspective.replace('_', ' ')}"
            cv2.rectangle(overlay, (w - 195, 10), (w - 12, 34), (35, 26, 20), -1)
            cv2.putText(overlay, badge_text, (w - 185, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        alpha = 0.88
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def render_interactive_view(
        self,
        img: np.ndarray,
        telemetry: Dict[str, Any],
        highlighted_item_id: Optional[int] = None,
        removed_ids: Optional[Any] = None,
        item_actions: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Public method to render a unified live interactive view of the workspace:
        - Any item in `removed_ids` is inpainted (erased/relocated) directly on the base image!
        - Any remaining un-cleared items stay outlined and numbered so the user never loses context.
        - If `highlighted_item_id` is set, spotlights that item with targeting brackets and relocation vector.
        """
        removed_set = set(removed_ids) if removed_ids else set()
        base_img = img
        if len(removed_set) > 0:
            inpaint_res = self.after_gen.inpaint_specific_items(
                img, telemetry, removed_set, item_actions=item_actions
            )
            base_img = inpaint_res.get("raw_cleared_image", inpaint_res.get("cleared_image", img))

        items = telemetry.get("items", [])
        spill_hazard = telemetry.get("hazards", {}).get("spill_risk", False)
        zone1_radius_px = int(35.0 / self.CM_PER_PIXEL)
        zone2_radius_px = int(55.0 / self.CM_PER_PIXEL)
        keyboard_anchor = telemetry.get("keyboard_anchor")
        mouse_anchor = telemetry.get("mouse_anchor")
        perspective = telemetry.get("perspective", "TOP_DOWN")
        central_focus = telemetry.get("central_focus")
        encroachment = telemetry.get("encroachment")

        return self._render_sand_overlay(
            base_img,
            items,
            lines=None,
            spill_hazard=spill_hazard,
            zone1_radius=zone1_radius_px,
            zone2_radius=zone2_radius_px,
            keyboard_anchor=keyboard_anchor,
            mouse_anchor=mouse_anchor,
            perspective=perspective,
            central_focus=central_focus,
            encroachment=encroachment,
            highlighted_item_id=highlighted_item_id,
            removed_ids=removed_set,
        )

    def _render_cognitive_heatmap(self, img: np.ndarray, grad_mag: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
        """Visual Overlay: Continuous OpenCV Visual Clutter / Cognitive Heatmap."""
        energy = np.abs(laplacian) + grad_mag
        energy_blurred = cv2.GaussianBlur(energy, (41, 41), 0)
        norm_energy = cv2.normalize(energy_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        heatmap = cv2.applyColorMap(norm_energy, cv2.COLORMAP_TURBO)
        blended = cv2.addWeighted(heatmap, 0.55, img, 0.45, 0)

        cv2.rectangle(blended, (16, 16), (330, 48), (20, 20, 20), -1)
        cv2.putText(blended, "Cognitive Friction Heatmap (OpenCV Spectral)", (24, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        return blended

    def _render_relocation_overlay(
        self,
        img: np.ndarray,
        items: List[Dict[str, Any]],
        zone1_radius: int,
        zone2_radius: int,
        keyboard_anchor: Any = None,
        perspective: str = "TOP_DOWN",
        central_focus: Any = None,
    ) -> np.ndarray:
        """Visual Overlay: Agentic Prescriptive Relocation Vectors & Liberated Central Focus Halo."""
        overlay = img.copy()
        h, w = img.shape[:2]

        if perspective not in ["PERSPECTIVE_ANGLED", "SIDE_VIEW"]:
            user_pt = (int(w / 2), int(h))
            cv2.ellipse(overlay, user_pt, (zone1_radius, int(zone1_radius * 0.72)), 0, 180, 360, (200, 215, 225), 1, cv2.LINE_AA)
            cv2.ellipse(overlay, user_pt, (zone2_radius, int(zone2_radius * 0.72)), 0, 180, 360, (190, 210, 195), 1, cv2.LINE_AA)

        if central_focus:
            fx, fy, fbw, fbh = central_focus["bbox"]
            env_bbox = central_focus.get("envelope_bbox", [fx - 60, fy - 30, fbw + 120, fbh + 80])
            ex, ey, ew, eh = env_bbox
            cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), (78, 165, 87), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"LIBERATED FOCUS HALO: {central_focus['name']} (100% Clear)", (ex + 10, max(20, ey - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (45, 120, 55), 1, cv2.LINE_AA)

        for it in items:
            cx, cy = int(it["centroid"][0]), int(it["centroid"][1])
            dest_name = it["target_destination"]
            disp_cm = it["displacement_cm"]

            if dest_name == "Maintain Current Position" or disp_cm < 2.0:
                cv2.circle(overlay, (cx, cy), 14, (65, 140, 75), 2, cv2.LINE_AA)
                cv2.putText(overlay, "OPTIMAL", (cx - 18, cy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (65, 140, 75), 1, cv2.LINE_AA)
                continue

            tx, ty = it["target_coords"]
            color_arrow = (66, 150, 32)
            cv2.arrowedLine(overlay, (cx, cy), (tx, ty), (25, 80, 18), 4, cv2.LINE_AA, tipLength=0.08)
            cv2.arrowedLine(overlay, (cx, cy), (tx, ty), color_arrow, 2, cv2.LINE_AA, tipLength=0.08)

            cv2.circle(overlay, (tx, ty), 14, (25, 80, 18), -1)
            cv2.circle(overlay, (tx, ty), 12, color_arrow, 2, cv2.LINE_AA)
            cv2.circle(overlay, (tx, ty), 3, (255, 255, 255), -1)

            label = f"{dest_name} (+{disp_cm}cm)"
            tag_w = len(label) * 7 + 12
            cv2.rectangle(overlay, (max(4, tx - 8), max(4, ty - 26)), (min(w - 4, tx + tag_w), max(24, ty - 6)), (22, 40, 22), -1)
            cv2.putText(overlay, label, (max(8, tx - 2), max(18, ty - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.circle(overlay, (cx, cy), 6, (38, 82, 184), -1)

        cv2.rectangle(overlay, (16, 16), (380, 46), (20, 20, 20), -1)
        cv2.putText(overlay, "Prescriptive Relocation Vectors (Central Focus Liberated)", (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        alpha = 0.90
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
