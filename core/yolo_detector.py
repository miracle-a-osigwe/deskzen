"""
DeskZen YOLO Object & Perspective Detector
Detects workspace objects from any camera angle (top-down, 45-degree angled, or side view)
using YOLOv8 and calculates spatial clustering and perspective orientation.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO

    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class WorkspaceYOLODetector:
    """
    Detects workspace objects across any camera orientation (top-down, side-view, angled),
    classifies perspective, and computes spatial clustering metrics.
    """

    # COCO class IDs of interest
    TARGET_CLASSES = {
        39: "bottle",
        41: "cup",
        64: "potted plant",
        66: "keyboard",
        67: "cell phone",
        63: "laptop",
        65: "mouse",
        73: "book",
        76: "scissors",
        62: "tv",  # monitor
    }

    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.model = None
        if _YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(
                    f"[WorkspaceYOLODetector] Note: YOLO load failed: {e}. Falling back gracefully."
                )

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def detect(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Runs YOLO detection on an image of any orientation.
        Returns detected objects, detected perspective, spatial clustering metrics,
        and central focus identification with peripheral encroachment analysis.
        """
        h, w = img_bgr.shape[:2]
        if not self.is_available:
            return {
                "detected": False,
                "objects": [],
                "perspective": "TOP_DOWN",
                "clustering_index": 0.0,
                "clustering_severity": "LOW_SPACED",
                "clusters": [],
                "proximity_pairs": [],
                "central_focus": None,
                "encroachment": {
                    "status": "ZEN_BREATHING_ROOM",
                    "wrist_clearance_cm": 35.0,
                    "mouse_clearance_cm": 35.0,
                    "left_flank_clearance_cm": 35.0,
                    "breathing_room_pct": 100.0,
                    "encroaching_items": [],
                },
            }

        try:
            results = self.model(img_bgr, conf=self.conf_thresh, verbose=False)
            boxes_data = results[0].boxes
        except Exception as e:
            print(f"[WorkspaceYOLODetector] Detection exception: {e}")
            return {
                "detected": False,
                "objects": [],
                "perspective": "TOP_DOWN",
                "clustering_index": 0.0,
                "clustering_severity": "LOW_SPACED",
                "clusters": [],
                "proximity_pairs": [],
                "central_focus": None,
                "encroachment": {
                    "status": "ZEN_BREATHING_ROOM",
                    "wrist_clearance_cm": 35.0,
                    "mouse_clearance_cm": 35.0,
                    "left_flank_clearance_cm": 35.0,
                    "breathing_room_pct": 100.0,
                    "encroaching_items": [],
                },
            }

        objects = []
        for i, box in enumerate(boxes_data):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names.get(cls_id, "object")

            # Check if relevant workspace object
            is_relevant = (cls_id in self.TARGET_CLASSES) or (
                cls_name
                in [
                    "cup",
                    "bottle",
                    "keyboard",
                    "laptop",
                    "mouse",
                    "cell phone",
                    "book",
                    "scissors",
                    "tv",
                ]
            )
            if not is_relevant and conf < 0.35:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Determine ergonomic category
            category = "CLUTTER"
            if cls_name in ["keyboard", "laptop", "tv"]:
                category = "ANCHOR_PRIMARY"
            elif cls_name in ["mouse"]:
                category = "ANCHOR_MOUSE"
            elif cls_name in ["cup", "bottle"]:
                category = "HAZARD_LIQUID"
            elif cls_name in ["cell phone"]:
                category = "DISTRACTION_MOBILE"
            elif cls_name in ["book"]:
                category = "CLUTTER_PAPERWORK"
            elif cls_name in ["potted plant"]:
                category = "DECOR_ZEN"

            objects.append(
                {
                    "id": i + 1,
                    "name": cls_name,
                    "category": category,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, bw, bh],
                    "centroid": [cx, cy],
                    "area": bw * bh,
                    "aspect_ratio": round(bw / max(bh, 1), 2),
                }
            )

        # 1. Perspective & Viewpoint Analysis
        perspective = self._estimate_perspective(objects, w, h)

        # 2. Central Work Focus Identification
        central_focus = self._identify_central_focus(objects, w, h)

        # 3. Peripheral Encroachment & Workspace Congestion Analysis
        encroachment = self._calculate_encroachment(central_focus, objects, w, h)

        # 4. Spatial Clustering Analysis (OpenCV inter-object distance matrix)
        clustering_info = self._calculate_clustering(objects, w, h, perspective)

        return {
            "detected": len(objects) > 0,
            "objects": objects,
            "perspective": perspective,
            "central_focus": central_focus,
            "encroachment": encroachment,
            "clustering_index": clustering_info["clustering_index"],
            "clustering_severity": clustering_info["clustering_severity"],
            "clusters": clustering_info["clusters"],
            "proximity_pairs": clustering_info["proximity_pairs"],
        }

    def _estimate_perspective(
        self, objects: List[Dict[str, Any]], w: int, h: int
    ) -> str:
        """
        Determines if the camera is top-down, angled/oblique, or side-view.
        Side view: objects have significant vertical overlap and depth gradient where y indicates reach.
        Top-down: camera directly overhead.
        """
        if not objects:
            return "TOP_DOWN"

        # Check for keyboard or laptop anchor
        anchors = [o for o in objects if o["category"] == "ANCHOR_PRIMARY"]
        if anchors:
            anchor = anchors[0]
            # In side view, keyboard is foreshortened and located in lower half
            # In top-down, keyboard is broad rectangular (ar > 2.2)
            ar = anchor["aspect_ratio"]
            cy = anchor["centroid"][1]
            if ar < 1.9 and cy > h * 0.40:
                return "PERSPECTIVE_ANGLED"
            elif cy > h * 0.60 and ar < 1.6:
                return "SIDE_VIEW"

        # Check vertical distribution of all objects: if objects span entire height with depth gradient
        y_positions = [o["centroid"][1] for o in objects]
        if max(y_positions) - min(y_positions) > h * 0.65:
            return "PERSPECTIVE_ANGLED"

        return "TOP_DOWN"

    def _calculate_clustering(
        self, objects: List[Dict[str, Any]], w: int, h: int, perspective: str
    ) -> Dict[str, Any]:
        """
        Calculates spatial clustering matrix and identifies congestion clusters.
        """
        n = len(objects)
        if n < 2:
            return {
                "clustering_index": 0.0,
                "clustering_severity": "LOW_ISOLATED",
                "clusters": [],
                "proximity_pairs": [],
            }

        # Threshold for objects being "clustered together"
        # In typical desk, items closer than ~18% of width are huddled together
        thresh_dist = w * 0.18
        proximity_pairs = []
        close_pairs_count = 0
        total_pairs = (n * (n - 1)) // 2

        adjacency = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                c1 = objects[i]["centroid"]
                c2 = objects[j]["centroid"]
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
                # In perspective/side-view, vertical distance represents depth, weight accordingly
                if perspective in ["PERSPECTIVE_ANGLED", "SIDE_VIEW"]:
                    dist = np.sqrt(dx**2 + (dy * 1.5) ** 2)
                else:
                    dist = np.sqrt(dx**2 + dy**2)

                if dist < thresh_dist:
                    close_pairs_count += 1
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                    proximity_pairs.append(
                        {
                            "item1": objects[i]["name"],
                            "item2": objects[j]["name"],
                            "distance_px": round(float(dist), 1),
                            "distance_cm": round(float(dist * 0.15), 1),
                        }
                    )

        clustering_index = round((close_pairs_count / max(total_pairs, 1)) * 100, 1)

        # Form connected component clusters (BFS)
        visited = set()
        clusters = []
        for i in range(n):
            if i not in visited:
                component = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adjacency[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                if len(component) > 1:
                    cluster_items = [objects[idx]["name"] for idx in component]
                    clusters.append(
                        {
                            "items": cluster_items,
                            "size": len(component),
                            "description": f"Cluster of {len(component)} items: {', '.join(cluster_items)}",
                        }
                    )

        if clustering_index > 40:
            severity = "HIGH_CONGESTION"
        elif clustering_index > 15:
            severity = "MODERATE_CLUSTER"
        else:
            severity = "LOW_SPACED"

        return {
            "clustering_index": clustering_index,
            "clustering_severity": severity,
            "clusters": clusters,
            "proximity_pairs": proximity_pairs,
        }

    def _identify_central_focus(
        self, objects: List[Dict[str, Any]], w: int, h: int
    ) -> Optional[Dict[str, Any]]:
        """
        Identifies the primary central work focus:
        1. Laptop (integrated work center)
        2. Desktop Setup (Keyboard + Monitor / TV)
        3. Keyboard (tactile typing anchor)
        4. Monitor / TV (visual display anchor)
        5. Work Gadget / Tablet near center
        """
        if not objects:
            return None

        laptops = [o for o in objects if o["name"] == "laptop"]
        keyboards = [o for o in objects if o["name"] == "keyboard"]
        monitors = [o for o in objects if o["name"] in ["tv", "monitor"]]

        if laptops:
            laptop = max(laptops, key=lambda o: o["area"])
            x, y, bw, bh = laptop["bbox"]
            cx, cy = laptop["centroid"]
            return self._build_focus_envelope(
                focus_type="LAPTOP",
                name="Laptop Workstation (Central Anchor)",
                bbox=[x, y, bw, bh],
                centroid=[cx, cy],
                confidence=laptop.get("confidence", 0.90),
                components=["laptop"],
                w=w,
                h=h,
            )

        if keyboards and monitors:
            kb = max(keyboards, key=lambda o: o["area"])
            mon = max(monitors, key=lambda o: o["area"])
            kx, ky, kw, kh = kb["bbox"]
            mx, my, mw, mh = mon["bbox"]
            min_x = min(kx, mx)
            min_y = min(ky, my)
            max_x = max(kx + kw, mx + mw)
            max_y = max(ky + kh, my + mh)
            return self._build_focus_envelope(
                focus_type="DESKTOP_SETUP",
                name="Desktop Workstation (Keyboard + Display)",
                bbox=[min_x, min_y, max_x - min_x, max_y - min_y],
                centroid=[(min_x + max_x) / 2.0, (min_y + max_y) / 2.0],
                confidence=round(
                    (kb.get("confidence", 0.85) + mon.get("confidence", 0.85)) / 2, 2
                ),
                components=["keyboard", "monitor"],
                keyboard_bbox=[kx, ky, kw, kh],
                monitor_bbox=[mx, my, mw, mh],
                w=w,
                h=h,
            )

        if keyboards:
            kb = max(keyboards, key=lambda o: o["area"])
            x, y, bw, bh = kb["bbox"]
            cx, cy = kb["centroid"]
            return self._build_focus_envelope(
                focus_type="KEYBOARD",
                name="Primary Keyboard Anchor",
                bbox=[x, y, bw, bh],
                centroid=[cx, cy],
                confidence=kb.get("confidence", 0.85),
                components=["keyboard"],
                w=w,
                h=h,
            )

        if monitors:
            mon = max(monitors, key=lambda o: o["area"])
            x, y, bw, bh = mon["bbox"]
            cx, cy = mon["centroid"]
            return self._build_focus_envelope(
                focus_type="MONITOR",
                name="Primary Display Anchor",
                bbox=[x, y, bw, bh],
                centroid=[cx, cy],
                confidence=mon.get("confidence", 0.85),
                components=["monitor"],
                w=w,
                h=h,
            )

        gadgets = [
            o
            for o in objects
            if o["category"] == "ANCHOR_PRIMARY"
            or o["name"] in ["tablet", "cell phone"]
        ]
        if gadgets:
            g = min(
                gadgets,
                key=lambda o: abs(o["centroid"][0] - w / 2.0)
                + abs(o["centroid"][1] - h * 0.55),
            )
            x, y, bw, bh = g["bbox"]
            cx, cy = g["centroid"]
            return self._build_focus_envelope(
                focus_type="WORK_GADGET",
                name=f"Work Device ({g['name'].title()})",
                bbox=[x, y, bw, bh],
                centroid=[cx, cy],
                confidence=g.get("confidence", 0.75),
                components=[g["name"]],
                w=w,
                h=h,
            )

        return None

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
        """
        Computes active ergonomic envelope:
        - Wrist rest zone (foreground, 12-15cm)
        - Mouse sweep corridor (right flank, 20-25cm)
        - Left flank secondary reach (15-18cm)
        - Rear margin (cooling & cable dock)
        """
        px_per_cm = max(1.0, w / 120.0)
        x, y, bw, bh = bbox

        wrist_h_px = int(14.0 * px_per_cm)
        mouse_w_px = int(22.0 * px_per_cm)
        left_w_px = int(18.0 * px_per_cm)
        rear_h_px = int(10.0 * px_per_cm)

        env_x1 = max(0, x - left_w_px)
        env_y1 = max(0, y - rear_h_px)
        env_x2 = min(w, x + bw + mouse_w_px)
        env_y2 = min(h, y + bh + wrist_h_px)

        envelope_data = {
            "type": focus_type,
            "name": name,
            "confidence": confidence,
            "bbox": bbox,
            "centroid": centroid,
            "components": components,
            "envelope_bbox": [env_x1, env_y1, env_x2 - env_x1, env_y2 - env_y1],
            "wrist_zone_bbox": [x, min(h, y + bh), bw, min(h - (y + bh), wrist_h_px)],
            "mouse_zone_bbox": [
                min(w, x + bw),
                max(0, y - int(4 * px_per_cm)),
                min(w - (x + bw), mouse_w_px),
                min(h, bh + wrist_h_px),
            ],
            "left_flank_bbox": [env_x1, y, left_w_px, bh],
        }
        envelope_data.update(kwargs)
        return envelope_data

    def _calculate_encroachment(
        self,
        central_focus: Optional[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> Dict[str, Any]:
        """
        Analyzes how peripheral elements crowd or encroach upon the central work focus.
        Calculates edge distance, relative quadrant, encroachment severity,
        and individual congestion contribution scores.
        """
        if central_focus is None:
            return {
                "status": "NO_CENTRAL_DEVICE_IDENTIFIED",
                "wrist_clearance_cm": 35.0,
                "mouse_clearance_cm": 35.0,
                "left_flank_clearance_cm": 35.0,
                "breathing_room_pct": 100.0,
                "encroaching_items": [],
                "congestion_penalty_total": 0.0,
                "critical_encroachment_count": 0,
            }

        px_per_cm = max(1.0, w / 120.0)
        cm_per_px = 1.0 / px_per_cm

        fx, fy, fbw, fbh = central_focus["bbox"]
        fcx, fcy = central_focus["centroid"]
        focus_components = set(central_focus.get("components", []))

        encroaching_items = []
        min_wrist_dist_cm = 99.0
        min_mouse_dist_cm = 99.0
        min_left_dist_cm = 99.0
        congestion_penalty_total = 0.0

        for obj in objects:
            if (
                obj.get("category") == "ANCHOR_PRIMARY"
                and obj["name"] in focus_components
            ):
                continue

            ox, oy, obw, obh = obj["bbox"]
            ocx, ocy = obj["centroid"]

            # Edge-to-edge distance calculation
            if ox + obw < fx:
                dx = fx - (ox + obw)
                rel_h = "LEFT"
            elif ox > fx + fbw:
                dx = ox - (fx + fbw)
                rel_h = "RIGHT"
            else:
                dx = 0
                rel_h = "ALIGNED"

            if oy + obh < fy:
                dy = fy - (oy + obh)
                rel_v = "BACKGROUND"
            elif oy > fy + fbh:
                dy = oy - (fy + fbh)
                rel_v = "FOREGROUND"
            else:
                dy = 0
                rel_v = "ALIGNED"

            edge_dist_px = float(np.sqrt(dx**2 + dy**2))
            edge_dist_cm = round(edge_dist_px * cm_per_px, 1)

            # Determine ergonomic quadrant relative to central focus
            if rel_v == "FOREGROUND" and (rel_h == "ALIGNED" or dx < 40):
                sector = "FOREGROUND_WRIST_ZONE"
                min_wrist_dist_cm = min(min_wrist_dist_cm, edge_dist_cm)
            elif rel_h == "RIGHT" and (rel_v == "ALIGNED" or dy < 70):
                sector = "RIGHT_FLANK_MOUSE_SWEEP"
                min_mouse_dist_cm = min(min_mouse_dist_cm, edge_dist_cm)
            elif rel_h == "LEFT" and (rel_v == "ALIGNED" or dy < 70):
                sector = "LEFT_FLANK_SECONDARY"
                min_left_dist_cm = min(min_left_dist_cm, edge_dist_cm)
            elif rel_v == "BACKGROUND":
                sector = "BACKGROUND_DISPLAY_ZONE"
            else:
                sector = "PERIPHERAL_CORNER"

            is_liquid = obj.get("category") == "HAZARD_LIQUID" or obj["name"] in [
                "cup",
                "bottle",
            ]
            if edge_dist_cm < 12.0:
                severity = "CRITICAL_ENCROACHMENT"
                base_penalty = (
                    35.0
                    if sector in ["FOREGROUND_WRIST_ZONE", "RIGHT_FLANK_MOUSE_SWEEP"]
                    else 20.0
                )
                if is_liquid:
                    base_penalty += 20.0
            elif edge_dist_cm < 24.0:
                severity = "MODERATE_ENCROACHMENT"
                base_penalty = (
                    18.0
                    if sector in ["FOREGROUND_WRIST_ZONE", "RIGHT_FLANK_MOUSE_SWEEP"]
                    else 10.0
                )
                if is_liquid:
                    base_penalty += 12.0
            else:
                severity = "SAFE_PERIPHERY"
                base_penalty = 0.0

            congestion_penalty_total += base_penalty

            if sector == "FOREGROUND_WRIST_ZONE":
                directive = (
                    "Clear completely from wrist-rest area to eliminate typing strain."
                )
            elif sector == "RIGHT_FLANK_MOUSE_SWEEP":
                directive = (
                    "Relocate at least 25cm outward to unblock dominant mouse travel."
                )
            elif sector == "LEFT_FLANK_SECONDARY":
                directive = "Consolidate onto vertical organizer or left coaster."
            else:
                directive = "Maintain at desk perimeter or stow."

            encroaching_items.append(
                {
                    "id": obj["id"],
                    "name": obj["name"],
                    "edge_distance_cm": edge_dist_cm,
                    "sector": sector,
                    "severity": severity,
                    "is_liquid": is_liquid,
                    "congestion_penalty": round(base_penalty, 1),
                    "directive": directive,
                    "bbox": obj["bbox"],
                    "centroid": obj["centroid"],
                }
            )

        wrist_score = (
            min(1.0, min_wrist_dist_cm / 15.0) if min_wrist_dist_cm < 90 else 1.0
        )
        mouse_score = (
            min(1.0, min_mouse_dist_cm / 22.0) if min_mouse_dist_cm < 90 else 1.0
        )
        left_score = min(1.0, min_left_dist_cm / 18.0) if min_left_dist_cm < 90 else 1.0
        breathing_room_pct = round(
            ((wrist_score * 0.4) + (mouse_score * 0.4) + (left_score * 0.2)) * 100, 1
        )

        critical_count = sum(
            1 for it in encroaching_items if it["severity"] == "CRITICAL_ENCROACHMENT"
        )
        if critical_count > 0 or breathing_room_pct < 55.0:
            status = "CRITICALLY_CONSTRICTED"
        elif breathing_room_pct < 85.0:
            status = "MODERATELY_CONGESTED"
        else:
            status = "ZEN_BREATHING_ROOM"

        return {
            "status": status,
            "wrist_clearance_cm": round(min_wrist_dist_cm, 1)
            if min_wrist_dist_cm < 90
            else 35.0,
            "mouse_clearance_cm": round(min_mouse_dist_cm, 1)
            if min_mouse_dist_cm < 90
            else 35.0,
            "left_flank_clearance_cm": round(min_left_dist_cm, 1)
            if min_left_dist_cm < 90
            else 35.0,
            "breathing_room_pct": breathing_room_pct,
            "congestion_penalty_total": round(congestion_penalty_total, 1),
            "critical_encroachment_count": critical_count,
            "encroaching_items": sorted(
                encroaching_items,
                key=lambda x: (-x["congestion_penalty"], x["edge_distance_cm"]),
            ),
        }
