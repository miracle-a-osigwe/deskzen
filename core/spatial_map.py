"""
DeskZen Spatial Digital Twin Map Engine (2.5D Layered Scene Graph)
Decomposes workspace photos into:
1. Base Tabletop Surface Layer (clean Telea-inpainted wood/matte plane)
2. Fixed Central Work Anchors (Laptop, Monitor, Keyboard, Mouse Pad)
3. Movable Object Sprites (RGBA foreground patches with soft alpha feathering)

Provides sub-10ms deterministic spatial compositing, contact drop shadows,
safe-zone repositioning, and photorealistic version synthesis without generative AI hallucination.
"""

import os
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Set, Union
import cv2
import numpy as np


@dataclass
class ObjectSprite:
    """Represents an isolated physical object in the 2.5D Spatial Digital Twin Map."""
    id: int
    semantic_name: str
    original_bbox: List[int]            # [x, y, w, h]
    original_centroid: Tuple[int, int]  # (cx, cy)
    target_coords: Tuple[int, int]      # (tx, ty) safe zone destination
    current_coords: Tuple[int, int]     # (x, y) active position on map
    rgba_patch: np.ndarray              # RGBA array with feathered alpha
    area_px: int
    is_liquid: bool = False
    is_phone: bool = False
    is_cable: bool = False
    zone: str = "ZONE_1_PRIMARY_REACH"
    target_destination: str = "Zone 2 Safe Area"
    displacement_cm: float = 0.0
    status: str = "ORIGINAL"            # "ORIGINAL", "RELOCATED", "STOWED"
    action_mode: str = "relocate"       # "relocate" or "stow"


class WorkspaceDigitalTwinMap:
    """
    2.5D Layered Scene Graph representing the physical workspace.
    Enables instant (<10ms) repositioning, inpainting, and compositing of desk elements.
    """

    CM_PER_PIXEL = 0.15

    def __init__(
        self,
        raw_original_image: np.ndarray,
        base_surface_bgr: np.ndarray,
        sprites: Dict[int, ObjectSprite],
        central_focus: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
    ):
        self.raw_original_image = raw_original_image
        self.base_surface_bgr = base_surface_bgr
        self.sprites = sprites
        self.central_focus = central_focus or {}
        self.telemetry = telemetry or {}
        self.plan = plan or {}
        self.height, self.width = raw_original_image.shape[:2]

    @classmethod
    def create_from_analysis(
        cls,
        image: np.ndarray,
        telemetry: Dict[str, Any],
        plan: Optional[Dict[str, Any]] = None,
    ) -> "WorkspaceDigitalTwinMap":
        """
        Decomposes an analyzed workspace photo into a clean Base Tabletop Canvas
        and isolated RGBA Object Sprites for every detected clutter element.
        """
        img = image.copy()
        h, w = img.shape[:2]
        items = telemetry.get("items", [])
        central_focus = telemetry.get("central_focus", {})

        # 1. Build Inpainting Mask for all movable clutter objects
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        sprites: Dict[int, ObjectSprite] = {}

        for it in items:
            x, y, bw, bh = it["bbox"]
            cx, cy = int(it["centroid"][0]), int(it["centroid"][1])
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            cv2.rectangle(inpaint_mask, (x1, y1), (x2, y2), 255, -1)

            # Extract RGBA sprite with feathered alpha
            rgba = cls._extract_object_rgba(img, it["bbox"])

            sem = it.get("semantic_name", "").lower()
            is_liq = it.get("is_potential_liquid_container", False) or any(
                k in sem for k in ["mug", "cup", "tumbler", "bottle", "drink", "coffee", "tea", "can"]
            )
            is_ph = any(k in sem for k in ["phone", "smartphone", "iphone", "android"])
            is_cb = any(k in sem for k in ["cable", "cord", "wire", "charger", "usb"])

            target_c = it.get("target_coords", (cx, cy))
            if not target_c or target_c == (cx, cy):
                if is_liq:
                    target_c = (int(w * 0.88), int(h * 0.46))
                elif is_ph:
                    target_c = (int(w * 0.86), int(h * 0.40))
                else:
                    target_c = (int(w * 0.14), int(h * 0.22))

            sprite = ObjectSprite(
                id=it["id"],
                semantic_name=it.get("semantic_name", f"Item #{it['id']}"),
                original_bbox=it["bbox"],
                original_centroid=(cx, cy),
                target_coords=target_c,
                current_coords=(cx, cy),
                rgba_patch=rgba,
                area_px=it.get("area_px", bw * bh),
                is_liquid=is_liq,
                is_phone=is_ph,
                is_cable=is_cb,
                zone=it.get("zone", "ZONE_1_PRIMARY_REACH"),
                target_destination=it.get("target_destination", "Zone 2 Safe Area"),
                displacement_cm=it.get("displacement_cm", 0.0),
                status="ORIGINAL",
                action_mode="relocate" if (is_liq or is_ph or "Safe" in it.get("target_destination", "")) else "stow",
            )
            sprites[it["id"]] = sprite

        # 2. Synthesize Clean Base Tabletop Canvas via Telea Inpainting
        if np.any(inpaint_mask > 0):
            base_surface = cv2.inpaint(img, inpaint_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
            # Smooth inpaint boundaries
            mask_boundary = cv2.Canny(inpaint_mask, 100, 200)
            mask_boundary_dil = cv2.dilate(mask_boundary, np.ones((7, 7), np.uint8))
            smooth_blend = cv2.GaussianBlur(base_surface, (7, 7), 0)
            base_surface = np.where(mask_boundary_dil[:, :, None] > 0, smooth_blend, base_surface)
        else:
            base_surface = img.copy()

        return cls(
            raw_original_image=img,
            base_surface_bgr=base_surface,
            sprites=sprites,
            central_focus=central_focus,
            telemetry=telemetry,
            plan=plan,
        )

    @staticmethod
    def _extract_object_rgba(img: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Segments physical object from tabletop with soft Gaussian alpha feathering.
        """
        h_img, w_img = img.shape[:2]
        x, y, bw, bh = bbox
        x = max(0, min(w_img - 1, x))
        y = max(0, min(h_img - 1, y))
        bw = max(1, min(w_img - x, bw))
        bh = max(1, min(h_img - y, bh))

        crop = img[y:y + bh, x:x + bw]
        ch, cw = crop.shape[:2]

        if ch < 4 or cw < 4:
            rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = 255
            return rgba

        # Estimate local desk background color from outer 3px border
        top_b = crop[:min(3, ch), :]
        bot_b = crop[max(0, ch - 3):, :]
        left_b = crop[:, :min(3, cw)]
        right_b = crop[:, max(0, cw - 3):]
        border_pts = np.vstack([
            top_b.reshape(-1, 3),
            bot_b.reshape(-1, 3),
            left_b.reshape(-1, 3),
            right_b.reshape(-1, 3),
        ])
        bg_color = np.median(border_pts, axis=0).astype(np.float32)

        # Color difference map
        crop_float = crop.astype(np.float32)
        diff = np.sqrt(np.sum((crop_float - bg_color) ** 2, axis=2))

        # Adaptive alpha thresholding
        thresh_val = max(18.0, float(np.percentile(diff, 20)))
        alpha_raw = np.clip((diff - thresh_val * 0.4) / (thresh_val * 1.2), 0.0, 1.0) * 255.0
        alpha_u8 = alpha_raw.astype(np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        alpha_clean = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
        alpha_feathered = cv2.GaussianBlur(alpha_clean, (5, 5), 1.5)

        rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
        rgba[:, :, :3] = crop
        rgba[:, :, 3] = alpha_feathered
        return rgba

    def apply_checklist_state(
        self,
        removed_ids: Set[int],
        action_modes: Optional[Dict[int, str]] = None,
    ):
        """
        Batch synchronizes the digital twin map state with checklist user selections.
        """
        action_map = action_modes or {}
        for item_id, sprite in self.sprites.items():
            is_checked = (item_id in removed_ids)
            mode = action_map.get(item_id, sprite.action_mode)
            sprite.action_mode = mode

            if is_checked:
                if mode == "stow":
                    sprite.status = "STOWED"
                else:
                    sprite.status = "RELOCATED"
                    sprite.current_coords = sprite.target_coords
            else:
                sprite.status = "ORIGINAL"
                sprite.current_coords = sprite.original_centroid

    def reposition_sprite(self, item_id: int, new_coords: Tuple[int, int]):
        """Reposition a specific sprite to arbitrary (x, y) coordinates."""
        if item_id in self.sprites:
            sprite = self.sprites[item_id]
            sprite.current_coords = new_coords
            sprite.status = "RELOCATED"

    def render_photorealistic(self) -> np.ndarray:
        """
        Renders the active 2.5D Digital Twin Scene Graph into a clean,
        photorealistic BGR workspace image (<10ms, zero HUD overlays).
        """
        canvas = self.base_surface_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Pass 1: Render Drop Shadows and Coasters for all active objects
        for sprite in self.sprites.values():
            if sprite.status == "STOWED":
                continue

            cx, cy = sprite.current_coords
            pw = sprite.rgba_patch.shape[1]
            ph = sprite.rgba_patch.shape[0]

            # Coaster for liquids when relocated
            if sprite.is_liquid and sprite.status == "RELOCATED":
                c_rad = max(24, int(max(pw, ph) * 0.62))
                c_y = cy + int(ph * 0.32)
                # Slate/cork styled coaster
                cv2.circle(canvas, (cx, c_y), c_rad + 3, (120, 110, 100), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, c_y), c_rad, (165, 155, 145), -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, c_y), c_rad - 3, (140, 130, 120), 1, cv2.LINE_AA)

            # Contact Drop Shadow
            s_w = max(10, int(pw * 0.45))
            s_h = max(5, int(ph * 0.20))
            s_y = cy + int(ph * 0.36)
            s_x = cx + 2

            shadow_overlay = canvas.copy()
            cv2.ellipse(shadow_overlay, (s_x, s_y), (s_w, s_h), 0, 0, 360, (25, 20, 15), -1, cv2.LINE_AA)
            cv2.addWeighted(shadow_overlay, 0.35, canvas, 0.65, 0, canvas)

        # 2. Pass 2: Alpha Composite RGBA Sprites onto Canvas
        for sprite in self.sprites.values():
            if sprite.status == "STOWED":
                continue

            cx, cy = sprite.current_coords
            rgba = sprite.rgba_patch
            sh, sw = rgba.shape[:2]

            x1 = cx - sw // 2
            y1 = cy - sh // 2
            x2 = x1 + sw
            y2 = y1 + sh

            # Canvas clipping
            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(w, x2)
            cy2 = min(h, y2)

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            sx1 = cx1 - x1
            sy1 = cy1 - y1
            sx2 = sx1 + (cx2 - cx1)
            sy2 = sy1 + (cy2 - cy1)

            patch_bgr = rgba[sy1:sy2, sx1:sx2, :3]
            patch_alpha = rgba[sy1:sy2, sx1:sx2, 3:4].astype(np.float32) / 255.0

            roi = canvas[cy1:cy2, cx1:cx2].astype(np.float32)
            blended = patch_bgr.astype(np.float32) * patch_alpha + roi * (1.0 - patch_alpha)
            canvas[cy1:cy2, cx1:cx2] = np.clip(blended, 0, 255).astype(np.uint8)

        return canvas

    def render_interactive_view(
        self,
        highlighted_id: Optional[int] = None,
        removed_ids: Optional[Set[int]] = None,
        action_modes: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """
        Renders the unified interactive workbench view:
        - Photorealistic composite as base
        - Central Focus Halo and reach zones
        - Highlighting reticle & vectors for focused item
        - Visual status badges for active elements
        """
        if removed_ids is not None:
            self.apply_checklist_state(removed_ids, action_modes)

        base = self.render_photorealistic()
        overlay = base.copy()
        h, w = base.shape[:2]

        # Draw Central Focus Halo
        if self.central_focus:
            fx, fy, fbw, fbh = self.central_focus.get("bbox", [int(w * 0.28), int(h * 0.45), int(w * 0.44), int(h * 0.42)])
            env_bbox = self.central_focus.get("envelope_bbox", [fx - 40, fy - 25, fbw + 80, fbh + 50])
            ex, ey, ew, eh = env_bbox
            cv2.rectangle(overlay, (ex, ey), (ex + ew, ey + eh), (78, 165, 87), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"CENTRAL FOCUS HALO: {self.central_focus.get('name', 'Workstation')}",
                (ex + 8, max(22, ey - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (45, 120, 55),
                1,
                cv2.LINE_AA,
            )

        # Reach Zone Radii
        user_pt = (int(w / 2), int(h))
        z1_r = int(35.0 / self.CM_PER_PIXEL)
        z2_r = int(55.0 / self.CM_PER_PIXEL)
        cv2.ellipse(overlay, user_pt, (z1_r, int(z1_r * 0.72)), 0, 180, 360, (200, 215, 225), 1, cv2.LINE_AA)
        cv2.ellipse(overlay, user_pt, (z2_r, int(z2_r * 0.72)), 0, 180, 360, (190, 210, 195), 1, cv2.LINE_AA)

        # Draw HUD overlays on items
        for it_id, sprite in self.sprites.items():
            cx, cy = sprite.current_coords
            pw, ph = sprite.rgba_patch.shape[1], sprite.rgba_patch.shape[0]
            x, y = cx - pw // 2, cy - ph // 2
            is_hl = (it_id == highlighted_id)

            if sprite.status == "STOWED":
                continue

            if is_hl:
                # Spotlight reticle
                b_len = 14
                cv2.line(overlay, (x - 6, y - 6), (x - 6 + b_len, y - 6), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x - 6, y - 6), (x - 6, y - 6 + b_len), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + pw + 6, y - 6), (x + pw + 6 - b_len, y - 6), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + pw + 6, y - 6), (x + pw + 6, y - 6 + b_len), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x - 6, y + ph + 6), (x - 6 + b_len, y + ph + 6), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x - 6, y + ph + 6), (x - 6, y + ph + 6 - b_len), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + pw + 6, y + ph + 6), (x + pw + 6 - b_len, y + ph + 6), (0, 220, 255), 3, cv2.LINE_AA)
                cv2.line(overlay, (x + pw + 6, y + ph + 6), (x + pw + 6, y + ph + 6 - b_len), (0, 220, 255), 3, cv2.LINE_AA)

                # Targeting vector
                if sprite.status == "ORIGINAL" and sprite.target_coords != (cx, cy):
                    tx, ty = sprite.target_coords
                    cv2.arrowedLine(overlay, (cx, cy), (tx, ty), (45, 180, 85), 2, cv2.LINE_AA, tipLength=0.12)
                    dest_lbl = f"-> {sprite.target_destination}"
                    cv2.rectangle(overlay, (max(4, tx - 10), max(4, ty - 22)), (min(w - 4, tx + len(dest_lbl) * 7 + 4), max(22, ty - 2)), (20, 50, 25), -1)
                    cv2.putText(overlay, dest_lbl, (max(8, tx - 6), max(16, ty - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                tag_lbl = f"#{it_id} {sprite.semantic_name} [ACTIVE]"
                cv2.rectangle(overlay, (x, max(0, y - 22)), (x + len(tag_lbl) * 7 + 12, y), (20, 20, 20), -1)
                cv2.putText(overlay, tag_lbl, (x + 6, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
            elif sprite.status == "RELOCATED":
                tag_lbl = f"#{it_id} Relocated (Safe Zone)"
                cv2.rectangle(overlay, (x, max(0, y - 20)), (x + len(tag_lbl) * 7 + 10, y), (30, 80, 35), -1)
                cv2.putText(overlay, tag_lbl, (x + 5, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 255, 220), 1, cv2.LINE_AA)
            else:
                # Original un-cleared item
                if sprite.is_liquid:
                    box_color = (32, 70, 200)
                    tag_lbl = f"#{it_id} Spill Threat"
                elif sprite.zone == "ZONE_1_PRIMARY_REACH":
                    box_color = (45, 125, 215)
                    tag_lbl = f"#{it_id} Encroaching"
                else:
                    box_color = (78, 117, 87)
                    tag_lbl = f"#{it_id} Peripheral"

                cv2.rectangle(overlay, (x, y), (x + pw, y + ph), box_color, 1, cv2.LINE_AA)
                cv2.rectangle(overlay, (x, max(0, y - 20)), (x + len(tag_lbl) * 7 + 10, y), box_color, -1)
                cv2.putText(overlay, tag_lbl, (x + 5, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        alpha = 0.88
        return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    def get_provisional_telemetry(self) -> Dict[str, Any]:
        """
        Computes real-time provisional Zen score, cleared counts, and hazard resolutions.
        """
        total = len(self.sprites)
        relocated = sum(1 for s in self.sprites.values() if s.status == "RELOCATED")
        stowed = sum(1 for s in self.sprites.values() if s.status == "STOWED")
        cleared = relocated + stowed

        base_score = self.telemetry.get("zen_score", 30)
        provisional = min(100, int(round(base_score + (100 - base_score) * (cleared / max(1, total)) * 0.94))) if total > 0 else 95
        score_delta = f"+{provisional - base_score} pts" if provisional >= base_score else "+0 pts"

        liquid_threat = any(s.is_liquid and s.status == "ORIGINAL" for s in self.sprites.values())

        return {
            "total_items": total,
            "cleared_count": cleared,
            "relocated_count": relocated,
            "stowed_count": stowed,
            "remaining_count": total - cleared,
            "provisional_score": provisional,
            "score_delta": score_delta,
            "spill_risk_active": liquid_threat,
            "is_fully_cleared": (cleared == total and total > 0),
        }
