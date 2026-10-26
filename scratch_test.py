import cv2, math
import numpy as np
from core.desk_scenarios import (
    generate_developer_chaos_desk,
    generate_paperwork_avalanche_desk,
    generate_minimalist_flow_desk,
    generate_cleaned_after_desk
)

CM_PER_PIXEL = 0.15

def analyze_test(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_pixels = float(h * w)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(np.var(laplacian))

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_hist, _ = np.histogram(grad_mag, bins=32, range=(0, 256), density=True)
    grad_hist = grad_hist[grad_hist > 0]
    shannon_entropy = -float(np.sum(grad_hist * np.log2(grad_hist)))

    blurred = cv2.bilateralFilter(gray, 7, 50, 50)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

    min_area = 400
    max_area = total_pixels * 0.45

    user_x = w / 2.0
    user_y = float(h)
    zone1_radius_px = int(35.0 / CM_PER_PIXEL)
    zone2_radius_px = int(55.0 / CM_PER_PIXEL)

    keyboard_anchor = None
    monitor_anchor = None
    mouse_anchor = None
    raw_candidates = []

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

        if bw > w * 0.70 or bh > h * 0.70 or bh <= 8 or bw <= 8:
            continue
        ar = float(bw) / float(bh + 1e-5)
        sol = float(area) / float(bw * bh + 1e-5)

        if (0.20 * w < cx < 0.80 * w) and (cy > h * 0.45) and (ar > 1.8) and (area > 4500):
            if keyboard_anchor is None or area > keyboard_anchor["area_px"]:
                keyboard_anchor = {"bbox": [x, y, bw, bh], "centroid": [cx, cy], "area_px": area}
                continue

        if (0.20 * w < cx < 0.80 * w) and (cy < h * 0.45) and (area > 5500):
            if monitor_anchor is None or area > monitor_anchor["area_px"]:
                monitor_anchor = {"bbox": [x, y, bw, bh], "centroid": [cx, cy], "area_px": area}
                continue

        if (0.58 * w < cx < 0.85 * w) and (cy > h * 0.45) and (0.40 < ar < 0.85) and (800 < area < 3000):
            if mouse_anchor is None:
                mouse_anchor = {"bbox": [x, y, bw, bh], "centroid": [cx, cy], "area_px": area}
                continue

        raw_candidates.append({
            "bbox": [x, y, bw, bh], "centroid": [cx, cy], "area_px": int(area), "ar": ar, "sol": sol
        })

    # Group overlapping items (e.g. mug sitting on coaster)
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
            # Check center distance
            dist = math.sqrt((c1["centroid"][0] - c2["centroid"][0])**2 + (c1["centroid"][1] - c2["centroid"][1])**2)
            if dist < 32.0:
                # Merge into single item (e.g. coaster + mug)
                skip_cands.add(j)
                merged["area_px"] = max(c1["area_px"], c2["area_px"])
                merged["is_composite_dock"] = True
                break
        merged_cands.append(merged)

    items = []
    total_clutter_pixels = 0

    kb_box = keyboard_anchor["bbox"] if keyboard_anchor else [int(w * 0.32), int(h * 0.60), int(w * 0.36), int(h * 0.16)]
    kx_min, ky_min, kw_b, kh_b = kb_box
    kx_max, ky_max = kx_min + kw_b, ky_min + kh_b

    for cand in merged_cands:
        x, y, bw, bh = cand["bbox"]
        cx, cy = cand["centroid"]
        area = cand["area_px"]
        ar = cand["ar"]
        sol = cand["sol"]

        dist_px = math.sqrt((cx - user_x)**2 + (cy - user_y)**2)
        dist_cm = round(dist_px * CM_PER_PIXEL, 1)

        if dist_px <= zone1_radius_px:
            zone = "ZONE_1_PRIMARY_REACH"
        elif dist_px <= zone2_radius_px:
            zone = "ZONE_2_SECONDARY_REACH"
        else:
            zone = "ZONE_3_PERIPHERY"

        side = "LEFT" if cx < (w / 2.0) else "RIGHT"

        pad = 8
        crop = img[max(0, y-pad):min(h, y+bh+pad), max(0, x-pad):min(w, x+bw+pad)]
        mean_bgr = cv2.mean(crop)[:3] if crop.size > 0 else (128, 128, 128)
        is_yellow = (mean_bgr[2] > 190 and mean_bgr[1] > 190 and mean_bgr[0] < 170)
        is_white_paper = (mean_bgr[0] > 210 and mean_bgr[1] > 210 and mean_bgr[2] > 210)

        is_liquid = False
        target_dest = "Drawer / Storage Tray"
        target_coords = (int(w * 0.12), int(h * 0.18))

        # Distance to keyboard boundary
        closest_x = max(kx_min, min(cx, kx_max))
        closest_y = max(ky_min, min(cy, ky_max))
        edge_dist_cm = round(math.sqrt((cx - closest_x)**2 + (cy - closest_y)**2) * CM_PER_PIXEL, 1)

        if is_white_paper and area > 1200:
            if zone == "ZONE_3_PERIPHERY":
                semantic_name = "Document Organizer (Zone 3 Safe)"
                target_dest = "Maintain Current Position"
                target_coords = (int(cx), int(cy))
            else:
                semantic_name = "Loose Documents & Invoices"
                target_dest = "Zone 3 Vertical Document Tray"
                target_coords = (int(w * 0.12), int(h * 0.18))
        elif sol < 0.22 and bw > 65 and bh > 65:
            if cy < h * 0.45 and area < 2000:
                semantic_name = "Neat Desk Cable Run"
                target_dest = "Maintain Current Position"
                target_coords = (int(cx), int(cy))
            else:
                semantic_name = "Tangled USB-C & Power Cords"
                target_dest = "Zone 3 Cable Clip / Grommet"
                target_coords = (int(w * 0.50), int(h * 0.08))
        elif is_yellow and bw < 75 and bh < 75:
            semantic_name = "Sticky Note Reminder Cluster"
            target_dest = "Monitor Bezel / Digital Kanban"
            target_coords = (int(w * 0.50), int(h * 0.12))
        elif (ar < 0.65 or ar > 1.55) and (sol > 0.35) and bw < 100 and bh < 100:
            semantic_name = "Smartphone (Notification Distraction)"
            target_dest = "Zone 2 Phone Dock / Stand"
            target_coords = (int(w * 0.86), int(h * 0.45))
        elif (0.70 < ar < 1.35) and (bw < 85 and bh < 85):
            is_liquid = True
            if edge_dist_cm >= 25.0 and zone in ("ZONE_2_SECONDARY_REACH", "ZONE_3_PERIPHERY"):
                semantic_name = "Ceramic Mug on Safe Coaster"
                target_dest = "Maintain Current Position"
                target_coords = (int(cx), int(cy))
            else:
                semantic_name = "Ceramic Coffee Mug (Spill Hazard)"
                target_dest = "Zone 2 Dedicated Coaster"
                target_coords = (int(w * 0.88), int(h * 0.65))
        elif (0.65 < ar < 1.4) and (area > 800) and cx > (w * 0.65):
            semantic_name = "Beverage Tumbler / Can"
            is_liquid = True
            target_dest = "Zone 2 Dedicated Coaster"
            target_coords = (int(w * 0.88), int(h * 0.65))
        elif zone == "ZONE_3_PERIPHERY" and area < 1200:
            semantic_name = "Desk Plant / Minimal Decor"
            target_dest = "Maintain Current Position"
            target_coords = (int(cx), int(cy))
        else:
            semantic_name = f"Peripheral Item #{len(items) + 1}"
            target_dest = "Stow in Drawer"
            target_coords = (int(w * 0.14), int(h * 0.86))

        disp_px = math.sqrt((target_coords[0] - cx)**2 + (target_coords[1] - cy)**2)
        disp_cm = round(disp_px * CM_PER_PIXEL, 1)

        items.append({
            "id": len(items) + 1,
            "semantic_name": semantic_name,
            "bbox": [x, y, bw, bh],
            "centroid": [round(cx, 1), round(cy, 1)],
            "area_px": int(area),
            "zone": zone,
            "dist_from_torso_cm": dist_cm,
            "edge_dist_to_kb_cm": edge_dist_cm,
            "side": side,
            "is_potential_liquid_container": bool(is_liquid),
            "target_destination": target_dest,
            "displacement_cm": disp_cm
        })
        total_clutter_pixels += area

    surface_blocked_pct = round((total_clutter_pixels / total_pixels) * 100, 1)

    # Balance
    left_items = [it for it in items if it["side"] == "LEFT"]
    right_items = [it for it in items if it["side"] == "RIGHT"]
    left_area = sum(it["area_px"] for it in left_items)
    right_area = sum(it["area_px"] for it in right_items)
    combined_area = max(1.0, float(left_area + right_area))
    left_ratio = round(left_area / combined_area, 2)
    right_ratio = round(right_area / combined_area, 2)

    if left_ratio > 0.68:
        balance_verdict = "HEAVY_LEFT_BIAS"
    elif right_ratio > 0.68:
        balance_verdict = "HEAVY_RIGHT_BIAS"
    else:
        balance_verdict = "HARMONIOUS_BALANCE"

    # Cable snarl: filter out monitor box, keyboard box, and upper edge
    edges = cv2.Canny(blurred, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=28, maxLineGap=10)
    desk_cables = []
    if lines is not None:
        for l in lines:
            c = np.asarray(l).ravel()
            x1, y1, x2, y2 = int(c[0]), int(c[1]), int(c[2]), int(c[3])
            # Exclude monitor box
            if (230 < x1 < 570 and y1 < 210) or (230 < x2 < 570 and y2 < 210):
                continue
            # Exclude keyboard box
            if (250 < x1 < 530 and 310 < y1 < 400) or (250 < x2 < 530 and 310 < y2 < 400):
                continue
            # Exclude upper perimeter
            if y1 < 120 and y2 < 120:
                continue
            desk_cables.append(l)
    cable_snarl_detected = len(desk_cables) > 4

    # Spill Hazard
    liquid_containers = [it for it in items if it["is_potential_liquid_container"]]
    spill_hazard = False
    for l_item in liquid_containers:
        if l_item["edge_dist_to_kb_cm"] < 25.0:
            spill_hazard = True

    actionable_items = [it for it in items if it["target_destination"] != "Maintain Current Position"]
    actionable_zone1 = [it for it in actionable_items if it["zone"] == "ZONE_1_PRIMARY_REACH"]

    raw_zen = 96.0 - (len(actionable_zone1) * 16.0) - (len(actionable_items) * 4.0) - (surface_blocked_pct * 1.5)
    if spill_hazard:
        raw_zen -= 22.0
    if cable_snarl_detected:
        raw_zen -= 12.0
    if balance_verdict != "HARMONIOUS_BALANCE":
        raw_zen -= 14.0

    entropy_normalized = min(95.0, max(15.0, (shannon_entropy * 12.0) + (lap_var / 70.0) + (len(actionable_items) * 4.0)))
    raw_zen -= max(0.0, (entropy_normalized - 22.0) * 0.18)

    zen_score = max(5, min(96, int(round(raw_zen))))

    return {
        "zen_score": zen_score,
        "spill_hazard": spill_hazard,
        "cable_snarl": cable_snarl_detected,
        "balance": balance_verdict,
        "items": items,
        "actionable_count": len(actionable_items)
    }

for name, sc in [
    ("Developer Chaos", generate_developer_chaos_desk()),
    ("Paperwork Avalanche", generate_paperwork_avalanche_desk()),
    ("Minimalist Flow", generate_minimalist_flow_desk()),
    ("Cleaned After", generate_cleaned_after_desk())
]:
    res = analyze_test(sc)
    print(f"=== {name} ===")
    print(f"Zen Score: {res['zen_score']}/100 | Spill: {res['spill_hazard']} | Cables: {res['cable_snarl']} | Balance: {res['balance']} | Actionable: {res['actionable_count']}")
    for it in res["items"]:
        print(f"  - #{it['id']}: {it['semantic_name']} [{it['zone']}] -> {it['target_destination']} (+{it['displacement_cm']}cm)")
    print()
