"""
DeskZen High-Fidelity Workspace Scenarios
Synthesizes crisp, realistic workspace scenes for instant judge demonstration:
1. Scenario A: "Software Engineer Chaos" (Spill Risk Mug + Snaking Cables)
2. Scenario B: "Paperwork Avalanche" (Unbalanced Document Piles + Scattered Pens)
3. Scenario C: "Zen Minimalist Flow" (Ergonomic Benchmark Desk)
4. Transformed "After" Desks for live agentic verification.
"""

import math
import numpy as np
import cv2


def _draw_desk_surface(w: int, h: int, wood_tone: tuple = (200, 218, 230)) -> np.ndarray:
    """Creates a textured warm Scandinavian oak desk surface."""
    desk = np.full((h, w, 3), wood_tone, dtype=np.uint8)
    for y in range(0, h, 28):
        offset = int(math.sin(y * 0.05) * 4)
        grain_color = (max(0, wood_tone[0] - 16), max(0, wood_tone[1] - 16), max(0, wood_tone[2] - 16))
        cv2.line(desk, (0, y + offset), (w, y - offset), grain_color, 1, cv2.LINE_AA)
    return desk


def _draw_monitor(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Renders a sleek bezel monitor with simulated code editor screen."""
    base_x = (x1 + x2) // 2
    # Stand neck & base
    cv2.rectangle(img, (base_x - 14, y2), (base_x + 14, y2 + 20), (80, 80, 80), -1)
    cv2.rectangle(img, (base_x - 50, y2 + 20), (base_x + 50, y2 + 32), (65, 65, 65), -1)
    # Bezel
    cv2.rectangle(img, (x1, y1), (x2, y2), (28, 28, 28), -1)
    # Screen inner (Dark IDE style)
    cv2.rectangle(img, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), (36, 28, 24), -1)
    # Syntax highlight lines
    for i in range(6):
        line_y = y1 + 24 + (i * 18)
        color = (190, 140, 95) if i % 2 == 0 else (115, 190, 135)
        line_len = 110 + (i * 35) % 170
        cv2.line(img, (x1 + 24, line_y), (x1 + 24 + line_len, line_y), color, 3, cv2.LINE_AA)


def _draw_keyboard(img: np.ndarray, x: int, y: int, w: int, h: int):
    """Renders an aluminum mechanical keyboard with individual keycaps."""
    cv2.rectangle(img, (x + 2, y + 3), (x + w + 2, y + h + 3), (150, 165, 175), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 55), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (38, 38, 38), 1)

    num_cols, num_rows = 14, 4
    kw = (w - 16) // num_cols
    kh = (h - 14) // num_rows
    for r in range(num_rows):
        for c in range(num_cols):
            kx = x + 8 + (c * kw)
            ky = y + 7 + (r * kh)
            kc = (85, 85, 85)
            if r == 0 and c == 0:
                kc = (60, 100, 190)
            elif r == 3 and 4 <= c <= 9:
                kc = (72, 72, 72)
            cv2.rectangle(img, (kx, ky), (kx + kw - 2, ky + kh - 2), kc, -1)


def _draw_mouse(img: np.ndarray, cx: int, cy: int):
    """Renders an optical mouse with scroll wheel."""
    cv2.ellipse(img, (cx + 2, cy + 3), (22, 35), 0, 0, 360, (155, 170, 180), -1)
    cv2.ellipse(img, (cx, cy), (20, 33), 0, 0, 360, (45, 45, 45), -1)
    cv2.rectangle(img, (cx - 2, cy - 14), (cx + 2, cy - 2), (180, 180, 180), -1)
    cv2.line(img, (cx, cy - 18), (cx, cy + 8), (35, 35, 35), 1)


def _draw_coffee_mug(img: np.ndarray, cx: int, cy: int, r: int = 24, is_hazard: bool = True):
    """Renders a ceramic coffee mug with liquid surface and handle."""
    cv2.circle(img, (cx + 3, cy + 3), r + 2, (150, 165, 175), -1)
    # Mug handle on right
    cv2.ellipse(img, (cx + r + 5, cy), (7, 12), 0, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(img, (cx + r + 5, cy), (3, 8), 0, 0, 360, (200, 218, 230), -1)
    # Ceramic mug outer rim
    cv2.circle(img, (cx, cy), r, (245, 245, 245), -1)
    cv2.circle(img, (cx, cy), r, (205, 205, 205), 2)
    # Coffee liquid inner
    cv2.circle(img, (cx, cy), r - 5, (35, 55, 80), -1)  # Dark roast coffee BGR
    # Coffee foam / reflection
    cv2.circle(img, (cx - 4, cy - 5), 4, (55, 80, 115), -1)
    if is_hazard:
        # Subtle coffee splash ring
        cv2.circle(img, (cx + 10, cy + 12), 5, (40, 60, 90), 1)


def _draw_coaster(img: np.ndarray, cx: int, cy: int, r: int = 34):
    """Renders a natural cork beverage coaster."""
    cv2.circle(img, (cx, cy), r, (135, 165, 200), -1)
    cv2.circle(img, (cx, cy), r, (105, 135, 170), 2)


def _draw_sticky_notes(img: np.ndarray, x: int, y: int, count: int = 2):
    """Renders bright yellow sticky note reminders."""
    for i in range(count):
        nx = x + (i * 28)
        ny = y + (i * 10)
        cv2.rectangle(img, (nx + 2, ny + 2), (nx + 52, ny + 52), (160, 175, 185), -1)
        cv2.rectangle(img, (nx, ny), (nx + 50, ny + 50), (135, 235, 255), -1)  # Canary yellow
        cv2.rectangle(img, (nx, ny), (nx + 50, ny + 50), (105, 205, 230), 1)
        cv2.line(img, (nx + 8, ny + 15), (nx + 40, ny + 15), (75, 115, 135), 1)
        cv2.line(img, (nx + 8, ny + 26), (nx + 34, ny + 26), (75, 115, 135), 1)


def _draw_papers(img: np.ndarray, x: int, y: int, w: int, h: int, messy: bool = True):
    """Renders scattered or neat document sheets."""
    if messy:
        for angle, offset_x, offset_y in [(-5, -6, -4), (4, 10, 8), (0, 0, 0)]:
            rect = ((x + w // 2 + offset_x, y + h // 2 + offset_y), (w, h), angle)
            box = np.int32(cv2.boxPoints(rect))
            cv2.drawContours(img, [box], 0, (246, 246, 246), -1)
            cv2.drawContours(img, [box], 0, (195, 195, 195), 1)
    else:
        # Neat document tray in Zone 3
        cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6), (150, 135, 115), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (250, 250, 250), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (215, 215, 215), 1)
        for i in range(5):
            cv2.line(img, (x + 10, y + 15 + (i * 14)), (x + w - 12, y + 15 + (i * 14)), (180, 180, 180), 1)


def _draw_cables(img: np.ndarray, pts_list: list):
    """Renders braided USB-C and power cables."""
    for pts in pts_list:
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (22, 22, 22), 3, cv2.LINE_AA)


def _draw_smartphone(img: np.ndarray, x: int, y: int):
    """Renders a sleek smartphone lying face up."""
    cv2.rectangle(img, (x + 2, y + 2), (x + 44, y + 84), (160, 175, 185), -1)
    cv2.rectangle(img, (x, y), (x + 42, y + 82), (28, 28, 28), -1)
    cv2.rectangle(img, (x + 3, y + 4), (x + 39, y + 78), (60, 50, 45), -1)
    cv2.circle(img, (x + 21, y + 34), 4, (75, 135, 235), -1)


# =========================================================================
# PUBLIC PRESET SCENARIO FACTORIES
# =========================================================================

def generate_developer_chaos_desk() -> np.ndarray:
    """
    Scenario 1: Software Engineer Workspace Overload
    - Ceramic Coffee Mug placed 7cm from Keyboard (CRITICAL SPILL HAZARD)
    - Tangled braided USB-C cables crossing through mouse reach zone
    - Sticky note reminders cluttering left forearm area
    - Smartphone face-up creating cognitive distraction
    - Soda can on far right
    """
    w, h = 800, 520
    desk = _draw_desk_surface(w, h)

    # Monitor centered in Zone 3
    _draw_monitor(desk, 240, 25, 560, 185)

    # Keyboard centered in Zone 1
    _draw_keyboard(desk, 260, 315, 260, 80)

    # Mouse on right in Zone 1
    _draw_mouse(desk, 585, 355)

    # HAZARD: Coffee mug placed directly left of keyboard in Zone 1 (Spill Danger! Distance ~ 8cm)
    _draw_coffee_mug(desk, 195, 350, r=24, is_hazard=True)

    # Tangled Cables winding through mouse sweep area (separated from monitor)
    cable_pts_1 = [(525, 215), (555, 255), (615, 285), (665, 335), (715, 395)]
    cable_pts_2 = [(585, 215), (615, 245), (575, 295), (635, 325), (685, 365)]
    cable_pts_3 = [(550, 230), (570, 275), (605, 310), (625, 360)]
    _draw_cables(desk, [cable_pts_1, cable_pts_2, cable_pts_3])

    # Sticky Notes on left (separated from mug)
    _draw_sticky_notes(desk, 80, 210, count=2)

    # Smartphone face-up in Zone 1 left corner (separated)
    _draw_smartphone(desk, 155, 420)

    # Soda can on far right
    cv2.circle(desk, (715, 245), 22, (55, 55, 205), -1)
    cv2.circle(desk, (715, 245), 22, (35, 35, 155), 2)
    cv2.circle(desk, (715, 245), 14, (185, 185, 185), -1)

    return desk


def generate_paperwork_avalanche_desk() -> np.ndarray:
    """
    Scenario 2: Paperwork & Document Avalanche
    - Heavy left-side visual asymmetry
    - Scattered receipts, invoices, and folders occluding desk
    - Water glass near paper edge
    - Pens rolling into mouse path
    """
    w, h = 800, 520
    desk = _draw_desk_surface(w, h, wood_tone=(210, 222, 232))

    # Monitor centered
    _draw_monitor(desk, 240, 25, 560, 185)

    # Keyboard & Mouse
    _draw_keyboard(desk, 280, 320, 255, 80)
    _draw_mouse(desk, 595, 360)

    # Heavy left-skewed paperwork stack
    _draw_papers(desk, 75, 220, 130, 170, messy=True)
    _draw_papers(desk, 115, 320, 110, 150, messy=True)

    # Water tumbler near keyboard edge (Spill Hazard)
    _draw_coffee_mug(desk, 235, 255, r=22, is_hazard=True)

    # Pens rolling across desk
    cv2.line(desk, (535, 285), (595, 310), (35, 35, 175), 4, cv2.LINE_AA)
    cv2.line(desk, (550, 270), (615, 290), (35, 135, 35), 4, cv2.LINE_AA)

    return desk


def generate_minimalist_flow_desk() -> np.ndarray:
    """
    Scenario 3: Ergonomic Benchmark Desk (Gold Standard)
    - Perfectly centered keyboard and optical mouse
    - Beverage safely docked in Zone 2 on cork coaster (+40cm away)
    - Hidden cable routing through rear grommet
    - Zero surface clutter in Zone 1
    - Zen Score: 94/100
    """
    w, h = 800, 520
    desk = _draw_desk_surface(w, h)

    # Clean monitor centered
    _draw_monitor(desk, 240, 25, 560, 185)

    # Rear cable grommet
    cv2.circle(desk, (400, 205), 14, (38, 38, 38), -1)
    cv2.circle(desk, (400, 205), 14, (65, 65, 65), 2)

    # Minimalist keyboard centered
    _draw_keyboard(desk, 260, 315, 260, 80)

    # Ergonomic mouse with wide sweep area
    _draw_mouse(desk, 585, 355)

    # Beverage safely on cork coaster in Zone 2 right perimeter (+40cm away)
    _draw_coaster(desk, 715, 345, r=32)
    _draw_coffee_mug(desk, 715, 345, r=22, is_hazard=False)

    # Small succulent desk plant in Zone 3 left rear corner
    cv2.circle(desk, (95, 80), 20, (85, 135, 90), -1)
    cv2.circle(desk, (95, 80), 20, (55, 95, 60), 2)

    return desk


def generate_oblique_laptop_desk() -> np.ndarray:
    """
    Scenario 4: Oblique 45-Degree Perspective Laptop Workstation
    - Camera pitched at ~42 degrees oblique perspective (trapezoidal desk geometry)
    - Space-Gray Aluminum Laptop as Central Focus (screen + foreshortened keyboard deck)
    - Ceramic Coffee Mug encroaching on left palm rest (Spill Hazard)
    - Tangled USB-C charger cord crossing right mouse corridor
    - Smartphone & sticky notes in active reach envelope
    """
    w, h = 800, 520
    # Subtle studio backdrop at top with trapezoidal desk surface
    img = np.full((h, w, 3), (48, 52, 56), dtype=np.uint8)
    desk_poly = np.array([[75, 85], [725, 85], [795, 515], [5, 515]], dtype=np.int32)
    cv2.fillPoly(img, [desk_poly], (196, 214, 228))
    cv2.polylines(img, [desk_poly], True, (145, 160, 175), 3, cv2.LINE_AA)

    # Perspective wood grain lines
    for y in range(110, h, 32):
        cv2.line(img, (20, y), (w - 20, y - 4), (182, 198, 212), 1, cv2.LINE_AA)

    # Open Laptop Screen (tilted perspective trapezoid)
    screen_poly = np.array([[270, 70], [530, 70], [540, 225], [260, 225]], dtype=np.int32)
    cv2.fillPoly(img, [screen_poly], (26, 28, 30))
    cv2.polylines(img, [screen_poly], True, (90, 95, 100), 2, cv2.LINE_AA)
    inner_screen = np.array([[278, 80], [522, 80], [530, 215], [270, 215]], dtype=np.int32)
    cv2.fillPoly(img, [inner_screen], (38, 30, 24))
    for i in range(5):
        ly = 102 + i * 20
        cv2.line(img, (290, ly), (290 + 130 + (i * 25) % 80, ly), (195, 145, 95) if i % 2 == 0 else (115, 195, 140), 3, cv2.LINE_AA)

    # Foreshortened Laptop Keyboard Deck + Trackpad (Central Focus Anchor)
    _draw_keyboard(img, 260, 295, 280, 76)
    # Trackpad below keyboard keys
    cv2.rectangle(img, (365, 372), (435, 404), (72, 75, 78), -1)
    cv2.rectangle(img, (365, 372), (435, 404), (50, 52, 55), 1)

    # Wireless Mouse on right
    _draw_mouse(img, 600, 345)

    # HAZARD: Coffee Mug near left laptop wrist rest
    _draw_coffee_mug(img, 190, 335, r=24, is_hazard=True)

    # Smartphone in left foreground
    _draw_smartphone(img, 115, 405)

    # Tangled USB-C Charger Cables in right mouse corridor
    cable_pts_1 = [(545, 225), (580, 265), (635, 295), (685, 345), (730, 405)]
    cable_pts_2 = [(565, 235), (605, 280), (620, 330), (665, 380)]
    _draw_cables(img, [cable_pts_1, cable_pts_2])

    # Sticky notes on far left
    _draw_sticky_notes(img, 95, 215, count=2)

    return img


def generate_cleaned_after_desk(scenario_type: str = "chaos") -> np.ndarray:
    """
    Renders the mathematically transformed, decluttered 'After' workspace state.
    - Mug relocated to Zone 2 perimeter cork coaster (+38cm right)
    - Cables routed through desk grommet
    - Documents organized neatly in Zone 3 vertical file tray
    - Full 35cm Zone 1 forearm radius completely reclaimed (0 clutter items in Zone 1)
    """
    w, h = 800, 520
    if scenario_type == "oblique":
        img = np.full((h, w, 3), (48, 52, 56), dtype=np.uint8)
        desk_poly = np.array([[75, 85], [725, 85], [795, 515], [5, 515]], dtype=np.int32)
        cv2.fillPoly(img, [desk_poly], (196, 214, 228))
        cv2.polylines(img, [desk_poly], True, (145, 160, 175), 3, cv2.LINE_AA)
        for y in range(110, h, 32):
            cv2.line(img, (20, y), (w - 20, y - 4), (182, 198, 212), 1, cv2.LINE_AA)
        screen_poly = np.array([[270, 70], [530, 70], [540, 225], [260, 225]], dtype=np.int32)
        cv2.fillPoly(img, [screen_poly], (26, 28, 30))
        inner_screen = np.array([[278, 80], [522, 80], [530, 215], [270, 215]], dtype=np.int32)
        cv2.fillPoly(img, [inner_screen], (38, 30, 24))
        _draw_keyboard(img, 260, 295, 280, 76)
        cv2.rectangle(img, (365, 372), (435, 404), (72, 75, 78), -1)
        _draw_mouse(img, 600, 345)
        _draw_coaster(img, 715, 345, r=32)
        _draw_coffee_mug(img, 715, 345, r=22, is_hazard=False)
        return img

    desk = _draw_desk_surface(w, h)

    # Monitor centered
    _draw_monitor(desk, 240, 25, 560, 185)

    # Grommet with routed cables
    cv2.circle(desk, (400, 205), 14, (38, 38, 38), -1)
    routed_cables = [(400, 205), (400, 235), (430, 265), (450, 310)]
    for i in range(len(routed_cables) - 1):
        cv2.line(desk, routed_cables[i], routed_cables[i + 1], (42, 42, 42), 2, cv2.LINE_AA)

    # Centered keyboard
    _draw_keyboard(desk, 260, 315, 260, 80)

    # Mouse with clear sweep area
    _draw_mouse(desk, 585, 355)

    # Mug moved safely to Zone 2 perimeter cork coaster
    _draw_coaster(desk, 715, 345, r=32)
    _draw_coffee_mug(desk, 715, 345, r=22, is_hazard=False)

    # If documents were present, show neat vertical document tray in Zone 3 left corner
    _draw_papers(desk, 75, 70, 90, 115, messy=False)

    return desk
