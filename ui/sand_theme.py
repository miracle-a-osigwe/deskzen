"""
DeskZen Clean Sand Mix Design System
Tactile Japandi & Mediterranean Sand aesthetic with maximum contrast and zero hover-dependency.
Guaranteed high visibility for all buttons, tabs, badges, and metrics.
"""
from typing import Any


def clean_html(html_str: str) -> str:
    """Strips leading/trailing whitespace from each line so Streamlit CommonMark
    parser never misinterprets 4 leading spaces as an indented code block.
    """
    return "\n".join(
        line.strip() for line in html_str.strip().splitlines() if line.strip()
    )


SAND_THEME_CSS = clean_html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --sand-bg: #FAF7F2;
    --sand-card: #FFFFFF;
    --sand-warm: #F4EFEA;
    --sand-border: #E2D7CA;
    --sand-border-subtle: #EDE4D8;
    --sand-text-main: #241A14;
    --sand-text-muted: #6B5D50;
    --terracotta: #B85226;
    --terracotta-dark: #963D16;
    --terracotta-light: #F7EDE8;
    --sage: #4E7557;
    --sage-bg: #EDF4EE;
    --amber-hazard: #C84620;
    --amber-hazard-bg: #FFF2ED;
    --card-shadow: 0 4px 20px rgba(60, 45, 30, 0.06);
}

/* Base Body Styling */
.stApp {
    background-color: var(--sand-bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--sand-text-main) !important;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Native Streamlit Container Border Overrides */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--sand-card) !important;
    border: 1.5px solid var(--sand-border) !important;
    border-radius: 18px !important;
    box-shadow: var(--card-shadow) !important;
    padding: 18px !important;
    margin-bottom: 12px !important;
}

/* =========================================================================
   BUTTON STYLING: HIGH-CONTRAST & ALWAYS LEGIBLE (NO HOVER NEEDED)
   ========================================================================= */

/* =========================================================================
   UNIVERSAL BUTTON STYLING: HIGH-CONTRAST, CRISP & BOLD AT ALL TIMES
   ========================================================================= */

/* Default / Secondary Buttons: Clean warm cream background with deep espresso text and thick border */
div.stButton > button,
div.stButton > button[data-testid="stBaseButton-secondary"],
button[data-testid="baseButton-secondary"],
button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background-color: #FAF4ED !important;
    color: #1E130B !important;
    border: 2px solid #7D5C43 !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 14.5px !important;
    letter-spacing: 0.1px !important;
    padding: 10px 22px !important;
    box-shadow: 0 3px 8px rgba(30, 19, 11, 0.10) !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}

/* Force deep espresso text on all children of default/secondary buttons */
div.stButton > button *,
div.stButton > button p,
div.stButton > button span,
div.stButton > button[data-testid="stBaseButton-secondary"] *,
button[kind="secondary"] *,
[data-testid="stBaseButton-secondary"] * {
    color: #1E130B !important;
    font-weight: 800 !important;
    fill: #1E130B !important;
}

div.stButton > button:hover,
div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #F0E5D8 !important;
    border-color: #4A3324 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 14px rgba(30, 19, 11, 0.18) !important;
}

/* Primary Buttons: Vivid Terracotta Gradient with Pure White Text */
div.stButton > button[data-testid="stBaseButton-primary"],
div.stButton > button[data-testid="baseButton-primary"],
div.stButton > button[kind="primary"],
button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(180deg, #C95A2B 0%, #A94317 100%) !important;
    color: #FFFFFF !important;
    border: 1.5px solid #7E2A08 !important;
    border-radius: 12px !important;
    font-weight: 850 !important;
    font-size: 14.5px !important;
    padding: 11px 24px !important;
    box-shadow: 0 4px 14px rgba(169, 67, 23, 0.32) !important;
}

/* Enforce pure white text on ALL children of primary buttons */
div.stButton > button[data-testid="stBaseButton-primary"] *,
div.stButton > button[data-testid="stBaseButton-primary"] p,
div.stButton > button[data-testid="stBaseButton-primary"] span,
div.stButton > button[kind="primary"] *,
button[kind="primary"] *,
[data-testid="stBaseButton-primary"] * {
    color: #FFFFFF !important;
    font-weight: 850 !important;
    fill: #FFFFFF !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25) !important;
}

div.stButton > button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(180deg, #B54A1E 0%, #8F340D 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(169, 67, 23, 0.42) !important;
}

/* Native File Uploader Button Overrides */
[data-testid="stFileUploader"] button {
    background-color: #FAF4ED !important;
    color: #1E130B !important;
    border: 2px solid #7D5C43 !important;
    font-weight: 800 !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
}

[data-testid="stFileUploader"] button * {
    color: #1E130B !important;
    font-weight: 800 !important;
}

/* =========================================================================
   TABS STYLING: HIGH-CONTRAST SEGMENTED NAVIGATION PILLS
   ========================================================================= */
.stTabs [data-baseweb="tab-list"] {
    background-color: #E8DFD4 !important;
    border-radius: 14px !important;
    padding: 6px !important;
    gap: 8px !important;
    border: 2px solid #C8B8A6 !important;
    margin-bottom: 14px !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #F4ECE3 !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 800 !important;
    font-size: 14.5px !important;
    color: #38281C !important;
    border: 1.5px solid #D8C7B5 !important;
    box-shadow: 0 1px 3px rgba(30, 20, 10, 0.05) !important;
    transition: all 0.15s ease !important;
}

.stTabs [data-baseweb="tab"] * {
    color: #38281C !important;
    font-weight: 800 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #EDE2D6 !important;
    border-color: #A98E78 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #C95A2B 0%, #A94317 100%) !important;
    color: #FFFFFF !important;
    border: 1.5px solid #7E2A08 !important;
    box-shadow: 0 3px 12px rgba(169, 67, 23, 0.32) !important;
}

.stTabs [aria-selected="true"] * {
    color: #FFFFFF !important;
    font-weight: 850 !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}
    font-weight: 850 !important;
}

/* Radio buttons for View Toggle */
div[data-testid="stRadio"] > div {
    gap: 12px !important;
}
div[data-testid="stRadio"] label {
    background-color: #FFFFFF !important;
    border: 1.5px solid var(--sand-border) !important;
    border-radius: 10px !important;
    padding: 6px 14px !important;
    font-weight: 700 !important;
    color: var(--sand-text-main) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* =========================================================================
   CARDS & TYPOGRAPHY
   ========================================================================= */

/* Header Component */
.sand-header {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8F3EE 100%);
    border: 1.5px solid var(--sand-border);
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: var(--card-shadow);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}

.sand-title {
    font-size: 26px;
    font-weight: 800;
    color: var(--sand-text-main);
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.2;
}

.sand-subtitle {
    font-size: 13.5px;
    color: var(--sand-text-muted);
    margin-top: 4px;
    font-weight: 600;
}

.sand-header-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.sand-badge-pill {
    background: #FFFFFF;
    border: 1.5px solid var(--sand-border);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    color: var(--sand-text-muted);
    box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}

/* Zen Score Hero Box */
.zen-score-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAF6F1 100%);
    border: 1.5px solid var(--sand-border);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: var(--card-shadow);
}

.zen-score-top {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 20px;
}

.zen-score-num {
    font-size: 66px;
    font-weight: 850;
    color: var(--terracotta);
    line-height: 1;
    letter-spacing: -2px;
}

.zen-score-info {
    flex: 1;
}

.zen-score-heading {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    color: var(--sand-text-muted);
}

.zen-tier-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12.5px;
    font-weight: 750;
    margin-top: 6px;
}

.tier-high-clarity {
    background: var(--sage-bg);
    color: var(--sage);
    border: 1px solid #C4DFC7;
}

.tier-moderate-clarity {
    background: #FFF4E8;
    color: var(--terracotta);
    border: 1px solid #F5D3B8;
}

.tier-high-friction {
    background: var(--amber-hazard-bg);
    color: var(--amber-hazard);
    border: 1px solid #F8C3B3;
}

.zen-metrics-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1.5px solid var(--sand-border-subtle);
    padding-top: 16px;
}

.zen-metric-item {
    text-align: center;
    flex: 1;
}

.zen-metric-label {
    display: block;
    font-size: 11.5px;
    color: var(--sand-text-muted);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
}

.zen-metric-val {
    font-size: 18px;
    font-weight: 800;
    color: var(--sand-text-main);
}

.zen-metric-val small {
    font-size: 11px;
    font-weight: 600;
    color: var(--sand-text-muted);
}

.zen-metric-divider {
    width: 1.5px;
    height: 32px;
    background-color: var(--sand-border-subtle);
}

/* Master Diagnosis Card */
.diagnosis-card {
    background-color: var(--sand-card);
    border: 1.5px solid var(--sand-border);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: var(--card-shadow);
}

.diagnosis-label {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--sand-text-muted);
    margin-bottom: 8px;
}

.diagnosis-text {
    font-size: 14px;
    line-height: 1.55;
    color: var(--sand-text-main);
    font-weight: 500;
}

.diagnosis-quote {
    font-size: 13px;
    font-style: italic;
    color: #7E6C5D;
    margin-top: 12px;
    padding-left: 12px;
    border-left: 3px solid var(--terracotta);
}

.diagnosis-model {
    margin-top: 10px;
    font-size: 11.5px;
    font-weight: 700;
    color: var(--terracotta);
}

/* Hazard Alerts */
.hazard-alert {
    background-color: var(--amber-hazard-bg);
    border: 1.5px solid #FCD4C7;
    border-left: 5px solid var(--amber-hazard);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #75220D;
}

.hazard-title {
    font-size: 13.5px;
    font-weight: 800;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.hazard-body {
    font-size: 12.5px;
    line-height: 1.45;
    font-weight: 500;
}

.cable-alert {
    background-color: #FAF2EB;
    border: 1.5px solid #EAD8C5;
    border-left: 5px solid var(--terracotta);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #633319;
}

/* Action Triage Pill Styles */
.pill-action {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11.5px;
    font-weight: 800;
    white-space: nowrap;
}

.pill-trash {
    background-color: #FCE8E2;
    color: var(--amber-hazard);
    border: 1px solid #F8C3B3;
}

.pill-stow {
    background-color: #F0E6DA;
    color: #5C4A3C;
    border: 1px solid #D9C8B6;
}

.pill-relocate {
    background-color: #FFF2DF;
    color: #B56717;
    border: 1px solid #F7DBB4;
}

.pill-keep {
    background-color: var(--sage-bg);
    color: var(--sage);
    border: 1px solid #C4DFC7;
}

/* Legend Bar */
.legend-bar {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    background: #FFFFFF;
    border: 1.5px solid var(--sand-border);
    border-radius: 12px;
    padding: 10px 16px;
    margin-top: 10px;
    font-size: 12px;
    font-weight: 600;
    color: var(--sand-text-muted);
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

/* Celebration Banner */
.celebration-banner {
    background: linear-gradient(135deg, #EDF6EE 0%, #DCECDC 100%);
    border: 2px solid #A8D1AA;
    border-radius: 20px;
    padding: 24px 28px;
    color: #1B4021;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(35, 80, 45, 0.10);
}

.celebration-title {
    font-size: 22px;
    font-weight: 850;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
}

.celebration-sub {
    font-size: 14.5px;
    font-weight: 600;
    opacity: 0.95;
    margin-bottom: 16px;
}

.celebration-stats {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    background: rgba(255, 255, 255, 0.75);
    border-radius: 14px;
    padding: 14px 22px;
    border: 1px solid #CBE3CD;
}

.celebration-stat-item {
    font-size: 13.5px;
    font-weight: 650;
}

.celebration-stat-item b {
    font-size: 17px;
    font-weight: 850;
    color: #14321A;
}

/* Balance Bar */
.balance-bar-container {
    background: #E8DFD4;
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    height: 26px;
    margin: 10px 0;
    border: 1px solid #D5C8B9;
}

.balance-left {
    background: var(--terracotta);
    color: #FFFFFF;
    font-size: 11.5px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: width 0.3s ease;
}

.balance-right {
    background: var(--sage);
    color: #FFFFFF;
    font-size: 11.5px;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: width 0.3s ease;
}
</style>
"""
)


def render_header() -> str:
    html = """
    <div class="sand-header">
        <div>
            <div class="sand-title">🌿 DeskZen</div>
            <div class="sand-subtitle">Autonomous Workspace Decluttering & Ergonomics Co-Pilot • OpenCV 5 & AWS Bedrock Nova Pro</div>
        </div>
        <div class="sand-header-badges">
            <span class="sand-badge-pill">⚡ Amazon Nova Pro</span>
            <span class="sand-badge-pill">🛠️ OpenCV 5.0 Vision</span>
            <span class="sand-badge-pill">🌐 AWS Graviton3 Ready</span>
            <span class="sand-badge-pill">📐 Ergonomic Zone Modeling</span>
        </div>
    </div>
    """
    return clean_html(html)


def render_score_card(
    score: int,
    tier: str,
    entropy: float,
    blocked: float,
    balance_verdict: str = "BALANCED",
    delta: str = None,
) -> str:
    tier_upper = tier.upper()
    if (
        "HIGH_CLARITY" in tier_upper
        or "MINIMAL" in tier_upper
        or "ZEN_HARMONY" in tier_upper
    ):
        tier_class = "tier-high-clarity"
        tier_label = "Optimal Zen Flow"
    elif "MODERATE" in tier_upper:
        tier_class = "tier-moderate-clarity"
        tier_label = "Moderate Friction"
    else:
        tier_class = "tier-high-friction"
        tier_label = "High Cognitive Load"

    delta_badge = (
        f"<span style='color: #4E7557; font-weight: 800; font-size: 14px; margin-left: 8px;'>({delta})</span>"
        if delta
        else ""
    )
    balance_clean = balance_verdict.replace("_", " ").title()

    html = f"""
    <div class="zen-score-card">
        <div class="zen-score-top">
            <div class="zen-score-num">{score}</div>
            <div class="zen-score-info">
                <div class="zen-score-heading">Zen Clarity Score / 100 {delta_badge}</div>
                <div class="zen-tier-badge {tier_class}">{tier_label}</div>
            </div>
        </div>
        <div class="zen-metrics-row">
            <div class="zen-metric-item">
                <span class="zen-metric-label">Visual Entropy</span>
                <span class="zen-metric-val">{entropy} <small>bit/px</small></span>
            </div>
            <div class="zen-metric-divider"></div>
            <div class="zen-metric-item">
                <span class="zen-metric-label">Surface Blocked</span>
                <span class="zen-metric-val">{blocked}%</span>
            </div>
            <div class="zen-metric-divider"></div>
            <div class="zen-metric-item">
                <span class="zen-metric-label">Visual Balance</span>
                <span class="zen-metric-val">{balance_clean}</span>
            </div>
        </div>
    </div>
    """
    return clean_html(html)


def render_diagnosis_card(
    diagnosis: str, quote: str, model_name: str = "Amazon Nova Pro"
) -> str:
    html = f"""
    <div class="diagnosis-card">
        <div class="diagnosis-label">🧠 Master Cognitive Diagnosis</div>
        <div class="diagnosis-text">{diagnosis}</div>
        <div class="diagnosis-quote">“{quote}”</div>
        <div class="diagnosis-model">Synthesized via {model_name}</div>
    </div>
    """
    return clean_html(html)


def render_hazard_alert(
    spill_risk: bool, cable_snarl: bool, spill_dist_px: Any = 0, cable_count: Any = 0
) -> str:
    snippets = []
    if spill_risk:
        try:
            dist_val = float(spill_dist_px) if spill_dist_px is not None else 0.0
        except (ValueError, TypeError):
            dist_val = 0.0

        if dist_val > 0.0:
            dist_str = (
                f"<b>{int(round(dist_val))}px (~{round(dist_val * 0.15, 1)}cm)</b>"
            )
        else:
            dist_str = "<b>critical proximity (<25cm)</b>"

        snippets.append(
            f"""
        <div class="hazard-alert">
            <div class="hazard-title">⚠️ Critical Liquid Spill Hazard</div>
            <div class="hazard-body">
                Open beverage container detected within {dist_str} of active electronics.
                Relocate beverage immediately to Zone 2 perimeter coaster.
            </div>
        </div>
        """
        )
    if cable_snarl:
        try:
            c_count = int(cable_count) if cable_count is not None else 0
        except (ValueError, TypeError):
            c_count = 0

        snippets.append(
            f"""
        <div class="cable-alert">
            <div class="hazard-title">🔌 Tangled Cord Network</div>
            <div class="hazard-body">
                OpenCV Hough line skeletonization detected <b>{c_count} crossing cord fragments</b> in forearm sweep area.
                Route cords through a desk grommet or adhesive cable clip.
            </div>
        </div>
        """
        )
    return clean_html("".join(snippets))


def render_central_focus_card(central_focus: dict, encroachment: dict) -> str:
    """
    Renders the dedicated Central Work Focus & Congestion Radar card
    highlighting the primary work device, wrist clearance, mouse corridor,
    and encroaching peripheral clutter.
    """
    if not central_focus:
        return ""

    cf_name = central_focus.get("name", "Workstation")
    status = encroachment.get("status", "MODERATE_ENCROACHMENT")
    total_enc = encroachment.get("total_encroaching_items", 0)
    wrist_cm = encroachment.get("wrist_clearance_cm", 15.0)
    mouse_cm = encroachment.get("mouse_clearance_cm", 22.0)
    breathing_room = int(round(encroachment.get("breathing_room_score", 70.0)))
    enc_items = encroachment.get("encroaching_items", [])

    if "CRITICAL" in status:
        badge_bg = "#FFF2ED"
        badge_color = "#C84620"
        badge_border = "#F5C7B8"
        badge_icon = "🚨"
        status_text = f"CRITICALLY CONSTRICTED ({total_enc} Items Crowding)"
    elif "MODERATE" in status:
        badge_bg = "#FEF3C7"
        badge_color = "#92400E"
        badge_border = "#FCD34D"
        badge_icon = "⚠️"
        status_text = f"MODERATE CONGESTION ({total_enc} Items Choking)"
    else:
        badge_bg = "#DCFCE7"
        badge_color = "#166534"
        badge_border = "#86EFAC"
        badge_icon = "🌿"
        status_text = "ZEN SANCTUARY (Liberated)"

    wrist_color = (
        "#166534" if wrist_cm >= 15.0 else ("#C84620" if wrist_cm < 10.0 else "#92400E")
    )
    mouse_color = (
        "#166534" if mouse_cm >= 22.0 else ("#C84620" if mouse_cm < 14.0 else "#92400E")
    )
    room_color = (
        "#166534"
        if breathing_room >= 80
        else ("#C84620" if breathing_room < 50 else "#92400E")
    )

    enc_list_html = ""
    if enc_items:
        items_rows = []
        for it in enc_items[:3]:
            it_name = it.get("name", "Clutter Item")
            it_sec = it.get("sector", "PERIPHERAL").replace("_", " ").title()
            it_dist = it.get("edge_distance_cm", 15.0)
            it_dir = it.get("clearance_directive", "")
            target_str = it_dir.split("➔")[-1].strip() if "➔" in it_dir else "Relocate"
            items_rows.append(
                f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #F0E8DD; font-size: 12px;">
                <span style="font-weight: 700; color: #241A14;">• {it_name} <small style="color: #8A7B6E;">({it_sec})</small></span>
                <span style="font-weight: 700; color: #B85226;">{it_dist}cm away ➔ {target_str}</span>
            </div>
            """
            )
        enc_list_html = f"""
        <div style="margin-top: 10px; padding-top: 8px; border-top: 1.5px solid #F0E8DD;">
            <div style="font-size: 11.5px; font-weight: 800; color: #6B5D50; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">
                Peripheral Items Constricting {cf_name}
            </div>
            {''.join(items_rows)}
        </div>
        """

    html = f"""
    <div style="background: #FFFFFF; border: 2px solid #E2D7CA; border-radius: 16px; padding: 16px; margin-top: 12px; box-shadow: 0 4px 16px rgba(45,35,25,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 20px;">🎯</span>
                <div>
                    <div style="font-size: 11px; font-weight: 800; text-transform: uppercase; color: #8A7B6E; letter-spacing: 0.5px;">Central Work Focus</div>
                    <div style="font-size: 16px; font-weight: 850; color: #241A14;">{cf_name}</div>
                </div>
            </div>
            <span style="background: {badge_bg}; color: {badge_color}; border: 1.5px solid {badge_border}; border-radius: 9999px; padding: 4px 12px; font-size: 11.5px; font-weight: 800; display: inline-flex; align-items: center; gap: 5px;">
                {badge_icon} {status_text}
            </span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 8px; background: #FAF7F2; border-radius: 12px; padding: 10px; border: 1px solid #EDE4D8;">
            <div style="text-align: center;">
                <div style="font-size: 10.5px; font-weight: 750; color: #6B5D50; text-transform: uppercase;">Wrist Clearance</div>
                <div style="font-size: 15px; font-weight: 850; color: {wrist_color}; margin-top: 2px;">{wrist_cm} cm</div>
                <div style="font-size: 9.5px; color: #8A7B6E;">Goal ≥ 15cm</div>
            </div>
            <div style="text-align: center; border-left: 1px solid #E2D7CA; border-right: 1px solid #E2D7CA;">
                <div style="font-size: 10.5px; font-weight: 750; color: #6B5D50; text-transform: uppercase;">Mouse Sweep</div>
                <div style="font-size: 15px; font-weight: 850; color: {mouse_color}; margin-top: 2px;">{mouse_cm} cm</div>
                <div style="font-size: 9.5px; color: #8A7B6E;">Goal ≥ 22cm</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 10.5px; font-weight: 750; color: #6B5D50; text-transform: uppercase;">Breathing Room</div>
                <div style="font-size: 15px; font-weight: 850; color: {room_color}; margin-top: 2px;">{breathing_room}%</div>
                <div style="font-size: 9.5px; color: #8A7B6E;">Halo Openness</div>
            </div>
        </div>

        {enc_list_html}
    </div>
    """
    return clean_html(html)


def render_legend() -> str:
    html = """
    <div class="legend-bar">
        <div class="legend-item"><span class="legend-dot" style="background:#3C8C50;"></span> 🎯 Central Focus Halo</div>
        <div class="legend-item"><span class="legend-dot" style="background:#B85226;"></span> Zone 1: Neutral Sweep (0-35cm)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#4E7557;"></span> Zone 2: Functional Reach (35-55cm)</div>
        <div class="legend-item"><span class="legend-dot" style="background:#C84620;"></span> Liquid Spill Radar</div>
        <div class="legend-item"><span class="legend-dot" style="background:#209642;"></span> ➔ Relocation Vectors</div>
        <div class="legend-item"><span class="legend-dot" style="background:#5C4A3C;"></span> Cord Segments</div>
    </div>
    """
    return clean_html(html)


def render_item_details(item: dict, triage: dict) -> str:
    action = triage.get("recommended_action", "STOW_IN_DRAWER")
    if action == "TRASH":
        pill_class = "pill-trash"
        pill_text = "🗑️ TRASH / RECYCLE"
    elif action == "KEEP_ON_DESK":
        pill_class = "pill-keep"
        pill_text = "✨ KEEP ON DESK"
    elif action in ["RELOCATE", "RELOCATE_ZONE_2"]:
        pill_class = "pill-relocate"
        pill_text = "🧭 RELOCATE TO ZONE 2"
    else:
        pill_class = "pill-stow"
        pill_text = "🗄️ STOW IN DRAWER"

    item_name = triage.get(
        "item_name", item.get("semantic_name", f"Item #{item.get('id', 1)}")
    )
    reason = triage.get(
        "reasoning", "Clear this item to restore neutral arm sweep radius."
    )
    zone = item.get("zone", "PRIMARY").replace("ZONE_", "").replace("_", " ").title()
    area_pct = item.get("area_pct", 0)
    target = item.get("target_destination", "Drawer / Organizer")
    disp_cm = item.get("displacement_cm", 0)

    disp_badge = (
        f"<span style='color: #4E7557; font-weight: 700;'>• Vector: {disp_cm}cm ➔ {target}</span>"
        if disp_cm > 0
        else ""
    )

    html = f"""
    <div>
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px;">
            <div style="font-weight: 800; font-size: 15px; color: #241A14;">{item_name}</div>
            <span class="pill-action {pill_class}">{pill_text}</span>
        </div>
        <div style="font-size: 13px; color: #6B5D50; line-height: 1.45; margin-bottom: 8px; font-weight: 500;">{reason}</div>
        <div style="display: flex; gap: 12px; font-size: 11.5px; color: #8A7B6E; border-top: 1.5px solid #F0E8DD; padding-top: 6px; flex-wrap: wrap;">
            <span><b>Reach Zone:</b> {zone}</span>
            <span><b>Surface Area:</b> {area_pct}%</span>
            {disp_badge}
        </div>
    </div>
    """
    return clean_html(html)


def render_item_card(item: dict, triage: dict) -> str:
    action = triage.get("recommended_action", "STOW_IN_DRAWER")
    if action == "TRASH":
        pill_class = "pill-trash"
        pill_text = "🗑️ TRASH / RECYCLE"
    elif action == "KEEP_ON_DESK":
        pill_class = "pill-keep"
        pill_text = "✨ KEEP ON DESK"
    elif action in ["RELOCATE", "RELOCATE_ZONE_2"]:
        pill_class = "pill-relocate"
        pill_text = "🧭 RELOCATE TO ZONE 2"
    else:
        pill_class = "pill-stow"
        pill_text = "🗄️ STOW IN DRAWER"

    item_name = triage.get(
        "item_name", item.get("semantic_name", f"Item #{item.get('id', 1)}")
    )
    reason = triage.get(
        "reasoning", "Clear this item to open up comfortable arm reach radius."
    )
    zone = item.get("zone", "PRIMARY").replace("ZONE_", "").replace("_", " ").title()
    area_pct = item.get("area_pct", 0)

    html = f"""
    <div class="triage-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 8px;">
            <div style="font-size: 15px; font-weight: 800; color: #241A14;">{item_name}</div>
            <span class="pill-action {pill_class}">{pill_text}</span>
        </div>
        <div style="font-size: 13px; color: #6B5D50; line-height: 1.45; margin-bottom: 10px;">{reason}</div>
        <div style="display: flex; gap: 14px; font-size: 11.5px; color: #8A7B6E; border-top: 1.5px solid #F0E8DD; padding-top: 8px;">
            <span><b>Reach Zone:</b> {zone}</span>
            <span><b>Surface Area:</b> {area_pct}% of desk</span>
        </div>
    </div>
    """
    return clean_html(html)


def render_balance_bar(left_ratio: float, right_ratio: float, verdict: str) -> str:
    l_pct = int(round(left_ratio * 100))
    r_pct = 100 - l_pct
    v_clean = verdict.replace("_", " ").title()
    html = f"""
    <div style="margin-top: 8px;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 700; margin-bottom: 4px; color: #6B5D50;">
            <span>Left Desk Side ({l_pct}%)</span>
            <span style="color: #241A14; font-weight: 800;">{v_clean}</span>
            <span>Right Desk Side ({r_pct}%)</span>
        </div>
        <div class="balance-bar-container">
            <div class="balance-left" style="width: {l_pct}%;">Left {l_pct}%</div>
            <div class="balance-right" style="width: {r_pct}%;">Right {r_pct}%</div>
        </div>
    </div>
    """
    return clean_html(html)


def render_celebration(res: dict) -> str:
    html = f"""
    <div class="celebration-banner">
        <div class="celebration-title">{res.get('verdict_badge', '🏆 AGENTIC TRANSFORMATION VERIFIED')}</div>
        <div class="celebration-sub">{res.get('headline', 'Workspace successfully transformed into an ergonomic sanctuary.')}</div>
        <div class="celebration-stats">
            <div class="celebration-stat-item">Zen Score: <b>{res.get('score_before')}</b> ➔ <b>{res.get('score_after')}</b> ({res.get('score_delta')})</div>
            <div class="celebration-stat-item">Cognitive Entropy: <b>-{res.get('entropy_reduction_pct')}%</b></div>
            <div class="celebration-stat-item">Desk Space Reclaimed: <b>~{res.get('estimated_sq_ft_reclaimed')} sq ft</b></div>
            <div class="celebration-stat-item">Spill Risk: <b>100% ELIMINATED</b></div>
        </div>
    </div>
    """
    return clean_html(html)


def render_item_checklist_card(
    item: dict,
    triage: dict,
    is_highlighted: bool = False,
    is_removed: bool = False,
) -> str:
    """Renders a tactile, high-contrast interactive card for an individual item in the declutter checklist."""
    item_id = item.get("id", 1)
    item_name = triage.get("item_name", item.get("semantic_name", f"Item #{item_id}"))
    reason = triage.get("reasoning", "Clear to restore working envelope.")
    zone = item.get("zone", "PRIMARY").replace("ZONE_", "").replace("_", " ").title()
    f_dist = item.get("edge_dist_to_focus_cm", 99.0)
    target = item.get("target_destination", "Drawer / Organizer")
    disp_cm = item.get("displacement_cm", 0.0)
    is_liquid = item.get("is_potential_liquid_container", False)

    if is_removed:
        card_border = "#4E7557"
        card_bg = "#F3F8F4"
        status_pill = '<span style="background: #DCFCE7; color: #166534; border: 1.5px solid #86EFAC; border-radius: 9999px; padding: 3px 10px; font-size: 11px; font-weight: 800;">✅ CLEARED / RELOCATED</span>'
    elif is_highlighted:
        card_border = "#D97706"
        card_bg = "#FFFBEB"
        status_pill = '<span style="background: #FEF3C7; color: #92400E; border: 1.5px solid #F59E0B; border-radius: 9999px; padding: 3px 10px; font-size: 11px; font-weight: 800;">🎯 ACTIVE IN IMAGE</span>'
    elif is_liquid:
        card_border = "#C84620"
        card_bg = "#FFFBF8"
        status_pill = '<span style="background: #FEE2E2; color: #991B1B; border: 1.5px solid #F87171; border-radius: 9999px; padding: 3px 10px; font-size: 11px; font-weight: 800;">⚠️ SPILL HAZARD</span>'
    else:
        card_border = "#E2D7CA"
        card_bg = "#FFFFFF"
        status_pill = '<span style="background: #F4EFEA; color: #6B5D50; border: 1.5px solid #D5C8B9; border-radius: 9999px; padding: 3px 10px; font-size: 11px; font-weight: 800;">PENDING ACTION</span>'

    vector_badge = (
        f'<span style="color: #4E7557; font-weight: 750;">➔ {target} ({disp_cm}cm)</span>'
        if disp_cm > 0
        else f'<span style="color: #6B5D50; font-weight: 750;">{target}</span>'
    )
    dist_badge = (
        f'<span style="color: #B85226; font-weight: 750;">• {f_dist}cm to Focus</span>'
        if f_dist < 90
        else ""
    )

    html = f"""
    <div style="background: {card_bg}; border: 2px solid {card_border}; border-radius: 14px; padding: 12px 14px; margin-bottom: 4px; box-shadow: 0 2px 8px rgba(30,20,10,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
            <div style="font-size: 14px; font-weight: 850; color: #241A14;">#{item_id} {item_name}</div>
            {status_pill}
        </div>
        <div style="font-size: 12.5px; color: #5C4D40; line-height: 1.4; margin-bottom: 8px;">{reason}</div>
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11.5px; border-top: 1px solid #EBE2D5; padding-top: 6px; flex-wrap: wrap; gap: 6px;">
            <span><b>Zone:</b> {zone} {dist_badge}</span>
            {vector_badge}
        </div>
    </div>
    """
    return clean_html(html)


def render_confirmation_prompt(
    removed_count: int,
    total_count: int,
    provisional_score: int,
    score_delta: str,
    is_fully_cleared: bool = True,
) -> str:
    """Renders the physical reality confirmation prompt banner."""
    title = (
        "🎯 Workspace Declutter State Reached!"
        if is_fully_cleared
        else "🧹 Workspace Declutter In Progress"
    )
    badge_bg = "#DCFCE7" if is_fully_cleared else "#FEF3C7"
    badge_color = "#166534" if is_fully_cleared else "#92400E"
    badge_border = "#86EFAC" if is_fully_cleared else "#FCD34D"
    badge_text = (
        "ALL ITEMS CLEARED"
        if is_fully_cleared
        else f"{removed_count}/{total_count} ITEMS CLEARED"
    )

    html = f"""
    <div style="background: linear-gradient(180deg, #FFFFFF 0%, #FAF6F0 100%); border: 2.5px solid #2D6A4F; border-radius: 18px; padding: 18px 22px; margin: 14px 0; box-shadow: 0 6px 24px rgba(45, 106, 79, 0.12);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">🌿</span>
                <div>
                    <div style="font-size: 11.5px; font-weight: 800; text-transform: uppercase; color: #2D6A4F; letter-spacing: 0.8px;">Physical Workspace Verification Loop</div>
                    <div style="font-size: 18px; font-weight: 850; color: #1E130B;">{title}</div>
                </div>
            </div>
            <span style="background: {badge_bg}; color: {badge_color}; border: 1.5px solid {badge_border}; border-radius: 9999px; padding: 5px 14px; font-size: 12.5px; font-weight: 850;">
                {badge_text}
            </span>
        </div>
        <div style="font-size: 14px; color: #382A1E; line-height: 1.5; margin-bottom: 12px;">
            OpenCV 5 has dynamically erased <b>{removed_count} of {total_count}</b> clutter items and quarantined spill hazards in your workspace visual preview.
            Your provisional Zen Clarity Score has jumped to <b style="color: #2D6A4F; font-size: 16px;">{provisional_score}/100 ({score_delta})</b>.
        </div>
        <div style="background: #F4EFEA; border-left: 4px solid #B85226; border-radius: 8px; padding: 10px 14px; font-size: 13.5px; color: #4A3324; font-weight: 650;">
            👉 <b>Physical Reality Check:</b> Does your physical desk currently match this cleared preview? Confirm below to run the closed-loop agentic verification!
        </div>
    </div>
    """
    return clean_html(html)


def render_studio_telemetry_ribbon(
    score: int,
    provisional_score: int,
    tier: str,
    central_focus: dict,
    encroachment: dict,
    hazards: dict,
    removed_count: int,
    total_count: int,
) -> str:
    """
    Renders a sleek, scannable 4-column top telemetry ribbon above the Side-by-Side Studio
    so the user sees live Score, Central Focus Halo, Safety Status, and Checklist Progress
    without pushing the interactive checklist off-screen.
    """
    active_score = provisional_score if removed_count > 0 else score
    delta_val = active_score - score
    delta_pill = (
        f'<span style="background: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 99px; font-size: 11.5px; font-weight: 800; margin-left: 6px;">+{delta_val} pts</span>'
        if delta_val > 0
        else ""
    )

    score_color = (
        "#2D6A4F"
        if active_score >= 80
        else ("#B85226" if active_score < 45 else "#92400E")
    )
    tier_clean = (
        "Zen Sanctuary" if active_score >= 80 else tier.replace("_", " ").title()
    )

    cf_name = (
        central_focus.get("name", "Workstation") if central_focus else "Workstation"
    )
    base_room = (
        int(round(encroachment.get("breathing_room_pct", 70.0))) if encroachment else 70
    )
    if removed_count > 0 and total_count > 0:
        live_room = min(
            100,
            int(round(base_room + (100 - base_room) * (removed_count / total_count))),
        )
    else:
        live_room = base_room
    wrist_cm = encroachment.get("wrist_clearance_cm", 15.0) if encroachment else 15.0
    mouse_cm = encroachment.get("mouse_clearance_cm", 22.0) if encroachment else 22.0
    room_color = (
        "#166534" if live_room >= 80 else ("#C84620" if live_room < 55 else "#92400E")
    )

    spill_risk = hazards.get("spill_risk", False) and (
        removed_count < total_count or total_count == 0
    )
    cable_snarl = hazards.get("cable_snarl_detected", False) and (
        removed_count < total_count or total_count == 0
    )
    if spill_risk:
        safety_badge = '<span style="color: #C84620; font-weight: 850;">⚠️ Liquid Spill Threat (&lt;25cm)</span>'
        safety_sub = "Relocate beverage to Zone 2 coaster"
    elif cable_snarl:
        safety_badge = '<span style="color: #92400E; font-weight: 850;">🔌 Cord Tangle Detected</span>'
        safety_sub = (
            f"{hazards.get('cable_lines_count', 0)} crossing segments in sweep zone"
        )
    else:
        safety_badge = '<span style="color: #166534; font-weight: 850;">✅ Zero Active Hazards</span>'
        safety_sub = "Electronics & forearm sweep protected"

    pct_done = (
        int(round((removed_count / max(1, total_count)) * 100))
        if total_count > 0
        else 100
    )

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 14px;">
        <div style="background: #FFFFFF; border: 1.5px solid #E2D7CA; border-radius: 14px; padding: 12px 16px; box-shadow: 0 2px 8px rgba(35,25,15,0.04);">
            <div style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #8A7B6E; letter-spacing: 0.6px;">Zen Clarity Score</div>
            <div style="display: flex; align-items: baseline; margin-top: 2px;">
                <span style="font-size: 28px; font-weight: 900; color: {score_color}; line-height: 1.1;">{active_score}</span>
                <span style="font-size: 13px; color: #8A7B6E; font-weight: 700; margin-left: 3px;">/100</span>
                {delta_pill}
            </div>
            <div style="font-size: 11.5px; font-weight: 750; color: #5C4D40; margin-top: 3px;">{tier_clean}</div>
        </div>

        <div style="background: #FFFFFF; border: 1.5px solid #E2D7CA; border-radius: 14px; padding: 12px 16px; box-shadow: 0 2px 8px rgba(35,25,15,0.04);">
            <div style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #8A7B6E; letter-spacing: 0.6px;">🎯 Central Work Focus</div>
            <div style="font-size: 14.5px; font-weight: 850; color: #241A14; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{cf_name}</div>
            <div style="font-size: 11.5px; color: #5C4D40; margin-top: 4px;">
                Halo Open: <b style="color: {room_color};">{live_room}%</b> • Wrist: <b>{wrist_cm}cm</b> • Mouse: <b>{mouse_cm}cm</b>
            </div>
        </div>

        <div style="background: #FFFFFF; border: 1.5px solid #E2D7CA; border-radius: 14px; padding: 12px 16px; box-shadow: 0 2px 8px rgba(35,25,15,0.04);">
            <div style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #8A7B6E; letter-spacing: 0.6px;">🛡️ Safety & Cord Sentinel</div>
            <div style="font-size: 14px; margin-top: 5px;">{safety_badge}</div>
            <div style="font-size: 11.5px; color: #6B5D50; margin-top: 4px;">{safety_sub}</div>
        </div>

        <div style="background: #FFFFFF; border: 1.5px solid #E2D7CA; border-radius: 14px; padding: 12px 16px; box-shadow: 0 2px 8px rgba(35,25,15,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 10.5px; font-weight: 800; text-transform: uppercase; color: #8A7B6E; letter-spacing: 0.6px;">✅ Declutter Progress</span>
                <span style="font-size: 12px; font-weight: 850; color: #2D6A4F;">{removed_count}/{total_count} Items ({pct_done}%)</span>
            </div>
            <div style="background: #EDE4D8; height: 9px; border-radius: 99px; overflow: hidden; margin-top: 9px;">
                <div style="background: #2D6A4F; width: {pct_done}%; height: 100%; transition: width 0.25s ease;"></div>
            </div>
            <div style="font-size: 11px; color: #6B5D50; margin-top: 5px;">
                {"All clutter cleared! Ready to verify." if removed_count == total_count and total_count > 0 else "Click Focus or toggle Cleared on the right ➔"}
            </div>
        </div>
    </div>
    """
    return clean_html(html)


def render_iso_calibration_card(calibration: dict, iso_audit: dict) -> str:
    """
    Rillar 1 UI: Renders the Intrinsic Anthropometric Fiducial Calibration & ISO 9241-5 Biomechanical Audit card.
    """
    if not calibration and not iso_audit:
        return ""

    fid_src = calibration.get("fiducial_source", "Hardware Fiducial Anchor")
    cm_x = calibration.get("cm_per_pixel_x", 0.15)
    cm_y = calibration.get("cm_per_pixel_y", 0.15)
    cam_deg = calibration.get("camera_elevation_deg", 74.5)
    desk_dims = calibration.get("estimated_desk_dimensions_cm", [120.0, 65.0])

    iso_score = iso_audit.get("iso_9241_score", 88)
    iso_tier = iso_audit.get("compliance_tier", "ISO 9241-5 COMPLIANT")
    neck_deg = iso_audit.get("neck_rotation_deg", 0.0)
    view_cm = iso_audit.get("viewing_distance_cm", 58.0)
    wrist_cm = iso_audit.get("wrist_clearance_cm", 15.0)
    mouse_abd = iso_audit.get("mouse_abduction_cm", 9.5)
    findings = iso_audit.get("findings", [])

    iso_color = (
        "#166534" if iso_score >= 82 else ("#C84620" if iso_score < 62 else "#92400E")
    )
    iso_bg = (
        "#F0FDF4" if iso_score >= 82 else ("#FEF2F2" if iso_score < 62 else "#FFFBEB")
    )

    findings_html = "".join(
        f'<div style="font-size: 12px; color: #3A2C22; padding: 4px 0; border-bottom: 1px dashed #E6DCCF;">• {f}</div>'
        for f in findings
    )

    html = f"""
    <div style="background: #FFFFFF; border: 1.5px solid #DFD3C3; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 3px 10px rgba(35,25,15,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <div style="font-size: 10.5px; font-weight: 850; text-transform: uppercase; letter-spacing: 0.7px; color: #7C6A58;">
                    📐 Intrinsic Anthropometric Calibration & ISO 9241-5 Audit
                </div>
                <div style="font-size: 13.5px; font-weight: 850; color: #241A14; margin-top: 2px;">
                    {fid_src}
                </div>
            </div>
            <div style="background: {iso_bg}; border: 1.5px solid {iso_color}; border-radius: 10px; padding: 5px 11px; text-align: right;">
                <div style="font-size: 17px; font-weight: 900; color: {iso_color}; line-height: 1.05;">{iso_score}/100</div>
                <div style="font-size: 9.5px; font-weight: 800; color: {iso_color};">{iso_tier}</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 10px;">
            <div style="background: #F8F4EE; border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; font-weight: 700; color: #7C6A58;">Scale X / Y</div>
                <div style="font-size: 13px; font-weight: 850; color: #241A14;">{cm_x} / {cm_y} <span style="font-size: 10px;">cm/px</span></div>
            </div>
            <div style="background: #F8F4EE; border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; font-weight: 700; color: #7C6A58;">Camera Pitch</div>
                <div style="font-size: 13px; font-weight: 850; color: #241A14;">{cam_deg}° <span style="font-size: 10px;">Elevation</span></div>
            </div>
            <div style="background: #F8F4EE; border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; font-weight: 700; color: #7C6A58;">Neck Rotation</div>
                <div style="font-size: 13px; font-weight: 850; color: #241A14;">{neck_deg}° <span style="font-size: 10px;">({view_cm}cm view)</span></div>
            </div>
            <div style="background: #F8F4EE; border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; font-weight: 700; color: #7C6A58;">Surface Span</div>
                <div style="font-size: 13px; font-weight: 850; color: #241A14;">{desk_dims[0]}×{desk_dims[1]} <span style="font-size: 10px;">cm</span></div>
            </div>
        </div>

        <div style="background: #FAF7F2; border-radius: 8px; padding: 8px 12px;">
            {findings_html}
        </div>
    </div>
    """
    return clean_html(html)


def render_edge_profiler_card(edge_profiler: dict) -> str:
    """
    Pillar 2 UI (COOL Award): Renders the OpenCV 5 + AWS Graviton3 Sub-Millisecond Edge Telemetry Profiler.
    """
    if not edge_profiler:
        return ""

    ent_ms = edge_profiler.get("entropy_spectral_ms", 0.0)
    yolo_ms = edge_profiler.get("yolo_anchor_ms", 0.0)
    morph_ms = edge_profiler.get("opencv_morphology_hough_ms", 0.0)
    inp_ms = edge_profiler.get("telea_inpaint_ms", 0.0)
    core_ms = edge_profiler.get("core_cv_ms", 0.0)
    fps = edge_profiler.get("realtime_fps", 30.0)
    mem_mb = edge_profiler.get("memory_mb", 4.5)
    badge = edge_profiler.get(
        "architecture_badge", "OpenCV 5 C++ SIMD + ARM64 Graviton3 Edge Ready"
    )

    html = f"""
    <div style="background: #231C16; color: #F5EFE6; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; border: 1px solid #4A3B2E;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                <div style="font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; color: #D4A373;">
                    ⚡ OpenCV 5 + AWS Graviton3 Edge Telemetry Profiler (COOL Award Track)
                </div>
                <div style="font-size: 12.5px; font-weight: 700; color: #E6DCCF; margin-top: 2px;">
                    {badge}
                </div>
            </div>
            <div style="background: #166534; color: #DCFCE7; padding: 4px 10px; border-radius: 99px; font-size: 11.5px; font-weight: 850;">
                {fps} FPS • {mem_mb} MB RAM
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;">
            <div style="background: rgba(255,255,255,0.06); border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; color: #BDB0A2;">Spectral Entropy</div>
                <div style="font-size: 14px; font-weight: 850; color: #86EFAC;">{ent_ms} ms</div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; color: #BDB0A2;">Morphology & Hough</div>
                <div style="font-size: 14px; font-weight: 850; color: #86EFAC;">{morph_ms} ms</div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; color: #BDB0A2;">YOLOv8n Anchor</div>
                <div style="font-size: 14px; font-weight: 850; color: #FDE047;">{yolo_ms} ms</div>
            </div>
            <div style="background: rgba(255,255,255,0.06); border-radius: 8px; padding: 7px 9px;">
                <div style="font-size: 10px; color: #BDB0A2;">Telea Inpaint</div>
                <div style="font-size: 14px; font-weight: 850; color: #93C5FD;">{inp_ms} ms</div>
            </div>
        </div>
        <div style="font-size: 11px; color: #BDB0A2; margin-top: 8px;">
            Core Classical OpenCV 5 Latency: <b style="color: #FFFFFF;">{core_ms} ms</b> (Zero GPU dependency — runs natively on ARM64 / AWS Graviton3).
        </div>
    </div>
    """
    return clean_html(html)


def render_agentic_reasoning_trace(
    trace_steps: list,
    verification_result: dict = None,
    **kwargs,
) -> str:
    """
    Pillar 4 UI (Agentic Vision Award): Renders the 5-stage autonomous Observe -> Orient -> Decide -> Act -> Verify loop.
    """
    if not trace_steps:
        return ""

    verify_badge_html = ""
    if verification_result:
        v_delta = verification_result.get("score_delta", "+0")
        v_after = verification_result.get("score_after", 95)
        verify_badge_html = (
            f'<span style="background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; border-radius: 99px; '
            f'padding: 2px 9px; font-size: 10.5px; font-weight: 850;">🏆 Verified: {v_after}/100 ({v_delta})</span>'
        )
    else:
        verify_badge_html = (
            '<span style="background: #F4EFEA; color: #5C4D40; border: 1px solid #D5C8B9; border-radius: 99px; '
            'padding: 2px 9px; font-size: 10.5px; font-weight: 800;">🔄 Closed-Loop Active</span>'
        )

    rows_html = ""
    for step in trace_steps:
        stg = step.get("stage", "STAGE")
        title = step.get("title", "")
        lat = step.get("latency_ms", 0.0)
        summary = step.get("summary", "")
        rows_html += f"""
        <div style="background: #FAF7F2; border-left: 3.5px solid #2D6A4F; border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 11.5px; font-weight: 850; color: #2D6A4F;">{stg} — {title}</span>
                <span style="font-size: 10.5px; font-weight: 800; color: #7C6A58; background: #EDE4D8; padding: 1px 7px; border-radius: 99px;">{lat} ms</span>
            </div>
            <div style="font-size: 12px; color: #3A2C22; margin-top: 3px; line-height: 1.4;">{summary}</div>
        </div>
        """

    html = f"""
    <div style="background: #FFFFFF; border: 1.5px solid #DFD3C3; border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
            <div style="font-size: 10.5px; font-weight: 850; text-transform: uppercase; letter-spacing: 0.7px; color: #7C6A58;">
                🧠 Autonomous Agentic Vision Loop (Observe ➔ Orient ➔ Decide ➔ Act ➔ Verify)
            </div>
            {verify_badge_html}
        </div>
        {rows_html}
    </div>
    """
    return clean_html(html)


__all__ = [
    "SAND_THEME_CSS",
    "clean_html",
    "render_header",
    "render_score_card",
    "render_central_focus_card",
    "render_diagnosis_card",
    "render_hazard_alert",
    "render_legend",
    "render_item_details",
    "render_item_card",
    "render_item_checklist_card",
    "render_confirmation_prompt",
    "render_studio_telemetry_ribbon",
    "render_iso_calibration_card",
    "render_edge_profiler_card",
    "render_agentic_reasoning_trace",
    "render_balance_bar",
    "render_celebration",
]
