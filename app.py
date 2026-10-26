"""
DeskZen: Autonomous Workspace Decluttering & Ergonomics Co-Pilot
Powered by OpenCV 5, YOLOv8, and AWS Bedrock (Amazon Nova Pro)
Clean Sand Mix Aesthetic • Side-by-Side Interactive Studio Layout
"""

import os
import io
import json
import textwrap
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import importlib
import core.after_generator
import core.cv_engine
import core.desk_scenarios

if not hasattr(core.cv_engine.WorkspaceVisionEngine, "generate_sanctuary_blueprint_card"):
    importlib.reload(core.after_generator)
    importlib.reload(core.cv_engine)
if not hasattr(core.desk_scenarios, "generate_oblique_laptop_desk"):
    importlib.reload(core.desk_scenarios)

from core.cv_engine import WorkspaceVisionEngine
from core.bedrock_agent import BedrockNovaAgent
from core.verification import AgenticVerifier
from core.desk_scenarios import (
    generate_developer_chaos_desk,
    generate_paperwork_avalanche_desk,
    generate_minimalist_flow_desk,
    generate_oblique_laptop_desk,
    generate_cleaned_after_desk,
)

import ui.sand_theme
importlib.reload(ui.sand_theme)
from ui.sand_theme import (
    SAND_THEME_CSS,
    render_header,
    render_score_card,
    render_central_focus_card,
    render_diagnosis_card,
    render_hazard_alert,
    render_legend,
    render_item_details,
    render_item_checklist_card,
    render_confirmation_prompt,
    render_studio_telemetry_ribbon,
    render_iso_calibration_card,
    render_edge_profiler_card,
    render_agentic_reasoning_trace,
    render_balance_bar,
    render_celebration,
)

# Page Configuration
st.set_page_config(
    page_title="DeskZen • Workspace Ergonomics Co-Pilot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply Clean Sand Mix CSS Theme
st.markdown(SAND_THEME_CSS, unsafe_allow_html=True)


# Initialize Core Services (cached with version key to auto-refresh across code upgrades)
@st.cache_resource
def get_services(_cache_version: str = "v6_workspace_studio_inpaint"):
    importlib.reload(core.after_generator)
    importlib.reload(core.cv_engine)
    from core.cv_engine import WorkspaceVisionEngine as _FreshEngine
    from core.after_generator import TargetAfterGenerator as _FreshAfterGen

    eng = _FreshEngine(crops_dir="static/crops")
    eng.after_gen = _FreshAfterGen(output_dir="static/crops")
    ag = BedrockNovaAgent(
        model_id=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
    )
    ver = AgenticVerifier()
    return eng, ag, ver


engine, agent, verifier = get_services("v6_workspace_studio_inpaint")
if (
    not hasattr(engine, "generate_sanctuary_blueprint_card")
    or not hasattr(engine, "after_gen")
    or engine.after_gen is None
    or not hasattr(engine.after_gen, "generate_sanctuary_blueprint_card")
):
    importlib.reload(core.after_generator)
    importlib.reload(core.cv_engine)
    from core.cv_engine import WorkspaceVisionEngine as _FallbackEngine
    from core.after_generator import TargetAfterGenerator as _FallbackAfterGen

    engine = _FallbackEngine(crops_dir="static/crops")
    engine.after_gen = _FallbackAfterGen(output_dir="static/crops")


def safe_inpaint_items(
    eng: WorkspaceVisionEngine,
    img: np.ndarray,
    telemetry: dict,
    removed_ids: set,
    plan: dict = None,
    item_actions: dict = None,
) -> dict:
    """Robust inpaint wrapper supporting both direct engine calls and after_gen delegates."""
    if hasattr(eng, "after_gen") and eng.after_gen is not None:
        return eng.after_gen.inpaint_specific_items(
            img, telemetry, removed_ids, plan or {}, item_actions=item_actions
        )
    try:
        return eng.inpaint_specific_items(
            img, telemetry, removed_ids, plan or {}, item_actions=item_actions
        )
    except TypeError:
        return eng.inpaint_specific_items(img, telemetry, removed_ids, plan or {})

# Session State Storage
if "before_telemetry" not in st.session_state:
    st.session_state.before_telemetry = None
if "before_plan" not in st.session_state:
    st.session_state.before_plan = None
if "before_image" not in st.session_state:
    st.session_state.before_image = None
if "original_uploaded_image" not in st.session_state:
    st.session_state.original_uploaded_image = None
if "workspace_version" not in st.session_state:
    st.session_state.workspace_version = 1
if "item_action_modes" not in st.session_state:
    st.session_state.item_action_modes = {}
if "after_telemetry" not in st.session_state:
    st.session_state.after_telemetry = None
if "verification_result" not in st.session_state:
    st.session_state.verification_result = None
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "custom"
if "display_layer" not in st.session_state:
    st.session_state.display_layer = "👁️ Interactive Triage"
if "input_source" not in st.session_state:
    st.session_state.input_source = "upload"
if "checklist_removed_ids" not in st.session_state:
    st.session_state.checklist_removed_ids = set()
if "highlighted_item_id" not in st.session_state:
    st.session_state.highlighted_item_id = None
if "digital_twin_map" not in st.session_state:
    st.session_state.digital_twin_map = None


def process_desk_image(
    bgr_img: np.ndarray,
    scenario_name: str = "custom",
    is_version_promotion: bool = False,
):
    """Run the end-to-end OpenCV 5 + YOLOv8 + Nova Pro pipeline without duplicate compute."""
    with st.spinner("Executing OpenCV 5 + YOLOv8 spatial analysis and Amazon Nova Pro reasoning..."):
        try:
            telem = engine.analyze(bgr_img, generate_after=False)
        except TypeError:
            telem = engine.analyze(bgr_img)
        plan = agent.synthesize_plan(telem)
        target_after_result = engine.generate_target_after(bgr_img, telem, plan)

        telem["target_after_image"] = target_after_result["target_image"]
        telem["target_after_metadata"] = target_after_result

        # Build 2.5D Spatial Digital Twin Map for instant (<10ms) deterministic relocation and compositing
        try:
            dt_map = engine.build_digital_twin_map(bgr_img, telem, plan)
        except Exception:
            dt_map = None

        if not is_version_promotion and st.session_state.original_uploaded_image is None:
            st.session_state.original_uploaded_image = bgr_img.copy()

        st.session_state.before_image = bgr_img
        st.session_state.before_telemetry = telem
        st.session_state.before_plan = plan
        st.session_state.digital_twin_map = dt_map
        st.session_state.current_scenario = scenario_name
        st.session_state.after_telemetry = None
        st.session_state.verification_result = None
        st.session_state.checklist_removed_ids = set()
        st.session_state.highlighted_item_id = None
        st.session_state.display_layer = "👁️ Interactive Triage"
        if not is_version_promotion:
            st.session_state.item_action_modes = {}


# Sidebar for Optional Advanced Controls
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    model_choice = st.selectbox(
        "AWS Bedrock Model",
        ["amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0"],
        index=0,
        help="Amazon Nova models are AWS-native and require no third-party vendor consent.",
    )
    agent.model_id = model_choice

    aws_region = st.selectbox(
        "AWS Region", ["us-east-1", "us-west-2", "eu-central-1"], index=0
    )
    agent.region_name = aws_region

    st.markdown("---")
    st.markdown(
        "<div style='font-size: 11.5px; color: #6B5D50; line-height: 1.5; font-weight: 600;'>"
        "<b>DeskZen v2.0 • OpenCV AI Competition 2026</b><br>"
        "Tracks: Agentic Vision Award • COOL Award (AWS Graviton3)<br>"
        "OpenCV 5 + YOLOv8: Central Focus Isolation, Shannon Entropy, Hough Lines, Inpainting & Relocation"
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================================
# TOP HEADER & WORKSPACE INPUT CONTROLS
# =========================================================================
st.markdown(render_header(), unsafe_allow_html=True)

# Clean, modern top action bar
top_c1, top_c2, top_c3, top_c_spacer = st.columns([1.1, 1.1, 1.0, 3.2], gap="small")

with top_c1:
    is_up_active = (st.session_state.input_source == "upload" and st.session_state.before_telemetry is None)
    if st.button(
        "📁 Upload Photo",
        use_container_width=True,
        type="primary" if is_up_active else "secondary",
        help="Upload a desk image from your computer",
    ):
        st.session_state.input_source = "upload"
        st.session_state.before_telemetry = None
        st.rerun()

with top_c2:
    is_cam_active = (st.session_state.input_source == "webcam" and st.session_state.before_telemetry is None)
    if st.button(
        "📷 Live Webcam",
        use_container_width=True,
        type="primary" if is_cam_active else "secondary",
        help="Capture your desk in real time using your webcam",
    ):
        st.session_state.input_source = "webcam"
        st.session_state.before_telemetry = None
        st.rerun()

with top_c3:
    if st.button("🔄 Reset Desk", use_container_width=True, type="secondary", help="Clear all state and start fresh"):
        st.session_state.clear()
        st.rerun()


# =========================================================================
# STATE 1: WELCOME / INPUT STATE (When no image has been analyzed yet)
# =========================================================================
if st.session_state.before_telemetry is None:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    c_left, c_right = st.columns([1.1, 1], gap="large")

    with c_left:
        with st.container(border=True):
            st.markdown("#### 📸 Analyze Your Physical Workspace")
            st.caption(
                "Upload any top-down, angled, or side-view photo of your desk showing your laptop, monitor, or keyboard and surrounding items."
            )

            uploaded_file = None
            if st.session_state.input_source == "upload":
                uploaded_file = st.file_uploader(
                    "Upload desk photo (JPG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    help="Snap a photo showing your tabletop work surface",
                )
            else:
                uploaded_file = st.camera_input("Capture desk photo via webcam")

            if uploaded_file is not None:
                image_bytes = uploaded_file.read()
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                st.image(pil_img, caption="Workspace Input Preview", use_container_width=True)

                if st.button(
                    "✨ Execute OpenCV 5 + YOLOv8 & Nova Pro Analysis",
                    key="btn_run_analysis",
                    use_container_width=True,
                    type="primary",
                ):
                    process_desk_image(bgr_img, scenario_name="user_workspace")
                    st.rerun()

    with c_right:
        with st.container(border=True):
            st.markdown("#### 🌿 How DeskZen Reclaims Deep Cognitive Focus")
            st.markdown(
                "DeskZen transforms passive computer vision into an **interactive ergonomic co-pilot**. "
                "Using YOLOv8 object detection, OpenCV 5 spatial mathematics, and Amazon Nova Pro:"
            )

            st.markdown(
                textwrap.dedent(
                    """
                    - 🎯 **Central Work Focus Isolation**: Locks onto your primary laptop or keyboard and computes its required **15cm wrist clearance** and **22cm mouse sweep corridor**.
                    - 🧹 **Inpaint & Relocate Engine**: Inpaints and heals congested areas using OpenCV Telea, extracting and relocating items to safe peripheral zones.
                    - 🔍 **Interactive Checklist**: Click any clutter item to spotlight it in 3D perspective or toggle removal/relocation in real time.
                    - ⚠️ **Liquid Spill & Cord Sentinel**: Flags open beverages within 25cm of active electronics and detects tangled cable networks.
                    - 💾 **Version Studio**: Manage, download, or promote your relocated desk as the new active workspace.
                    """
                ).strip()
            )

            st.markdown(
                "<div style='background: #F0FDF4; border: 1.5px solid #BBF7D0; border-radius: 12px; padding: 12px 16px; margin-top: 14px; font-size: 13px; color: #166534; font-weight: 600;'>"
                "💡 <b>Tip:</b> Take a top-down or angled photo of your desk showing your laptop or keyboard and surrounding items to begin!"
                "</div>",
                unsafe_allow_html=True,
            )


# =========================================================================
# STATE 2: SIDE-BY-SIDE INTERACTIVE STUDIO (Zero Scroll Ping-Pong)
# =========================================================================
else:
    if (
        st.session_state.before_telemetry is not None
        and "intrinsic_calibration" not in st.session_state.before_telemetry
        and st.session_state.before_image is not None
    ):
        process_desk_image(
            st.session_state.before_image,
            scenario_name=st.session_state.get("current_scenario", "chaos"),
        )

    telem = st.session_state.before_telemetry
    plan = st.session_state.before_plan
    hazards = telem.get("hazards", {})
    balance = telem.get("visual_mass_balance", {})
    central_focus = telem.get("central_focus", {})
    encroachment = telem.get("encroachment", {})
    items = telem.get("items", [])
    triage_map = {t["item_id"]: t for t in plan.get("triage_deck", [])}

    total_items = len(items)
    valid_item_ids = {it["id"] for it in items}
    active_removed = st.session_state.checklist_removed_ids.intersection(valid_item_ids)
    cleared_count = len(active_removed)
    progress_frac = float(cleared_count) / float(max(1, total_items)) if total_items > 0 else 1.0
    is_fully_cleared = (cleared_count == total_items and total_items > 0)

    # Pre-compute progressive digital twin state and inpainting result
    if st.session_state.digital_twin_map is None and st.session_state.before_image is not None:
        try:
            st.session_state.digital_twin_map = engine.build_digital_twin_map(
                st.session_state.before_image, telem, plan
            )
        except Exception:
            st.session_state.digital_twin_map = None

    if st.session_state.digital_twin_map is not None:
        st.session_state.digital_twin_map.apply_checklist_state(
            active_removed, st.session_state.item_action_modes
        )
        twin_telem = st.session_state.digital_twin_map.get_provisional_telemetry()
        raw_clean_desk = st.session_state.digital_twin_map.render_photorealistic()
        interactive_twin_img = st.session_state.digital_twin_map.render_interactive_view(
            highlighted_id=st.session_state.highlighted_item_id,
            removed_ids=active_removed,
            action_modes=st.session_state.item_action_modes,
        )
        provisional_score = twin_telem["provisional_score"]
        score_delta = twin_telem["score_delta"]
        relocated_cnt = twin_telem["relocated_count"]
        stowed_cnt = twin_telem["stowed_count"]
        clean_res = {
            "raw_cleared_image": raw_clean_desk,
            "cleared_image": interactive_twin_img,
            "provisional_score": provisional_score,
            "score_delta": score_delta,
            "relocated_count": relocated_cnt,
            "stowed_count": stowed_cnt,
            "removed_count": cleared_count,
            "remaining_count": twin_telem["remaining_count"],
            "total_count": total_items,
            "is_fully_cleared": is_fully_cleared,
        }
    else:
        clean_res = safe_inpaint_items(
            engine,
            st.session_state.before_image,
            telem,
            active_removed,
            plan,
            item_actions=st.session_state.item_action_modes,
        )
        provisional_score = clean_res.get("provisional_score", telem.get("zen_score", 30))
        score_delta = clean_res.get("score_delta", "+0 pts")
        relocated_cnt = clean_res.get("relocated_count", 0)
        stowed_cnt = clean_res.get("stowed_count", 0)
        raw_clean_desk = clean_res.get("raw_cleared_image", clean_res["cleared_image"])
        interactive_twin_img = clean_res["cleared_image"]

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # 1. TOP EXECUTIVE TELEMETRY RIBBON (Compact 4-Card Bar)
    # ---------------------------------------------------------------------
    st.markdown(
        render_studio_telemetry_ribbon(
            score=telem["zen_score"],
            provisional_score=provisional_score,
            tier=telem["cognitive_load_tier"],
            central_focus=central_focus,
            encroachment=encroachment,
            hazards=hazards,
            removed_count=cleared_count,
            total_count=total_items,
        ),
        unsafe_allow_html=True,
    )

    # Show Verified Celebration Banner at top if user completed verification
    if st.session_state.verification_result:
        st.markdown(render_celebration(st.session_state.verification_result), unsafe_allow_html=True)

    # ---------------------------------------------------------------------
    # 2. SIDE-BY-SIDE INTERACTIVE WORKSPACE STUDIO
    #    Left: Interactive Workspace Canvas | Right: Checklist & Triage Workbench
    # ---------------------------------------------------------------------
    col_canvas, col_workbench = st.columns([1.18, 1], gap="large")

    with col_canvas:
        with st.container(border=True):
            layer_options = [
                "👁️ Interactive Triage",
                "🧹 Inpainted & Relocated Preview",
                "🎯 Target Blueprint",
                "🧭 Relocation Vectors",
                "🔥 Clutter Heatmap",
                "⚖️ Before / After Card",
            ]
            if st.session_state.display_layer not in layer_options:
                st.session_state.display_layer = "👁️ Interactive Triage"
            default_idx = layer_options.index(st.session_state.display_layer)

            layer_choice = st.radio(
                "Workspace View Mode",
                layer_options,
                index=default_idx,
                horizontal=True,
                label_visibility="collapsed",
                key="radio_display_layer",
            )
            st.session_state.display_layer = layer_choice

            # Pre-generate composite Before -> After Sanctuary Blueprint Card
            if hasattr(engine, "generate_sanctuary_blueprint_card"):
                blueprint_card_res = engine.generate_sanctuary_blueprint_card(
                    telem["annotated_image"],
                    telem.get("target_after_image", clean_res["cleared_image"]),
                    telem,
                    plan,
                )
            else:
                blueprint_card_res = engine.after_gen.generate_sanctuary_blueprint_card(
                    telem["annotated_image"],
                    telem.get("target_after_image", clean_res["cleared_image"]),
                    telem,
                    plan,
                )

            # Select active image to display
            if layer_choice == "🧹 Inpainted & Relocated Preview":
                active_img = raw_clean_desk
                badge_info = f"Inpainted & Relocated Preview ({relocated_cnt} Relocated • {stowed_cnt} Stowed)"
            elif layer_choice == "🔥 Clutter Heatmap":
                active_img = telem.get("heatmap_image", telem["annotated_image"])
                badge_info = "OpenCV Gradient Magnitude & Laplacian Energy Heatmap"
            elif layer_choice == "🧭 Relocation Vectors":
                active_img = telem.get("relocation_image", telem["annotated_image"])
                badge_info = "Prescriptive Relocation Vectors & Safe Zone Targets"
            elif layer_choice == "🎯 Target Blueprint":
                active_img = telem.get("target_after_image", telem["annotated_image"])
                badge_info = "Synthesized Digital Twin Goal (100% Decluttered Target)"
            elif layer_choice == "⚖️ Before / After Card":
                active_img = blueprint_card_res["card_image"]
                badge_info = "Side-by-Side Ergonomic Sanctuary Blueprint Card (Downloadable)"
            else:
                # Default: Unified Live Interactive Triage View!
                # Uses 2.5D Digital Twin Scene Graph for instant (<10ms) zero-hallucination compositing
                if st.session_state.digital_twin_map is not None:
                    active_img = interactive_twin_img
                elif st.session_state.highlighted_item_id is not None or cleared_count > 0:
                    active_img = engine.render_interactive_view(
                        st.session_state.before_image,
                        telem,
                        highlighted_item_id=st.session_state.highlighted_item_id,
                        removed_ids=active_removed,
                        item_actions=st.session_state.item_action_modes,
                    )
                else:
                    active_img = telem["annotated_image"]

                hl_txt = f" • Spotlight on #{st.session_state.highlighted_item_id}" if st.session_state.highlighted_item_id else ""
                rem_txt = f" • {cleared_count}/{total_items} Inpainted & Relocated" if cleared_count > 0 else ""
                badge_info = f"Live Interactive Canvas ({total_items - cleared_count} Remaining{rem_txt}{hl_txt})"

            # Compact Metadata Pills
            persp = telem.get("perspective", "TOP_DOWN").replace("_", " ").title()
            cal_info = telem.get("intrinsic_calibration", {})
            cm_px_val = cal_info.get("cm_per_pixel_x", 0.15)
            cam_deg_val = cal_info.get("camera_elevation_deg", 74.5)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; flex-wrap: wrap; gap: 6px;">
                    <span style="font-size: 12.5px; font-weight: 750; color: #5C4D40;">📸 {badge_info}</span>
                    <span style="background: #F4EFEA; color: #4A3B32; border: 1px solid #D5C8B9; border-radius: 99px; padding: 2px 10px; font-size: 11px; font-weight: 750;">📐 {persp} ({cam_deg_val}° • {cm_px_val} cm/px)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rgb_view = cv2.cvtColor(active_img, cv2.COLOR_BGR2RGB)
            st.image(rgb_view, use_container_width=True)

            # Quick Spotlight Action Bar directly beneath image when an item is focused
            if st.session_state.highlighted_item_id is not None:
                hl_id = st.session_state.highlighted_item_id
                hl_item = next((x for x in items if x["id"] == hl_id), None)
                hl_name = hl_item["semantic_name"] if hl_item else f"Item #{hl_id}"
                is_hl_removed = hl_id in active_removed

                c_hl1, c_hl2, c_hl3 = st.columns([2.2, 1.2, 0.9], gap="small")
                with c_hl1:
                    st.markdown(
                        f"<div style='background:#FFFBEB; border:1.5px solid #F59E0B; border-radius:10px; padding:7px 12px; font-size:12.5px; color:#92400E; font-weight:750;'>"
                        f"🎯 Focused: <b>#{hl_id} {hl_name}</b>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c_hl2:
                    toggle_lbl = "↩️ Undo Action" if is_hl_removed else f"✅ Act on #{hl_id}"
                    if st.button(toggle_lbl, key="btn_quick_clear_under_img", use_container_width=True, type="primary"):
                        if is_hl_removed:
                            st.session_state.checklist_removed_ids.discard(hl_id)
                        else:
                            st.session_state.checklist_removed_ids.add(hl_id)
                        st.rerun()
                with c_hl3:
                    if st.button("👁️ Unfocus", key="btn_clear_focus_under_img", use_container_width=True):
                        st.session_state.highlighted_item_id = None
                        st.rerun()

            st.markdown(render_legend(), unsafe_allow_html=True)

            # 1-Click Export Bar: Download Sanctuary Blueprint PNG & Spatial Telemetry JSON
            export_payload = {
                "zen_score": telem.get("zen_score"),
                "provisional_score": provisional_score,
                "cognitive_load_tier": telem.get("cognitive_load_tier"),
                "perspective": telem.get("perspective"),
                "intrinsic_calibration": telem.get("intrinsic_calibration", {}),
                "iso_9241_audit": telem.get("iso_9241_audit", {}),
                "edge_profiler": telem.get("edge_profiler", {}),
                "central_focus": telem.get("central_focus", {}),
                "encroachment": telem.get("encroachment", {}),
                "hazards": telem.get("hazards", {}),
                "agentic_loop_trace": telem.get("agentic_loop_trace", []),
                "items": telem.get("items", []),
            }
            exp_c1, exp_c2 = st.columns(2, gap="small")
            with exp_c1:
                st.download_button(
                    "📥 Download Sanctuary Blueprint (PNG)",
                    data=blueprint_card_res["png_bytes"],
                    file_name=f"deskzen_sanctuary_blueprint_{st.session_state.current_scenario}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with exp_c2:
                st.download_button(
                    "📊 Export Spatial Telemetry (JSON)",
                    data=json.dumps(export_payload, indent=2),
                    file_name=f"deskzen_telemetry_{st.session_state.current_scenario}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # -------------------------------------------------------------
            # WORKSPACE VERSION MANAGEMENT STUDIO
            # -------------------------------------------------------------
            with st.container(border=True):
                ver_num = st.session_state.workspace_version
                ver_tag = f"v{ver_num}.0 (Original Upload)" if ver_num == 1 else f"v{ver_num}.0 (Relocated Desk State)"
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 11px; font-weight: 850; text-transform: uppercase; color: #2D6A4F; letter-spacing: 0.7px;">
                            💾 Workspace Version Studio
                        </span>
                        <span style="background: #E8F5E9; color: #1B5E20; border: 1px solid #A5D6A7; border-radius: 99px; padding: 2px 10px; font-size: 11px; font-weight: 800;">
                            Active Baseline: {ver_tag}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                raw_clean_desk = clean_res.get("raw_cleared_image", clean_res["cleared_image"])
                ok_enc, clean_png_bytes = cv2.imencode(".png", raw_clean_desk)

                if cleared_count > 0:
                    st.markdown(
                        f"<div style='font-size: 12px; color: #4A3B32; margin-bottom: 8px; line-height: 1.4;'>"
                        f"OpenCV inpainting and spatial relocation synthesized <b>{relocated_cnt} items moved to safe zones</b> "
                        f"and <b>{stowed_cnt} items stowed away</b>."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    v_col1, v_col2 = st.columns([1.4, 1.1], gap="small")
                    with v_col1:
                        if st.button(
                            "🔄 Set Relocated Desk as Active Workspace",
                            key="btn_promote_version",
                            type="primary",
                            use_container_width=True,
                            help="Promotes this inpainted & relocated workspace to active baseline and re-runs full CV telemetry",
                        ):
                            st.session_state.workspace_version += 1
                            process_desk_image(
                                raw_clean_desk,
                                scenario_name=f"relocated_v{st.session_state.workspace_version}",
                                is_version_promotion=True,
                            )
                            st.rerun()
                    with v_col2:
                        if ok_enc:
                            st.download_button(
                                "📥 Download Clean Photo (.PNG)",
                                data=clean_png_bytes.tobytes(),
                                file_name=f"deskzen_relocated_v{st.session_state.workspace_version}.png",
                                mime="image/png",
                                use_container_width=True,
                            )

                if st.session_state.workspace_version > 1 and st.session_state.original_uploaded_image is not None:
                    if st.button("↩️ Revert to Original Uploaded Desk Photo", key="btn_revert_original", use_container_width=True):
                        st.session_state.workspace_version = 1
                        process_desk_image(
                            st.session_state.original_uploaded_image,
                            scenario_name="original_upload",
                            is_version_promotion=True,
                        )
                        st.rerun()

            # Physical Reality Confirmation Prompt right below the canvas when items are cleared!
            if cleared_count > 0:
                st.markdown(
                    render_confirmation_prompt(
                        removed_count=cleared_count,
                        total_count=total_items,
                        provisional_score=provisional_score,
                        score_delta=score_delta,
                        is_fully_cleared=is_fully_cleared,
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    "✅ Yes! Confirm Physical Reality & Run Agentic Verification",
                    key="btn_confirm_canvas_verify",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Verifying decluttered physical workspace with OpenCV 5..."):
                        # Use raw_cleared_image (without HUD drawings) so CV engine accurately scores the clean desk!
                        telem_after = engine.analyze(raw_clean_desk)
                        v_res = verifier.verify_transformation(
                            st.session_state.before_telemetry, telem_after
                        )
                        st.session_state.after_telemetry = telem_after
                        st.session_state.verification_result = v_res
                        st.rerun()

    # ---------------------------------------------------------------------
    # RIGHT COLUMN: INTERACTIVE WORKBENCH (Directly beside the canvas!)
    # ---------------------------------------------------------------------
    with col_workbench:
        tab_checklist, tab_diagnosis, tab_blueprint, tab_telemetry = st.tabs(
            [
                f"✅ 1. Action Checklist ({cleared_count}/{total_items})",
                "🧠 2. Focus & Ergonomics",
                "📐 3. Blueprint & Verify",
                "⚡ 4. Agentic & Edge Audit",
            ]
        )

        # -----------------------------------------------------------------
        # TAB 1: SIDE-BY-SIDE INTERACTIVE ITEM CHECKLIST
        # -----------------------------------------------------------------
        with tab_checklist:
            if not items:
                st.success(
                    "✨ Zero clutter items encroaching on your Central Work Focus! Your workspace is in optimal Zen harmony."
                )
            else:
                # Top Quick Action Bar
                c_qa1, c_qa2, c_qa3 = st.columns([1.4, 1, 0.9], gap="small")
                with c_qa1:
                    st.markdown(
                        f"<div style='font-size: 13px; color: #5C4D40; font-weight: 700; padding-top: 6px;'>"
                        f"Click <b>🎯 Focus</b> to locate or check <b>✅ Cleared</b> to erase:"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with c_qa2:
                    if st.button(
                        "⚡ Clear All Items",
                        key="btn_clear_all_items",
                        use_container_width=True,
                        type="primary" if cleared_count < total_items else "secondary",
                        help="Inpaint and erase all detected clutter items at once",
                    ):
                        st.session_state.checklist_removed_ids.update(valid_item_ids)
                        st.session_state.highlighted_item_id = None
                        st.rerun()
                with c_qa3:
                    if st.button(
                        "🔄 Reset",
                        key="btn_reset_checklist",
                        use_container_width=True,
                        help="Restore all items and clear focus",
                    ):
                        st.session_state.checklist_removed_ids = set()
                        st.session_state.highlighted_item_id = None
                        st.rerun()

                # Scrollable container for checklist items so it aligns with canvas height
                with st.container(height=540, border=False):
                    for it in items:
                        t_data = triage_map.get(
                            it["id"],
                            {
                                "item_name": it.get("semantic_name", f"Clutter Object #{it['id']}"),
                                "recommended_action": "STOW_IN_DRAWER",
                                "reasoning": it.get("clearance_directive", "Clear to restore working arm radius."),
                            },
                        )
                        is_hl = (it["id"] == st.session_state.highlighted_item_id)
                        is_rem = (it["id"] in active_removed)

                        with st.container(border=True):
                            c_thumb, c_body = st.columns([0.85, 2.8], gap="small")
                            with c_thumb:
                                thumb_path = it.get("thumbnail_path", "")
                                if os.path.exists(thumb_path):
                                    st.image(thumb_path, use_container_width=True)
                                else:
                                    st.markdown(
                                        "<div style='height:64px; background:#F4EFEA; border:1px solid #E2D7CA; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:20px;'>📦</div>",
                                        unsafe_allow_html=True,
                                    )

                                btn_label = "👁️ Unfocus" if is_hl else "🎯 Focus"
                                if st.button(
                                    btn_label,
                                    key=f"btn_hl_{it['id']}",
                                    use_container_width=True,
                                    type="primary" if is_hl else "secondary",
                                ):
                                    if is_hl:
                                        st.session_state.highlighted_item_id = None
                                    else:
                                        st.session_state.highlighted_item_id = it["id"]
                                        if st.session_state.display_layer not in ["👁️ Interactive Triage", "🧹 Cleared Preview"]:
                                            st.session_state.display_layer = "👁️ Interactive Triage"
                                    st.rerun()

                            with c_body:
                                st.markdown(
                                    render_item_checklist_card(
                                        item=it,
                                        triage=t_data,
                                        is_highlighted=is_hl,
                                        is_removed=is_rem,
                                    ),
                                    unsafe_allow_html=True,
                                )

                                # Determine default action mode (Relocate to safe perimeter vs Stow off-desk)
                                rec_action = t_data.get("recommended_action", "STOW_IN_DRAWER")
                                default_mode = "relocate" if ("RELOCATE" in rec_action or it.get("target_destination", "") not in ["Off-Desk / Drawer", "Stow Off-Desk"]) else "stow"
                                cur_mode = st.session_state.item_action_modes.get(it["id"], default_mode)

                                col_act_chk, col_act_rad = st.columns([1.15, 1.45], gap="small")
                                with col_act_chk:
                                    new_chk = st.checkbox(
                                        f"Act on #{it['id']}",
                                        value=is_rem,
                                        key=f"chk_rem_{it['id']}",
                                        help=f"Execute inpaint & relocation on #{it['id']}",
                                    )
                                with col_act_rad:
                                    action_opts = ["🧭 Relocate", "🗄️ Stow"]
                                    selected_opt = st.radio(
                                        f"Action #{it['id']}",
                                        options=action_opts,
                                        index=0 if cur_mode == "relocate" else 1,
                                        key=f"rad_act_{it['id']}",
                                        horizontal=True,
                                        label_visibility="collapsed",
                                    )
                                    new_mode = "relocate" if "Relocate" in selected_opt else "stow"
                                    if new_mode != cur_mode:
                                        st.session_state.item_action_modes[it["id"]] = new_mode
                                        st.rerun()

                                if new_chk != is_rem:
                                    if new_chk:
                                        st.session_state.checklist_removed_ids.add(it["id"])
                                    else:
                                        st.session_state.checklist_removed_ids.discard(it["id"])
                                    # Keep user in Interactive Triage view so checked items vanish in-place while remaining items stay visible!
                                    if st.session_state.display_layer not in ["👁️ Interactive Triage", "🧹 Inpainted & Relocated Preview"]:
                                        st.session_state.display_layer = "👁️ Interactive Triage"
                                    st.rerun()

        # -----------------------------------------------------------------
        # TAB 2: CENTRAL FOCUS & AI COGNITIVE DIAGNOSIS
        # -----------------------------------------------------------------
        with tab_diagnosis:
            if central_focus:
                st.markdown(render_central_focus_card(central_focus, encroachment), unsafe_allow_html=True)

            st.markdown(
                render_iso_calibration_card(
                    telem.get("intrinsic_calibration", {}),
                    telem.get("iso_9241_audit", {}),
                ),
                unsafe_allow_html=True,
            )

            hazard_html = render_hazard_alert(
                spill_risk=hazards.get("spill_risk", False),
                cable_snarl=hazards.get("cable_snarl_detected", False),
                spill_dist_px=hazards.get("min_spill_distance_px") or 0,
                cable_count=hazards.get("cable_lines_count") or 0,
            )
            if hazard_html:
                st.markdown(hazard_html, unsafe_allow_html=True)

            st.markdown(
                render_diagnosis_card(
                    diagnosis=plan.get(
                        "cognitive_state_diagnosis",
                        "Workspace requires decluttering to minimize cognitive friction.",
                    ),
                    quote=plan.get("zen_philosophy_quote", "A clear desk reflects a clear mind."),
                    model_name=plan.get("model_used", "DeskZen Spatial Synthesizer"),
                ),
                unsafe_allow_html=True,
            )

            with st.container(border=True):
                st.markdown("##### ⚖️ Visual Mass Distribution")
                st.markdown(
                    render_balance_bar(
                        left_ratio=balance.get("left_ratio", 0.5),
                        right_ratio=balance.get("right_ratio", 0.5),
                        verdict=balance.get("verdict", "BALANCED"),
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(plan.get("ergonomic_blueprint_directives", ""))

        # -----------------------------------------------------------------
        # TAB 3: SPATIAL BLUEPRINT & CUSTOM VERIFICATION
        # -----------------------------------------------------------------
        with tab_blueprint:
            focus_lib = plan.get("central_focus_liberation_strategy", {})
            if focus_lib:
                with st.container(border=True):
                    st.markdown(f"##### 🎯 Liberation Strategy: {focus_lib.get('focus_device', 'Workstation')}")
                    st.markdown(f"- **⌨️ Wrist Zone:** {focus_lib.get('wrist_zone_liberation', '')}")
                    st.markdown(f"- **🖱️ Mouse Sweep:** {focus_lib.get('mouse_corridor_liberation', '')}")
                    st.markdown(f"- **☕ Beverage Safety:** {focus_lib.get('beverage_quarantine', '')}")

            with st.container(border=True):
                st.markdown("##### ⏱️ 5-Minute Physical Replication Steps")
                for s_idx, step in enumerate(plan.get("five_minute_challenge", []), 1):
                    st.markdown(f"**{s_idx}.** {step}")

                if st.button(
                    "▶ Simulate Full Physical Declutter & Verify",
                    key="btn_simulate_clean",
                    use_container_width=True,
                    type="primary",
                ):
                    with st.spinner("Simulating decluttered workspace and running OpenCV 5 verification..."):
                        st.session_state.checklist_removed_ids = set(valid_item_ids)
                        full_clean = safe_inpaint_items(
                            engine,
                            st.session_state.before_image,
                            telem,
                            valid_item_ids,
                            plan,
                            item_actions=st.session_state.item_action_modes,
                        )
                        telem_after = engine.analyze(full_clean["raw_cleared_image"])
                        v_res = verifier.verify_transformation(
                            st.session_state.before_telemetry, telem_after
                        )
                        st.session_state.after_telemetry = telem_after
                        st.session_state.verification_result = v_res
                        st.rerun()

                st.markdown("---")
                after_file = st.file_uploader(
                    "Or Upload Real 'After' Photo (JPG, PNG)",
                    type=["jpg", "jpeg", "png"],
                    key="after_upload_photo",
                )
                if after_file is not None:
                    if st.button("✨ Verify Uploaded 'After' Photo", key="btn_run_verify", use_container_width=True, type="primary"):
                        with st.spinner("Analyzing uploaded 'After' photo..."):
                            img_after = Image.open(io.BytesIO(after_file.read())).convert("RGB")
                            bgr_after = cv2.cvtColor(np.array(img_after), cv2.COLOR_RGB2BGR)
                            telem_after = engine.analyze(bgr_after)
                            v_res = verifier.verify_transformation(
                                st.session_state.before_telemetry, telem_after
                            )
                            st.session_state.after_telemetry = telem_after
                            st.session_state.verification_result = v_res
                            st.rerun()

        # -----------------------------------------------------------------
        # TAB 4: AGENTIC LOOP TRACE & OPENCV 5 / AWS GRAVITON3 TELEMETRY
        # -----------------------------------------------------------------
        with tab_telemetry:
            st.markdown(
                render_agentic_reasoning_trace(
                    telem.get("agentic_loop_trace", []),
                    verification_result=st.session_state.verification_result,
                ),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_edge_profiler_card(telem.get("edge_profiler", {})),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_iso_calibration_card(
                    telem.get("intrinsic_calibration", {}),
                    telem.get("iso_9241_audit", {}),
                ),
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------------
    # 3. BEFORE VS. VERIFIED AFTER COMPARISON (When Verified)
    # ---------------------------------------------------------------------
    if st.session_state.verification_result and st.session_state.after_telemetry is not None:
        res = st.session_state.verification_result
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_before_card, col_after_card = st.columns(2, gap="large")

        with col_before_card:
            with st.container(border=True):
                st.markdown(
                    f"##### 🔴 Before (Cluttered State) — Zen Score: `{res['score_before']}/100`"
                )
                if st.session_state.before_image is not None:
                    st.image(
                        cv2.cvtColor(st.session_state.before_image, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )

        with col_after_card:
            with st.container(border=True):
                st.markdown(
                    f"##### 🟢 After (Verified Sanctuary) — Zen Score: `{res['score_after']}/100 ({res['score_delta']})`"
                )
                st.image(
                    cv2.cvtColor(
                        st.session_state.after_telemetry["annotated_image"],
                        cv2.COLOR_BGR2RGB,
                    ),
                    use_container_width=True,
                )
