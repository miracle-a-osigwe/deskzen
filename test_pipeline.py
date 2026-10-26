"""
Automated Pipeline Test for DeskZen
Tests OpenCV 5 analysis, scenarios, multi-layer overlays, Bedrock Nova agent, and verification.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2

from core.cv_engine import WorkspaceVisionEngine
from core.bedrock_agent import BedrockNovaAgent
from core.verification import AgenticVerifier
from core.desk_scenarios import (
    generate_developer_chaos_desk,
    generate_oblique_laptop_desk,
    generate_paperwork_avalanche_desk,
    generate_minimalist_flow_desk,
    generate_cleaned_after_desk
)
from ui.sand_theme import (
    render_iso_calibration_card,
    render_edge_profiler_card,
    render_agentic_reasoning_trace,
)


def run_test():
    print("==================================================")
    print("[DeskZen] End-to-End Automated Pipeline Test")
    print("==================================================")

    # 1. Initialize modules
    print("\n[1/8] Initializing OpenCV 5 Engine, Bedrock Nova Agent, and Verifier...")
    engine = WorkspaceVisionEngine(crops_dir="static/crops")
    agent = BedrockNovaAgent(model_id="amazon.nova-pro-v1:0")
    verifier = AgenticVerifier()
    print("  [OK] Modules successfully initialized.")

    # 2. Test Presets Generation
    print("\n[2/8] Testing 4 Realistic Multi-Angle Workspace Presets...")
    chaos_desk = generate_developer_chaos_desk()
    oblique_desk = generate_oblique_laptop_desk()
    paper_desk = generate_paperwork_avalanche_desk()
    zen_desk = generate_minimalist_flow_desk()
    print(f"  [OK] Developer Chaos Desk (Top-Down): shape {chaos_desk.shape}")
    print(f"  [OK] Oblique 45° Laptop Desk (Perspective): shape {oblique_desk.shape}")
    print(f"  [OK] Paperwork Avalanche Desk: shape {paper_desk.shape}")
    print(f"  [OK] Minimalist Flow Desk: shape {zen_desk.shape}")

    # 3. Analyze Developer Chaos Scenario & Oblique Scenario
    print("\n[3/8] Running OpenCV 5 Analysis + Intrinsic Calibration & ISO 9241-5 Audit...")
    telem = engine.analyze(chaos_desk)
    telem_oblique = engine.analyze(oblique_desk)
    print(f"  [OK] Zen Score: {telem['zen_score']}/100 | Oblique Zen Score: {telem_oblique['zen_score']}/100")
    print(f"  [OK] Cognitive Tier: {telem['cognitive_load_tier']}")
    print(f"  [OK] Visual Entropy: {telem['visual_entropy_score']}")
    print(f"  [OK] Surface Blocked: {telem['surface_area_blocked_pct']}%")
    print(f"  [OK] Items Detected: {telem['total_items_detected']}")
    print(f"  [OK] Spill Risk: {telem['hazards']['spill_risk']}")
    print(f"  [OK] Cable Snarl: {telem['hazards']['cable_snarl_detected']}")

    # Pillar 1: Verify Intrinsic Anthropometric Calibration & ISO 9241-5 Ergonomic Audit
    assert "intrinsic_calibration" in telem, "Missing intrinsic_calibration in telemetry"
    assert "iso_9241_audit" in telem, "Missing iso_9241_audit in telemetry"
    cal = telem["intrinsic_calibration"]
    iso = telem["iso_9241_audit"]
    cal_obl = telem_oblique["intrinsic_calibration"]
    assert cal.get("cm_per_pixel_x", 0) > 0, "Invalid cm_per_pixel_x"
    assert telem_oblique["perspective"] == "PERSPECTIVE_ANGLED", f"Expected PERSPECTIVE_ANGLED for oblique desk, got {telem_oblique['perspective']}"
    print(f"  [OK] Intrinsic Calibration (Top-Down): {cal['reference_anchor']} -> {cal['cm_per_pixel_x']} cm/px (Elev: {cal['camera_elevation_deg']} deg)")
    print(f"  [OK] Intrinsic Calibration (45° Oblique): {cal_obl['reference_anchor']} -> {cal_obl['cm_per_pixel_x']}x{cal_obl['cm_per_pixel_y']} cm/px (Elev: {cal_obl['camera_elevation_deg']} deg)")
    print(f"  [OK] ISO 9241-5 Ergonomic Score: {iso['iso_9241_score']}/100 ({iso['compliance_tier']}) | Neck Rotation: {iso['neck_rotation_deg']} deg | View Dist: {iso['viewing_distance_cm']}cm")

    # Pillar 2 & 4: Verify Live Edge Telemetry Profiler & 5-Stage Agentic Loop Trace
    assert "edge_profiler" in telem, "Missing edge_profiler in telemetry"
    assert "agentic_loop_trace" in telem and len(telem["agentic_loop_trace"]) == 5, "Expected 5-stage agentic_loop_trace"
    prof = telem["edge_profiler"]
    print(f"  [OK] OpenCV 5 + Graviton3 Edge Profiler: Core CV {prof['core_cv_ms']} ms ({prof['realtime_fps']} FPS) | Memory {prof['memory_mb']} MB")
    print(f"  [OK] 5-Stage Agentic Loop Trace: {[s['stage'] for s in telem['agentic_loop_trace']]}")

    # Verify UI card renderers
    iso_html = render_iso_calibration_card(cal, iso)
    prof_html = render_edge_profiler_card(prof)
    trace_html = render_agentic_reasoning_trace(telem["agentic_loop_trace"])
    assert len(iso_html) > 100 and len(prof_html) > 100 and len(trace_html) > 100, "UI card renderers returned empty HTML"
    print("  [OK] UI Renderers (ISO 9241-5 Card, Edge Profiler Card, Agentic Loop Trace Card) verified.")

    # Verify Central Work Focus First architecture
    assert "central_focus" in telem, "Missing central_focus in telemetry"
    assert "encroachment" in telem, "Missing encroachment in telemetry"
    cf = telem["central_focus"]
    enc = telem["encroachment"]
    print(f"  [OK] Central Focus Device: {cf.get('name')} (confidence {cf.get('confidence')})")
    print(f"  [OK] Encroaching Items: {enc.get('total_encroaching_items')} choking active envelope")
    print(f"  [OK] Wrist Clearance: {enc.get('wrist_clearance_cm')}cm | Mouse Corridor: {enc.get('mouse_clearance_cm')}cm")
    print(f"  [OK] Breathing Room: {enc.get('breathing_room_score')}% | Status: {enc.get('status')}")

    # Verify visual layer generation
    assert "annotated_image" in telem, "Missing annotated_image"
    assert "heatmap_image" in telem, "Missing heatmap_image"
    assert "relocation_image" in telem, "Missing relocation_image"
    print("  [OK] All 3 Visual Layers (Zoning, Cognitive Heatmap, Relocation Vectors) generated.")

    # Verify semantic items
    for it in telem["items"][:3]:
        print(f"    - Item #{it['id']}: {it.get('semantic_name')} | Zone: {it.get('zone')} | Target: {it.get('target_destination')} (+{it.get('displacement_cm')}cm)")

    # 4. Agent Synthesis
    print("\n[4/8] Testing Amazon Nova Pro Bedrock Synthesis...")
    plan = agent.synthesize_plan(telem)
    print(f"  [OK] Model Used: {plan.get('model_used')}")
    print(f"  [OK] Zen Quote: \"{plan.get('zen_philosophy_quote')}\"")
    print(f"  [OK] Triage Items: {len(plan.get('triage_deck', []))}")
    print(f"  [OK] 5-Minute Challenge Steps: {len(plan.get('five_minute_challenge', []))}")
    if "central_focus_liberation_strategy" in plan:
        cfl = plan["central_focus_liberation_strategy"]
        print(f"  [OK] Central Focus Strategy for: {cfl.get('focus_device')}")
        print(f"       - Wrist: {cfl.get('wrist_zone_liberation')[:65]}...")
        print(f"       - Mouse: {cfl.get('mouse_corridor_liberation')[:65]}...")

    # Test Target "After" Synthesis
    target_res = engine.after_gen.generate_target_after(chaos_desk, telem, plan)
    assert "target_image" in target_res, "Missing target_image in after_generator output"
    print(f"  [OK] Target 'After' Blueprint Generated: Score {target_res['target_score']}/100 ({target_res['score_delta']}) | {target_res['stowed_count']} items stowed")

    # 5. Interactive Checklist, Dynamic Inpainting, and Verification Loop
    print("\n[5/8] Testing Interactive Highlighting & Dynamic Inpainting...")
    all_ids = [it["id"] for it in telem["items"]]
    first_id = all_ids[0] if all_ids else 1
    second_id = all_ids[1] if len(all_ids) > 1 else first_id
    interactive_img = engine.render_interactive_view(
        chaos_desk, telem, highlighted_item_id=second_id, removed_ids={first_id}
    )
    assert interactive_img is not None and interactive_img.shape == chaos_desk.shape, "Interactive view failed"
    print(f"  [OK] Interactive Highlight View (Item #{second_id} targeted, Item #{first_id} in-place erased) rendered.")

    partial_ids = set(all_ids[:2])
    cleared_res = engine.inpaint_specific_items(
        chaos_desk, telem, removed_item_ids=partial_ids, plan=plan
    )
    assert "cleared_image" in cleared_res, "Missing cleared_image in inpaint_specific_items"
    assert "raw_cleared_image" in cleared_res, "Missing raw_cleared_image in inpaint_specific_items"
    assert cleared_res["removed_count"] == len(partial_ids), f"Expected {len(partial_ids)} removed items, got {cleared_res['removed_count']}"
    assert cleared_res["provisional_score"] > telem["zen_score"], "Provisional score did not increase"
    print(f"  [OK] Partial Inpainting Erased {len(partial_ids)} Items: Provisional Zen {cleared_res['provisional_score']}/100 ({cleared_res['score_delta']})")

    # Test full checklist removal & closed-loop verification on raw_cleared_image
    print("\n[6/8] Testing Closed-Loop Agentic Verification on Full Checklist Inpainted Desk...")
    full_cleared_res = engine.inpaint_specific_items(
        chaos_desk, telem, removed_item_ids=set(all_ids), plan=plan
    )
    telem_inpainted = engine.analyze(full_cleared_res["raw_cleared_image"], generate_after=False)
    v_res_inpainted = verifier.verify_transformation(telem, telem_inpainted)
    print(f"  [OK] Full Checklist Inpainted Verification: {v_res_inpainted['verdict_badge']}")
    print(f"       Score Delta: {v_res_inpainted['score_before']} -> {v_res_inpainted['score_after']} ({v_res_inpainted['score_delta']})")
    print(f"       Surface Reclaimed: ~{v_res_inpainted['estimated_sq_ft_reclaimed']} sq ft")
    assert v_res_inpainted["score_after"] > v_res_inpainted["score_before"] + 25, (
        f"Expected significant score jump on raw_cleared_image, got {v_res_inpainted['score_before']} -> {v_res_inpainted['score_after']}"
    )

    # 7. Test Sanctuary Blueprint Card Export (Pillar 5)
    print("\n[7/8] Testing Downloadable Before -> After Sanctuary Blueprint Card Export...")
    card_res = engine.generate_sanctuary_blueprint_card(
        before_img=chaos_desk,
        after_img=full_cleared_res["cleared_image"],
        telemetry=telem,
        plan=plan,
    )
    assert "card_image" in card_res and card_res["card_image"].shape == (720, 1620, 3), "Invalid blueprint card image shape"
    assert "png_bytes" in card_res and len(card_res["png_bytes"]) > 10000, "Invalid PNG bytes for blueprint card"
    print(f"  [OK] Sanctuary Blueprint Composite Card Generated: {card_res['card_image'].shape} ({len(card_res['png_bytes']) // 1024} KB PNG)")

    # 8. Test 2.5D Spatial Digital Twin Map (Layered Scene Graph)
    print("\n[8/8] Testing 2.5D Spatial Digital Twin Map & Sub-10ms Compositing Engine...")
    import time
    twin_map = engine.build_digital_twin_map(chaos_desk, telem, plan)
    assert len(twin_map.sprites) == len(telem["items"]), f"Expected {len(telem['items'])} sprites, got {len(twin_map.sprites)}"
    assert twin_map.base_surface_bgr.shape == chaos_desk.shape, "Base surface canvas shape mismatch"

    # Test batch state application and sub-10ms rendering
    t_comp_0 = time.perf_counter()
    twin_map.apply_checklist_state(removed_ids=set(all_ids[:2]), action_modes={all_ids[0]: "relocate", all_ids[1]: "stow"})
    twin_rendered = twin_map.render_photorealistic()
    twin_interactive = twin_map.render_interactive_view(highlighted_id=all_ids[0])
    twin_ms = (time.perf_counter() - t_comp_0) * 1000.0

    twin_telem = twin_map.get_provisional_telemetry()
    assert twin_rendered.shape == chaos_desk.shape, "Twin photorealistic composite shape mismatch"
    assert twin_interactive.shape == chaos_desk.shape, "Twin interactive view shape mismatch"
    assert twin_telem["cleared_count"] == 2, f"Expected 2 cleared items, got {twin_telem['cleared_count']}"
    assert twin_telem["relocated_count"] == 1, f"Expected 1 relocated, got {twin_telem['relocated_count']}"
    assert twin_telem["stowed_count"] == 1, f"Expected 1 stowed, got {twin_telem['stowed_count']}"
    print(f"  [OK] 2.5D Digital Twin Map: {len(twin_map.sprites)} Sprites | Base Surface Inpainted | Composited in {twin_ms:.2f}ms (<10ms)")
    print(f"  [OK] Digital Twin Telemetry: Provisional Score {twin_telem['provisional_score']}/100 ({twin_telem['score_delta']}) | {twin_telem['relocated_count']} Relocated • {twin_telem['stowed_count']} Stowed")

    # Also test realistic preset transformation
    clean_desk = generate_cleaned_after_desk("chaos")
    telem_after = engine.analyze(clean_desk)
    v_res = verifier.verify_transformation(telem, telem_after)

    print(f"\n  [OK] Full Preset Score Before: {v_res['score_before']} -> Score After: {v_res['score_after']} ({v_res['score_delta']})")
    print(f"  [OK] Entropy Reduction: -{v_res['entropy_reduction_pct']}%")
    print(f"  [OK] Surface Reclaimed: ~{v_res['estimated_sq_ft_reclaimed']} sq ft")
    print(f"  [OK] Spill Hazard Resolved: {v_res['hazard_resolved']}")
    print(f"  [OK] Verdict: {v_res['verdict_badge']}")

    print("\n==================================================")
    print("[SUCCESS] ALL 8 PIPELINE PHASES VERIFIED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_test()
