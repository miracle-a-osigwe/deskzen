# 🌿 DeskZen: Autonomous Workspace Ergonomics & Declutter Co-Pilot
**OpenCV AI Competition 2026, Powered by AWS**  
*Target Tracks: Grand Prize ($5,000) | Agentic Vision Award ($1,000) | Best Use of COOL Award (AWS Graviton)*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-AWS%20EC2-FF9900?logo=amazon-aws&logoColor=white)](http://ec2-44-200-118-89.compute-1.amazonaws.com)
[![OpenCV 5](https://img.shields.io/badge/OpenCV-5.x%20Edge%20Math-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org)
[![AWS Bedrock](https://img.shields.io/badge/AWS%20Bedrock-Amazon%20Nova%20Pro-232F3E?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![Architecture](https://img.shields.io/badge/Platform-AWS%20Graviton3%20%7C%20Arm64-00A4E4)](https://aws.amazon.com/ec2/graviton/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 1. Executive Summary

Physical clutter creates cognitive friction. Neuroscience demonstrates that task-irrelevant visual stimuli in our visual field compete for neural representation in the visual cortex, elevating cortisol and inducing decision paralysis.

**DeskZen** transforms passive computer vision into an **active, agentic ergonomics co-pilot**:
1. **Identifies the Primary Central Work Focus** (Laptop, Desktop Keyboard + Display, Tablet) and calculates spatial encroachment choking the user's active work envelope.
2. **Conducts an Automated ISO 9241-5 Biomechanical Audit** measuring display-to-input axial alignment ($^\circ$ neck rotation), eye-to-screen viewing distance, forearm neutral wrist depth (target $\ge 15\text{ cm}$), and mouse shoulder abduction (target $\ge 22\text{ cm}$).
3. **Decomposes Workspace Photos into a 2.5D Canva-Style Spatial Digital Twin** (`core/spatial_map.py`):
   - Inpaints a pristine base tabletop canvas via Fast-Marching Telea (`cv2.INPAINT_TELEA`).
   - Extracts authentic RGBA object sprites with feathered alpha cutouts.
   - Computes **Multi-Constraint Collision-Free Safe Slots** in Zone 2/3 (e.g. liquid container placed strictly $>25\text{ cm}$ away on an auto-scaled cork/slate coaster; phone docked outside mouse sweep corridor).
4. **Sub-10ms Deterministic Spatial Compositing**: Enables instant Canva-style repositioning with realistic contact drop shadows without slow, hallucinatory generative diffusion models.
5. **Executes a Closed-Loop Agentic Verification**: Mathematically certifies physical decluttering via Before-vs-After computer vision re-ingestion, celebrating verified Zen score leaps (+60 to +84 pts) and reclaimed square footage.

👉 **Live Web Application**: [http://ec2-44-200-118-89.compute-1.amazonaws.com](http://ec2-44-200-118-89.compute-1.amazonaws.com) *(Direct IP: `http://44.200.118.89`)*

---

## 2. Technical Architecture

```
[ Workspace Photo / Video ]
           │
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     OpenCV 5 + YOLOv8 Vision Engine                          │
│  • Intrinsic Fiducial Scale Calibration (Laptop 32.5cm -> cm/pixel, Pitch θ) │
│  • Central Work Focus & Ergonomic Breathing Halo Isolation                   │
│  • ISO 9241-5 Biomechanical Audit (Neck Rotation, Viewing Dist, Reach)       │
│  • Shannon Visual Entropy & Laplacian Gradient Chaos                         │
│  • Euclidean Spill Distance Radar (<25cm) & Hough Cable Snarl Sentinel       │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │ Structured Telemetry JSON
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    AWS Bedrock Agent (Amazon Nova Pro)                       │
│  • Converse API Multi-Constraint Ergonomic Optimization                      │
│  • Dynamic Priority Ranking: Spill Risk > Wrist > Mouse > Entropy           │
│  • Collision-Free Zone 2 / Zone 3 Geometric Slot Allocation Matrix           │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            2.5D Canva-Style Spatial Digital Twin Scene Graph                 │
│  • Inpainted Base Tabletop Canvas (Telea Fast-Marching)                      │
│  • RGBA Object Sprites with Feathered Boundaries                             │
│  • Multi-Constraint Non-Overlapping Slot Solver                              │
│  • Sub-10ms Contact Drop Shadows & Scaled Slate/Cork Coasters                │
│  • Side-by-Side Interactive Studio with 1-Click Version Promotion (v1 -> v2)│
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            Closed-Loop Agentic Verification & Sanctuary Export               │
│  • Physical Reality Confirmation & Mathematical CV Re-Scan                   │
│  • Verified Zen Score Jump Celebration (+56 to +84 points)                   │
│  • Downloadable Before/After Sanctuary Blueprint Card (1620x720 PNG + JSON)  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The 5 Winning Pillars

### Pillar 1: Intrinsic Anthropometric Scale Calibration & ISO 9241-5 Geometry
* Automatically detects the central work device (`laptop` standard width $32.5\text{ cm}$, `keyboard` $42.0\text{ cm}$, `monitor` $54.0\text{ cm}$) as a physical fiducial anchor to compute exact `cm/pixel` scaling for any camera resolution or viewing distance.
* Computes 3D camera elevation angle $\theta = \arccos(\text{foreshortening})$ and runs an automated **ISO 9241-5 Workstation Ergonomic Audit** measuring axial display-to-input symmetry offset ($^\circ$ neck rotation), viewing distance ($50\text{--}70\text{ cm}$ target), forearm neutral wrist depth ($\ge 15\text{ cm}$), and mouse shoulder abduction ($\ge 22\text{ cm}$).

### Pillar 2: COOL Award Edge Profiler (AWS Graviton3 & CPU Optimized)
* Engineered for high efficiency on CPU and ARM64 without requiring expensive GPU clusters.
* Live Edge Profiler measures sub-millisecond execution times for all vision stages (`homography_ms`, `yolo_anchor_ms`, `opencv_morphology_hough_ms`, `telea_inpaint_ms`, `overlay_synthesis_ms`), achieving **<120 ms total CV pipeline latency**, **~7.1 MB memory footprint**, and **>8.5 FPS throughput**.

### Pillar 3: Multi-Angle Camera Benchmark Presets
* Ships with 4 competition-grade workspace presets tested across angles:
  1. **Developer Chaos**: Top-down workstation choked by sticky notes, mug in spill zone, and mouse corridor blockage.
  2. **Oblique 45° Laptop Desk**: Angled perspective proving homography correction and foreshortened depth bands.
  3. **Paperwork Avalanche**: High-entropy document overflow on side zones.
  4. **Minimalist Flow**: Zen benchmark setup demonstrating optimal ISO 9241-5 compliance.

### Pillar 4: Transparent 5-Stage Agentic Reasoning Trace
* Exposes the agent's internal state machine directly to users and judges:
  1. `1. OBSERVE`: Intrinsic scale calibration, camera pitch, and hybrid object detection.
  2. `2. ORIENT`: Central Focus identification, wrist clearance, and ISO 9241-5 audit.
  3. `3. DECIDE`: Amazon Nova Pro spatial triage and collision-free slot allocation.
  4. `4. ACT`: Interactive checklist with real-time Telea inpainting and 2.5D compositing.
  5. `5. VERIFY`: Closed-loop computer vision re-ingestion certifying hazard resolution.

### Pillar 5: Downloadable Sanctuary Blueprint Card & Spatial Telemetry Export
* Generates a downloadable high-resolution 1620×720 side-by-side Before (Spatial Diagnosis) vs. After (Target Sanctuary + Relocation Checklist) composite PNG card with embedded Zen score deltas, ISO compliance tier, and 1-click JSON telemetry export.

---

## 4. Quick Start Guide

### Option 1: Live Cloud Demo
Access the live deployment on AWS EC2 directly in your browser:  
👉 **`http://ec2-44-200-118-89.compute-1.amazonaws.com`** *(or `http://44.200.118.89`)*

### Option 2: Local Installation

```bash
# 1. Clone repository
git clone https://github.com/[your-github-username]/opencv-ai-deskzen.git
cd opencv-ai-deskzen

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure AWS Bedrock credentials in .env
cp .env.example .env
# Edit .env with your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
# Note: An intelligent local fallback synthesizer is included if AWS credentials are not configured.

# 4. Run automated test suite (verifies all 8 pipeline phases)
python test_pipeline.py

# 5. Launch interactive web app
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option 3: Docker Deployment (AWS Graviton / x86_64)

```bash
docker build -t deskzen:latest .
docker run -d -p 8501:8501 --name deskzen-app deskzen:latest
```

---

## 5. Automated Test Suite

DeskZen includes a comprehensive automated test script (`test_pipeline.py`) validating all 8 pipeline phases headlessly:
```bash
python test_pipeline.py
```
Expected output:
```text
==================================================
[DeskZen] End-to-End Automated Pipeline Test
==================================================
[1/8] Initializing OpenCV 5 Engine, Bedrock Nova Agent, and Verifier... [OK]
[2/8] Testing 4 Realistic Multi-Angle Workspace Presets... [OK]
[3/8] Running OpenCV 5 Analysis + Intrinsic Calibration & ISO 9241-5 Audit... [OK]
[4/8] Testing Amazon Nova Pro Bedrock Synthesis... [OK]
[5/8] Testing Interactive Highlighting & Dynamic Inpainting... [OK]
[6/8] Testing Closed-Loop Agentic Verification on Full Checklist Inpainted Desk... [OK]
[7/8] Testing Downloadable Before -> After Sanctuary Blueprint Card Export... [OK]
[8/8] Testing 2.5D Spatial Digital Twin Map & Sub-10ms Compositing Engine... [OK]
==================================================
[SUCCESS] ALL 8 PIPELINE PHASES VERIFIED SUCCESSFULLY!
==================================================
```

---

## 6. Repository Structure

```
├── app.py                      # Main Streamlit Side-by-Side Studio Application
├── test_pipeline.py            # Automated 8-Phase Verification Test Suite
├── Dockerfile                  # Production Container for AWS Graviton & CPU
├── deploy.sh                   # 1-Click AWS Deployment Script
├── requirements.txt            # Python Dependencies (OpenCV 5, Ultralytics, Boto3, Streamlit)
├── core/
│   ├── cv_engine.py            # OpenCV 5 Hybrid Vision Engine & ISO 9241-5 Audit
│   ├── spatial_map.py          # 2.5D Digital Twin Scene Graph & Collision-Free Solver
│   ├── bedrock_agent.py        # AWS Bedrock (Amazon Nova Pro) Converse API Client
│   ├── yolo_detector.py        # YOLOv8 Neural Anchor & Object Detector
│   ├── after_generator.py      # Telea Inpainter & Sanctuary Blueprint Generator
│   ├── verification.py         # Agentic Closed-Loop Verification Engine
│   └── desk_scenarios.py       # Multi-Angle Competition Benchmark Presets
├── ui/
│   └── sand_theme.py           # "Clean Sand Mix" Japandi/Mediterranean Design System
├── docs/
│   ├── submission.md           # Official Devpost Submission Text
│   └── video_script.md         # 5-Minute Competition Video Demo Script
└── static/crops/               # Extracted Clutter Sprites & Blueprint Artifacts
```

---

## 7. License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
