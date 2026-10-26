# 🌿 DeskZen: Autonomous Workspace Decluttering & Ergonomics Co-Pilot
**OpenCV AI Competition 2026, powered by AWS**  
*Track Target: Grand Prize ($5,000) & Agentic Vision Award ($1,000) | AWS Graviton & Bedrock*

---

## 1. Overview
Physical clutter creates cognitive friction. **DeskZen** is an intelligent vision assistant that analyzes workspace photos, computes a scientific **Cognitive Clutter & Visual Entropy Index**, detects dangerous **liquid spill risks and cable snarls**, generates a swipeable **"Keep, Stow, Trash" Triage Deck**, and conducts an interactive **Agentic Verification Loop** that visually validates physical decluttering when the user submits an "After" photo.

Designed with a **Clean Sand Mix** aesthetic (warm alabaster, desert linen, terracotta clay, and calm sage), DeskZen offers a supportive, non-judgmental experience.

---

## 2. Core Technical Architecture

```
[ User Desk Photo ] ──▶ [ OpenCV 5 Vision Engine ] ──▶ [ Structured Telemetry JSON ]
                               │                                     │
                               ▼                                     ▼
                     • Surface Homography                 [ AWS Bedrock Agent ]
                     • Shannon Gradient Entropy             (Amazon Nova Pro)
                     • Connected Components (Item Crops)             │
                     • Spatial Moments (Visual Mass)                 ▼
                     • Hough Line Skeletonization        [ Clean Sand Streamlit UI ]
                     • Liquid-to-Electronics Proximity      • Triage Deck Cards
                                                            • Spill & Cable Warnings
                                                            • Spatial Blueprint
                                                            • Before/After Verification
```

### OpenCV 5 Algorithmic Depth
1. **Visual Entropy (`cv2.Laplacian` + `cv2.Sobel`)**: Quantifies visual clutter using Shannon entropy across image gradient magnitudes.
2. **Item Segmentation & Thumbnail Cropping (`cv2.connectedComponentsWithStats`)**: Isolates clutter items, extracts bounding boxes `(x, y, w, h)`, and saves cropped PNG thumbnails.
3. **Ergonomic Reach Zones**: Projects concentric arcs from user seating position to categorize items into *Zone 1 (Primary)*, *Zone 2 (Secondary)*, and *Zone 3 (Periphery)*.
4. **Visual Mass Balance (`cv2.moments`)**: Calculates spatial moments to determine bilateral weight distribution (left-heavy vs. right-heavy bias).
5. **Spill & Cable Sentinel (`cv2.Canny` + `cv2.HoughLinesP` + Euclidean Distance)**: Measures distance between cylindrical drink containers and Zone 1 electronics to proactively prevent liquid damage.

### AWS Bedrock (Amazon Nova Pro)
* **Model ID**: `amazon.nova-pro-v1:0` (AWS 1st-party foundation model; requires no third-party consent).
* **Converse API**: Formats telemetry and yields structured triage cards, ergonomic recommendations, and a personalized 5-Minute Challenge.
* **Resilient Architecture**: Includes an automatic fallback engine if AWS credentials are not yet configured.

---

## 3. Quick Start Guide

### 3.1 Prerequisites & Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 3.2 Configure AWS Bedrock Credentials
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Fill in your AWS credentials or run `aws configure`.
3. Ensure your IAM user/role has the permissions specified in `iam_policy.json`.

### 3.3 Run Automated Pipeline Test
Verify all modules headlessly:
```bash
python test_pipeline.py
```

### 3.4 Launch Streamlit Web UI
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 4. The Agentic Verification Loop

1. **Step 1 (Inspection)**: Upload or capture your cluttered desk photo. DeskZen analyzes the workspace, scores your clutter (e.g. 42/100), and outputs your 3-step action checklist.
2. **Step 2 (Physical Action)**: Follow the instructions (e.g., move coffee cup, stow loose papers into drawer).
3. **Step 3 (Verification)**: Upload the cleaned "After" photo in the second tab. DeskZen recalculates all metrics, verifies physical removal, and celebrates the score jump (e.g., 42 ➔ 86/100) and square footage reclaimed!
