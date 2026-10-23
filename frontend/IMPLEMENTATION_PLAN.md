# High-Performance Minimalist CCTV Surveillance UI Implementation Plan

> **Goal:** Transform the CCTV monitoring frontend into an ultra-lightweight, zero-latency, high-craft surveillance interface that operates at maximum FPS, completely purges fake data, eliminates parallel stream floods, and provides seamless 1-click camera management.

**Architecture:** 
- Decoupled, zero-re-render `StreamViewport` with hardware-accelerated time and FPS telemetry overlay.
- Complete removal of heavy canvas re-renders, backdrop filters, and parallel snapshot polling.
- Pure real-time camera registry strictly linked to active channels on `142.67.45.150`, equipped with an instant channel-templated camera creation modal.
- Strictly isolated state tree ensuring UI interactions (buttons, tabs, switches) execute instantaneously on the first click without main-thread blocking.

**Tech Stack:** Next.js 15, React 19, Vanilla CSS / Tailwind utilities (minimal CPU footprint, hardware-composited `transform: translateZ(0)`), TypeScript.

---

## Phase 1: Fake Data Purge & Channel-Templated Camera Registry (Frontend)

### Context & Problem:
The frontend currently loads 1,245 lines of simulated border cameras (`borderCameras.ts`) and merges them with backend cameras. Furthermore, `cameracard.tsx` automatically queries `/api/backend/stream/snapshot` for every camera simultaneously on page load, firing 20–40 parallel RTSP connections that saturate the NVR at `142.67.45.150`, choking bandwidth and dropping live stream framerates to 5 FPS.

### Deliverables:
1. **Purge Fake Cameras from Frontend (`frontend/src/lib/borderCameras.ts`, `frontend/src/lib/camerasStore.ts`)**:
   - Limit default initial camera pool strictly to the real cameras 1–10 (`bop-jk-01` to `bop-jk-10`).
   - Filter out all dummy cameras (indices >= 11 or non-existent IPs).
2. **Prevent Parallel NVR Stream Floods (`frontend/src/components/surveillance/cameracard.tsx`, `cameragrid.tsx`)**:
   - Eliminate unconditional `<img src="/api/backend/stream/snapshot">` on all cards.
   - Replace with ultra-lightweight status badges (Name, Channel, Status, Lat/Lng).
   - Only fetch snapshot or video for the single camera currently inspected or selected.
3. **1-Click RTSP Channel-Templated Camera Addition (`frontend/src/components/cameras/AddCameraModal.tsx` / `frontend/src/lib/camerasStore.ts`)**:
   - Implement input form with:
     - Camera Name (e.g. "Gate 02 Forward")
     - Channel Number (e.g. `11`)
     - Latitude & Longitude (e.g. `23.15421`, `72.885257`)
   - Auto-generate RTSP URL using template:
     `rtsp://admin:Admin123@142.67.45.150:554/cam/realmonitor?channel={CHANNEL}&subtype=0`
   - Store in `camerasStore` and sync to backend API (`POST /api/cameras`).

---

## Phase 2: Ultra-Lightweight Live Stream Viewport & Feed HUD (Frontend)

### Context & Problem:
The live surveillance page (`liveworkspace.tsx`) is currently 2,095 lines long and contains heavy DOM trees, animated SVGs, backdrop-blur filters, canvas overlays, and a 350ms interval that polls telemetry and triggers full component re-renders. This locks the browser's JavaScript thread, starving the video stream decoder and causing frames to drop to 5 FPS and clicks to feel unresponsive.

### Deliverables:
1. **Isolated `LightweightStreamViewport` Component (`frontend/src/components/surveillance/LightweightStreamViewport.tsx`)**:
   - Encapsulate the `<img src={streamUrl}>` inside an isolated component that never re-renders when parent states change.
   - Utilize CSS `transform: translateZ(0)` and `contain: strict` to isolate browser compositing and paint operations to the GPU.
2. **Minimalist Feed Overlay: ONLY Time & FPS**:
   - Strip all unnecessary visual clutter from the video feed: remove pulsating green radar dots, corner brackets, animated bounding boxes, glowing filters, and floating drawers.
   - Keep ONLY two clean, high-contrast, low-overhead HUD readouts:
     - **Real-Time Clock** (e.g. `22:52:15 IST`) updated via a single uncoupled 1-second interval.
     - **Live Stream FPS** (measured directly from stream response timing / requestAnimationFrame).
3. **Instant Single-Click Controls**:
   - Zero debounce or event queue backlog: camera switches, full-screen toggles, and stream start/stop respond on the exact frame the click occurs.

---

## Phase 3: Minimalist Industrial CCTV Command Interface (Frontend)

### Context & Problem:
The UI currently features heavy consumer-grade styling (rounded glassmorphic badges, multi-layer shadows, gradient overlays) that look like generic AI prototypes and consume excessive GPU rendering cycles during video playback.

### Deliverables:
1. **Redesign the Surveillance Console (`frontend/src/components/surveillance/LiveSurveillanceConsole.tsx`)**:
   - Streamlined, high-craft, utilitarian layout built specifically for 24/7 security control rooms.
   - Quick-switch Camera Bar: Single-click camera selection ribbon with clear status indicators (Online/Alert/Offline) and channel numbers.
   - Clean stream controls: Fullscreen (F), Stream Restart, Snapshot Capture, and Minimal Inference Toggle.
2. **Minimalist AI Inference Mode**:
   - Best-in-class mode: Clean toggle between raw stream and server-rendered inference.
   - When AI overlay is enabled, display clean, high-contrast bounding boxes without DOM overhead, preserving 20+ FPS.
3. **Route Integration**:
   - Connect the new streamlined console directly to `frontend/src/app/(operator)/live/page.tsx` and ensure responsive performance across all screen resolutions.

---

## Phase 4: Backend Optimization Analysis & Consultation (Awaiting Approval)

### Context & Identified Bottlenecks:
While investigating backend streaming (`backend/app/api/stream.py`), we identified key areas that directly impact stream FPS and latency:
1. **Dummy Cameras in Backend**: `backend/app/data/cameras.json` contains dummy cameras 11–40. Trimming them ensures the backend doesn't waste memory or threads on offline simulated cameras.
2. **OpenCV In-Frame HUD vs Client-Side HUD**: The backend currently runs OpenCV text and rectangle rendering on every frame in Python (`draw_tactical_hud` in `stream.py`). Removing redundant backend drawings saves CPU/GPU cycle time per frame.
3. **RTSP Buffer & Drop Settings**: Setting `cv2.CAP_PROP_BUFFERSIZE = 1` and optimizing RTSP transport flags (`rtsp_transport=tcp`) prevents frame buffering delay.

> **CRITICAL GATE:** As instructed, NO backend files will be touched without prior review and explicit approval from you stating the exact reasons.

---

## Phase 5: End-to-End Validation & Performance Benchmarking

### Deliverables:
1. Measure stream framerate (confirming smooth 20–30 FPS playback from `142.67.45.150`).
2. Verify camera addition with custom channel number (`channel={CHANNEL}`).
3. Verify single-click camera switching with instant feed transition.
4. Verify complete absence of fake/dummy camera entries.
5. Validate production build: `npm run build` and `npx tsc --noEmit`.
