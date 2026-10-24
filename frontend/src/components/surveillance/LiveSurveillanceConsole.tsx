"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { useCameras } from "@/lib/camerasStore";
import { Camera as CameraType } from "@/types/camera";
import { StreamViewport } from "./StreamViewport";
import AddCameraModal from "@/components/cameras/AddCameraModal";
import {
  Play,
  RefreshCw,
  Camera,
  Maximize2,
  Minimize2,
  X,
  Plus,
  ScanFace,
  Download,
  Shield,
  Layers,
  ChevronDown,
  ChevronUp,
  MapPin,
  Wifi,
  Sparkles,
  Activity,
} from "lucide-react";

/** Build the backend MJPEG stream URL (proxied Next.js → FastAPI). */
function buildStreamUrl(
  rtspUrl: string,
  drawDetections: boolean,
  enableFaceBiometrics: boolean,
  sessionId?: string,
  cameraId?: string
): string {
  const isAi = Boolean(drawDetections);
  const params = new URLSearchParams({
    rtsp_url: rtspUrl,
    draw_detections: String(isAi),
    enable_face_recognition: String(isAi && enableFaceBiometrics),
    enable_vehicle_detection: "false",
    fps: "20",
    _t: String(Date.now()),
  });
  if (sessionId) params.set("session_id", sessionId);
  if (cameraId) params.set("camera_id", cameraId);
  return `/api/backend/stream/live?${params.toString()}`;
}

export function LiveSurveillanceConsole() {
  const { cameras } = useCameras();

  // Selected camera state
  const [selectedCamera, setSelectedCamera] = useState<CameraType | null>(null);
  const [activeMjpegSrc, setActiveMjpegSrc] = useState<string | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [, setStreamError] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);

  // Viewport display frame size: "compact" (70% width, 30% smaller to fit screen without scrolling) or "full" (100% wide)
  const [frameSize, setFrameSize] = useState<"compact" | "full">("compact");

  // Real-time telemetry (FPS & Latency)
  const [streamFps, setStreamFps] = useState<number | null>(null);
  const [streamLatency, setStreamLatency] = useState<number | null>(null);

  // AI & Stream mode toggles
  const [aiMode, setAiMode] = useState<"raw" | "inference">("inference");
  const [enableFaceBiometrics, setEnableFaceBiometrics] = useState(true);

  // Manual RTSP toggle & input
  const [showManualRtsp, setShowManualRtsp] = useState(false);
  const [rtspInputValue, setRtspInputValue] = useState("");
  const [rtspUrl, setRtspUrl] = useState("");

  // Add Camera modal
  const [showAddCamera, setShowAddCamera] = useState(false);

  // Fullscreen
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // View mode for camera list: "ribbon" or "matrix"
  const [viewMode, setViewMode] = useState<"compact" | "grid">("compact");

  // Stable session ID
  const sessionIdRef = useRef(
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2)
  );

  // Sort cameras strictly by channel number (1 to 10)
  const sortedCameras = useMemo(() => {
    return [...cameras].sort((a, b) => {
      const chA = a.channel ?? 999;
      const chB = b.channel ?? 999;
      return chA - chB;
    });
  }, [cameras]);

  // Auto-select first camera on initial load if none selected
  useEffect(() => {
    if (!selectedCamera && sortedCameras.length > 0) {
      const firstCam = sortedCameras[0];
      handleSelectCamera(firstCam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sortedCameras]);

  // Stream teardown on unmount
  useEffect(() => {
    const sid = sessionIdRef.current;
    return () => {
      fetch(
        `/api/backend/stream/live/stop?session_id=${encodeURIComponent(sid)}`,
        { method: "DELETE", keepalive: true }
      ).catch(() => {});
    };
  }, []);

  // Fullscreen event listener
  useEffect(() => {
    const onFs = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener("fullscreenchange", onFs);
    return () => document.removeEventListener("fullscreenchange", onFs);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    if (!viewportRef.current) return;
    if (!document.fullscreenElement) {
      try {
        await viewportRef.current.requestFullscreen();
      } catch {
        setIsFullscreen(true);
      }
    } else {
      await document.exitFullscreen().catch(() => {});
    }
  }, []);

  // Connect to a camera stream with socket cleanup and race condition prevention
  const connectToStream = useCallback(
    (targetUrl: string, camName?: string) => {
      if (!targetUrl) return;

      // 1. Immediately signal backend to stop previous session
      const oldSid = sessionIdRef.current;
      if (oldSid) {
        fetch(
          `/api/backend/stream/live/stop?session_id=${encodeURIComponent(oldSid)}`,
          { method: "DELETE" }
        ).catch(() => {});
      }

      // 2. Generate new unique session ID
      const newSid =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).slice(2);
      sessionIdRef.current = newSid;

      // 3. Temporarily clear active source to unbind old Chromium HTTP socket
      setActiveMjpegSrc(null);
      setStreamFps(null);
      setStreamLatency(null);

      const drawDetections = aiMode === "inference";
      const src = buildStreamUrl(
        targetUrl,
        drawDetections,
        enableFaceBiometrics,
        newSid,
        selectedCamera?.id
      );

      setRtspUrl(targetUrl);
      setRtspInputValue(targetUrl);
      setIsStreamActive(true);
      setStreamError(false);
      setStatusMessage(camName ? `Viewing: ${camName}` : "Connected to stream");

      // 4. Yield to browser render loop before mounting new MJPEG connection
      requestAnimationFrame(() => {
        setActiveMjpegSrc(src);
        setStreamKey((k) => k + 1);
      });
    },
    [aiMode, enableFaceBiometrics, selectedCamera?.id]
  );

  // Switch camera selection
  const handleSelectCamera = useCallback(
    (cam: CameraType) => {
      setSelectedCamera(cam);
      if (cam.streamUrl) {
        connectToStream(cam.streamUrl, cam.name);
      } else {
        setIsStreamActive(false);
        setActiveMjpegSrc(null);
        setStatusMessage(`No RTSP URL configured for ${cam.name}`);
      }
    },
    [connectToStream]
  );

  // Switch between RAW and AI INFERENCE modes with socket reset
  const handleSwitchAiMode = useCallback(
    (newMode: "raw" | "inference") => {
      setAiMode(newMode);
      if (isStreamActive && rtspUrl) {
        const oldSid = sessionIdRef.current;
        if (oldSid) {
          fetch(
            `/api/backend/stream/live/stop?session_id=${encodeURIComponent(oldSid)}`,
            { method: "DELETE" }
          ).catch(() => {});
        }
        const newSid =
          typeof crypto !== "undefined" && crypto.randomUUID
            ? crypto.randomUUID()
            : Math.random().toString(36).slice(2);
        sessionIdRef.current = newSid;

        // Clean unbind old socket
        setActiveMjpegSrc(null);
        setStreamFps(null);
        setStreamLatency(null);

        const drawDetections = newMode === "inference";
        const src = buildStreamUrl(
          rtspUrl,
          drawDetections,
          enableFaceBiometrics,
          newSid,
          selectedCamera?.id
        );

        setStatusMessage(
          newMode === "raw"
            ? "Raw Stream active (Zero AI overlays)"
            : "AI Inference active"
        );

        requestAnimationFrame(() => {
          setActiveMjpegSrc(src);
          setStreamKey((k) => k + 1);
        });
      }
    },
    [isStreamActive, rtspUrl, enableFaceBiometrics, selectedCamera?.id]
  );

  // Toggle biometric faces checkbox with socket reset
  const handleToggleFaceBiometrics = useCallback(
    (enabled: boolean) => {
      setEnableFaceBiometrics(enabled);
      if (isStreamActive && rtspUrl && aiMode === "inference") {
        const oldSid = sessionIdRef.current;
        if (oldSid) {
          fetch(
            `/api/backend/stream/live/stop?session_id=${encodeURIComponent(oldSid)}`,
            { method: "DELETE" }
          ).catch(() => {});
        }
        const newSid =
          typeof crypto !== "undefined" && crypto.randomUUID
            ? crypto.randomUUID()
            : Math.random().toString(36).slice(2);
        sessionIdRef.current = newSid;

        setActiveMjpegSrc(null);
        setStreamFps(null);
        setStreamLatency(null);

        const src = buildStreamUrl(rtspUrl, true, enabled, newSid, selectedCamera?.id);

        requestAnimationFrame(() => {
          setActiveMjpegSrc(src);
          setStreamKey((k) => k + 1);
        });
      }
    },
    [isStreamActive, rtspUrl, aiMode, selectedCamera?.id]
  );

  // Restart / Reconnect active stream
  const handleRestartStream = useCallback(() => {
    if (rtspUrl) {
      setStreamFps(null);
      setStreamLatency(null);
      connectToStream(rtspUrl, selectedCamera?.name);
      setStatusMessage("Stream refreshed");
    }
  }, [connectToStream, rtspUrl, selectedCamera?.name]);

  // Stop active stream
  const handleStopStream = useCallback(() => {
    setIsStreamActive(false);
    setActiveMjpegSrc(null);
    setStreamError(false);
    setStreamFps(null);
    setStreamLatency(null);
    setStatusMessage("Stream stopped");
    const sid = sessionIdRef.current;
    fetch(
      `/api/backend/stream/live/stop?session_id=${encodeURIComponent(sid)}`,
      { method: "DELETE" }
    ).catch(() => {});
  }, []);

  // Real-time stream telemetry (FPS & Latency) polling with overload protection
  useEffect(() => {
    if (!isStreamActive || !activeMjpegSrc) {
      setStreamFps(null);
      setStreamLatency(null);
      return;
    }

    let isMounted = true;
    let pollTimer: NodeJS.Timeout | null = null;
    let inFlight = false;
    let abortController: AbortController | null = null;

    const pollTelemetry = async () => {
      if (!isMounted || inFlight) return;
      const sid = sessionIdRef.current;
      if (!sid) return;

      inFlight = true;
      abortController = new AbortController();
      const timeoutId = setTimeout(() => {
        abortController?.abort();
      }, 1800);

      try {
        const res = await fetch(
          `/api/backend/stream/live/detections?session_id=${encodeURIComponent(sid)}`,
          {
            cache: "no-store",
            signal: abortController.signal,
          }
        );
        clearTimeout(timeoutId);
        if (res.ok && isMounted) {
          const data = await res.json();
          if (isMounted && data) {
            if (typeof data.fps === "number" && data.status === "active") {
              setStreamFps(data.fps);
              if (typeof data.latency_ms === "number") {
                setStreamLatency(data.latency_ms);
              }
            } else if (data.status === "stale" || data.status === "idle") {
              setStreamFps(0);
            }
          }
        }
      } catch {
        // Handled: aborted or transient backend load spike
      } finally {
        inFlight = false;
        if (isMounted) {
          // Schedule next poll only AFTER current finishes — prevents network socket congestion
          pollTimer = setTimeout(pollTelemetry, 1000);
        }
      }
    };

    pollTimer = setTimeout(pollTelemetry, 600);

    return () => {
      isMounted = false;
      if (pollTimer) clearTimeout(pollTimer);
      if (abortController) abortController.abort();
    };
  }, [isStreamActive, activeMjpegSrc, streamKey]);

  // Snapshot capture
  const handleCaptureSnapshot = useCallback(async () => {
    if (!rtspUrl) return;
    setIsCapturing(true);
    setStatusMessage("Capturing snapshot...");
    try {
      const res = await fetch(
        `/api/backend/stream/snapshot?rtsp_url=${encodeURIComponent(rtspUrl)}`
      );
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const namePart = (selectedCamera?.name || "camera")
          .toLowerCase()
          .replace(/[^a-z0-9]/g, "_");
        const chPart = selectedCamera?.channel
          ? `_ch${selectedCamera.channel}`
          : "";
        a.download = `snapshot_${namePart}${chPart}_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setStatusMessage("Snapshot downloaded");
      } else {
        setStatusMessage("Snapshot failed: camera unreachable");
      }
    } catch {
      setStatusMessage("Snapshot failed");
    } finally {
      setIsCapturing(false);
    }
  }, [rtspUrl, selectedCamera?.name, selectedCamera?.channel]);

  // Keyboard navigation & controls:
  // 1-9: Channel 1-9, 0: Channel 10, F: Fullscreen, R: Reconnect, S: Snapshot, A: Toggle AI
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA") return;

      // Number keys: 1-9 (Channels 1-9), 0 (Channel 10)
      if (e.key >= "1" && e.key <= "9") {
        const channelNum = parseInt(e.key, 10);
        const cam = sortedCameras.find((c) => c.channel === channelNum);
        if (cam) {
          e.preventDefault();
          handleSelectCamera(cam);
        }
      } else if (e.key === "0") {
        const cam = sortedCameras.find((c) => c.channel === 10);
        if (cam) {
          e.preventDefault();
          handleSelectCamera(cam);
        }
      } else if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        toggleFullscreen();
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        handleRestartStream();
      } else if (e.key === "s" || e.key === "S") {
        e.preventDefault();
        handleCaptureSnapshot();
      } else if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        handleSwitchAiMode(aiMode === "raw" ? "inference" : "raw");
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    sortedCameras,
    handleSelectCamera,
    toggleFullscreen,
    handleRestartStream,
    handleCaptureSnapshot,
  ]);

  return (
    <div className="space-y-3 font-sans">
      {/* ═══ 1. QUICK-SWITCH CAMERA BUTTON GRID ═══ */}
      <div className="bg-white/80 backdrop-blur-xs border border-slate-200/90 rounded-xl p-2.5 space-y-2 shadow-2xs">
        {/* Header Bar */}
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2 min-w-0">
            <span className="px-2 py-0.5 rounded-md bg-emerald-100/90 border border-emerald-200/80 text-emerald-900 text-[11px] font-mono font-bold tracking-tight">
              CHANNELS ({sortedCameras.length})
            </span>
            {selectedCamera ? (
              <span className="text-xs text-slate-600 truncate">
                Active:{" "}
                <strong className="text-slate-900 font-bold font-mono">
                  [CH {String(selectedCamera.channel ?? 0).padStart(2, "0")}]
                </strong>{" "}
                <span className="font-semibold text-slate-800">{selectedCamera.name}</span>
              </span>
            ) : (
              <span className="text-xs text-slate-500 italic">Select a channel to view live feed</span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="hidden md:inline text-[11px] text-slate-500 font-mono">
              Press keys <kbd className="px-1 py-0.5 bg-slate-50 border border-slate-200 rounded text-slate-700 font-bold shadow-2xs">1-0</kbd> to switch
            </span>

            <button
              type="button"
              onClick={() => setShowAddCamera(true)}
              className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-semibold border border-dashed border-slate-300 bg-white text-slate-700 hover:text-emerald-700 hover:border-emerald-500 hover:bg-emerald-50/50 transition-colors cursor-pointer shadow-2xs"
              title="Add New Channel"
            >
              <Plus className="w-3.5 h-3.5 text-emerald-600" />
              <span>Add Channel</span>
            </button>

            {/* View toggle */}
            <div className="flex items-center gap-0.5 border border-slate-200 rounded-lg p-0.5 bg-slate-50 shadow-2xs">
              <button
                type="button"
                onClick={() => setViewMode("compact")}
                className={`px-2.5 py-1 text-[11px] rounded-md cursor-pointer transition-colors ${
                  viewMode === "compact"
                    ? "bg-white text-slate-900 font-bold shadow-2xs"
                    : "text-slate-600 hover:text-slate-900 font-medium"
                }`}
              >
                Feed
              </button>
              <button
                type="button"
                onClick={() => setViewMode("grid")}
                className={`px-2.5 py-1 text-[11px] rounded-md cursor-pointer transition-colors ${
                  viewMode === "grid"
                    ? "bg-white text-slate-900 font-bold shadow-2xs"
                    : "text-slate-600 hover:text-slate-900 font-medium"
                }`}
              >
                Matrix
              </button>
            </div>
          </div>
        </div>

        {/* Responsive Zero-Scroll Camera Button Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-5 xl:grid-cols-10 gap-1.5">
          {sortedCameras.map((cam) => {
            const isSelected = selectedCamera?.id === cam.id;
            const ch = cam.channel ?? 0;
            const shortcutKey = ch === 10 ? "0" : ch >= 1 && ch <= 9 ? String(ch) : null;

            return (
              <button
                key={cam.id}
                type="button"
                onClick={() => handleSelectCamera(cam)}
                title={`${cam.name} (Channel ${ch} • Shortcut: ${shortcutKey ? `Key ${shortcutKey}` : "None"})`}
                className={`flex flex-col justify-between p-2 rounded-xl text-left transition-all cursor-pointer select-none border relative group min-h-[52px] ${
                  isSelected
                    ? "bg-slate-950 border-emerald-500 text-white shadow-md ring-2 ring-emerald-500/40"
                    : "bg-white border-slate-200 text-slate-700 hover:border-emerald-400 hover:bg-emerald-50/20 hover:text-slate-950 hover:shadow-2xs"
                }`}
              >
                <div className="flex items-center justify-between gap-1 w-full">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`w-2 h-2 rounded-full shrink-0 ${
                        cam.status === "offline"
                          ? "bg-slate-400"
                          : isSelected
                          ? "bg-emerald-400 ring-2 ring-emerald-400/40 animate-pulse"
                          : "bg-emerald-500"
                      }`}
                    />
                    <span
                      className={`font-mono text-[10px] font-bold px-1 py-0.2 rounded ${
                        isSelected
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                          : "bg-slate-100 text-slate-600 group-hover:bg-emerald-100 group-hover:text-emerald-800"
                      }`}
                    >
                      {ch ? `CH ${String(ch).padStart(2, "0")}` : "CAM"}
                    </span>
                  </div>
                  {shortcutKey && (
                    <span
                      className={`font-mono text-[9px] px-1 rounded font-bold ${
                        isSelected
                          ? "text-slate-300 bg-slate-800 border border-slate-700"
                          : "text-slate-400 bg-slate-50 border border-slate-200"
                      }`}
                    >
                      {shortcutKey}
                    </span>
                  )}
                </div>

                <div
                  className={`text-xs font-semibold truncate leading-tight w-full mt-1 ${
                    isSelected ? "text-white font-bold" : "text-slate-800 group-hover:text-emerald-950"
                  }`}
                  title={cam.name}
                >
                  {cam.name}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* ═══ 2. ACTIVE FEED VIEWPORT (30% Compact default or 100% Full Width) ═══ */}
      <div
        ref={viewportRef}
        className={`relative rounded-xl overflow-hidden bg-black border border-slate-800/90 shadow-md transition-all duration-300 ${
          isFullscreen
            ? "fixed inset-0 z-50 rounded-none border-none w-full h-full"
            : frameSize === "compact"
            ? "w-full md:w-[70%] max-w-[70%] mx-auto"
            : "w-full"
        }`}
        style={{
          aspectRatio: isFullscreen ? undefined : "16/9",
          contain: "strict",
        }}
      >
        <StreamViewport
          src={activeMjpegSrc}
          label={
            selectedCamera
              ? `[CH ${String(selectedCamera.channel || 0).padStart(2, "0")}] ${selectedCamera.name}`
              : undefined
          }
          streamKey={streamKey}
          fps={streamFps}
          onError={() => setStreamError(true)}
          onLoad={() => setStreamError(false)}
        />

        {/* Tactical In-Viewport Controls (Top & Bottom subtle overlays) */}
        <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 z-20">
          {/* Frame Size Toggle (30% Compact vs 100% Full Width) */}
          {!isFullscreen && (
            <button
              type="button"
              onClick={() => setFrameSize((s) => (s === "compact" ? "full" : "compact"))}
              className="px-2 py-1 rounded-md bg-black/60 hover:bg-black/90 text-white/80 hover:text-white border border-white/10 text-[10px] font-mono font-bold transition-colors cursor-pointer select-none"
              title={
                frameSize === "compact"
                  ? "Currently 70% Compact (Click for 100% Full Width)"
                  : "Currently 100% Full Width (Click for 70% Compact)"
              }
            >
              {frameSize === "compact" ? "70% FIT" : "100% WIDE"}
            </button>
          )}

          {isStreamActive && (
            <button
              type="button"
              onClick={handleStopStream}
              className="p-1.5 rounded-md bg-black/60 hover:bg-rose-950/80 text-white/70 hover:text-rose-300 border border-white/10 transition-colors cursor-pointer"
              title="Stop Stream"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            type="button"
            onClick={toggleFullscreen}
            className="p-1.5 rounded-md bg-black/60 hover:bg-black/90 text-white/70 hover:text-white border border-white/10 transition-colors cursor-pointer"
            title={isFullscreen ? "Exit Fullscreen (F)" : "Fullscreen (F)"}
          >
            {isFullscreen ? (
              <Minimize2 className="w-3.5 h-3.5" />
            ) : (
              <Maximize2 className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* ═══ 3. INDUSTRIAL COMMAND & INFERENCE DECK ═══ */}
      <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-2xs space-y-3">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 flex-wrap">
          {/* Mode Switcher: RAW vs AI INFERENCE */}
          <div className="flex items-center gap-2 flex-wrap">
            <div className="inline-flex items-center p-0.5 rounded-lg bg-slate-100 border border-slate-200">
              <button
                type="button"
                onClick={() => handleSwitchAiMode("raw")}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors cursor-pointer ${
                  aiMode === "raw"
                    ? "bg-slate-900 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title="Direct RTSP stream without AI inference overlay (Max FPS)"
              >
                Raw Stream
              </button>
              <button
                type="button"
                onClick={() => handleSwitchAiMode("inference")}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors cursor-pointer flex items-center gap-1.5 ${
                  aiMode === "inference"
                    ? "bg-emerald-600 text-white shadow-2xs"
                    : "text-slate-600 hover:text-slate-900"
                }`}
                title="AI Detection Inference active on stream (YOLO & Biometrics)"
              >
                <Sparkles className="w-3 h-3" />
                AI Inference
              </button>
            </div>

            {/* AI Sub-options when inference mode is selected */}
            {aiMode === "inference" && (
              <div className="flex items-center gap-2 pl-1 border-l border-slate-200">
                <label className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium text-slate-700 hover:bg-slate-50 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={enableFaceBiometrics}
                    onChange={(e) => handleToggleFaceBiometrics(e.target.checked)}
                    className="w-3.5 h-3.5 rounded text-emerald-600 border-slate-300 focus:ring-0 cursor-pointer"
                  />
                  <ScanFace className="w-3.5 h-3.5 text-slate-500" />
                  <span>Biometric Faces</span>
                </label>
              </div>
            )}
          </div>

          {/* Real-time Output FPS Monitoring */}
          <div
            className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50/90 shadow-2xs"
            title="Real-time output stream framerate and pipeline latency"
          >
            <div className="flex items-center gap-1.5">
              <span
                className={`w-2 h-2 rounded-full transition-colors ${
                  !isStreamActive
                    ? "bg-slate-300"
                    : streamFps === null
                    ? "bg-amber-400 animate-pulse"
                    : streamFps >= 18
                    ? "bg-emerald-500 animate-pulse"
                    : streamFps >= 10
                    ? "bg-blue-500 animate-pulse"
                    : "bg-rose-500 animate-pulse"
                }`}
              />
              <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1">
                <Activity className="w-3 h-3 text-slate-500" />
                OUTPUT
              </span>
            </div>

            <div className="flex items-baseline gap-1 font-mono">
              <span
                className={`text-sm font-bold tabular-nums ${
                  !isStreamActive
                    ? "text-slate-400"
                    : streamFps === null
                    ? "text-amber-600"
                    : streamFps >= 18
                    ? "text-emerald-700"
                    : streamFps >= 10
                    ? "text-blue-700"
                    : "text-rose-600"
                }`}
              >
                {!isStreamActive
                  ? "--"
                  : streamFps === null
                  ? "..."
                  : streamFps.toFixed(1)}
              </span>
              <span className="text-[10px] font-semibold text-slate-400">
                FPS
              </span>
            </div>

            {isStreamActive && streamLatency !== null && streamLatency > 0 && (
              <span
                className="text-[10px] font-mono text-slate-500 border-l border-slate-200 pl-2 tabular-nums"
                title="Pipeline frame processing latency"
              >
                {Math.round(streamLatency)}ms
              </span>
            )}

            {isStreamActive && (
              <span
                className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase tracking-tight ${
                  streamFps === null
                    ? "bg-amber-100 text-amber-800"
                    : streamFps >= 18
                    ? "bg-emerald-100 text-emerald-800"
                    : streamFps >= 10
                    ? "bg-blue-100 text-blue-800"
                    : "bg-rose-100 text-rose-800"
                }`}
              >
                {streamFps === null
                  ? "SYNC"
                  : streamFps >= 18
                  ? "REALTIME"
                  : streamFps >= 10
                  ? "STABLE"
                  : "LOW"}
              </span>
            )}
          </div>

          {/* Quick Actions (Restart, Snapshot, Manual RTSP) */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={handleRestartStream}
              disabled={!rtspUrl}
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 disabled:opacity-40 transition-colors cursor-pointer flex items-center gap-1.5"
              title="Restart stream pipeline (Hotkey: R)"
            >
              <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
              <span>Reconnect (R)</span>
            </button>

            <button
              type="button"
              onClick={handleCaptureSnapshot}
              disabled={!rtspUrl || isCapturing}
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 disabled:opacity-40 transition-colors cursor-pointer flex items-center gap-1.5"
              title="Download full-res frame snapshot (Hotkey: S)"
            >
              <Camera className="w-3.5 h-3.5 text-slate-500" />
              <span>{isCapturing ? "Saving..." : "Snapshot (S)"}</span>
            </button>

            <button
              type="button"
              onClick={() => setShowManualRtsp((s) => !s)}
              className="px-2.5 py-1.5 rounded-lg text-xs font-medium border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 transition-colors cursor-pointer flex items-center gap-1"
              title="Manual RTSP URL Override"
            >
              <Wifi className="w-3.5 h-3.5 text-slate-500" />
              <span>URL</span>
              {showManualRtsp ? (
                <ChevronUp className="w-3 h-3 text-slate-400" />
              ) : (
                <ChevronDown className="w-3 h-3 text-slate-400" />
              )}
            </button>
          </div>
        </div>

        {/* Collapsible Manual RTSP URL input */}
        {showManualRtsp && (
          <div className="pt-2 border-t border-slate-100 flex items-center gap-2">
            <input
              type="text"
              placeholder="rtsp://admin:Admin123@142.67.45.150:554/cam/realmonitor?channel=1&subtype=0"
              value={rtspInputValue}
              onChange={(e) => setRtspInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && rtspInputValue.trim()) {
                  connectToStream(rtspInputValue.trim(), "Custom RTSP");
                }
              }}
              className="flex-1 px-3 py-1.5 text-xs font-mono rounded-lg border border-slate-200 bg-slate-50 text-slate-800 placeholder-slate-400 focus:outline-none focus:bg-white focus:border-emerald-500"
            />
            <button
              type="button"
              onClick={() => {
                if (rtspInputValue.trim()) {
                  connectToStream(rtspInputValue.trim(), "Custom RTSP");
                }
              }}
              disabled={!rtspInputValue.trim()}
              className="px-3 py-1.5 text-xs font-bold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-40 transition-colors cursor-pointer"
            >
              Apply
            </button>
          </div>
        )}

        {/* Status Line */}
        <div className="flex items-center justify-between text-[11px] font-mono text-slate-500 pt-1 border-t border-slate-100">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                isStreamActive ? "bg-emerald-500 animate-pulse" : "bg-slate-300"
              }`}
            />
            <span>
              {statusMessage ||
                (isStreamActive
                  ? "Live Feed Active"
                  : "Idle — Select a camera to stream")}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span>
              Hotkeys: <kbd className="font-semibold text-slate-700">1-0</kbd> Ch{" "}
              • <kbd className="font-semibold text-slate-700">A</kbd> AI •{" "}
              <kbd className="font-semibold text-slate-700">R</kbd> Reset •{" "}
              <kbd className="font-semibold text-slate-700">F</kbd> Max
            </span>
          </div>
        </div>
      </div>

      {/* ═══ 4. 10-CAMERA MATRIX VIEW (When Matrix mode selected) ═══ */}
      {viewMode === "grid" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5 pt-2">
          {sortedCameras.map((cam) => {
            const isSelected = selectedCamera?.id === cam.id;
            const ch = cam.channel ?? 0;
            const hasCoords = cam.latitude !== null && cam.longitude !== null;

            return (
              <div
                key={cam.id}
                onClick={() => handleSelectCamera(cam)}
                className={`p-3 rounded-xl border transition-all cursor-pointer select-none ${
                  isSelected
                    ? "bg-slate-900 border-emerald-500 text-white shadow-sm"
                    : "bg-white border-slate-200 text-slate-800 hover:border-slate-300 hover:bg-slate-50/80"
                }`}
              >
                <div className="flex items-center justify-between gap-1 mb-1.5">
                  <span
                    className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${
                      isSelected
                        ? "bg-emerald-600 text-white"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    CH {String(ch).padStart(2, "0")}
                  </span>
                  <span
                    className={`w-2 h-2 rounded-full ${
                      cam.status === "offline"
                        ? "bg-slate-400"
                        : "bg-emerald-500"
                    }`}
                  />
                </div>

                <div className="font-semibold text-xs truncate mb-1">
                  {cam.name}
                </div>

                <div
                  className={`text-[10px] font-mono truncate flex items-center justify-between ${
                    isSelected ? "text-slate-400" : "text-slate-500"
                  }`}
                >
                  <span>{cam.type || "Optical 4K"}</span>
                  {hasCoords ? (
                    <span className="flex items-center gap-0.5 text-emerald-600 font-medium">
                      <MapPin className="w-2.5 h-2.5" />
                      GPS
                    </span>
                  ) : (
                    <span className="text-slate-400">No GPS</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ═══ 5. ADD CAMERA MODAL ═══ */}
      <AddCameraModal
        isOpen={showAddCamera}
        onClose={() => setShowAddCamera(false)}
        onAdded={() => {
          setStatusMessage("New channel configured successfully.");
        }}
      />
    </div>
  );
}

export default LiveSurveillanceConsole;
