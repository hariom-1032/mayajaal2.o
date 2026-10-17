"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { useBackendStatus } from "@/lib/hooks/useBackendStatus";
import { Detection, VideoFrameResult, VideoMetadata } from "@/types/backend";
import { Camera as CameraType } from "@/types/camera";
import { alertsStore } from "@/lib/alertsStore";
import { formatConfidence, normalizeConfidence } from "@/lib/utils";
import { DetectionCanvas } from "./detectioncanvas";
import { CameraGrid } from "./cameragrid";
import { SuspectTrajectoryModal } from "@/components/map/SuspectTrajectoryModal";
import {
  Play,
  Upload,
  Radio,
  Sliders,
  RefreshCw,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  Camera,
  Video,
  Film,
  WifiOff,
  Wifi,
  Link2,
  Link2Off,
  Maximize2,
  Minimize2,
  AlertTriangle,
  X,
  Globe,
  HelpCircle,
  Compass,
  ShieldAlert,
  ScanFace,
  UploadCloud,
  Trash2,
  Car,
} from "lucide-react";

/** Build the backend MJPEG stream URL (proxied Next.js → FastAPI). */
function buildStreamUrl(
  rtspUrl: string,
  drawDetections = true,
  enableFaceBiometrics = true,
  enableVehicleDetector = false,
  sessionId?: string
): string {
  const params = new URLSearchParams({
    rtsp_url: rtspUrl,
    draw_detections: String(drawDetections),
    enable_face_recognition: String(enableFaceBiometrics),
    enable_vehicle_detection: String(enableVehicleDetector),
    fps: "20",
    _t: String(Date.now()),
  });
  if (sessionId) params.set("session_id", sessionId);
  return `/api/backend/stream/live?${params.toString()}`;
}

export function LiveWorkspace() {
  const { isOnline, loadedModels, refetch } = useBackendStatus();

  // Mode: "upload-image" | "upload-video" | "rtsp"
  const [sourceMode, setSourceMode] = useState<
    "upload-image" | "upload-video" | "rtsp"
  >("upload-image");

  // Selected Model
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [loadingModel, setLoadingModel] = useState(false);

  // Confidence & IOU Sliders
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);

  // Multi-Object Tracking (ByteTrack) toggle
  const [enableTracking, setEnableTracking] = useState<boolean>(true);

  // ─── Live Stream (RTSP & IP Webcam Pro) state ───────────────────────────
  const [rtspUrl, setRtspUrl] = useState<string>("sample");
  const [rtspInputValue, setRtspInputValue] = useState<string>("sample");
  const [streamKey, setStreamKey] = useState<number>(0);
  const [isStreamActive, setIsStreamActive] = useState<boolean>(false);
  const [streamError, setStreamError] = useState<boolean>(false);
  const [isValidatingStream, setIsValidatingStream] = useState<boolean>(false);
  const [streamDiagnostics, setStreamDiagnostics] = useState<string | null>(null);
  const [streamProtocol, setStreamProtocol] = useState<string>("Live Stream");
  const [detectedResolvedUrl, setDetectedResolvedUrl] = useState<string | null>(null);
  const [drawDetectionsOnStream, setDrawDetectionsOnStream] = useState<boolean>(true);
  const [enableFaceBiometrics, setEnableFaceBiometrics] = useState<boolean>(true);
  const [enableVehicleDetector, setEnableVehicleDetector] = useState<boolean>(false);
  const [selectedCamera, setSelectedCamera] = useState<CameraType | null>(null);
  const [activeMjpegSrc, setActiveMjpegSrc] = useState<string | null>(null);

  // Image source state
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string>("");
  const [isDraggingOver, setIsDraggingOver] = useState<boolean>(false);

  // Video source state
  const [selectedVideoFile, setSelectedVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string>(
    "/videos/himalayan-border-animated.mp4"
  );
  const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
  const [videoResults, setVideoResults] = useState<VideoFrameResult[]>([]);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number>(0);
  const [videoMetadata, setVideoMetadata] = useState<VideoMetadata | null>(null);
  // Max frames for video tracking / inference (default 300 = ~10s at 30fps)
  const [videoMaxFrames, setVideoMaxFrames] = useState<number>(300);

  // Resolution dimensions
  const [sourceDims, setSourceDims] = useState<{ width: number; height: number }>({
    width: 1280,
    height: 720,
  });

  // Inference state
  const [isInferencing, setIsInferencing] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const router = useRouter();
  // Biometric Suspect Detection state for uploaded image/video
  const [detectedSuspectsInMedia, setDetectedSuspectsInMedia] = useState<
    Array<{ name: string; threat_level: string; confidence: number; category?: string; frame_index?: number }>
  >([]);
  const [selectedSuspectForTrajectory, setSelectedSuspectForTrajectory] = useState<string | null>(null);

  const imageFileInputRef = useRef<HTMLInputElement>(null);
  const videoFileInputRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamImgRef = useRef<HTMLImageElement>(null);
  // Stable session ID for this component mount — uniquely identifies the live
  // stream connection so the backend can stop inference when we navigate away.
  const sessionIdRef = useRef<string>(
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2)
  );

  // Fullscreen viewport container ref & state
  const viewportContainerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showFullscreenInspector, setShowFullscreenInspector] = useState<boolean>(false);

  // Fullscreen toggle handler (uses native Fullscreen API with state fallback)
  const toggleFullscreen = useCallback(async () => {
    if (!viewportContainerRef.current) return;

    if (!document.fullscreenElement && !isFullscreen) {
      try {
        if (viewportContainerRef.current.requestFullscreen) {
          await viewportContainerRef.current.requestFullscreen();
        }
      } catch (err) {
        console.warn("Native fullscreen request failed, falling back to CSS fullscreen:", err);
      }
      setIsFullscreen(true);
    } else {
      if (document.fullscreenElement && document.exitFullscreen) {
        await document.exitFullscreen().catch(() => {});
      }
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  // Sync fullscreen state with browser events
  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", handleFsChange);
    document.addEventListener("webkitfullscreenchange", handleFsChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFsChange);
      document.removeEventListener("webkitfullscreenchange", handleFsChange);
    };
  }, []);

  // Keyboard shortcut listeners (ESC to exit, F to toggle)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.isContentEditable) {
        return;
      }
      if (e.key === "Escape" && isFullscreen) {
        if (document.fullscreenElement && document.exitFullscreen) {
          document.exitFullscreen().catch(() => {});
        }
        setIsFullscreen(false);
      } else if (e.key === "f" || e.key === "F") {
        e.preventDefault();
        toggleFullscreen();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isFullscreen, toggleFullscreen]);

  // Set default model when models list updates (prioritize 'best')
  useEffect(() => {
    if (loadedModels.length > 0) {
      if (!selectedModel || !loadedModels.includes(selectedModel)) {
        const preferred = loadedModels.includes("best") ? "best" : loadedModels[0];
        setSelectedModel(preferred);
      }
    }
  }, [loadedModels, selectedModel]);

  // Auto-stop stream when leaving RTSP mode
  useEffect(() => {
    if (sourceMode !== "rtsp") {
      setIsStreamActive(false);
      setActiveMjpegSrc(null);
      setStreamError(false);
      setStreamDiagnostics(null);
    }
  }, [sourceMode]);

  // ── Reliable Stream Teardown on Unmount ───────────────────────────────────
  // When the user navigates away from Live Surveillance:
  //  1. Call DELETE /api/backend/stream/live/stop?session_id=X with keepalive:true
  //     so the request completes even during page navigation.  The backend
  //     generator exits within one frame interval (~40 ms), freeing GPU/CPU.
  //  2. Clear img.src as belt-and-suspenders to close the HTTP connection.
  useEffect(() => {
    const sid = sessionIdRef.current;
    return () => {
      fetch(
        `/api/backend/stream/live/stop?session_id=${encodeURIComponent(sid)}`,
        { method: "DELETE", keepalive: true }
      ).catch(() => {});
      if (streamImgRef.current) {
        streamImgRef.current.src = "";
      }
    };
  }, []);

  // Rebuild stream URL when AI overlay, Face Biometrics, or Vehicle Detector toggle changes while stream is live
  useEffect(() => {
    if (isStreamActive && rtspUrl) {
      const src = buildStreamUrl(
        rtspUrl,
        drawDetectionsOnStream,
        enableFaceBiometrics,
        enableVehicleDetector,
        sessionIdRef.current
      );
      setActiveMjpegSrc(src);
      setStreamKey((k) => k + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawDetectionsOnStream, enableFaceBiometrics, enableVehicleDetector]);

  // ── Real-Time Target Inspector Sync for Live Stream ──────────────────────────
  useEffect(() => {
    if (sourceMode !== "rtsp" || !isStreamActive || streamError) {
      return;
    }

    let isMounted = true;
    const sid = sessionIdRef.current;

    const pollLiveDetections = async () => {
      try {
        const res = await api.getLiveStreamDetections(sid);
        if (!isMounted) return;
        if (res && res.status === "active" && Array.isArray(res.detections)) {
          setDetections(res.detections);
          if (res.latency_ms) setLatencyMs(Math.round(res.latency_ms));
          if (res.suspects && res.suspects.length > 0) {
            setDetectedSuspectsInMedia(res.suspects);
          }
        }
      } catch (err) {
        console.debug("Live stream detection sync poll:", err);
      }
    };

    const interval = setInterval(pollLiveDetections, 350);
    pollLiveDetections();

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [sourceMode, isStreamActive, streamError]);

  // Camera grid click → populate RTSP URL + auto-connect
  const handleCameraSelect = useCallback(
    (camera: CameraType) => {
      const url = camera.streamUrl || "";
      setSelectedCamera(camera);
      setRtspUrl(url);
      setRtspInputValue(url);
      setSourceMode("rtsp");
      if (url) {
        const src = buildStreamUrl(
          url,
          drawDetectionsOnStream,
          enableFaceBiometrics,
          enableVehicleDetector,
          sessionIdRef.current
        );
        setActiveMjpegSrc(src);
        setIsStreamActive(true);
        setStreamError(false);
        setStreamDiagnostics(`Connected to registered node: ${camera.name}`);
        setStreamProtocol("RTSP Camera");
        setStatusMessage(`Connecting to: ${camera.name} — ${camera.sector}`);
        setStreamKey((k) => k + 1);
      }
    },
    [drawDetectionsOnStream, enableFaceBiometrics, enableVehicleDetector]
  );

  const handleConnectStream = useCallback(async (customUrl?: string) => {
    const rawUrl = (customUrl !== undefined ? customUrl : rtspInputValue).trim();
    if (!rawUrl) {
      setStatusMessage("Please enter a valid RTSP, IP Webcam (http://ip:8080), or 'sample' URL.");
      return;
    }

    setIsValidatingStream(true);
    setStreamError(false);
    setStreamDiagnostics(null);
    setStatusMessage(`Testing stream connection for: ${rawUrl}...`);

    try {
      // Pre-flight check via backend validator
      const validation = await api.validateStream(rawUrl);
      setStreamProtocol(validation.protocol || "Live Stream");
      setDetectedResolvedUrl(validation.resolved_url || rawUrl);

      if (validation.reachable) {
        setRtspUrl(rawUrl);
        setRtspInputValue(rawUrl);
        setStreamError(false);
        const src = buildStreamUrl(
          rawUrl,
          drawDetectionsOnStream,
          enableFaceBiometrics,
          enableVehicleDetector,
          sessionIdRef.current
        );
        setActiveMjpegSrc(src);
        setIsStreamActive(true);
        setStreamKey((k) => k + 1);
        setStatusMessage(validation.message || `Connected to ${validation.protocol}`);
        setStreamDiagnostics(validation.message);
      } else {
        // Unreachable host/port
        setRtspUrl(rawUrl);
        setRtspInputValue(rawUrl);
        setStreamError(true);
        setStatusMessage(validation.message);
        setStreamDiagnostics(validation.message);
        // Start standby stream so tactical overlay with diagnostic text is displayed
        const src = buildStreamUrl(
          rawUrl,
          drawDetectionsOnStream,
          enableFaceBiometrics,
          enableVehicleDetector,
          sessionIdRef.current
        );
        setActiveMjpegSrc(src);
        setIsStreamActive(true);
        setStreamKey((k) => k + 1);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Validation failed";
      setRtspUrl(rawUrl);
      setRtspInputValue(rawUrl);
      const src = buildStreamUrl(
        rawUrl,
        drawDetectionsOnStream,
        enableFaceBiometrics,
        enableVehicleDetector,
        sessionIdRef.current
      );
      setActiveMjpegSrc(src);
      setIsStreamActive(true);
      setStreamKey((k) => k + 1);
      setStatusMessage(`Stream initialising: ${rawUrl}`);
      setStreamDiagnostics(`Connection initiated. (${msg})`);
    } finally {
      setIsValidatingStream(false);
    }
  }, [rtspInputValue, drawDetectionsOnStream, enableFaceBiometrics, enableVehicleDetector]);

  const handleDisconnectStream = useCallback(() => {
    setIsStreamActive(false);
    setActiveMjpegSrc(null);
    setStreamError(false);
    setStreamDiagnostics(null);
    setDetections([]);
    setDetectedSuspectsInMedia([]);
    setStatusMessage("Stream disconnected.");
    const sid = sessionIdRef.current;
    fetch(`/api/backend/stream/live/stop?session_id=${encodeURIComponent(sid)}`, {
      method: "DELETE",
    }).catch(() => {});
  }, []);

  // Load default model helper if no models loaded
  const handleLoadDefaultModel = async () => {
    setLoadingModel(true);
    setStatusMessage("Loading best.onnx into engine...");
    try {
      await api.loadModel("best", "best.onnx");
      await refetch();
      setSelectedModel("best");
      setStatusMessage("Model 'best' loaded successfully.");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load model";
      setStatusMessage(`Error: ${msg}`);
    } finally {
      setLoadingModel(false);
    }
  };

  // Handle local image selection
  const handleImageFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedImageFile(file);
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
    setSourceMode("upload-image");
    setDetections([]);
    setDetectedSuspectsInMedia([]);
    setLatencyMs(null);
    setStatusMessage(`Loaded image: ${file.name}`);
  };

  // Handle local video selection
  const handleVideoFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedVideoFile(file);
    const url = URL.createObjectURL(file);
    setVideoPreviewUrl(url);
    setAnnotatedVideoUrl(null);
    setSourceMode("upload-video");
    setDetections([]);
    setVideoResults([]);
    setDetectedSuspectsInMedia([]);
    setLatencyMs(null);
    setStatusMessage(
      `Loaded video: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`
    );
  };

  // Measure natural dimensions when preview image loads
  const handleImageLoaded = () => {
    if (imageRef.current) {
      setSourceDims({
        width: imageRef.current.naturalWidth || 1280,
        height: imageRef.current.naturalHeight || 720,
      });
    }
  };

  // Measure natural dimensions when preview video loads
  const handleVideoLoaded = () => {
    if (videoRef.current) {
      setSourceDims({
        width: videoRef.current.videoWidth || 1280,
        height: videoRef.current.videoHeight || 720,
      });
      const dur = videoRef.current.duration;
      if (dur && !isNaN(dur) && isFinite(dur) && dur > 0) {
        // Automatically set frame limit to cover the video duration (up to 1800 frames / 60s)
        const estFrames = Math.min(1800, Math.max(30, Math.round(dur * 30)));
        setVideoMaxFrames(estFrames);
      }
    }
  };

  // Switch active video frame inspector
  const handleSelectFrame = (frameIndex: number) => {
    setSelectedFrameIndex(frameIndex);
    const target = videoResults.find((fr) => fr.frame_index === frameIndex);
    if (target) {
      const dets = target.tracked_objects || target.inference?.detections || target.detections || [];
      setDetections(dets);
    }

    if (videoRef.current && videoMetadata?.fps && videoMetadata.fps > 0) {
      videoRef.current.currentTime = frameIndex / videoMetadata.fps;
    }
  };

  // Synchronize target bounding boxes and inspector in real-time as video plays
  const handleVideoTimeUpdate = () => {
    if (sourceMode !== "upload-video" || videoResults.length === 0 || !videoRef.current) return;
    const currentTime = videoRef.current.currentTime;
    const fps = videoMetadata?.fps && videoMetadata.fps > 0 ? videoMetadata.fps : 30;
    const currentFrameIdx = Math.min(
      videoResults.length - 1,
      Math.max(0, Math.floor(currentTime * fps))
    );
    if (currentFrameIdx !== selectedFrameIndex) {
      setSelectedFrameIndex(currentFrameIdx);
      const target = videoResults.find((fr) => fr.frame_index === currentFrameIdx);
      if (target) {
        const dets = target.tracked_objects || target.inference?.detections || target.detections || [];
        setDetections(dets);
      }
    }
  };

  // Video stream error telemetry
  const handleVideoError = () => {
    if (sourceMode === "upload-video") {
      setStatusMessage(
        "Server video stream playback error. Ensure backend video server is reachable and FFmpeg H.264 is enabled."
      );
    }
  };

  // Trigger AI Inference (Image, Video, or RTSP)
  const handleRunInference = async () => {
    if (!isOnline) {
      setStatusMessage("Cannot run inference: Backend is offline.");
      return;
    }

    const modelToUse = selectedModel || loadedModels[0];
    if (!modelToUse) {
      setStatusMessage("No model loaded. Please load or select an ONNX model.");
      return;
    }

    setIsInferencing(true);
    const t0 = performance.now();

    try {
      if (sourceMode === "upload-image") {
        let fileToSubmit = selectedImageFile;
        if (!fileToSubmit) {
          if (!imagePreviewUrl) {
            setStatusMessage("Please upload an image first.");
            imageFileInputRef.current?.click();
            setIsInferencing(false);
            return;
          }
          const res = await fetch(imagePreviewUrl);
          const blob = await res.blob();
          fileToSubmit = new File([blob], "surveillance_sample.jpg", {
            type: "image/jpeg",
          });
        }
        setStatusMessage("Executing dual-pipeline: ONNX threats + Biometric Face Scan...");

        const [result, faceScanResult] = await Promise.all([
          api.runImageInference({
            file: fileToSubmit,
            modelName: modelToUse,
            confThreshold,
            iouThreshold,
            postprocess: true,
          }),
          api.scanFaceImage(fileToSubmit, 0.40, 35).catch((err) => {
            console.debug("Facial scan fallback:", err);
            return null;
          }),
        ]);

        const elapsed = Math.round(performance.now() - t0);
        setLatencyMs(
          result.inference_time_ms
            ? Math.round(result.inference_time_ms)
            : elapsed
        );

        // Convert detected faces into Detection objects
        const faceDetections: Detection[] = [];
        const suspectsFound: Array<{ name: string; threat_level: string; confidence: number; category?: string }> = [];

        if (faceScanResult && faceScanResult.faces && faceScanResult.faces.length > 0) {
          for (const face of faceScanResult.faces) {
            const isThreat = Boolean(face.is_threat);
            const isKnown = Boolean(face.is_known);
            const conf = normalizeConfidence(
              face.calibrated_conf || face.match_score || face.confidence || 0.85
            );

            if (isThreat || isKnown) {
              const faceDet: Detection = {
                box: face.bbox,
                confidence: conf,
                class_id: 999,
                class_name: isThreat ? `suspect_${face.name.toLowerCase()}` : `face_${face.name.toLowerCase()}`,
                is_threat: isThreat,
                threat_level: face.threat_level || (isThreat ? "HIGH" : undefined),
                suspect_name: face.name,
                category: face.category || (isThreat ? "Suspect Target" : "Personnel"),
              };
              faceDetections.push(faceDet);

              if (isThreat) {
                const tLevel = face.threat_level || "HIGH";
                const sev: "High" | "Medium" | "Low" = "High";
                suspectsFound.push({
                  name: face.name,
                  threat_level: tLevel,
                  confidence: conf,
                  category: face.category,
                });

                // Persist alert to backend so it triggers WebSocket broadcast and Global Threat HUD Toast chime
                api.createAlert({
                  title: `SUSPECT DETECTED: ${face.name.toUpperCase()}`,
                  location: "Live Surveillance (Sector Image Scan)",
                  severity: sev,
                  cameraName: "Live Image Workspace",
                  className: "suspect",
                  confidence: conf,
                  box: face.bbox,
                  suspectName: face.name,
                  threatLevel: tLevel,
                  category: face.category,
                  notes: `Biometric facial match verified via YuNet+SFace (${formatConfidence(conf)} match).`,
                }).catch((err) => {
                  console.debug("Failed to dispatch alert to backend, adding locally:", err);
                  if (!alertsStore.isConnected()) {
                    alertsStore.addAlert({
                      title: `CRITICAL SUSPECT: ${face.name.toUpperCase()}`,
                      location: "Live Surveillance (Sector Image Scan)",
                      severity: sev,
                      className: "suspect",
                      confidence: conf,
                      cameraName: "Live Image Workspace",
                      box: face.bbox,
                      suspectName: face.name,
                      threatLevel: tLevel,
                      category: face.category,
                    });
                  }
                });
              }
            }
          }
        }

        const combinedDetections = [...(result.detections || []), ...faceDetections];
        setDetections(combinedDetections);
        setDetectedSuspectsInMedia(suspectsFound);

        const suspectNotice = suspectsFound.length > 0
          ? ` • 🚨 ${suspectsFound.length} SUSPECT(S) IDENTIFIED: ${suspectsFound.map((s) => s.name).join(", ")}`
          : "";
        setStatusMessage(
          `Detected ${result.detections?.length || 0} objects, ${faceDetections.length} faces in ${elapsed}ms${suspectNotice}.`
        );

        // Register high threat alerts
        if (result.detections && result.detections.length > 0) {
          const highThreatClasses = [
            "firearm",
            "explosive",
            "melee_weapon",
            "person",
            "blunt_weapon",
          ];
          result.detections.forEach((det) => {
            if (det.confidence >= 0.4) {
              const isHigh = highThreatClasses.includes(
                det.class_name.toLowerCase()
              );
              alertsStore.addAlert({
                title: `${det.class_name
                  .replace(/_/g, " ")
                  .toUpperCase()} Identified in Sector`,
                location: "Live Surveillance Feed (Cam 01)",
                severity: isHigh ? "High" : "Medium",
                className: det.class_name,
                confidence: det.confidence,
                cameraName: "Live Stream Cam 01",
                box: det.box,
              });
            }
          });
        }
      } else if (sourceMode === "upload-video") {
        let fileToSubmit = selectedVideoFile;
        if (!fileToSubmit) {
          const res = await fetch(videoPreviewUrl);
          const blob = await res.blob();
          fileToSubmit = new File([blob], "surveillance_sample.mp4", {
            type: "video/mp4",
          });
        }

        let rawResults: VideoFrameResult[] = [];
        let videoMeta: VideoMetadata | undefined;
        let framesProcessed = 0;
        let suspectsDetected: Array<{ name: string; threat_level: string; frame_index: number; confidence: number; category?: string }> = [];

        if (enableTracking) {
          setStatusMessage(
            `Tracking objects across video sequence with ByteTrack (${videoMaxFrames} frames requested)...`
          );
          const result = await api.runVideoTracking({
            file: fileToSubmit,
            modelName: modelToUse,
            confThreshold,
            iouThreshold,
            maxFrames: videoMaxFrames,
          });
          videoMeta = result.video;
          framesProcessed = result.frames_processed ?? result.results.length;
          suspectsDetected = result.suspects_detected || [];
          rawResults = (result.results || []).map((fr) => ({
            ...fr,
            detections: fr.tracked_objects || fr.detections || [],
          }));

          // Server-Side Bounding Box Integration
          if (result.annotated_video_url) {
            const backendBase = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
            const fullAnnUrl = `${backendBase}${result.annotated_video_url}?_t=${Date.now()}`;
            setAnnotatedVideoUrl(fullAnnUrl);
            setVideoPreviewUrl(fullAnnUrl);
          }
        } else {
          setStatusMessage(
            `Sending video to backend endpoint /api/inference/video (${videoMaxFrames} frames requested)...`
          );
          const result = await api.runVideoInference({
            file: fileToSubmit,
            modelName: modelToUse,
            confThreshold,
            iouThreshold,
            maxFrames: videoMaxFrames,
            postprocess: true,
          });
          videoMeta = result.video;
          framesProcessed =
            result.frames_processed ??
            result.total_frames_processed ??
            (result.results || []).length;
          suspectsDetected = result.suspects_detected || [];
          rawResults = (result.results || result.frame_results || []).map((fr) => ({
            ...fr,
            detections: fr.inference?.detections || fr.detections || [],
          }));
        }

        const elapsed = Math.round(performance.now() - t0);
        setLatencyMs(elapsed);
        setVideoResults(rawResults);

        if (videoMeta) {
          setVideoMetadata(videoMeta);
          if (videoMeta.width && videoMeta.height) {
            setSourceDims({
              width: videoMeta.width,
              height: videoMeta.height,
            });
          }
        }

        // Collect all detections / tracked objects across all frames
        const allDetections: Detection[] = [];
        const uniqueTrackIds = new Set<number>();
        rawResults.forEach((fr) => {
          const dets = fr.tracked_objects || fr.inference?.detections || fr.detections || [];
          dets.forEach((d) => {
            allDetections.push(d);
            if (d.track_id !== undefined) {
              uniqueTrackIds.add(d.track_id);
            }
          });
        });

        // Set active frame to first frame with detections
        const firstWithDets =
          rawResults.find(
            (fr) => (fr.tracked_objects || fr.inference?.detections || fr.detections || []).length > 0
          ) || rawResults[0];

        const activeDets = firstWithDets
          ? firstWithDets.tracked_objects || firstWithDets.inference?.detections || firstWithDets.detections || []
          : [];

        setSelectedFrameIndex(firstWithDets?.frame_index ?? 0);
        setDetections(activeDets);

        // Check for suspects detected in video
        if (suspectsDetected.length > 0) {
          setDetectedSuspectsInMedia(suspectsDetected);
        }

        const trackSummary =
          uniqueTrackIds.size > 0
            ? ` • ${uniqueTrackIds.size} unique targets tracked (ByteTrack MOT)`
            : "";

        const suspectSummary =
          suspectsDetected.length > 0
            ? ` • 🚨 ${suspectsDetected.length} SUSPECT(S) IDENTIFIED: ${suspectsDetected.map((s) => s.name).join(", ")}`
            : "";

        const durationSecs =
          videoMeta?.fps && videoMeta.fps > 0
            ? (framesProcessed / videoMeta.fps).toFixed(1)
            : (framesProcessed / 30).toFixed(1);

        setStatusMessage(
          `Processed ${framesProcessed} video frames (~${durationSecs}s) in ${elapsed}ms. Found ${allDetections.length} targets${trackSummary}${suspectSummary}.`
        );

        // Register alerts in store
        const highThreatClasses = [
          "firearm",
          "explosive",
          "melee_weapon",
          "person",
          "blunt_weapon",
          "car",
        ];
        allDetections.forEach((det) => {
          if (det.confidence >= 0.45) {
            const isHigh = highThreatClasses.includes(
              det.class_name.toLowerCase()
            );
            const trackPrefix = det.track_id !== undefined ? `[Track #${det.track_id}] ` : "";
            alertsStore.addAlert({
              title: `${trackPrefix}${det.class_name
                .replace(/_/g, " ")
                .toUpperCase()} Identified in Video Stream`,
              location: `Video Stream (${fileToSubmit.name})`,
              severity: isHigh ? "High" : "Medium",
              className: det.class_name,
              confidence: det.confidence,
              cameraName: "Video Upload Stream",
              box: det.box,
            });
          }
        });
      } else if (sourceMode === "rtsp") {
        if (enableTracking) {
          setStatusMessage("Connecting to RTSP stream and running ByteTrack tracking...");
          const result = await api.runRtspTracking({
            rtspUrl,
            cameraId: selectedCamera?.id || "live_workspace_rtsp",
            modelName: modelToUse,
            confThreshold,
            iouThreshold,
            maxFrames: 5,
          });

          const elapsed = Math.round(performance.now() - t0);
          setLatencyMs(elapsed);
          const lastFrame = result.results?.[result.results.length - 1];
          const trackedDets = lastFrame?.tracked_objects || lastFrame?.detections || [];
          setDetections(trackedDets);
          setStatusMessage(
            `ByteTrack tracked ${trackedDets.length} targets across ${result.frames_processed ?? 5} RTSP frames (Session: ${result.camera_id}).`
          );
        } else {
          setStatusMessage("Connecting to RTSP stream and sampling frames...");
          const result = await api.runRtspInference({
            rtspUrl,
            modelName: modelToUse,
            confThreshold,
            iouThreshold,
            maxFrames: 5,
          });

          const elapsed = Math.round(performance.now() - t0);
          setLatencyMs(elapsed);
          const lastFrame = result.frame_results?.[result.frame_results.length - 1];
          setDetections(lastFrame?.detections || []);
          setStatusMessage(
            `Sampled ${result.frames_sampled} frames from RTSP stream.`
          );
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Inference failed";
      setStatusMessage(`Inference error: ${msg}`);
    } finally {
      setIsInferencing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Hidden File Inputs */}
      <input
        ref={imageFileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageFileChange}
        className="hidden"
      />
      <input
        ref={videoFileInputRef}
        type="file"
        accept="video/*,.mp4,.avi,.mov,.mkv"
        onChange={handleVideoFileChange}
        className="hidden"
      />

      {/* 1. Tactical Controls Bar */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs p-4 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
        {/* Source Mode Tabs */}
        <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg">
          <button
            type="button"
            onClick={() => setSourceMode("upload-image")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              sourceMode === "upload-image"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            Surveillance Image
          </button>

          <button
            type="button"
            onClick={() => {
              setSourceMode("upload-video");
              if (!selectedVideoFile) {
                videoFileInputRef.current?.click();
              }
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              sourceMode === "upload-video"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Video className="w-3.5 h-3.5" />
            Upload Video
          </button>

          <button
            type="button"
            onClick={() => setSourceMode("rtsp")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              sourceMode === "rtsp"
                ? "bg-white text-slate-900 shadow-xs"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Radio className="w-3.5 h-3.5" />
            Live Surveillance (Webcam / RTSP / IP Feed)
          </button>
        </div>

        {/* Model Selector, Thresholds, and Action Buttons */}
        <div className="flex flex-wrap items-center gap-3 w-full lg:w-auto">
          {/* Model selection */}
          <div className="flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-slate-500" />
            {loadedModels.length > 0 ? (
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="text-xs bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 font-medium text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-600"
              >
                {loadedModels.map((m) => (
                  <option key={m} value={m}>
                    Model: {m}
                  </option>
                ))}
              </select>
            ) : (
              <button
                type="button"
                onClick={handleLoadDefaultModel}
                disabled={loadingModel || !isOnline}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-300 hover:bg-emerald-100 transition-colors disabled:opacity-50"
              >
                <Cpu className="w-3.5 h-3.5" />
                {loadingModel ? "Loading Model..." : "Load Detection Model"}
              </button>
            )}
          </div>

          {/* Confidence Slider */}
          <div className="flex items-center gap-2 text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5">
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <span>Conf: {Math.round(confThreshold * 100)}%</span>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              className="w-16 accent-emerald-600 cursor-pointer"
            />
          </div>

          {/* Video Frames Limit Control */}
          {sourceMode === "upload-video" && (
            <div className="flex items-center gap-1.5 text-xs text-slate-700 bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5">
              <Film className="w-3.5 h-3.5 text-slate-500" />
              <span className="font-medium whitespace-nowrap">Frames:</span>
              <select
                value={videoMaxFrames}
                onChange={(e) => setVideoMaxFrames(parseInt(e.target.value, 10))}
                className="text-xs bg-white border border-slate-300 rounded px-1.5 py-0.5 font-mono text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-600 cursor-pointer"
                title="Number of video frames to annotate with ByteTrack"
              >
                <option value={30}>30 frames (~1s preview)</option>
                <option value={90}>90 frames (~3s)</option>
                <option value={150}>150 frames (~5s)</option>
                <option value={300}>300 frames (~10s full)</option>
                <option value={450}>450 frames (~15s)</option>
                <option value={600}>600 frames (~20s)</option>
                <option value={900}>900 frames (~30s)</option>
                <option value={1800}>1800 frames (~60s)</option>
                {![30, 90, 150, 300, 450, 600, 900, 1800].includes(videoMaxFrames) && (
                  <option value={videoMaxFrames}>
                    {videoMaxFrames} frames ({((videoMaxFrames) / 30).toFixed(1)}s auto)
                  </option>
                )}
              </select>
            </div>
          )}

          {/* Upload Button based on mode */}
          {sourceMode === "upload-image" ? (
            <button
              type="button"
              onClick={() => imageFileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-[#eaf4ed] hover:bg-[#dceee1] text-[#1b5032] border border-[#bcdbc5] transition-colors cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" />
              {selectedImageFile
                ? `Image: ${selectedImageFile.name.slice(0, 16)}...`
                : imagePreviewUrl
                ? "Replace Image"
                : "Upload Image"}
            </button>
          ) : sourceMode === "upload-video" ? (
            <button
              type="button"
              onClick={() => videoFileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 transition-colors"
            >
              <Upload className="w-3.5 h-3.5" />
              {selectedVideoFile
                ? `Video: ${selectedVideoFile.name.slice(0, 14)}...`
                : "Upload Video"}
            </button>
          ) : null}

          {/* ByteTrack MOT Toggle (for Video & RTSP) */}
          {(sourceMode === "upload-video" || sourceMode === "rtsp") && (
            <button
              type="button"
              onClick={() => setEnableTracking((prev) => !prev)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                enableTracking
                  ? "bg-emerald-700 hover:bg-emerald-600 text-white border-emerald-500 shadow-xs"
                  : "bg-slate-100 hover:bg-slate-200 text-slate-600 border-slate-300"
              }`}
              title="ByteTrack Multi-Object Tracking (persistent Track IDs across frames)"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>ByteTrack: {enableTracking ? "ON" : "OFF"}</span>
            </button>
          )}

          {/* Run Inference CTA */}
          <button
            type="button"
            onClick={handleRunInference}
            disabled={isInferencing || !isOnline}
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-lg text-xs font-semibold bg-[#1e4b38] hover:bg-[#163a2b] text-white shadow-xs transition-all duration-150 disabled:opacity-50"
          >
            {isInferencing ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                {enableTracking && sourceMode !== "upload-image" ? "Tracking..." : "Inferencing..."}
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                {enableTracking && sourceMode !== "upload-image" ? "Run ByteTrack Tracking" : "Run AI Detection"}
              </>
            )}
          </button>
        </div>
      </div>

      {/* Live Stream URL Controls (RTSP & IP Webcam Pro) */}
      {sourceMode === "rtsp" && (
        <div className="space-y-2">
          <div className="bg-white text-slate-900 p-3.5 rounded-2xl flex flex-col gap-2.5 shadow-xs border border-slate-200/90">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
              {/* Status dot / Protocol Icon */}
              <div className="flex items-center gap-2 flex-shrink-0">
                {isValidatingStream ? (
                  <span className="flex items-center gap-1.5 text-amber-600">
                    <RefreshCw className="w-4 h-4 animate-spin text-amber-600" />
                    <span className="text-xs font-mono font-bold text-amber-700">Probing Link:</span>
                  </span>
                ) : isStreamActive && !streamError ? (
                  <span className="flex items-center gap-1.5">
                    <span className="flex h-2 w-2 relative">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                    </span>
                    <Wifi className="w-4 h-4 text-emerald-700" />
                    <span className="text-xs font-mono font-bold text-emerald-800 whitespace-nowrap">
                      {streamProtocol}:
                    </span>
                  </span>
                ) : streamError ? (
                  <span className="flex items-center gap-1.5">
                    <AlertTriangle className="w-4 h-4 text-amber-600" />
                    <span className="text-xs font-mono font-bold text-amber-700 whitespace-nowrap">
                      Feed Alert:
                    </span>
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5">
                    <WifiOff className="w-4 h-4 text-slate-400" />
                    <span className="text-xs font-mono font-bold text-slate-600 whitespace-nowrap">
                      Live Feed:
                    </span>
                  </span>
                )}
              </div>

              {/* URL input with Clear button */}
              <div className="relative flex-1 w-full">
                <input
                  id="rtsp-url-input"
                  type="text"
                  value={rtspInputValue}
                  onChange={(e) => {
                    setRtspInputValue(e.target.value);
                    if (streamError) setStreamError(false);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleConnectStream();
                    }
                  }}
                  placeholder="e.g. 0 (Local Webcam), http://192.168.1.5:8080 (IP Webcam), rtsp://..., or 'sample'"
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 pr-8 text-xs font-mono text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700 shadow-2xs"
                />
                {rtspInputValue && (
                  <button
                    type="button"
                    onClick={() => {
                      setRtspInputValue("");
                    }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-700 p-0.5 cursor-pointer"
                    title="Clear input"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>

              {/* Connect / Disconnect Action Buttons */}
              <div className="flex items-center gap-2 flex-shrink-0">
                {isStreamActive ? (
                  <button
                    type="button"
                    onClick={handleDisconnectStream}
                    className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-700 text-white transition-colors cursor-pointer shadow-xs"
                  >
                    <Link2Off className="w-3.5 h-3.5" />
                    Disconnect
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleConnectStream()}
                    disabled={!rtspInputValue.trim() || isValidatingStream}
                    className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white transition-colors disabled:opacity-40 cursor-pointer shadow-xs"
                  >
                    {isValidatingStream ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        Connecting...
                      </>
                    ) : (
                      <>
                        <Link2 className="w-3.5 h-3.5" />
                        Connect Feed
                      </>
                    )}
                  </button>
                )}

                {/* AI overlay toggle */}
                {loadedModels.length > 0 && (
                  <label className="flex items-center gap-1.5 cursor-pointer select-none text-xs font-semibold text-slate-700 ml-1 border-l border-slate-200 pl-2.5">
                    <input
                      type="checkbox"
                      checked={drawDetectionsOnStream}
                      onChange={(e) => setDrawDetectionsOnStream(e.target.checked)}
                      className="accent-emerald-600 w-3.5 h-3.5 cursor-pointer"
                    />
                    AI Overlay
                  </label>
                )}

                {/* Face biometrics overlay toggle */}
                <label className="flex items-center gap-1.5 cursor-pointer select-none text-xs font-semibold text-slate-700 ml-1 border-l border-slate-200 pl-2.5" title="Overlay real-time facial recognition & suspect identification">
                  <input
                    type="checkbox"
                    checked={enableFaceBiometrics}
                    onChange={(e) => setEnableFaceBiometrics(e.target.checked)}
                    className="accent-emerald-600 w-3.5 h-3.5 cursor-pointer"
                  />
                  <span className="flex items-center gap-1 text-emerald-800 font-semibold">
                    <ScanFace className="w-3.5 h-3.5 text-emerald-600" />
                    Face Biometrics
                  </span>
                </label>

                {/* Vehicle detector overlay toggle */}
                <label className="flex items-center gap-1.5 cursor-pointer select-none text-xs font-semibold text-slate-700 ml-1 border-l border-slate-200 pl-2.5" title="Overlay real-time vehicle and transport detection (vehicle_detector.onnx)">
                  <input
                    type="checkbox"
                    checked={enableVehicleDetector}
                    onChange={(e) => setEnableVehicleDetector(e.target.checked)}
                    className="accent-amber-600 w-3.5 h-3.5 cursor-pointer"
                  />
                  <span className="flex items-center gap-1 text-amber-800 font-semibold">
                    <Car className="w-3.5 h-3.5 text-amber-600" />
                    Vehicle Detector
                  </span>
                </label>
              </div>
            </div>

            {/* Quick preset links */}
            <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-slate-100 text-[11px]">
              <span className="text-slate-400 font-medium">Presets:</span>
              <button
                type="button"
                onClick={() => {
                  setRtspInputValue("0");
                  handleConnectStream("0");
                }}
                className="px-2.5 py-1 rounded-md bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-300 transition-colors font-mono font-bold cursor-pointer flex items-center gap-1 shadow-2xs"
              >
                <Camera className="w-3.5 h-3.5 text-emerald-700" />
                Local Webcam (Device 0)
              </button>
              <button
                type="button"
                onClick={() => {
                  setRtspInputValue("sample");
                  handleConnectStream("sample");
                }}
                className="px-2 py-0.5 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-800 border border-slate-200 transition-colors font-mono font-medium cursor-pointer"
              >
                Sample Simulation Feed
              </button>
              <button
                type="button"
                onClick={() => {
                  setRtspInputValue("http://192.168.1.5:8080");
                }}
                className="px-2 py-0.5 rounded-md bg-slate-100 hover:bg-slate-200 text-sky-800 border border-slate-200 transition-colors font-mono font-medium cursor-pointer"
              >
                IP Webcam Pro (Android/iOS)
              </button>
              <button
                type="button"
                onClick={() => {
                  setRtspInputValue("rtsp://10.20.72.101:554/live/ch0");
                }}
                className="px-2 py-0.5 rounded-md bg-slate-100 hover:bg-slate-200 text-amber-800 border border-slate-200 transition-colors font-mono font-medium cursor-pointer"
              >
                RTSP Camera (Port 554)
              </button>
            </div>
          </div>

          {/* Diagnostic & Error Feedback Banner */}
          {streamError && (
            <div className="bg-amber-950/40 border border-amber-500/50 rounded-lg p-3 text-xs text-amber-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <div className="flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-semibold text-amber-100">Feed Connection Issue</div>
                  <div className="text-amber-300 font-mono mt-0.5">
                    {streamDiagnostics || "Target address timed out or refused connection."}
                  </div>
                  <div className="text-slate-400 text-[11px] mt-1">
                    * If connecting to <strong className="text-slate-300">IP Webcam Pro</strong>: Make sure your phone is connected to the same Wi-Fi as this machine, and tap &quot;Start Server&quot; in the mobile app.
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => {
                    setRtspInputValue("sample");
                    handleConnectStream("sample");
                  }}
                  className="px-2.5 py-1 rounded bg-amber-900/80 hover:bg-amber-800 text-amber-100 font-medium transition-colors"
                >
                  Use Demo Feed
                </button>
                <button
                  type="button"
                  onClick={() => handleConnectStream()}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition-colors inline-flex items-center gap-1"
                >
                  <RefreshCw className="w-3 h-3" />
                  Retry
                </button>
              </div>
            </div>
          )}

          {isStreamActive && !streamError && streamDiagnostics && (
            <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-lg px-3 py-1.5 text-xs text-emerald-300 flex items-center justify-between">
              <span className="flex items-center gap-2 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {streamDiagnostics}
              </span>
              {detectedResolvedUrl && detectedResolvedUrl !== rtspUrl && (
                <span className="text-[11px] text-slate-400 font-mono hidden sm:inline">
                  Stream Endpoint: {detectedResolvedUrl}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* 🚨 Tactical Suspect Alert Banner for Uploaded Media */}
      {detectedSuspectsInMedia.length > 0 && (
        <div className="bg-white rounded-2xl border-l-4 border-l-rose-600 border border-slate-200/90 shadow-sm p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3.5 min-w-0">
            <div className="w-11 h-11 rounded-xl bg-rose-50 border border-rose-200 flex items-center justify-center shrink-0 text-rose-600 shadow-2xs">
              <ShieldAlert className="w-5 h-5 animate-pulse" />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-extrabold text-slate-900 flex flex-wrap items-center gap-2">
                <span className="text-rose-700 font-mono uppercase tracking-wider text-xs font-bold bg-rose-50 border border-rose-200 px-2 py-0.5 rounded">
                  🚨 Suspect Identified in Feed
                </span>
                {detectedSuspectsInMedia.map((s, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-0.5 rounded-md bg-rose-600 text-white text-xs font-mono font-bold shadow-2xs flex items-center gap-1.5"
                  >
                    <span>{s.name.toUpperCase()}</span>
                    <span className="text-rose-200 text-[10px] font-normal">[{s.threat_level || "HIGH"}]</span>
                    <span className="bg-rose-800/80 px-1 py-0.2 rounded text-[10px]">{formatConfidence(s.confidence)}</span>
                  </span>
                ))}
              </div>
              <div className="text-xs text-slate-500 font-medium mt-1">
                Facial biometric match verified via YuNet + SFace. Priority alert dispatched to Command Center & QRT.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2.5 shrink-0 w-full sm:w-auto justify-end">
            {detectedSuspectsInMedia[0] && (
              <button
                type="button"
                onClick={() => {
                  setSelectedSuspectForTrajectory(detectedSuspectsInMedia[0].name);
                }}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 shadow-2xs transition-all flex items-center gap-1.5 cursor-pointer"
              >
                <Compass className="w-3.5 h-3.5 text-rose-600" />
                Track Trajectory
              </button>
            )}
            <button
              type="button"
              onClick={() => router.push("/alerts")}
              className="px-3.5 py-2 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white shadow-xs transition-all flex items-center gap-1.5 cursor-pointer"
            >
              Command Center →
            </button>
          </div>
        </div>
      )}

      {/* 2. Primary Feed with Tactical Overlay Canvas & Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Main Viewport (Col 8) */}
        <div
          ref={viewportContainerRef}
          className={`bg-[#0d1215] flex flex-col relative select-none transition-all duration-200 ${
            isFullscreen
              ? "fixed inset-0 z-[9999] w-screen h-screen rounded-none border-none shadow-2xl overflow-hidden"
              : "lg:col-span-8 rounded-xl border border-slate-800 shadow-lg overflow-hidden"
          }`}
        >
          {/* Tactical Header Overlay */}
          <div className="p-3 bg-gradient-to-b from-black/90 via-black/60 to-transparent flex items-center justify-between text-xs text-white z-30 pointer-events-none">
            <div className="flex items-center gap-2 pointer-events-auto">
              <span className="flex h-2 w-2 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="font-mono font-bold tracking-wider">
                {sourceMode === "upload-video"
                  ? "VIDEO FEED • BORDER SURVEILLANCE"
                  : sourceMode === "rtsp"
                  ? `${streamProtocol.toUpperCase()} • LIVE STREAM`
                  : selectedImageFile
                  ? `IMAGE SCAN • ${selectedImageFile.name.toUpperCase()}`
                  : imagePreviewUrl
                  ? "IMAGE SCAN • SAMPLE FRAME"
                  : "IMAGE SCAN • DIRECT UPLOAD"}
              </span>
              {isFullscreen && (
                <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-emerald-950/90 text-emerald-300 border border-emerald-500/50 shadow-xs">
                  FULLSCREEN FOCUS
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 sm:gap-3 pointer-events-auto">
              <div className="font-mono text-[11px] text-slate-400 hidden sm:block">
                RES: {sourceMode === "upload-image" && !imagePreviewUrl ? "Awaiting Input" : `${sourceDims.width}x${sourceDims.height}`}
                {videoMetadata?.fps ? ` • ${Math.round(videoMetadata.fps)} FPS` : ""}
                {latencyMs !== null && ` • LATENCY: ${latencyMs}ms`}
              </div>

              {/* Detections quick drawer toggle in fullscreen */}
              {isFullscreen && detections.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowFullscreenInspector((prev) => !prev)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                    showFullscreenInspector
                      ? "bg-emerald-600 text-white ring-1 ring-emerald-400 shadow-xs"
                      : "bg-slate-800/90 text-emerald-300 hover:bg-slate-700 border border-emerald-500/40"
                  }`}
                  title="Toggle Tactical Target Inspector in fullscreen"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{detections.length} Targets</span>
                </button>
              )}

              {/* Primary Fullscreen Toggle Button */}
              <button
                type="button"
                onClick={toggleFullscreen}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer shadow-md ${
                  isFullscreen
                    ? "bg-rose-600/90 hover:bg-rose-600 text-white border border-rose-500/80 ring-2 ring-rose-500/30"
                    : "bg-emerald-900/60 hover:bg-emerald-800/90 text-emerald-200 border border-emerald-500/40 hover:border-emerald-400 hover:text-white"
                }`}
                title={isFullscreen ? "Exit Fullscreen (Esc or F)" : "View in Fullscreen (F)"}
              >
                {isFullscreen ? (
                  <>
                    <Minimize2 className="w-3.5 h-3.5 text-white" />
                    <span>Exit Fullscreen</span>
                    <span className="hidden md:inline-block text-[10px] bg-black/40 px-1.5 py-0.2 rounded border border-white/20">
                      ESC
                    </span>
                  </>
                ) : (
                  <>
                    <Maximize2 className="w-3.5 h-3.5 text-emerald-300" />
                    <span>Fullscreen</span>
                    <span className="hidden md:inline-block text-[10px] bg-black/40 px-1.5 py-0.2 rounded border border-emerald-500/30 text-emerald-300">
                      F
                    </span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Media Viewport + Overlay Canvas */}
          <div
            onDoubleClick={toggleFullscreen}
            className={`relative w-full flex items-center justify-center bg-black overflow-hidden transition-all ${
              isFullscreen
                ? "flex-1 min-h-0 h-full max-h-none"
                : "aspect-video min-h-[360px] max-h-[560px]"
            }`}
          >
            {/* Corner Reticles */}
            <div className="absolute top-3 left-3 w-5 h-5 border-t-2 border-l-2 border-emerald-500/80 z-30 pointer-events-none" />
            <div className="absolute top-3 right-3 w-5 h-5 border-t-2 border-r-2 border-emerald-500/80 z-30 pointer-events-none" />
            <div className="absolute bottom-3 left-3 w-5 h-5 border-b-2 border-l-2 border-emerald-500/80 z-30 pointer-events-none" />
            <div className="absolute bottom-3 right-3 w-5 h-5 border-b-2 border-r-2 border-emerald-500/80 z-30 pointer-events-none" />

            {/* ── RTSP / IP Webcam Live MJPEG Stream ── */}
            {sourceMode === "rtsp" ? (
              isStreamActive && activeMjpegSrc ? (
                <div className="relative w-full h-full">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    ref={streamImgRef}
                    key={streamKey}
                    src={activeMjpegSrc}
                    alt="Live Surveillance Stream"
                    className="w-full h-full object-contain"
                    onError={() => {
                      setStreamError(true);
                      setStatusMessage("Stream error — camera unreachable or backend offline.");
                    }}
                  />
                  {streamError && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/75 z-20 p-6 text-center">
                      <AlertTriangle className="w-10 h-10 text-amber-400 mb-2" />
                      <p className="text-white text-sm font-semibold">Stream Currently Offline</p>
                      <p className="text-slate-400 text-xs mt-1 max-w-md font-mono">{rtspUrl}</p>
                      {streamDiagnostics && (
                        <p className="text-amber-300 text-[11px] mt-2 max-w-lg font-mono bg-black/60 px-3 py-1.5 rounded border border-amber-500/30">
                          {streamDiagnostics}
                        </p>
                      )}
                      <div className="flex items-center gap-3 mt-4">
                        <button
                          type="button"
                          onClick={() => {
                            setRtspInputValue("sample");
                            handleConnectStream("sample");
                          }}
                          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-white"
                        >
                          Load Sample Feed
                        </button>
                        <button
                          type="button"
                          onClick={() => handleConnectStream()}
                          className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white"
                        >
                          <RefreshCw className="w-3.5 h-3.5" />
                          Retry Connection
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                /* Standby — RTSP mode selected but not yet connected */
                <div className="flex flex-col items-center justify-center gap-4 w-full h-full bg-[#0a0e12] p-8">
                  <div className="relative">
                    <div className="w-20 h-20 rounded-full border-2 border-slate-700 flex items-center justify-center">
                      <Maximize2 className="w-8 h-8 text-slate-600" />
                    </div>
                    <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-slate-800 border border-slate-600 flex items-center justify-center">
                      <WifiOff className="w-3 h-3 text-slate-500" />
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-slate-300 text-sm font-semibold font-mono">NO STREAM ACTIVE</p>
                    <p className="text-slate-500 text-xs mt-1">
                      Choose your local webcam, an IP Webcam / RTSP stream, or simulation demo.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={() => {
                        setRtspInputValue("0");
                        handleConnectStream("0");
                      }}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white transition-colors cursor-pointer shadow-xs"
                    >
                      <Camera className="w-3.5 h-3.5 text-emerald-400" />
                      Start Local Webcam
                    </button>
                    <button
                      type="button"
                      onClick={() => handleConnectStream()}
                      disabled={!rtspInputValue.trim()}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-white transition-colors disabled:opacity-40"
                    >
                      <Link2 className="w-3.5 h-3.5" />
                      Connect Feed
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setRtspInputValue("sample");
                        handleConnectStream("sample");
                      }}
                      className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors cursor-pointer"
                    >
                      Sample Feed
                    </button>
                  </div>
                </div>
              )
            ) : sourceMode === "upload-video" ? (
              <video
                ref={videoRef}
                key={videoPreviewUrl}
                src={videoPreviewUrl}
                controls
                playsInline
                loop
                onLoadedMetadata={handleVideoLoaded}
                onTimeUpdate={handleVideoTimeUpdate}
                onError={handleVideoError}
                className="w-full h-full object-contain"
              />
            ) : sourceMode === "upload-image" && !imagePreviewUrl ? (
              /* DIRECT UPLOAD DROPZONE INSTEAD OF STATIC IMAGE */
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDraggingOver(true);
                }}
                onDragLeave={() => setIsDraggingOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDraggingOver(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file && file.type.startsWith("image/")) {
                    setSelectedImageFile(file);
                    const url = URL.createObjectURL(file);
                    setImagePreviewUrl(url);
                    setDetections([]);
                    setDetectedSuspectsInMedia([]);
                    setStatusMessage(`Loaded image: ${file.name}`);
                  }
                }}
                onClick={() => imageFileInputRef.current?.click()}
                className={`w-full h-full min-h-[360px] flex flex-col items-center justify-center p-8 text-center cursor-pointer transition-all ${
                  isDraggingOver
                    ? "bg-[#13281c] border-2 border-dashed border-emerald-400"
                    : "bg-[#0b1013] hover:bg-[#0f171b] border-2 border-dashed border-slate-700/80 hover:border-emerald-500/60"
                }`}
              >
                <div className="w-16 h-16 rounded-2xl bg-emerald-950/60 border border-emerald-500/30 flex items-center justify-center text-emerald-400 mb-4 shadow-inner">
                  <UploadCloud className="w-8 h-8 text-emerald-400" />
                </div>
                <h4 className="text-white font-bold text-base mb-1 tracking-tight">
                  Upload Surveillance Image
                </h4>
                <p className="text-slate-400 text-xs max-w-sm mb-4 leading-relaxed">
                  Drag and drop a tactical frame, checkpoint still, or border surveillance capture here, or click to browse
                </p>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      imageFileInputRef.current?.click();
                    }}
                    className="px-4 py-2 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white flex items-center gap-2 shadow-xs transition-all cursor-pointer"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    Browse Image File
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setImagePreviewUrl("/images/himalayan-border-hero.jpg");
                      setSelectedImageFile(null);
                      setStatusMessage("Loaded demo border patrol sample frame.");
                    }}
                    className="px-3 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors cursor-pointer"
                  >
                    Use Demo Frame
                  </button>
                </div>
                <span className="text-[10px] font-mono text-slate-500 mt-4 uppercase tracking-wider">
                  Supports: JPG, PNG, WEBP, BMP (Max 25MB)
                </span>
              </div>
            ) : (
              /* Image Loaded State with Replace / Clear Controls */
              <div
                className="relative w-full h-full flex items-center justify-center"
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDraggingOver(true);
                }}
                onDragLeave={() => setIsDraggingOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDraggingOver(false);
                  const file = e.dataTransfer.files?.[0];
                  if (file && file.type.startsWith("image/")) {
                    setSelectedImageFile(file);
                    const url = URL.createObjectURL(file);
                    setImagePreviewUrl(url);
                    setDetections([]);
                    setDetectedSuspectsInMedia([]);
                    setStatusMessage(`Loaded image: ${file.name}`);
                  }
                }}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  ref={imageRef}
                  src={imagePreviewUrl}
                  alt="Surveillance Feed Frame"
                  onLoad={handleImageLoaded}
                  className="w-full h-full object-contain"
                />

                {/* Direct Upload / Replace Overlay Controls */}
                <div className="absolute top-3 right-3 z-30 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={toggleFullscreen}
                    className="p-1.5 rounded-lg text-xs font-semibold bg-slate-900/85 hover:bg-slate-800 text-slate-200 hover:text-white border border-slate-700 shadow-md transition-all cursor-pointer backdrop-blur-xs"
                    title={isFullscreen ? "Exit Fullscreen (Esc)" : "Expand to Fullscreen (F)"}
                  >
                    {isFullscreen ? <Minimize2 className="w-3.5 h-3.5 text-white" /> : <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />}
                  </button>
                  <button
                    type="button"
                    onClick={() => imageFileInputRef.current?.click()}
                    className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-900/85 hover:bg-slate-800 text-white border border-slate-700 shadow-md flex items-center gap-1.5 transition-all cursor-pointer backdrop-blur-xs"
                    title="Upload different image"
                  >
                    <Upload className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Replace Image</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setImagePreviewUrl("");
                      setSelectedImageFile(null);
                      setDetections([]);
                      setDetectedSuspectsInMedia([]);
                      setStatusMessage("Image cleared. Upload a new image to scan.");
                    }}
                    className="p-1.5 rounded-lg text-xs font-semibold bg-slate-900/85 hover:bg-rose-900 text-slate-300 hover:text-rose-200 border border-slate-700 shadow-md transition-all cursor-pointer backdrop-blur-xs"
                    title="Clear and show upload dropzone"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                {isDraggingOver && (
                  <div className="absolute inset-0 bg-black/75 border-2 border-dashed border-emerald-400 flex flex-col items-center justify-center z-40">
                    <UploadCloud className="w-12 h-12 text-emerald-400 animate-bounce mb-2" />
                    <p className="text-white text-sm font-bold">Drop Image to Load Frame</p>
                  </div>
                )}
              </div>
            )}

            {/* Detection overlay canvas — strictly for image mode; video mode uses server-side burned-in rendering */}
            {sourceMode === "upload-image" && imagePreviewUrl && (
              <DetectionCanvas
                detections={detections}
                sourceWidth={sourceDims.width}
                sourceHeight={sourceDims.height}
              />
            )}

            {/* Floating Target Inspector Drawer (Active in Fullscreen mode when targets exist) */}
            {isFullscreen && showFullscreenInspector && detections.length > 0 && (
              <div className="absolute top-16 right-4 z-40 w-84 max-h-[calc(100vh-140px)] bg-slate-950/90 border border-slate-800 backdrop-blur-md rounded-2xl p-4 shadow-2xl flex flex-col animate-in fade-in slide-in-from-right-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-2">
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                    Target Inspector ({detections.length})
                  </h4>
                  <button
                    type="button"
                    onClick={() => setShowFullscreenInspector(false)}
                    className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
                <div className="space-y-2 overflow-y-auto max-h-[calc(100vh-220px)] pr-1">
                  {detections.map((det, idx) => {
                    const isSuspect = Boolean(
                      det.is_threat ||
                      det.class_name.toLowerCase().startsWith("suspect") ||
                      det.threat_level
                    );
                    const suspectName = det.suspect_name || det.class_name.replace(/^suspect_/i, "");

                    return (
                      <div
                        key={idx}
                        className={`p-2.5 rounded-xl border text-xs ${
                          isSuspect
                            ? "bg-rose-950/40 border-rose-800/60 text-rose-200"
                            : "bg-slate-900/80 border-slate-800 text-slate-200"
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-mono font-bold uppercase truncate flex items-center gap-1.5">
                            {isSuspect ? (
                              <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse shrink-0" />
                            ) : null}
                            <span>{isSuspect ? `SUSPECT: ${suspectName.toUpperCase()}` : `${det.class_name} #${idx + 1}`}</span>
                          </span>
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                            isSuspect ? "bg-rose-600 text-white" : "bg-emerald-950 text-emerald-300 border border-emerald-700/50"
                          }`}>
                            {formatConfidence(det.confidence)}
                          </span>
                        </div>
                        <div className="text-[10px] font-mono text-slate-400 flex items-center justify-between mt-1">
                          <span>Box: [{det.box.map((n) => Math.round(n)).join(", ")}]</span>
                          {isSuspect && (
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedSuspectForTrajectory(suspectName);
                                toggleFullscreen();
                              }}
                              className="text-rose-400 hover:text-rose-300 font-semibold underline flex items-center gap-1 cursor-pointer"
                            >
                              <Compass className="w-3 h-3" /> Trajectory
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Video Frames Scrubber Bar (visible when video results are available) */}
          {sourceMode === "upload-video" && videoResults.length > 0 && (
            <div className="bg-slate-900 px-4 py-2.5 border-t border-slate-800">
              <div className="flex flex-wrap items-center justify-between text-xs text-slate-400 mb-2 gap-2">
                <span className="font-mono flex items-center gap-1.5 text-emerald-400 font-semibold">
                  <Film className="w-3.5 h-3.5" />
                  Processed Video Frames ({videoResults.length})
                </span>

                {annotatedVideoUrl && (
                  <div className="flex items-center gap-1.5 bg-emerald-950/80 text-emerald-400 border border-emerald-500/40 px-2.5 py-1 rounded-md text-[11px] font-mono font-semibold shadow-xs">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span>Server-Side Bounding Boxes Burned-In</span>
                  </div>
                )}

                <span className="text-[11px] text-slate-400">
                  Click frame to inspect targets
                </span>
              </div>
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 custom-scrollbar">
                {videoResults.map((fr) => {
                  const frameDets =
                    fr.inference?.detections || fr.detections || [];
                  const isSelected = fr.frame_index === selectedFrameIndex;
                  const hasDetections = frameDets.length > 0;

                  return (
                    <button
                      key={fr.frame_index}
                      onClick={() => handleSelectFrame(fr.frame_index)}
                      className={`px-2 py-1 rounded text-[11px] font-mono whitespace-nowrap transition-all flex items-center gap-1 ${
                        isSelected
                          ? "bg-emerald-600 text-white font-bold ring-1 ring-emerald-400"
                          : hasDetections
                          ? "bg-slate-800 text-emerald-300 hover:bg-slate-700 border border-emerald-800/60"
                          : "bg-slate-800/80 text-slate-400 hover:bg-slate-700"
                      }`}
                    >
                      <span>#{fr.frame_index}</span>
                      {hasDetections && (
                        <span className="px-1 rounded bg-emerald-950/80 text-[10px] text-emerald-300 font-semibold">
                          {frameDets.length}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Bottom Bar with Status Telemetry */}
          <div className="px-4 py-2.5 bg-slate-900/95 border-t border-slate-800 flex items-center justify-between text-xs text-slate-300 z-30 shrink-0">
            <span className="font-mono text-[11px] truncate mr-2">
              {statusMessage ||
                (isOnline
                  ? (sourceMode === "upload-image" && !imagePreviewUrl
                      ? "Awaiting Image — Upload a surveillance image to scan."
                      : "Engine Ready — Trigger detection to scan feed.")
                  : "AI Engine Offline — Start backend on port 8000.")}
            </span>
            <div className="flex items-center gap-3 shrink-0">
              {isFullscreen && (
                <span className="hidden sm:inline-block font-mono text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-700">
                  Double-click feed or press ESC / F to exit
                </span>
              )}
              {sourceMode === "rtsp" ? (
                <span className={`font-semibold text-xs font-mono ${
                  isStreamActive && !streamError ? "text-emerald-400" : "text-slate-500"
                }`}>
                  {isStreamActive && !streamError
                    ? `● LIVE • ${detections.length} TARGETS ACQUIRED`
                    : streamError
                    ? "● ERROR"
                    : "○ STANDBY"}
                </span>
              ) : (
                <span className="font-semibold text-emerald-400 text-xs font-mono">
                  {sourceMode === "upload-image" && !imagePreviewUrl
                    ? "NO IMAGE LOADED"
                    : `${detections.length} TARGETS ACQUIRED`}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Tactical Telemetry & Detection Inspector (Col 4) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs p-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-emerald-700" />
                Target Inspector
              </h3>
              <span className="text-xs font-mono font-medium text-slate-500 bg-slate-100 px-2 py-0.5 rounded-md">
                {sourceMode === "upload-video" && videoResults.length > 0
                  ? `Frame #${selectedFrameIndex} (${detections.length})`
                  : `${detections.length} Detected`}
              </span>
            </div>

            {detections.length === 0 ? (
              <div className="py-10 text-center text-xs text-slate-400">
                <CheckCircle2 className="w-8 h-8 text-slate-300 mx-auto mb-2" />
                No active threats or targets identified in current view.
              </div>
            ) : (
              <div className="divide-y divide-slate-100 max-h-[340px] overflow-y-auto mt-2">
                {detections.map((det, idx) => {
                  const isSuspect = Boolean(
                    det.is_threat ||
                    det.class_name.toLowerCase().startsWith("suspect") ||
                    det.threat_level
                  );
                  const suspectName = det.suspect_name || det.class_name.replace(/^suspect_/i, "");

                  return (
                    <div
                      key={idx}
                      className={`py-2.5 px-2 rounded-xl transition-all ${
                        isSuspect
                          ? "bg-rose-50/50 border border-rose-200/80 my-1"
                          : "hover:bg-slate-50 my-0.5"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span
                          className={`text-xs font-bold font-mono uppercase flex items-center gap-1.5 ${
                            isSuspect ? "text-rose-800" : "text-slate-800"
                          }`}
                        >
                          {isSuspect ? (
                            <span className="w-2 h-2 rounded-full bg-rose-600 animate-pulse shrink-0" />
                          ) : det.track_id !== undefined ? (
                            <span className="bg-emerald-950 text-emerald-300 border border-emerald-700/50 px-1.5 py-0.5 rounded text-[10px] font-bold font-mono">
                              ID #{det.track_id}
                            </span>
                          ) : null}
                          <span className="truncate">
                            {isSuspect ? `SUSPECT: ${suspectName.toUpperCase()}` : `${det.class_name} #${idx + 1}`}
                          </span>
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {isSuspect && det.threat_level && (
                            <span className="text-[10px] font-bold bg-rose-600 text-white px-1.5 py-0.2 rounded uppercase">
                              {det.threat_level}
                            </span>
                          )}
                          <span
                            className={`text-xs font-semibold px-2 py-0.5 rounded ${
                              isSuspect
                                ? "text-rose-700 bg-rose-100 border border-rose-200 font-bold"
                                : "text-emerald-800 bg-emerald-50 border border-emerald-200/60"
                            }`}
                          >
                            {formatConfidence(det.confidence)}
                          </span>
                        </div>
                      </div>
                      <div className="text-[11px] font-mono text-slate-500 flex items-center justify-between mt-1.5">
                        <span>
                          Box: [{det.box.map((n) => Math.round(n)).join(", ")}]
                        </span>
                        {isSuspect ? (
                          <button
                            type="button"
                            onClick={() => setSelectedSuspectForTrajectory(suspectName)}
                            className="text-[11px] text-rose-700 hover:text-rose-900 font-semibold underline flex items-center gap-1 cursor-pointer"
                          >
                            <Compass className="w-3 h-3" /> Trajectory
                          </button>
                        ) : (
                          <span>Class ID: {det.class_id}</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Engine Parameters Card */}
          <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs p-4 text-xs space-y-2 text-slate-600">
            <div className="font-bold text-slate-900 border-b border-slate-100 pb-2 flex items-center justify-between">
              <span>Runtime Specs</span>
              <span className="font-mono text-xs font-bold text-emerald-800 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded">
                {selectedModel || "None Selected"}
              </span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-slate-500">Mode:</span>
              <span className="font-mono font-medium text-slate-800 capitalize">
                {sourceMode.replace("-", " ")}
              </span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-slate-500">Inference Hardware:</span>
              <span className="font-mono font-medium text-slate-800">
                ONNX Runtime (CPU/CUDA)
              </span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-slate-500">NMS Threshold:</span>
              <span className="font-mono font-medium text-slate-800">
                IoU {iouThreshold}
              </span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-slate-500">Confidence Cutoff:</span>
              <span className="font-mono font-medium text-slate-800">
                Conf &gt;= {confThreshold}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Sector Camera Feeds Matrix */}
      <section aria-label="Camera Matrix" className="pt-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-xs font-semibold text-slate-600 tracking-wider uppercase">
            Sector Camera Feeds
          </h2>
          <span className="text-xs font-mono text-slate-500">
            {selectedCamera ? `Viewing: ${selectedCamera.name}` : "Click a camera to connect"}
          </span>
        </div>
        <CameraGrid
          selectedCameraId={selectedCamera?.id}
          onSelectCamera={handleCameraSelect}
        />
      </section>

      {/* Suspect Trajectory Modal */}
      {selectedSuspectForTrajectory && (
        <SuspectTrajectoryModal
          suspectName={selectedSuspectForTrajectory}
          onClose={() => setSelectedSuspectForTrajectory(null)}
        />
      )}
    </div>
  );
}

export default LiveWorkspace;
