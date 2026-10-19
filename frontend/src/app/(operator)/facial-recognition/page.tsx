"use client";

import { useState, useEffect, useRef } from "react";
import {
  ScanFace,
  UserCheck,
  UserX,
  Camera,
  Video,
  Upload,
  Search,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Plus,
  Trash2,
  Play,
  Pause,
  Layers,
  Clock,
  Sparkles,
  Zap,
  Maximize2,
  Minimize2,
  X,
  UserPlus,
  Eye,
  Activity,
  Cpu,
  Download,
  Filter,
  Grid,
  List,
  SlidersHorizontal,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  EnrolledPerson,
  FaceDetection,
  FaceEngineStatus,
  FaceEvent,
  FaceScanResponse,
  UnknownPerson,
  UnknownStats,
} from "@/types/face";

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

export default function FacialRecognitionPage() {
  const [activeTab, setActiveTab] = useState<"stream" | "directory" | "unknowns" | "scan">("stream");

  // Telemetry & Status
  const [engineStatus, setEngineStatus] = useState<FaceEngineStatus | null>(null);
  const [enrolledPersons, setEnrolledPersons] = useState<EnrolledPerson[]>([]);
  const [recentEvents, setRecentEvents] = useState<FaceEvent[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Unknown Persons Archival State
  const [unknowns, setUnknowns] = useState<UnknownPerson[]>([]);
  const [unknownStats, setUnknownStats] = useState<UnknownStats | null>(null);
  const [unknownSearch, setUnknownSearch] = useState<string>("");
  const [unknownCameraFilter, setUnknownCameraFilter] = useState<string>("all");
  const [unknownStatusFilter, setUnknownStatusFilter] = useState<string>("all");
  const [selectedUnknownForInspect, setSelectedUnknownForInspect] = useState<UnknownPerson | null>(null);
  const [selectedUnknownForEnroll, setSelectedUnknownForEnroll] = useState<UnknownPerson | null>(null);
  const [unknownViewMode, setUnknownViewMode] = useState<"grid" | "table">("grid");
  const [isLoadingUnknowns, setIsLoadingUnknowns] = useState<boolean>(false);
  const [isConsolidating, setIsConsolidating] = useState<boolean>(false);

  // Quick Enroll Unknown Person Form State
  const [quickEnrollName, setQuickEnrollName] = useState<string>("");
  const [quickEnrollIsSuspect, setQuickEnrollIsSuspect] = useState<boolean>(true);
  const [quickEnrollThreatLevel, setQuickEnrollThreatLevel] = useState<"CRITICAL" | "HIGH" | "MEDIUM" | "LOW">("HIGH");
  const [quickEnrollCategory, setQuickEnrollCategory] = useState<string>("Wanted / BOLO");
  const [quickEnrollNotes, setQuickEnrollNotes] = useState<string>("");
  const [isQuickEnrolling, setIsQuickEnrolling] = useState<boolean>(false);
  const [quickEnrollError, setQuickEnrollError] = useState<string | null>(null);
  const [quickEnrollSuccess, setQuickEnrollSuccess] = useState<string | null>(null);

  // Live Stream State
  const [streamSource, setStreamSource] = useState<string>("browser");
  const [isStreamPlaying, setIsStreamPlaying] = useState<boolean>(true);
  const [streamKey, setStreamKey] = useState<number>(0);
  const [streamError, setStreamError] = useState<boolean>(false);
  const [customStreamUrl, setCustomStreamUrl] = useState<string>("");
  const streamImgRef = useRef<HTMLImageElement | null>(null);

  // Client-Side Browser Webcam WebSocket Streaming State
  const clientVideoRef = useRef<HTMLVideoElement | null>(null);
  const clientCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const clientDisplayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const clientWsRef = useRef<WebSocket | null>(null);
  const clientWatchdogRef = useRef<NodeJS.Timeout | null>(null);
  const clientSendTimerRef = useRef<NodeJS.Timeout | null>(null);
  const isProcessingRef = useRef<boolean>(false);
  const lastSentTimeRef = useRef<number>(0);
  const hasReceivedFirstFrameRef = useRef<boolean>(false);
  const [hasClientFeed, setHasClientFeed] = useState<boolean>(false);
  const [isClientStreaming, setIsClientStreaming] = useState<boolean>(false);

  const stopClientStreaming = () => {
    if (clientWatchdogRef.current) {
      clearInterval(clientWatchdogRef.current);
      clientWatchdogRef.current = null;
    }
    if (clientSendTimerRef.current) {
      clearTimeout(clientSendTimerRef.current);
      clientSendTimerRef.current = null;
    }
    if (clientWsRef.current) {
      try {
        clientWsRef.current.close();
      } catch {}
      clientWsRef.current = null;
    }
    if (clientVideoRef.current && clientVideoRef.current.srcObject) {
      try {
        const stream = clientVideoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      } catch {}
      clientVideoRef.current.srcObject = null;
    }
    isProcessingRef.current = false;
    hasReceivedFirstFrameRef.current = false;
    setHasClientFeed(false);
    setIsClientStreaming(false);

    if (clientDisplayCanvasRef.current) {
      const ctx = clientDisplayCanvasRef.current.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, clientDisplayCanvasRef.current.width, clientDisplayCanvasRef.current.height);
      }
    }
  };

  const startClientStreaming = async () => {
    stopClientStreaming();
    setStreamError(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });

      if (clientVideoRef.current) {
        clientVideoRef.current.srcObject = stream;
        try {
          await clientVideoRef.current.play();
        } catch {}
      }

      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsHost = BACKEND_BASE_URL.replace(/^https?:\/\//i, "");
      const wsUrl = `${wsProtocol}//${wsHost}/api/faces/ws/stream`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "blob";
      clientWsRef.current = ws;

      const captureAndSend = () => {
        if (!clientWsRef.current || clientWsRef.current.readyState !== WebSocket.OPEN) {
          return;
        }

        // Backpressure flow control: Skip if backend is still processing previous frame
        if (isProcessingRef.current) {
          if (performance.now() - lastSentTimeRef.current > 350) {
            // Watchdog unlock if a frame or network reply was dropped
            isProcessingRef.current = false;
          } else {
            return;
          }
        }

        // Skip if browser WebSocket buffer is congested
        if (clientWsRef.current.bufferedAmount > 0) {
          return;
        }

        const video = clientVideoRef.current;
        const canvas = clientCanvasRef.current;
        if (!video || !canvas || !video.videoWidth || !video.videoHeight) {
          return;
        }

        // Downsample to max width 640 maintaining aspect ratio for ultra-fast encoding & inference
        const maxW = 640;
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        let targetW = vw;
        let targetH = vh;
        if (vw > maxW) {
          targetW = maxW;
          targetH = Math.round((vh * maxW) / vw);
        }

        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        ctx.drawImage(video, 0, 0, targetW, targetH);
        isProcessingRef.current = true;
        lastSentTimeRef.current = performance.now();

        canvas.toBlob(
          (blob) => {
            if (blob && clientWsRef.current && clientWsRef.current.readyState === WebSocket.OPEN) {
              clientWsRef.current.send(blob);
            } else {
              isProcessingRef.current = false;
            }
          },
          "image/jpeg",
          0.65
        );
      };

      const scheduleNextFrame = () => {
        if (clientSendTimerRef.current) return;
        const now = performance.now();
        const elapsed = now - lastSentTimeRef.current;
        // Paced interval: aim for smooth 20-22 FPS (~45ms) without queue buildup
        const wait = Math.max(0, 45 - elapsed);
        clientSendTimerRef.current = setTimeout(() => {
          clientSendTimerRef.current = null;
          captureAndSend();
        }, wait);
      };

      ws.onopen = () => {
        setIsClientStreaming(true);
        setStreamError(false);
        isProcessingRef.current = false;
        hasReceivedFirstFrameRef.current = false;
        captureAndSend();

        // Safety ticker to recover if a frame drop occurs
        clientWatchdogRef.current = setInterval(() => {
          if (!isProcessingRef.current) {
            scheduleNextFrame();
          } else if (performance.now() - lastSentTimeRef.current > 350) {
            isProcessingRef.current = false;
            scheduleNextFrame();
          }
        }, 100);
      };

      ws.onmessage = async (event) => {
        isProcessingRef.current = false;
        if (event.data instanceof Blob) {
          try {
            const bmp = await createImageBitmap(event.data);
            const displayCanvas = clientDisplayCanvasRef.current;
            if (displayCanvas) {
              if (displayCanvas.width !== bmp.width || displayCanvas.height !== bmp.height) {
                displayCanvas.width = bmp.width;
                displayCanvas.height = bmp.height;
              }
              const ctx = displayCanvas.getContext("2d");
              if (ctx) {
                ctx.drawImage(bmp, 0, 0);
              }
            }
            bmp.close();
            if (!hasReceivedFirstFrameRef.current) {
              hasReceivedFirstFrameRef.current = true;
              setHasClientFeed(true);
            }
          } catch {
            // Bitmap decode fallback
          }
        }
        // Immediately trigger next frame capture (Zero-lag request-response loop!)
        scheduleNextFrame();
      };

      ws.onerror = () => {
        setStreamError(true);
      };

      ws.onclose = () => {
        setIsClientStreaming(false);
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn("Client webcam stream error:", msg);
      setStreamError(true);
    }
  };

  const handleToggleStream = async () => {
    if (isStreamPlaying) {
      setIsStreamPlaying(false);
      stopClientStreaming();
      try {
        await api.stopFaceCamera();
      } catch {}
    } else {
      setStreamError(false);
      setStreamKey(Date.now());
      setIsStreamPlaying(true);
    }
  };

  const handleSourceChange = async (newSource: string) => {
    setStreamError(false);
    stopClientStreaming();
    try {
      await api.stopFaceCamera();
    } catch {}
    setStreamSource(newSource);
    setStreamKey(Date.now());
    setIsStreamPlaying(true);
  };

  // Enrollment Modal State
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);
  const [enrollMode, setEnrollMode] = useState<"webcam" | "upload">("webcam");
  const [enrollName, setEnrollName] = useState("");
  const [enrollIsSuspect, setEnrollIsSuspect] = useState<boolean>(true);
  const [enrollThreatLevel, setEnrollThreatLevel] = useState<"CRITICAL" | "HIGH" | "MEDIUM" | "LOW">("HIGH");
  const [enrollCategory, setEnrollCategory] = useState<string>("Wanted / BOLO");
  const [enrollNotes, setEnrollNotes] = useState<string>("");
  const [directoryFilter, setDirectoryFilter] = useState<"all" | "suspects" | "authorized">("all");
  const [enrollFile, setEnrollFile] = useState<File | null>(null);
  const [enrollPreview, setEnrollPreview] = useState<string | null>(null);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [enrollError, setEnrollError] = useState<string | null>(null);
  const [enrollSuccess, setEnrollSuccess] = useState<string | null>(null);
  const [targetPersonForSample, setTargetPersonForSample] = useState<EnrolledPerson | null>(null);

  // Webcam Capture Refs
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  // Forensic Image Scan State
  const [scanFile, setScanFile] = useState<File | null>(null);
  const [scanPreview, setScanPreview] = useState<string | null>(null);
  const [scanResult, setScanResult] = useState<FaceScanResponse | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [minMatchThreshold, setMinMatchThreshold] = useState<number>(0.40);

  // Fullscreen states & refs
  const streamContainerRef = useRef<HTMLDivElement | null>(null);
  const [isStreamFullscreen, setIsStreamFullscreen] = useState(false);
  const [isFaceScanFullscreen, setIsFaceScanFullscreen] = useState(false);

  const toggleStreamFullscreen = () => {
    if (!streamContainerRef.current) return;
    if (!document.fullscreenElement && !isStreamFullscreen) {
      streamContainerRef.current.requestFullscreen().catch(() => {});
      setIsStreamFullscreen(true);
    } else {
      if (document.fullscreenElement) {
        document.exitFullscreen().catch(() => {});
      }
      setIsStreamFullscreen(false);
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsStreamFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", handleFsChange);
    return () => document.removeEventListener("fullscreenchange", handleFsChange);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (isFaceScanFullscreen) setIsFaceScanFullscreen(false);
        if (isStreamFullscreen) {
          if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
          setIsStreamFullscreen(false);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isStreamFullscreen, isFaceScanFullscreen]);

  // Initial Data Fetch
  useEffect(() => {
    refreshAllData();
    const interval = setInterval(() => {
      fetchEvents();
      fetchStatus();
      fetchUnknowns();
    }, 3500);
    return () => clearInterval(interval);
  }, []);

  // Handle Webcam Stream when modal is open
  useEffect(() => {
    if (isEnrollModalOpen && enrollMode === "webcam") {
      startWebcam();
    } else {
      stopWebcam();
    }
    return () => stopWebcam();
  }, [isEnrollModalOpen, enrollMode]);

  // Release camera hardware on component unmount
  useEffect(() => {
    return () => {
      stopClientStreaming();
      api.stopFaceCamera().catch(() => {});
    };
  }, []);

  // When switching away from the Live Stream tab, pause and release camera hardware
  useEffect(() => {
    if (activeTab !== "stream" && isStreamPlaying) {
      handleToggleStream();
    }
  }, [activeTab]);

  // Handle client browser webcam streaming vs server stream
  useEffect(() => {
    if (activeTab === "stream" && isStreamPlaying && streamSource === "browser") {
      startClientStreaming();
    } else {
      stopClientStreaming();
    }
    return () => stopClientStreaming();
  }, [activeTab, isStreamPlaying, streamSource, streamKey]);

  const refreshAllData = async () => {
    setIsLoading(true);
    await Promise.all([fetchStatus(), fetchEnrolled(), fetchEvents(), fetchUnknowns()]);
    setIsLoading(false);
  };

  const fetchStatus = async () => {
    try {
      const res = await api.getFaceEngineStatus();
      setEngineStatus(res);
    } catch {
      // Backend temporarily reconnecting
    }
  };

  const fetchEnrolled = async () => {
    try {
      const res = await api.getEnrolledFaces();
      if (res.success) {
        setEnrolledPersons(res.persons);
      }
    } catch {
      // Backend temporarily reconnecting
    }
  };

  const fetchEvents = async () => {
    try {
      const res = await api.getRecentFaceEvents(20);
      if (res.success) {
        setRecentEvents(res.events);
      }
    } catch {
      // Backend temporarily reconnecting
    }
  };

  const fetchUnknowns = async () => {
    try {
      setIsLoadingUnknowns(true);
      const [listRes, statsRes] = await Promise.all([
        api.getUnknownFaces({ limit: 120 }),
        api.getUnknownFaceStats().catch(() => null),
      ]);
      if (listRes.success) {
        setUnknowns(listRes.unknowns);
      }
      if (statsRes && statsRes.success) {
        setUnknownStats(statsRes);
      }
    } catch {
      // Backend temporarily reconnecting
    } finally {
      setIsLoadingUnknowns(false);
    }
  };

  const handleDeleteUnknown = async (id: string) => {
    if (!confirm(`Are you sure you want to delete unknown record ${id}? The saved photograph will be deleted from disk.`)) {
      return;
    }
    try {
      await api.deleteUnknownFace(id);
      setUnknowns((prev) => prev.filter((u) => u.id !== id));
      if (selectedUnknownForInspect?.id === id) {
        setSelectedUnknownForInspect(null);
      }
      api.getUnknownFaceStats().then((res) => setUnknownStats(res)).catch(() => {});
    } catch {
      alert("Failed to delete unknown record.");
    }
  };

  const handleConsolidateDuplicates = async () => {
    try {
      setIsConsolidating(true);
      const res = await api.consolidateUnknownFaces(0.38);
      if (res.success) {
        alert(res.message);
        await fetchUnknowns();
      }
    } catch {
      alert("Failed to consolidate duplicate unknown records.");
    } finally {
      setIsConsolidating(false);
    }
  };

  const handleClearAllUnknowns = async () => {
    if (!confirm("Are you sure you want to clear ALL unknown person records and delete their saved photographs? This cannot be undone.")) {
      return;
    }
    try {
      await api.clearUnknownFaces();
      setUnknowns([]);
      setSelectedUnknownForInspect(null);
      api.getUnknownFaceStats().then((res) => setUnknownStats(res)).catch(() => {});
    } catch {
      alert("Failed to clear unknown records.");
    }
  };

  const openQuickEnrollModal = (unknown: UnknownPerson) => {
    setSelectedUnknownForEnroll(unknown);
    setQuickEnrollName(unknown.enrolled_name || "");
    setQuickEnrollIsSuspect(true);
    setQuickEnrollThreatLevel("HIGH");
    setQuickEnrollCategory("Wanted / BOLO");
    setQuickEnrollNotes(unknown.notes || `Detected on ${unknown.camera_name || unknown.camera_id} at ${unknown.timestamp}`);
    setQuickEnrollError(null);
    setQuickEnrollSuccess(null);
  };

  const closeQuickEnrollModal = () => {
    setSelectedUnknownForEnroll(null);
    setQuickEnrollName("");
    setQuickEnrollError(null);
    setQuickEnrollSuccess(null);
  };

  const handleQuickEnrollSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedUnknownForEnroll || !quickEnrollName.trim()) {
      setQuickEnrollError("Please enter an identity name.");
      return;
    }

    setIsQuickEnrolling(true);
    setQuickEnrollError(null);
    setQuickEnrollSuccess(null);

    try {
      const res = await api.enrollUnknownFace(selectedUnknownForEnroll.id, {
        name: quickEnrollName.trim(),
        is_suspect: quickEnrollIsSuspect,
        threat_level: quickEnrollThreatLevel,
        category: quickEnrollCategory,
        notes: quickEnrollNotes,
      });

      if (res.success) {
        setQuickEnrollSuccess(`Successfully registered ${quickEnrollName.trim()} into Facial Recognition database!`);
        await Promise.all([fetchEnrolled(), fetchUnknowns()]);
        setTimeout(() => {
          closeQuickEnrollModal();
        }, 1200);
      } else {
        setQuickEnrollError(res.error || "Enrollment failed.");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setQuickEnrollError(msg || "Enrollment failed.");
    } finally {
      setIsQuickEnrolling(false);
    }
  };

  const handleUpdateUnknownStatus = async (unknownId: string, newStatus: "unreviewed" | "flagged" | "dismissed") => {
    try {
      const res = await api.updateUnknownFace(unknownId, { status: newStatus });
      if (res.success) {
        setUnknowns((prev) => prev.map((u) => (u.id === unknownId ? res.unknown : u)));
        if (selectedUnknownForInspect?.id === unknownId) {
          setSelectedUnknownForInspect(res.unknown);
        }
      }
    } catch {
      alert("Failed to update status.");
    }
  };

  const startWebcam = async () => {
    setWebcamError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsWebcamActive(true);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setWebcamError(`Webcam access error: ${msg}. Try uploading a photo instead.`);
      setIsWebcamActive(false);
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsWebcamActive(false);
  };

  const captureWebcamSnapshot = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
      setEnrollPreview(dataUrl);
    }
  };

  const handleEnrollSubmit = async () => {
    if (!targetPersonForSample && (!enrollName || !enrollName.trim())) {
      setEnrollError("Please enter a person name");
      return;
    }

    const payloadImage = enrollPreview || enrollFile;
    if (!payloadImage) {
      setEnrollError("Please capture or upload a face photo");
      return;
    }

    setIsEnrolling(true);
    setEnrollError(null);
    setEnrollSuccess(null);

    try {
      if (targetPersonForSample) {
        // Add sample to existing person
        const res = await api.addFaceSample(targetPersonForSample.id, payloadImage);
        if (res.success) {
          setEnrollSuccess(`Sample added successfully to ${targetPersonForSample.name}!`);
          await fetchEnrolled();
          setTimeout(() => {
            closeEnrollModal();
          }, 1400);
        } else {
          setEnrollError(res.error || "Failed to add sample");
        }
      } else {
        // Register new person with suspect metadata
        const res = await api.registerFace(enrollName.trim(), payloadImage, {
          is_suspect: enrollIsSuspect,
          threat_level: enrollThreatLevel,
          category: enrollCategory,
          notes: enrollNotes,
        });
        if (res.success) {
          setEnrollSuccess(
            `Successfully registered ${res.person?.name || enrollName} as ${
              enrollIsSuspect ? "Wanted Suspect" : "Authorized Personnel"
            }!`
          );
          await fetchEnrolled();
          setTimeout(() => {
            closeEnrollModal();
          }, 1400);
        } else {
          setEnrollError(res.error || "Failed to enroll person");
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setEnrollError(msg || "Enrolment failed");
    } finally {
      setIsEnrolling(false);
    }
  };

  const handleToggleSuspect = async (person: EnrolledPerson) => {
    const newStatus = !person.is_suspect;
    try {
      await api.updateFaceMetadata(person.id, {
        is_suspect: newStatus,
        threat_level: newStatus ? (person.threat_level || "HIGH") : "LOW",
        category: newStatus ? "Wanted / BOLO" : "Authorized Personnel",
      });
      await fetchEnrolled();
    } catch {
      alert("Failed to update status");
    }
  };

  const handleDeletePerson = async (person: EnrolledPerson) => {
    if (!confirm(`Are you sure you want to remove ${person.name} from the facial recognition database?`)) {
      return;
    }
    try {
      await api.deleteFace(person.id);
      await fetchEnrolled();
    } catch (err) {
      alert("Failed to delete person from database");
    }
  };

  const openEnrollModal = (targetPerson: EnrolledPerson | null = null) => {
    setTargetPersonForSample(targetPerson);
    setEnrollName(targetPerson ? targetPerson.name : "");
    setEnrollIsSuspect(targetPerson ? (targetPerson.is_suspect ?? false) : true);
    setEnrollThreatLevel(targetPerson ? (targetPerson.threat_level || "HIGH") : "HIGH");
    setEnrollCategory(targetPerson ? (targetPerson.category || "Wanted / BOLO") : "Wanted / BOLO");
    setEnrollNotes(targetPerson ? (targetPerson.notes || "") : "");
    setEnrollPreview(null);
    setEnrollFile(null);
    setEnrollError(null);
    setEnrollSuccess(null);
    setIsEnrollModalOpen(true);
  };

  const closeEnrollModal = () => {
    stopWebcam();
    setIsEnrollModalOpen(false);
    setTargetPersonForSample(null);
    setEnrollIsSuspect(true);
    setEnrollThreatLevel("HIGH");
    setEnrollCategory("Wanted / BOLO");
    setEnrollNotes("");
    setEnrollPreview(null);
    setEnrollFile(null);
    setEnrollError(null);
    setEnrollSuccess(null);
  };

  const handleScanSubmit = async (file: File) => {
    setScanFile(file);
    const previewUrl = URL.createObjectURL(file);
    setScanPreview(previewUrl);
    setIsScanning(true);
    setScanError(null);
    setScanResult(null);

    try {
      const res = await api.scanFaceImage(file, minMatchThreshold, 40);
      setScanResult(res);
      await fetchEvents();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setScanError(msg || "Image analysis failed");
    } finally {
      setIsScanning(false);
    }
  };

  const suspectCount = enrolledPersons.filter((p) => p.is_suspect).length;
  const authorizedCount = enrolledPersons.filter((p) => !p.is_suspect).length;

  const filteredPersons = enrolledPersons.filter((p) => {
    const q = searchQuery.toLowerCase().trim();
    const matchesSearch =
      !q ||
      p.name.toLowerCase().includes(q) ||
      p.id.toLowerCase().includes(q) ||
      (p.notes && p.notes.toLowerCase().includes(q)) ||
      (p.category && p.category.toLowerCase().includes(q));

    if (!matchesSearch) return false;
    if (directoryFilter === "suspects") return !!p.is_suspect;
    if (directoryFilter === "authorized") return !p.is_suspect;
    return true;
  });

  return (
    <div className="space-y-6 pb-16">
      {/* Hidden elements for webcam frame capture and streaming */}
      <canvas ref={canvasRef} className="hidden" />
      <video ref={clientVideoRef} className="hidden" playsInline muted autoPlay />
      <canvas ref={clientCanvasRef} className="hidden" />

      {/* TOP COMMAND HEADER */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 md:p-6 shadow-xs flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#143724] text-white flex items-center justify-center shadow-xs">
              <ScanFace className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl md:text-2xl font-black tracking-tight text-slate-900">
                  Facial Recognition
                </h1>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-[11px] font-semibold text-emerald-800">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      engineStatus?.status === "ready"
                        ? "bg-emerald-500 animate-pulse"
                        : "bg-rose-500"
                    }`}
                  />
                  {engineStatus?.status === "ready" ? "Engine Active" : "Engine Offline"}
                </span>
                <span className="hidden sm:inline-flex items-center px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-600 text-[10px] font-mono font-semibold">
                  YuNet + SFace
                </span>
              </div>
              <p className="text-xs md:text-sm font-medium text-slate-500 mt-0.5">
                YuNet High-Res Face Detector & SFace 128-D Cosine Matching with Temporal Smoothing
              </p>
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={refreshAllData}
            disabled={isLoading}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors shadow-xs cursor-pointer disabled:opacity-50"
            title="Sync System Data"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${isLoading ? "animate-spin" : ""}`} />
            <span>Sync</span>
          </button>

          <button
            type="button"
            onClick={() => openEnrollModal(null)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold transition-all shadow-xs hover:shadow-emerald-900/10 cursor-pointer"
          >
            <UserPlus className="w-4 h-4 text-emerald-400" />
            <span>Enroll Identity</span>
          </button>
        </div>
      </div>

      {/* METRICS RIBBON (4 CARDS) */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1: Engine Status */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Biometric Engine
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5 flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  engineStatus?.status === "ready" ? "bg-emerald-500 animate-pulse" : "bg-rose-500"
                }`}
              />
              {engineStatus?.status === "ready" ? "Active" : "Offline"}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              YuNet & SFace 128-D
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-700">
            <Cpu className="w-5 h-5" />
          </div>
        </div>

        {/* Card 2: Enrolled Roster */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Enrolled Identities
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5 flex items-baseline gap-2">
              <span>{enrolledPersons.length}</span>
              {suspectCount > 0 && (
                <span className="text-xs font-bold text-rose-600 bg-rose-50 px-2 py-0.5 rounded-full border border-rose-200 flex items-center gap-1">
                  <ShieldAlert className="w-3 h-3" />
                  {suspectCount} Wanted
                </span>
              )}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              {authorizedCount} Authorized Personnel
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-700">
            <UserCheck className="w-5 h-5" />
          </div>
        </div>

        {/* Card 3: Unknown Person Detections */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Unknown Detections
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5 flex items-baseline gap-2">
              <span>{unknownStats?.total_unknowns ?? unknowns.length}</span>
              {(unknownStats?.unreviewed_count ?? unknowns.filter((u) => u.status === "unreviewed").length) > 0 && (
                <span className="text-xs font-bold text-amber-700 bg-amber-50 px-2 py-0.5 rounded-full border border-amber-200 flex items-center gap-1">
                  <UserX className="w-3 h-3 text-amber-600" />
                  {unknownStats?.unreviewed_count ?? unknowns.filter((u) => u.status === "unreviewed").length} New
                </span>
              )}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              {(unknownStats?.today_count ?? 0)} Detected Today Across Feeds
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center text-amber-700">
            <UserX className="w-5 h-5" />
          </div>
        </div>

        {/* Card 4: Match Sensitivity */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Match Sensitivity
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5 font-mono">
              {minMatchThreshold.toFixed(2)}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              Cosine similarity cutoff
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-emerald-700">
            <Sparkles className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* NAVIGATION TABS */}
      <div className="flex items-center gap-2 border-b border-slate-200 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveTab("stream")}
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer shrink-0 ${
            activeTab === "stream"
              ? "bg-white text-emerald-800 border-t-2 border-emerald-600 border-x border-slate-200 shadow-xs"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <Video className="w-4 h-4" />
          <span>Live Recognition HUD</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("directory")}
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer shrink-0 ${
            activeTab === "directory"
              ? "bg-white text-emerald-800 border-t-2 border-emerald-600 border-x border-slate-200 shadow-xs"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <UserCheck className="w-4 h-4" />
          <span>
            Personnel Roster ({enrolledPersons.length})
            {suspectCount > 0 && (
              <span className="ml-1.5 px-1.5 py-0.5 rounded-full bg-rose-600 text-white text-[10px] font-bold">
                {suspectCount}
              </span>
            )}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("unknowns")}
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer shrink-0 ${
            activeTab === "unknowns"
              ? "bg-white text-amber-800 border-t-2 border-amber-600 border-x border-slate-200 shadow-xs"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <UserX className="w-4 h-4 text-amber-600" />
          <span>
            Unknown Persons ({unknowns.length})
            {(unknownStats?.unreviewed_count ?? unknowns.filter((u) => u.status === "unreviewed").length) > 0 && (
              <span className="ml-1.5 px-1.5 py-0.5 rounded-full bg-amber-500 text-white text-[10px] font-bold">
                {unknownStats?.unreviewed_count ?? unknowns.filter((u) => u.status === "unreviewed").length} New
              </span>
            )}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab("scan")}
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer shrink-0 ${
            activeTab === "scan"
              ? "bg-white text-emerald-800 border-t-2 border-emerald-600 border-x border-slate-200 shadow-xs"
              : "text-slate-600 hover:text-slate-900"
          }`}
        >
          <Upload className="w-4 h-4" />
          <span>Photo Forensic Scanner</span>
        </button>
      </div>

      {/* TAB 1: LIVE RECOGNITION HUD */}
      {activeTab === "stream" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Video Screen Container (2 cols) */}
          <div className="lg:col-span-2 space-y-4">
            <div
              ref={streamContainerRef}
              className={`bg-slate-900 overflow-hidden relative transition-all duration-200 ${
                isStreamFullscreen
                  ? "fixed inset-0 z-[9999] w-screen h-screen flex flex-col rounded-none border-none shadow-2xl"
                  : "rounded-2xl border border-slate-800 shadow-lg"
              }`}
            >
              {/* Top Stream Control Bar */}
              <div className="bg-slate-950/80 backdrop-blur-md px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 shrink-0">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                  <span className="text-xs font-bold font-mono text-emerald-400">BIOMETRIC STREAM</span>
                  <span className="text-xs text-slate-500">|</span>
                  <span className="text-xs text-slate-300 font-mono">{streamSource.toUpperCase()}</span>
                  {isStreamFullscreen && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-500/40">
                      FULLSCREEN FOCUS
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <select
                    value={streamSource}
                    onChange={(e) => handleSourceChange(e.target.value)}
                    className="bg-slate-800 text-slate-200 text-xs px-2.5 py-1 rounded-lg border border-slate-700 outline-none cursor-pointer"
                  >
                    <option value="browser">This Device (Browser Webcam)</option>
                    <option value="sample">Demo Surveillance Feed (Server)</option>
                    <option value="0">Server Host Camera 0</option>
                    <option value="1">Server Host Camera 1</option>
                    <option value="custom">Custom RTSP / IP URL...</option>
                  </select>

                  {streamSource === "custom" && (
                    <div className="flex items-center gap-1.5">
                      <input
                        type="text"
                        placeholder="rtsp://... or http://..."
                        value={customStreamUrl}
                        onChange={(e) => setCustomStreamUrl(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") setStreamKey(Date.now());
                        }}
                        className="bg-slate-800 text-slate-200 text-xs px-2.5 py-1 rounded-lg border border-slate-700 w-36 sm:w-48 outline-none"
                      />
                      <button
                        type="button"
                        onClick={() => setStreamKey(Date.now())}
                        className="px-2.5 py-1 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-bold cursor-pointer"
                      >
                        Go
                      </button>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => {
                      if (streamSource === "browser") {
                        startClientStreaming();
                      } else {
                        setStreamKey(Date.now());
                      }
                    }}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition-colors cursor-pointer"
                    title="Reconnect"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>

                  <button
                    type="button"
                    onClick={handleToggleStream}
                    className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition-colors cursor-pointer"
                    title={isStreamPlaying ? "Pause Stream" : "Resume Stream"}
                  >
                    {isStreamPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  </button>

                  {/* Fullscreen Toggle Button */}
                  <button
                    type="button"
                    onClick={toggleStreamFullscreen}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                      isStreamFullscreen
                        ? "bg-rose-600 hover:bg-rose-500 text-white"
                        : "bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white"
                    }`}
                    title={isStreamFullscreen ? "Exit Fullscreen (Esc)" : "View in Fullscreen"}
                  >
                    {isStreamFullscreen ? (
                      <>
                        <Minimize2 className="w-3.5 h-3.5" />
                        <span>Exit Fullscreen</span>
                      </>
                    ) : (
                      <>
                        <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />
                        <span>Fullscreen</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Stream Feed Viewport */}
              <div
                onDoubleClick={toggleStreamFullscreen}
                className={`relative w-full bg-black flex items-center justify-center overflow-hidden ${
                  isStreamFullscreen ? "flex-1 min-h-0 h-full" : "aspect-video"
                }`}
              >
                {isStreamPlaying ? (
                  <>
                    {streamSource === "browser" ? (
                      <div className="relative w-full h-full flex items-center justify-center">
                        <canvas
                          ref={clientDisplayCanvasRef}
                          className={`w-full h-full object-contain ${hasClientFeed ? "block" : "hidden"}`}
                        />
                        {!hasClientFeed && (
                          <div className="flex flex-col items-center gap-3 text-slate-400 p-6 text-center">
                            <Camera className="w-10 h-10 text-emerald-400 animate-pulse" />
                            <p className="text-sm font-medium text-slate-200">Connecting Local Webcam to AI Engine...</p>
                            <p className="text-xs text-slate-400 max-w-sm">
                              Streaming browser webcam frames to {BACKEND_BASE_URL} for real-time YuNet & SFace biometric identification.
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img
                        ref={streamImgRef}
                        key={streamKey}
                        src={`${BACKEND_BASE_URL}/api/faces/stream?source=${encodeURIComponent(
                          streamSource === "custom" ? customStreamUrl || "sample" : streamSource
                        )}&fps=20${streamKey > 0 ? `&t=${streamKey}` : ""}`}
                        alt="Live Facial Recognition Stream"
                        className="w-full h-full object-contain"
                        onLoad={() => setStreamError(false)}
                        onError={() => {
                          setStreamError(true);
                        }}
                        suppressHydrationWarning
                      />
                    )}

                    {streamError && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/85 backdrop-blur-sm z-20 p-6 text-center gap-3">
                        <AlertTriangle className="w-10 h-10 text-amber-400 animate-bounce" />
                        <p className="text-sm text-slate-200 font-medium">
                          Unable to connect to feed ({streamSource === "browser" ? "This Device Webcam" : (streamSource === "0" ? "Server Host Camera 0" : streamSource)})
                        </p>
                        <p className="text-xs text-slate-400 max-w-sm">
                          {streamSource === "0" || streamSource === "1"
                            ? `The backend at ${BACKEND_BASE_URL} does not have a physical USB camera connected. Switch to "This Device (Browser Webcam)" to use your local camera.`
                            : "Please check camera permissions or ensure the RTSP/video source is online."}
                        </p>
                        <div className="flex flex-wrap items-center justify-center gap-3 mt-2">
                          <button
                            type="button"
                            onClick={() => {
                              setStreamError(false);
                              if (streamSource === "browser") {
                                startClientStreaming();
                              } else {
                                setStreamKey(Date.now());
                              }
                            }}
                            className="px-4 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white font-bold text-xs uppercase tracking-wider transition cursor-pointer"
                          >
                            Retry Reconnect
                          </button>
                          <button
                            type="button"
                            onClick={() => handleSourceChange("browser")}
                            className="px-4 py-2 rounded-xl bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-semibold hover:bg-emerald-500/30 transition cursor-pointer"
                          >
                            Use This Device&apos;s Webcam
                          </button>
                          <button
                            type="button"
                            onClick={() => handleSourceChange("sample")}
                            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition cursor-pointer"
                          >
                            Switch to Demo Video
                          </button>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col items-center gap-3 text-slate-400">
                    <Video className="w-10 h-10 text-slate-500" />
                    <p className="text-xs">Stream paused</p>
                    <button
                      type="button"
                      onClick={handleToggleStream}
                      className="px-4 py-1.5 rounded-lg bg-[#143724] text-white font-bold text-xs uppercase tracking-wider hover:bg-[#1a472f] cursor-pointer"
                    >
                      Resume Stream
                    </button>
                  </div>
                )}
              </div>

              {/* Bottom Stream Status Bar */}
              <div className="bg-slate-950/80 backdrop-blur-md px-4 py-2.5 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 font-mono">
                <span className="hidden sm:inline">SMOOTHING: TEMPORAL 5-FRAME</span>
                <span className="text-emerald-400 font-bold">YuNet Detector + SFace 128-D Cosine</span>
              </div>
            </div>

            {/* Server Hardware Camera Notice */}
            {(streamSource === "0" || streamSource === "1") && (
              <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs text-amber-900">
                <div className="flex items-start sm:items-center gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5 sm:mt-0" />
                  <div>
                    <span className="font-bold text-amber-950">Server Hardware Camera Mode:</span>{" "}
                    The backend at <code className="px-1.5 py-0.5 rounded bg-amber-100 font-mono text-amber-900">{BACKEND_BASE_URL}</code> is querying a physical USB webcam attached directly to that server machine. To stream your computer&apos;s webcam, switch to <strong>&quot;This Device (Browser Webcam)&quot;</strong>.
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleSourceChange("browser")}
                  className="px-3.5 py-1.5 rounded-lg bg-[#143724] hover:bg-[#1a472f] text-white font-bold text-xs whitespace-nowrap transition cursor-pointer self-start sm:self-center"
                >
                  Use This Device
                </button>
              </div>
            )}

            {/* Tactical Guidelines Bar (Matches ANPR style) */}
            <div className="bg-white rounded-xl border border-slate-200/90 p-3.5 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-600 shadow-xs">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
                <span>Green Reticle: Verified Identity Match</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm bg-amber-500" />
                <span>Amber Reticle: Unregistered / Unknown Face</span>
              </div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm bg-cyan-600" />
                <span>128-D Cosine Metric Smoothing</span>
              </div>
            </div>
          </div>

          {/* Side Panel: Live Recognition Log (Matches ANPR side panel style) */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col h-[580px]">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-3">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-800" />
                <h3 className="text-sm font-bold text-slate-900">Live Recognition Log</h3>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-semibold">
                Realtime
              </span>
            </div>

            {/* Scrollable Event List */}
            <div className="flex-1 overflow-y-auto space-y-2.5 pr-1">
              {recentEvents.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-400">
                  <UserX className="w-8 h-8 mb-2 opacity-40" />
                  <p className="text-xs">No face detections logged yet.</p>
                  <p className="text-[11px] text-slate-400 mt-1">Faces appearing in the stream will automatically register here.</p>
                </div>
              ) : (
                recentEvents.map((evt) => (
                  <div
                    key={evt.id}
                    className={`p-3 rounded-xl border transition-all ${
                      evt.is_threat
                        ? "bg-rose-50/80 border-rose-300 shadow-xs hover:border-rose-400"
                        : evt.is_known
                        ? "bg-emerald-50/60 border-emerald-200/80 hover:border-emerald-300"
                        : "bg-amber-50/60 border-amber-200/80 hover:border-amber-300"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div
                          className={`p-1.5 rounded-lg shrink-0 ${
                            evt.is_threat
                              ? "bg-rose-600 text-white"
                              : evt.is_known
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-amber-100 text-amber-800"
                          }`}
                        >
                          {evt.is_threat ? (
                            <ShieldAlert className="w-4 h-4 animate-pulse" />
                          ) : evt.is_known ? (
                            <UserCheck className="w-4 h-4" />
                          ) : (
                            <UserX className="w-4 h-4" />
                          )}
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <p className="text-xs font-bold text-slate-900 truncate leading-tight">
                              {evt.name}
                            </p>
                            {evt.is_threat && (
                              <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-rose-600 text-white shrink-0">
                                {evt.threat_level || "WANTED"}
                              </span>
                            )}
                          </div>
                          <p className="text-[10px] font-mono text-slate-500 mt-0.5">
                            {evt.is_threat
                              ? `🚨 High Threat Target (${evt.calibrated_conf}%)`
                              : evt.is_known
                              ? `Match: ${evt.calibrated_conf}%`
                              : "Unregistered Face"}
                          </p>
                        </div>
                      </div>
                      <span className="text-[10px] font-mono text-slate-400 shrink-0">{evt.timestamp}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: PERSONNEL DIRECTORY */}
      {activeTab === "directory" && (
        <div className="space-y-6">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3 flex-1">
              <div className="relative flex-1 min-w-[220px] max-w-md">
                <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  placeholder="Search name, category, FIR notes..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 text-xs rounded-xl border border-slate-200 bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700 shadow-xs text-slate-900 placeholder:text-slate-400"
                />
              </div>

              {/* Filter Pills */}
              <div className="flex items-center gap-1 p-1 rounded-xl bg-slate-100 border border-slate-200 text-xs font-semibold">
                <button
                  type="button"
                  onClick={() => setDirectoryFilter("all")}
                  className={`px-3 py-1.5 rounded-lg transition cursor-pointer ${
                    directoryFilter === "all"
                      ? "bg-white text-slate-900 shadow-xs"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  All ({enrolledPersons.length})
                </button>
                <button
                  type="button"
                  onClick={() => setDirectoryFilter("suspects")}
                  className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 cursor-pointer ${
                    directoryFilter === "suspects"
                      ? "bg-rose-600 text-white shadow-xs"
                      : "text-rose-700 hover:bg-rose-50"
                  }`}
                >
                  <ShieldAlert className="w-3.5 h-3.5" />
                  <span>Wanted Suspects ({suspectCount})</span>
                </button>
                <button
                  type="button"
                  onClick={() => setDirectoryFilter("authorized")}
                  className={`px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 cursor-pointer ${
                    directoryFilter === "authorized"
                      ? "bg-emerald-700 text-white shadow-xs"
                      : "text-emerald-700 hover:bg-emerald-50"
                  }`}
                >
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>Authorized ({authorizedCount})</span>
                </button>
              </div>
            </div>

            <button
              type="button"
              onClick={() => openEnrollModal(null)}
              className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold transition-all shadow-xs cursor-pointer self-start lg:self-center"
            >
              <Plus className="w-4 h-4 text-emerald-400" />
              <span>Register New Identity</span>
            </button>
          </div>

          {filteredPersons.length === 0 ? (
            <div className="bg-white rounded-2xl border border-dashed border-slate-200 p-16 text-center flex flex-col items-center gap-3 shadow-xs">
              <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center text-slate-400">
                <UserX className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-bold text-slate-900">No personnel found</h3>
              <p className="text-xs text-slate-500 max-w-sm">
                {directoryFilter === "suspects"
                  ? "No suspects or BOLO profiles match your current search criteria."
                  : directoryFilter === "authorized"
                  ? "No authorized personnel profiles match your current search criteria."
                  : "Enroll known operators, authorized personnel, or wanted suspects to monitor feeds."}
              </p>
              <button
                type="button"
                onClick={() => openEnrollModal(null)}
                className="mt-2 px-4 py-2 rounded-xl bg-[#143724] text-white font-bold text-xs uppercase tracking-wider hover:bg-[#1a472f] cursor-pointer"
              >
                Enroll First Person
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-5">
              {filteredPersons.map((person) => (
                <div
                  key={person.id}
                  className={`group relative flex flex-col rounded-2xl bg-white border overflow-hidden shadow-xs hover:shadow-md transition-all duration-200 ${
                    person.is_suspect
                      ? "border-rose-200 hover:border-rose-400 ring-1 ring-rose-200/50"
                      : "border-slate-200/90 hover:border-emerald-600/40"
                  }`}
                >
                  <div className="relative aspect-square w-full bg-slate-100 overflow-hidden flex items-center justify-center">
                    <img
                      src={`${BACKEND_BASE_URL}${person.image_url}`}
                      alt={person.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                      onError={(e) => {
                        (e.target as HTMLElement).style.display = "none";
                      }}
                    />

                    {/* Top status badges */}
                    <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5">
                      {person.is_suspect ? (
                        <span className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-600 text-white text-[10px] font-bold shadow-xs tracking-wider uppercase">
                          <ShieldAlert className="w-3 h-3" />
                          {person.threat_level || "HIGH"} THREAT
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-700/90 text-white text-[10px] font-bold shadow-xs">
                          <ShieldCheck className="w-3 h-3" />
                          AUTHORIZED
                        </span>
                      )}
                    </div>

                    <div className="absolute top-2.5 right-2.5 flex items-center gap-1 px-2 py-0.5 rounded-md bg-white/95 border border-slate-200 text-slate-700 text-[10px] font-mono font-bold shadow-2xs">
                      <span>{person.sample_count} sample{person.sample_count > 1 ? "s" : ""}</span>
                    </div>
                  </div>

                  <div className="p-4 flex flex-col justify-between flex-1 gap-3">
                    <div>
                      <div className="flex items-center justify-between gap-2">
                        <h3
                          className={`font-bold text-sm transition-colors ${
                            person.is_suspect
                              ? "text-rose-950 group-hover:text-rose-700"
                              : "text-slate-900 group-hover:text-emerald-800"
                          }`}
                        >
                          {person.name}
                        </h3>
                        <span className="text-[10px] font-medium px-2 py-0.5 rounded-md bg-slate-100 text-slate-600 truncate max-w-[120px]">
                          {person.category || (person.is_suspect ? "Wanted / BOLO" : "Personnel")}
                        </span>
                      </div>
                      <p className="text-[11px] font-mono text-slate-400 mt-0.5 truncate">ID: {person.id}</p>
                      {person.notes && (
                        <p className="text-[11px] text-slate-600 mt-1 line-clamp-1 italic bg-slate-50 px-2 py-0.5 rounded border border-slate-100">
                          &quot;{person.notes}&quot;
                        </p>
                      )}
                      <p className="text-[11px] text-slate-500 mt-1.5 flex items-center gap-1">
                        <Clock className="w-3 h-3 text-slate-400" />
                        {person.created_at || "Registered"}
                      </p>
                    </div>

                    <div className="flex items-center justify-between pt-3 border-t border-slate-100 gap-2">
                      <button
                        type="button"
                        onClick={() => handleToggleSuspect(person)}
                        className={`text-[11px] font-semibold px-2 py-1 rounded-lg border transition-colors cursor-pointer ${
                          person.is_suspect
                            ? "border-emerald-200 text-emerald-700 hover:bg-emerald-50"
                            : "border-rose-200 text-rose-700 hover:bg-rose-50"
                        }`}
                        title={person.is_suspect ? "Mark as Authorized Personnel" : "Mark as Wanted Suspect"}
                      >
                        {person.is_suspect ? "Mark Authorized" : "Flag Suspect"}
                      </button>

                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => openEnrollModal(person)}
                          className="p-1.5 rounded-lg text-emerald-700 hover:bg-emerald-50 transition-colors cursor-pointer"
                          title="Add Angle Sample"
                        >
                          <Plus className="w-3.5 h-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeletePerson(person)}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                          title="Delete Identity"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* TAB: UNKNOWN PERSON ARCHIVAL LOG */}
      {activeTab === "unknowns" && (
        <div className="flex flex-col gap-6">
          {/* Header Controls Banner */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-900">
                  Unidentified Personnel Log
                </h2>
                <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-amber-50 text-amber-800 border border-amber-200">
                  {unknowns.length} Captured
                </span>
                {unknownStats?.today_count ? (
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-emerald-50 text-emerald-800 border border-emerald-200">
                    {unknownStats.today_count} Today
                  </span>
                ) : null}
              </div>
              <p className="text-xs text-slate-500 mt-0.5">
                Automatic optical snapshots & real-time timestamps recorded when un-enrolled faces are detected across camera feeds.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              {/* View Mode Toggle */}
              <div className="inline-flex rounded-xl border border-slate-200 bg-slate-50 p-0.5">
                <button
                  type="button"
                  onClick={() => setUnknownViewMode("grid")}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                    unknownViewMode === "grid"
                      ? "bg-white text-slate-900 shadow-xs"
                      : "text-slate-500 hover:text-slate-900"
                  }`}
                >
                  <Grid className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Grid</span>
                </button>
                <button
                  type="button"
                  onClick={() => setUnknownViewMode("table")}
                  className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${
                    unknownViewMode === "table"
                      ? "bg-white text-slate-900 shadow-xs"
                      : "text-slate-500 hover:text-slate-900"
                  }`}
                >
                  <List className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">Table</span>
                </button>
              </div>

              {/* Sync Button */}
              <button
                type="button"
                onClick={fetchUnknowns}
                disabled={isLoadingUnknowns}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors cursor-pointer shadow-xs disabled:opacity-50"
                title="Refresh Records"
              >
                <RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${isLoadingUnknowns ? "animate-spin" : ""}`} />
                <span>Refresh</span>
              </button>

              {/* Deduplicate / Merge Button */}
              {unknowns.length > 1 && (
                <button
                  type="button"
                  onClick={handleConsolidateDuplicates}
                  disabled={isConsolidating}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-amber-200 bg-amber-50/70 text-xs font-semibold text-amber-800 hover:bg-amber-100 transition-colors cursor-pointer shadow-xs disabled:opacity-50"
                  title="Biometrically cluster and merge duplicate detections"
                >
                  <Sparkles className={`w-3.5 h-3.5 text-amber-600 ${isConsolidating ? "animate-spin" : ""}`} />
                  <span>{isConsolidating ? "Merging..." : "Merge Duplicates"}</span>
                </button>
              )}

              {/* Clear All */}
              {unknowns.length > 0 && (
                <button
                  type="button"
                  onClick={handleClearAllUnknowns}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-rose-200 bg-rose-50/60 text-xs font-semibold text-rose-700 hover:bg-rose-100/80 transition-colors cursor-pointer shadow-xs"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear All</span>
                </button>
              )}
            </div>
          </div>

          {/* Search & Filter Toolbar */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex flex-col md:flex-row gap-3">
            {/* Search Input */}
            <div className="relative flex-1">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search by Unknown ID (e.g. UNK-), camera name, or notes..."
                value={unknownSearch}
                onChange={(e) => setUnknownSearch(e.target.value)}
                className="w-full pl-9 pr-4 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/60 text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-amber-600 focus:ring-1 focus:ring-amber-600 outline-none transition"
              />
              {unknownSearch && (
                <button
                  type="button"
                  onClick={() => setUnknownSearch("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>

            {/* Camera Filter */}
            <div className="flex items-center gap-2">
              <Camera className="w-4 h-4 text-slate-400 shrink-0" />
              <select
                value={unknownCameraFilter}
                onChange={(e) => setUnknownCameraFilter(e.target.value)}
                className="px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/60 text-slate-700 font-semibold outline-none focus:bg-white cursor-pointer"
              >
                <option value="all">All Cameras ({unknowns.length})</option>
                {Array.from(new Set(unknowns.map((u) => u.camera_name || u.camera_id).filter(Boolean))).map((camName) => (
                  <option key={camName} value={camName}>
                    {camName}
                  </option>
                ))}
              </select>
            </div>

            {/* Status Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-400 shrink-0" />
              <select
                value={unknownStatusFilter}
                onChange={(e) => setUnknownStatusFilter(e.target.value)}
                className="px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/60 text-slate-700 font-semibold outline-none focus:bg-white cursor-pointer"
              >
                <option value="all">All Statuses</option>
                <option value="unreviewed">Pending Review</option>
                <option value="enrolled">Enrolled</option>
                <option value="flagged">Flagged</option>
              </select>
            </div>
          </div>

          {/* Records Display */}
          {(() => {
            const filteredUnknowns = unknowns.filter((u) => {
              const q = unknownSearch.toLowerCase().trim();
              const matchesSearch =
                !q ||
                u.id.toLowerCase().includes(q) ||
                (u.camera_name && u.camera_name.toLowerCase().includes(q)) ||
                (u.camera_id && u.camera_id.toLowerCase().includes(q)) ||
                (u.notes && u.notes.toLowerCase().includes(q)) ||
                (u.enrolled_name && u.enrolled_name.toLowerCase().includes(q));

              const matchesCam =
                unknownCameraFilter === "all" ||
                u.camera_name === unknownCameraFilter ||
                u.camera_id === unknownCameraFilter;

              const matchesStatus =
                unknownStatusFilter === "all" || u.status === unknownStatusFilter;

              return matchesSearch && matchesCam && matchesStatus;
            });

            if (filteredUnknowns.length === 0) {
              return (
                <div className="bg-white rounded-2xl border border-slate-200/90 p-12 text-center shadow-xs flex flex-col items-center justify-center">
                  <div className="w-16 h-16 rounded-2xl bg-amber-50 border border-amber-100 flex items-center justify-center text-amber-600 mb-4 shadow-inner">
                    <UserX className="w-8 h-8" />
                  </div>
                  <h3 className="text-base font-bold text-slate-900">
                    {unknowns.length === 0 ? "No Unknown Persons Detected Yet" : "No Matching Records Found"}
                  </h3>
                  <p className="text-xs text-slate-500 max-w-md mt-1">
                    {unknowns.length === 0
                      ? "When an unrecognized individual walks in front of any camera, their ID, real-time timestamp, and photograph are automatically saved here."
                      : "Try clearing your search query or camera filter to view other captured unidentified individuals."}
                  </p>
                  {unknownSearch && (
                    <button
                      type="button"
                      onClick={() => {
                        setUnknownSearch("");
                        setUnknownCameraFilter("all");
                        setUnknownStatusFilter("all");
                      }}
                      className="mt-4 px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-xs font-semibold text-slate-700 transition cursor-pointer"
                    >
                      Clear Filters
                    </button>
                  )}
                </div>
              );
            }

            if (unknownViewMode === "grid") {
              return (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {filteredUnknowns.map((u) => {
                    const photoSrc = `${BACKEND_BASE_URL}${u.photo_url}`;
                    return (
                      <div
                        key={u.id}
                        className="bg-white rounded-2xl border border-slate-200/90 shadow-xs hover:shadow-md transition-all flex flex-col overflow-hidden group hover:border-amber-300"
                      >
                        {/* Photo Container */}
                        <div className="relative aspect-square bg-slate-950 flex items-center justify-center overflow-hidden">
                          <img
                            src={photoSrc}
                            alt={`Detected Unknown ${u.id}`}
                            className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src =
                                "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='100' height='100' fill='%2364748b' viewBox='0 0 24 24'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";
                            }}
                          />

                          {/* Quick Inspect Button Overlay */}
                          <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                            <button
                              type="button"
                              onClick={() => setSelectedUnknownForInspect(u)}
                              className="px-3 py-1.5 rounded-xl bg-white/90 hover:bg-white text-slate-900 text-xs font-bold shadow-lg flex items-center gap-1.5 cursor-pointer backdrop-blur-xs transition"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              <span>Inspect</span>
                            </button>
                            <a
                              href={photoSrc}
                              download={`unknown_${u.id}.jpg`}
                              target="_blank"
                              rel="noreferrer"
                              className="p-1.5 rounded-xl bg-white/90 hover:bg-white text-slate-900 shadow-lg cursor-pointer backdrop-blur-xs transition"
                              title="Download Photograph"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </a>
                          </div>

                          {/* Top Badges */}
                          <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between gap-1 pointer-events-none">
                            <span className="font-mono font-black text-[11px] px-2 py-0.5 rounded-md bg-slate-950/85 text-amber-300 border border-amber-500/30 backdrop-blur-xs shadow-xs">
                              {u.id}
                            </span>
                            {u.status === "enrolled" ? (
                              <span className="px-2 py-0.5 rounded-md bg-emerald-950/85 text-emerald-300 border border-emerald-500/30 text-[10px] font-bold backdrop-blur-xs">
                                Enrolled
                              </span>
                            ) : u.status === "flagged" ? (
                              <span className="px-2 py-0.5 rounded-md bg-rose-950/85 text-rose-300 border border-rose-500/30 text-[10px] font-bold backdrop-blur-xs">
                                Flagged
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded-md bg-amber-950/85 text-amber-300 border border-amber-500/30 text-[10px] font-bold backdrop-blur-xs">
                                Unreviewed
                              </span>
                            )}
                          </div>

                          {/* Confidence Pill Bottom */}
                          <div className="absolute bottom-2.5 left-2.5 pointer-events-none">
                            <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md bg-slate-900/80 text-white border border-white/10 backdrop-blur-xs">
                              {u.confidence.toFixed(1)}% Conf
                            </span>
                          </div>
                        </div>

                        {/* Details Container */}
                        <div className="p-3.5 flex flex-col gap-2 flex-1 justify-between">
                          <div>
                            {/* Real Time Stamp */}
                            <div className="flex items-center gap-1.5 text-slate-700">
                              <Clock className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                              <span className="font-mono text-xs font-bold tracking-tight">
                                {u.timestamp}
                              </span>
                            </div>

                            {/* Camera Location & Trajectory */}
                            <div className="flex items-center gap-1.5 text-slate-500 text-[11px] mt-1">
                              <Camera className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                              <span className="truncate font-medium" title={u.camera_name || u.camera_id}>
                                {u.camera_name || u.camera_id}
                              </span>
                              {u.cameras_seen && u.cameras_seen.length > 1 && (
                                <span
                                  className="ml-auto px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 font-mono text-[9px] font-bold border border-blue-200"
                                  title={`Observed on feeds: ${u.cameras_seen.join(", ")}`}
                                >
                                  {u.cameras_seen.length} Cams
                                </span>
                              )}
                            </div>

                            {/* Occurrence & Last Seen */}
                            <div className="flex items-center justify-between text-[10px] text-slate-400 mt-1.5 pt-1.5 border-t border-slate-100 font-mono">
                              <span>Seen: {u.seen_count} time{u.seen_count > 1 ? "s" : ""}</span>
                              {u.last_seen && u.last_seen !== u.timestamp ? (
                                <span className="text-slate-500 text-[9px]" title={`Last seen at ${u.last_seen}`}>
                                  Last: {u.last_seen.split(" ")[1]}
                                </span>
                              ) : null}
                              {u.enrolled_name && (
                                <span className="text-emerald-700 font-bold truncate max-w-[120px]">
                                  As: {u.enrolled_name}
                                </span>
                              )}
                            </div>

                            {u.notes && (
                              <p className="text-[10px] text-slate-600 mt-1 line-clamp-1 italic bg-slate-50 px-2 py-0.5 rounded border border-slate-100">
                                &quot;{u.notes}&quot;
                              </p>
                            )}
                          </div>

                          {/* Actions */}
                          <div className="flex items-center justify-between pt-2 border-t border-slate-100 gap-1.5">
                            <button
                              type="button"
                              onClick={() => openQuickEnrollModal(u)}
                              className="flex-1 py-1.5 px-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-[11px] font-bold transition flex items-center justify-center gap-1 cursor-pointer shadow-xs"
                              title="Register this person with a name"
                            >
                              <UserPlus className="w-3.5 h-3.5 text-emerald-400" />
                              <span>{u.status === "enrolled" ? "Re-Enroll" : "Enroll Identity"}</span>
                            </button>

                            <button
                              type="button"
                              onClick={() => handleDeleteUnknown(u.id)}
                              className="p-1.5 rounded-xl text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition cursor-pointer"
                              title="Delete Record"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              );
            }

            // Table View Mode
            return (
              <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50/80 border-b border-slate-200 text-slate-500 font-bold uppercase tracking-wider text-[10px]">
                      <tr>
                        <th className="py-3 px-4">Photograph</th>
                        <th className="py-3 px-4">Unknown ID</th>
                        <th className="py-3 px-4">Real Timestamp</th>
                        <th className="py-3 px-4">Camera Feed</th>
                        <th className="py-3 px-4">Confidence</th>
                        <th className="py-3 px-4">Seen</th>
                        <th className="py-3 px-4">Status</th>
                        <th className="py-3 px-4 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {filteredUnknowns.map((u) => {
                        const photoSrc = `${BACKEND_BASE_URL}${u.photo_url}`;
                        return (
                          <tr key={u.id} className="hover:bg-slate-50/60 transition">
                            <td className="py-2.5 px-4">
                              <div
                                onClick={() => setSelectedUnknownForInspect(u)}
                                className="w-12 h-12 rounded-xl bg-slate-950 overflow-hidden cursor-pointer relative group border border-slate-200 shadow-xs"
                              >
                                <img
                                  src={photoSrc}
                                  alt={u.id}
                                  className="w-full h-full object-cover group-hover:scale-110 transition"
                                />
                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center text-white">
                                  <Eye className="w-3.5 h-3.5" />
                                </div>
                              </div>
                            </td>
                            <td className="py-3 px-4 font-mono font-bold text-slate-900">
                              <span className="px-2 py-1 rounded-md bg-slate-100 text-slate-800 border border-slate-200">
                                {u.id}
                              </span>
                            </td>
                            <td className="py-3 px-4 font-mono font-semibold text-slate-700 whitespace-nowrap">
                              {u.timestamp} IST
                            </td>
                            <td className="py-3 px-4 text-slate-600 font-medium whitespace-nowrap">
                              <div>{u.camera_name || u.camera_id}</div>
                              {u.cameras_seen && u.cameras_seen.length > 1 && (
                                <div className="text-[10px] text-blue-600 font-mono mt-0.5" title={u.cameras_seen.join(", ")}>
                                  {u.cameras_seen.length} Cams: {u.cameras_seen.join(" → ")}
                                </div>
                              )}
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-700 font-bold">
                              {u.confidence.toFixed(1)}%
                            </td>
                            <td className="py-3 px-4 font-mono text-slate-500">
                              {u.seen_count}x
                            </td>
                            <td className="py-3 px-4">
                              {u.status === "enrolled" ? (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                                  Enrolled
                                </span>
                              ) : u.status === "flagged" ? (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-50 text-rose-700 border border-rose-200">
                                  Flagged
                                </span>
                              ) : (
                                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-50 text-amber-700 border border-amber-200">
                                  Pending
                                </span>
                              )}
                            </td>
                            <td className="py-3 px-4 text-right">
                              <div className="flex items-center justify-end gap-1.5">
                                <button
                                  type="button"
                                  onClick={() => openQuickEnrollModal(u)}
                                  className="px-2.5 py-1 rounded-lg bg-[#143724] hover:bg-[#1a472f] text-white text-[11px] font-bold transition flex items-center gap-1 cursor-pointer"
                                >
                                  <UserPlus className="w-3 h-3 text-emerald-400" />
                                  <span>Enroll</span>
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setSelectedUnknownForInspect(u)}
                                  className="p-1 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition cursor-pointer"
                                  title="Inspect Photo"
                                >
                                  <Eye className="w-4 h-4" />
                                </button>
                                <button
                                  type="button"
                                  onClick={() => handleDeleteUnknown(u.id)}
                                  className="p-1 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition cursor-pointer"
                                  title="Delete"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* TAB 3: FORENSIC PHOTO SCAN */}
      {activeTab === "scan" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Upload & Controls Card */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs flex flex-col gap-5">
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Upload className="w-4 h-4 text-emerald-800" />
              Upload Photo for Inspection
            </h2>

            <label className="flex flex-col items-center justify-center p-8 rounded-xl border-2 border-dashed border-slate-300 hover:border-emerald-600 cursor-pointer bg-slate-50/60 hover:bg-emerald-50/20 transition-all group">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleScanSubmit(e.target.files[0]);
                  }
                }}
              />
              <Upload className="w-10 h-10 text-slate-400 group-hover:text-emerald-700 transition-colors" />
              <p className="mt-3 text-xs font-bold text-slate-700">Click or Drag Image Here</p>
              <p className="text-[10px] text-slate-400 mt-0.5">Supports JPG, PNG, WEBP</p>
            </label>

            {/* Threshold Slider */}
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-600 font-medium">Match Sensitivity Cutoff:</span>
                <span className="font-mono text-emerald-800 font-bold">
                  {minMatchThreshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min="0.25"
                max="0.65"
                step="0.05"
                value={minMatchThreshold}
                onChange={(e) => setMinMatchThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-700 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400">
                Default 0.40 recommended for calibrated SFace cosine similarity.
              </span>
            </div>

            {scanError && (
              <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs">
                {scanError}
              </div>
            )}
          </div>

          {/* Results Display */}
          <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs flex flex-col gap-5">
            {isScanning ? (
              <div className="p-16 flex flex-col items-center justify-center gap-3 text-slate-500">
                <RefreshCw className="w-8 h-8 animate-spin text-emerald-700" />
                <p className="text-xs font-medium">Extracting YuNet face anchors and computing SFace embeddings...</p>
              </div>
            ) : scanResult ? (
              <div className="flex flex-col gap-5">
                {/* Annotated Output Preview */}
                <div
                  onDoubleClick={() => setIsFaceScanFullscreen(true)}
                  className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-md group cursor-zoom-in"
                >
                  {scanResult.annotated_image ? (
                    <img
                      src={scanResult.annotated_image}
                      alt="Annotated Inspection Result"
                      className="w-full max-h-[480px] object-contain mx-auto"
                    />
                  ) : scanPreview ? (
                    <img
                      src={scanPreview}
                      alt="Source Preview"
                      className="w-full max-h-[480px] object-contain mx-auto"
                    />
                  ) : null}

                  <div className="absolute top-3 left-3 px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-700 text-[11px] font-mono text-emerald-400">
                    FACES DETECTED: {scanResult.face_count} • LATENCY: {scanResult.latency_ms}ms
                  </div>

                  <div className="absolute top-3 right-3 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setIsFaceScanFullscreen(true)}
                      className="px-2.5 py-1 rounded-lg bg-slate-900/90 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-200 hover:text-white flex items-center gap-1.5 backdrop-blur-xs cursor-pointer shadow-md"
                      title="View forensic scan in full screen"
                    >
                      <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />
                      <span>Fullscreen</span>
                    </button>
                  </div>
                </div>

                {/* Detected Identities List */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {scanResult.faces.map((f, idx) => (
                    <div
                      key={idx}
                      className={`p-3.5 rounded-xl border flex items-center justify-between ${
                        f.is_known
                          ? "bg-emerald-50/60 border-emerald-200 text-slate-900"
                          : "bg-amber-50/60 border-amber-200 text-slate-900"
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-2 rounded-xl ${
                            f.is_known
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-amber-100 text-amber-800"
                          }`}
                        >
                          {f.is_known ? <UserCheck className="w-4 h-4" /> : <UserX className="w-4 h-4" />}
                        </div>
                        <div>
                          <p className="font-bold text-xs text-slate-900">{f.name}</p>
                          <p className="text-[11px] font-mono text-slate-500">
                            Confidence: {f.calibrated_conf}%
                          </p>
                          <p className="text-[10px] font-mono text-slate-400">
                            Cosine: {f.match_score} | L2: {f.l2_distance ?? "N/A"}
                          </p>
                        </div>
                      </div>

                      <span
                        className={`px-2.5 py-1 text-[11px] font-bold rounded-lg ${
                          f.is_known
                            ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                            : "bg-amber-100 text-amber-800 border border-amber-300"
                        }`}
                      >
                        {f.is_known ? "VERIFIED" : "UNKNOWN"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-20 rounded-xl bg-slate-50 border border-dashed border-slate-200 text-center text-slate-400 text-xs">
                Upload or drop a photo on the left to scan for registered personnel.
              </div>
            )}
          </div>

          {/* Fullscreen Modal for Face Scan Inspection */}
          {isFaceScanFullscreen && scanResult && (
            <div
              className="fixed inset-0 z-[9999] bg-slate-950/95 flex flex-col p-4 backdrop-blur-md animate-in fade-in"
              onClick={() => setIsFaceScanFullscreen(false)}
            >
              <div
                className="flex items-center justify-between text-white p-2 pb-3 border-b border-slate-800"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center gap-2.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="font-mono text-sm font-bold tracking-wider">
                    BIOMETRIC FORENSIC SCAN • FULLSCREEN
                  </span>
                  <span className="text-xs font-mono text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-500/30">
                    {scanResult.face_count} Face(s) Detected • {scanResult.latency_ms}ms
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setIsFaceScanFullscreen(false)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-600/90 hover:bg-rose-600 text-white text-xs font-bold transition-colors cursor-pointer"
                >
                  <Minimize2 className="w-3.5 h-3.5" />
                  <span>Exit Fullscreen</span>
                  <span className="text-[10px] bg-black/40 px-1 rounded border border-white/20">ESC</span>
                </button>
              </div>
              <div
                className="flex-1 flex items-center justify-center min-h-0 p-4"
                onClick={(e) => e.stopPropagation()}
              >
                <img
                  src={scanResult.annotated_image || scanPreview || ""}
                  alt="Biometric Scan Fullscreen"
                  className="w-full h-full object-contain max-h-full"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* ENROLLMENT MODAL */}
      {isEnrollModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in">
          <div className="relative w-full max-w-lg rounded-2xl bg-white border border-slate-200 shadow-2xl p-6 flex flex-col gap-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800">
                  <UserPlus className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-base">
                    {targetPersonForSample
                      ? `Add Face Sample for ${targetPersonForSample.name}`
                      : "Enroll New Identity"}
                  </h3>
                  <p className="text-xs text-slate-500">
                    Quality requirements: Face &gt; 65x65 px, unblurred, good lighting
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={closeEnrollModal}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Mode Switcher */}
            <div className="flex rounded-xl bg-slate-100 p-1 border border-slate-200 text-xs font-semibold">
              <button
                type="button"
                onClick={() => setEnrollMode("webcam")}
                className={`flex-1 py-2 rounded-lg transition cursor-pointer ${
                  enrollMode === "webcam" ? "bg-white text-emerald-800 shadow-xs" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Direct Webcam Snap
              </button>
              <button
                type="button"
                onClick={() => setEnrollMode("upload")}
                className={`flex-1 py-2 rounded-lg transition cursor-pointer ${
                  enrollMode === "upload" ? "bg-white text-emerald-800 shadow-xs" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Upload Photo File
              </button>
            </div>

            {/* Name Input (hidden if adding sample to existing) */}
            {!targetPersonForSample && (
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-semibold text-slate-700">Personnel Full Name</label>
                <input
                  type="text"
                  placeholder="e.g. Hariom"
                  value={enrollName}
                  onChange={(e) => setEnrollName(e.target.value)}
                  className="px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700 text-slate-900"
                />
              </div>
            )}

            {/* Classification Selector: Wanted Suspect vs Authorized */}
            {!targetPersonForSample && (
              <div className="flex flex-col gap-3 p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 text-rose-600" />
                    Security Classification
                  </label>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setEnrollIsSuspect(true);
                        setEnrollCategory("Wanted / BOLO");
                        setEnrollThreatLevel("HIGH");
                      }}
                      className={`px-3 py-1 text-xs font-bold rounded-lg transition cursor-pointer ${
                        enrollIsSuspect
                          ? "bg-rose-600 text-white shadow-xs"
                          : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-100"
                      }`}
                    >
                      🚨 Wanted Suspect
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setEnrollIsSuspect(false);
                        setEnrollCategory("Authorized Personnel");
                        setEnrollThreatLevel("LOW");
                      }}
                      className={`px-3 py-1 text-xs font-bold rounded-lg transition cursor-pointer ${
                        !enrollIsSuspect
                          ? "bg-emerald-700 text-white shadow-xs"
                          : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-100"
                      }`}
                    >
                      🛡️ Authorized
                    </button>
                  </div>
                </div>

                {/* If Suspect: Threat Level & Category */}
                {enrollIsSuspect ? (
                  <div className="space-y-2.5 pt-2 border-t border-slate-200">
                    <div>
                      <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                        Threat Priority Level
                      </label>
                      <div className="grid grid-cols-4 gap-2">
                        {(["CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map((lvl) => (
                          <button
                            key={lvl}
                            type="button"
                            onClick={() => setEnrollThreatLevel(lvl)}
                            className={`py-1.5 text-xs font-bold rounded-lg border transition cursor-pointer ${
                              enrollThreatLevel === lvl
                                ? lvl === "CRITICAL"
                                  ? "bg-rose-700 border-rose-800 text-white shadow-xs"
                                  : lvl === "HIGH"
                                  ? "bg-rose-600 border-rose-700 text-white shadow-xs"
                                  : lvl === "MEDIUM"
                                  ? "bg-amber-600 border-amber-700 text-white shadow-xs"
                                  : "bg-slate-700 border-slate-800 text-white shadow-xs"
                                : "bg-white border-slate-200 text-slate-700 hover:bg-slate-100"
                            }`}
                          >
                            {lvl}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <div>
                        <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                          Suspect Category
                        </label>
                        <select
                          value={enrollCategory}
                          onChange={(e) => setEnrollCategory(e.target.value)}
                          className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none"
                        >
                          <option value="Wanted / BOLO">Wanted / BOLO</option>
                          <option value="Infiltrator / Poacher">Infiltrator / Poacher</option>
                          <option value="Smuggling Suspect">Smuggling Suspect</option>
                          <option value="Missing Person">Missing Person</option>
                          <option value="Person of Interest">Person of Interest</option>
                        </select>
                      </div>

                      <div>
                        <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                          Case / FIR / Reference ID
                        </label>
                        <input
                          type="text"
                          placeholder="e.g. FIR #402/26 - Sector 4"
                          value={enrollNotes}
                          onChange={(e) => setEnrollNotes(e.target.value)}
                          className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2 border-t border-slate-200">
                    <div>
                      <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                        Authorized Role
                      </label>
                      <select
                        value={enrollCategory}
                        onChange={(e) => setEnrollCategory(e.target.value)}
                        className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none"
                      >
                        <option value="Authorized Personnel">Authorized Personnel</option>
                        <option value="Border Security Force (BSF)">Border Security Force (BSF)</option>
                        <option value="Command Staff">Command Staff</option>
                        <option value="Technician / Operator">Technician / Operator</option>
                        <option value="VIP / Official Visitor">VIP / Official Visitor</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                        Department / Badge / Unit Notes
                      </label>
                      <input
                        type="text"
                        placeholder="e.g. Outpost Beta Team Alpha"
                        value={enrollNotes}
                        onChange={(e) => setEnrollNotes(e.target.value)}
                        className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Mode 1: Webcam Snap */}
            {enrollMode === "webcam" && (
              <div className="flex flex-col gap-3">
                <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 aspect-video flex items-center justify-center">
                  {enrollPreview ? (
                    <img
                      src={enrollPreview}
                      alt="Captured Face Preview"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />
                  )}

                  {enrollPreview && (
                    <button
                      type="button"
                      onClick={() => setEnrollPreview(null)}
                      className="absolute top-3 right-3 px-2.5 py-1 rounded-lg bg-slate-900/90 text-xs text-slate-200 border border-slate-700 hover:bg-slate-800 cursor-pointer"
                    >
                      Retake
                    </button>
                  )}
                </div>

                {webcamError && (
                  <p className="text-xs text-rose-600">{webcamError}</p>
                )}

                {!enrollPreview && (
                  <button
                    type="button"
                    onClick={captureWebcamSnapshot}
                    disabled={!isWebcamActive}
                    className="py-2.5 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 disabled:opacity-50 cursor-pointer shadow-xs"
                  >
                    <Camera className="w-4 h-4 text-emerald-400" />
                    Snap Face Photo
                  </button>
                )}
              </div>
            )}

            {/* Mode 2: File Upload */}
            {enrollMode === "upload" && (
              <div className="flex flex-col gap-3">
                <label className="flex flex-col items-center justify-center p-8 rounded-xl border-2 border-dashed border-slate-300 hover:border-emerald-600 cursor-pointer bg-slate-50/50 hover:bg-emerald-50/20 transition-all">
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        const file = e.target.files[0];
                        setEnrollFile(file);
                        setEnrollPreview(URL.createObjectURL(file));
                      }
                    }}
                  />
                  {enrollPreview ? (
                    <img
                      src={enrollPreview}
                      alt="Upload Preview"
                      className="max-h-48 rounded-xl object-contain"
                    />
                  ) : (
                    <>
                      <Upload className="w-8 h-8 text-slate-400" />
                      <p className="mt-2 text-xs text-slate-700 font-semibold">Select Clear Face Photo</p>
                      <p className="text-[10px] text-slate-400">JPG, PNG up to 10MB</p>
                    </>
                  )}
                </label>
              </div>
            )}

            {enrollError && (
              <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs">
                {enrollError}
              </div>
            )}

            {enrollSuccess && (
              <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2 font-medium">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                {enrollSuccess}
              </div>
            )}

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={closeEnrollModal}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleEnrollSubmit}
                disabled={isEnrolling || !enrollPreview}
                className="px-5 py-2.5 rounded-xl bg-[#143724] hover:bg-[#102d1d] text-white font-bold text-xs uppercase tracking-wider shadow-xs disabled:opacity-50 transition cursor-pointer"
              >
                {isEnrolling ? "Verifying & Enrolling..." : "Complete Enrollment"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 2: INSPECT UNKNOWN PERSON DOSSIER */}
      {selectedUnknownForInspect && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl border border-slate-200/90 shadow-2xl max-w-2xl w-full p-6 flex flex-col gap-5 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <span className="font-mono font-black text-xs px-2.5 py-1 rounded-lg bg-slate-900 text-amber-300 border border-amber-500/30">
                  {selectedUnknownForInspect.id}
                </span>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Unknown Person Dossier
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    Captured on {selectedUnknownForInspect.camera_name || selectedUnknownForInspect.camera_id}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={() => setSelectedUnknownForInspect(null)}
                className="p-1.5 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {/* Photo View */}
              <div className="flex flex-col gap-2">
                <div className="relative aspect-square rounded-2xl bg-slate-950 border border-slate-800 overflow-hidden flex items-center justify-center shadow-inner group">
                  <img
                    src={`${BACKEND_BASE_URL}${selectedUnknownForInspect.photo_url}`}
                    alt={selectedUnknownForInspect.id}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute bottom-3 right-3 flex items-center gap-2">
                    <a
                      href={`${BACKEND_BASE_URL}${selectedUnknownForInspect.photo_url}`}
                      download={`unknown_${selectedUnknownForInspect.id}.jpg`}
                      target="_blank"
                      rel="noreferrer"
                      className="px-3 py-1.5 rounded-xl bg-slate-900/90 text-white text-xs font-semibold backdrop-blur-xs flex items-center gap-1.5 shadow-md hover:bg-black transition cursor-pointer"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Download</span>
                    </a>
                  </div>
                </div>
                <span className="text-[10px] text-slate-400 font-mono text-center truncate">
                  {selectedUnknownForInspect.photo_filename}
                </span>
              </div>

              {/* Metadata Details */}
              <div className="flex flex-col justify-between gap-3 text-xs">
                <div className="flex flex-col gap-3">
                  <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80 flex flex-col gap-1.5">
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500 text-[11px]">Real Timestamp</span>
                      <span className="font-mono font-bold text-slate-800">
                        {selectedUnknownForInspect.timestamp} IST
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500 text-[11px]">Camera Feed</span>
                      <span className="font-semibold text-slate-800 truncate max-w-[150px]">
                        {selectedUnknownForInspect.camera_name || selectedUnknownForInspect.camera_id}
                      </span>
                    </div>
                    {selectedUnknownForInspect.cameras_seen && selectedUnknownForInspect.cameras_seen.length > 1 && (
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-[11px]">Traversed Feeds</span>
                        <span className="font-semibold text-blue-700 text-[10px] truncate max-w-[160px]" title={selectedUnknownForInspect.cameras_seen.join(", ")}>
                          {selectedUnknownForInspect.cameras_seen.join(", ")}
                        </span>
                      </div>
                    )}
                    {selectedUnknownForInspect.last_seen && selectedUnknownForInspect.last_seen !== selectedUnknownForInspect.timestamp && (
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500 text-[11px]">Last Sighting</span>
                        <span className="font-mono text-slate-700 text-[11px]">
                          {selectedUnknownForInspect.last_seen} IST
                        </span>
                      </div>
                    )}
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500 text-[11px]">Detection Confidence</span>
                      <span className="font-mono font-bold text-emerald-800">
                        {selectedUnknownForInspect.confidence.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500 text-[11px]">Sightings Count</span>
                      <span className="font-mono font-bold text-slate-800">
                        {selectedUnknownForInspect.seen_count} detection{selectedUnknownForInspect.seen_count > 1 ? "s" : ""}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-slate-500 text-[11px]">Bounding Box</span>
                      <span className="font-mono text-[10px] text-slate-600">
                        [{selectedUnknownForInspect.bbox.join(", ")}]
                      </span>
                    </div>
                  </div>

                  {/* Status Selection */}
                  <div>
                    <label className="text-[11px] font-bold text-slate-700 block mb-1">
                      Administrative Status
                    </label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {(["unreviewed", "flagged", "dismissed"] as const).map((st) => (
                        <button
                          key={st}
                          type="button"
                          onClick={() => handleUpdateUnknownStatus(selectedUnknownForInspect.id, st)}
                          className={`py-1.5 px-2 rounded-lg text-xs font-semibold capitalize border transition cursor-pointer ${
                            selectedUnknownForInspect.status === st
                              ? st === "flagged"
                                ? "bg-rose-50 border-rose-300 text-rose-700"
                                : st === "unreviewed"
                                ? "bg-amber-50 border-amber-300 text-amber-800"
                                : "bg-slate-100 border-slate-300 text-slate-800"
                              : "border-slate-200 text-slate-600 hover:bg-slate-50"
                          }`}
                        >
                          {st}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Notes */}
                  {selectedUnknownForInspect.notes && (
                    <div className="p-2.5 rounded-xl bg-amber-50/60 border border-amber-200/80 text-[11px] text-amber-900">
                      <span className="font-bold block mb-0.5">Notes / Context:</span>
                      {selectedUnknownForInspect.notes}
                    </div>
                  )}
                </div>

                {/* Action Buttons in Modal */}
                <div className="flex items-center gap-2 pt-2 border-t border-slate-100">
                  <button
                    type="button"
                    onClick={() => {
                      const target = selectedUnknownForInspect;
                      setSelectedUnknownForInspect(null);
                      openQuickEnrollModal(target);
                    }}
                    className="flex-1 py-2.5 px-3 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold transition flex items-center justify-center gap-1.5 cursor-pointer shadow-xs"
                  >
                    <UserPlus className="w-4 h-4 text-emerald-400" />
                    <span>Enroll as Person</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleDeleteUnknown(selectedUnknownForInspect.id)}
                    className="p-2.5 rounded-xl border border-rose-200 text-rose-600 hover:bg-rose-50 transition cursor-pointer"
                    title="Delete Record"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 3: QUICK ENROLL UNKNOWN PERSON */}
      {selectedUnknownForEnroll && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-3xl border border-slate-200/90 shadow-2xl max-w-lg w-full p-6 flex flex-col gap-4 animate-in fade-in zoom-in-95 duration-200">
            {/* Header */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <div className="w-9 h-9 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center">
                  <UserPlus className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Enroll Unknown Individual
                  </h3>
                  <p className="text-[11px] text-slate-500">
                    Auto-loaded photo from detection {selectedUnknownForEnroll.id}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={closeQuickEnrollModal}
                className="p-1.5 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Photo Preview Thumbnail & Metadata */}
            <div className="flex items-center gap-3 p-3 rounded-2xl bg-slate-50 border border-slate-200/80">
              <div className="w-16 h-16 rounded-xl bg-slate-950 overflow-hidden shrink-0 border border-slate-300">
                <img
                  src={`${BACKEND_BASE_URL}${selectedUnknownForEnroll.photo_url}`}
                  alt={selectedUnknownForEnroll.id}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="text-xs flex flex-col gap-0.5">
                <span className="font-mono font-bold text-slate-900">{selectedUnknownForEnroll.id}</span>
                <span className="text-slate-500 text-[11px] flex items-center gap-1">
                  <Clock className="w-3 h-3 text-slate-400" />
                  {selectedUnknownForEnroll.timestamp} IST
                </span>
                <span className="text-slate-500 text-[11px] flex items-center gap-1">
                  <Camera className="w-3 h-3 text-slate-400" />
                  {selectedUnknownForEnroll.camera_name || selectedUnknownForEnroll.camera_id}
                </span>
              </div>
            </div>

            {/* Form */}
            <form onSubmit={handleQuickEnrollSubmit} className="flex flex-col gap-3.5">
              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">
                  Full Name / Subject Codename *
                </label>
                <input
                  type="text"
                  placeholder="e.g. Infiltrator Alpha-1, John Doe..."
                  value={quickEnrollName}
                  onChange={(e) => setQuickEnrollName(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-white text-slate-900 font-semibold focus:border-emerald-600 focus:ring-1 focus:ring-emerald-600 outline-none"
                  autoFocus
                  required
                />
              </div>

              {/* Classification Toggle */}
              <div>
                <label className="text-[11px] font-bold text-slate-700 block mb-1">
                  Biometric Classification
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setQuickEnrollIsSuspect(true);
                      setQuickEnrollThreatLevel("HIGH");
                      setQuickEnrollCategory("Wanted / BOLO");
                    }}
                    className={`py-2 px-3 rounded-xl border text-xs font-bold flex items-center justify-center gap-1.5 transition cursor-pointer ${
                      quickEnrollIsSuspect
                        ? "bg-rose-50 border-rose-300 text-rose-700 shadow-xs"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    <ShieldAlert className="w-3.5 h-3.5" />
                    <span>Wanted Suspect</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setQuickEnrollIsSuspect(false);
                      setQuickEnrollThreatLevel("LOW");
                      setQuickEnrollCategory("Authorized Personnel");
                    }}
                    className={`py-2 px-3 rounded-xl border text-xs font-bold flex items-center justify-center gap-1.5 transition cursor-pointer ${
                      !quickEnrollIsSuspect
                        ? "bg-emerald-50 border-emerald-300 text-emerald-800 shadow-xs"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    <ShieldCheck className="w-3.5 h-3.5" />
                    <span>Authorized Personnel</span>
                  </button>
                </div>
              </div>

              {/* Threat Level & Category */}
              {quickEnrollIsSuspect && (
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                      Threat Level
                    </label>
                    <select
                      value={quickEnrollThreatLevel}
                      onChange={(e) => setQuickEnrollThreatLevel(e.target.value as any)}
                      className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none font-semibold"
                    >
                      <option value="CRITICAL">CRITICAL</option>
                      <option value="HIGH">HIGH</option>
                      <option value="MEDIUM">MEDIUM</option>
                      <option value="LOW">LOW</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                      Category
                    </label>
                    <select
                      value={quickEnrollCategory}
                      onChange={(e) => setQuickEnrollCategory(e.target.value)}
                      className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none"
                    >
                      <option value="Wanted / BOLO">Wanted / BOLO</option>
                      <option value="Infiltrator / Poacher">Infiltrator / Poacher</option>
                      <option value="Smuggling Suspect">Smuggling Suspect</option>
                      <option value="Trespasser">Trespasser</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Notes */}
              <div>
                <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                  Notes / Intelligence
                </label>
                <input
                  type="text"
                  placeholder="e.g. Captured near BOP Alpha perimeter fence"
                  value={quickEnrollNotes}
                  onChange={(e) => setQuickEnrollNotes(e.target.value)}
                  className="w-full px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-900 outline-none"
                />
              </div>

              {quickEnrollError && (
                <div className="p-2.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs">
                  {quickEnrollError}
                </div>
              )}

              {quickEnrollSuccess && (
                <div className="p-2.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2 font-medium">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  {quickEnrollSuccess}
                </div>
              )}

              {/* Footer */}
              <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-100">
                <button
                  type="button"
                  onClick={closeQuickEnrollModal}
                  className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isQuickEnrolling || !quickEnrollName.trim()}
                  className="px-5 py-2.5 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold transition flex items-center gap-1.5 cursor-pointer shadow-xs disabled:opacity-50"
                >
                  <UserCheck className="w-4 h-4 text-emerald-400" />
                  <span>{isQuickEnrolling ? "Registering..." : "Confirm & Enroll Identity"}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
