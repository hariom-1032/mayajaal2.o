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
  X,
  UserPlus,
  Eye,
  Activity,
  Cpu,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  EnrolledPerson,
  FaceDetection,
  FaceEngineStatus,
  FaceEvent,
  FaceScanResponse,
} from "@/types/face";

const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

export default function FacialRecognitionPage() {
  const [activeTab, setActiveTab] = useState<"stream" | "directory" | "scan">("stream");

  // Telemetry & Status
  const [engineStatus, setEngineStatus] = useState<FaceEngineStatus | null>(null);
  const [enrolledPersons, setEnrolledPersons] = useState<EnrolledPerson[]>([]);
  const [recentEvents, setRecentEvents] = useState<FaceEvent[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);

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
  const clientWsRef = useRef<WebSocket | null>(null);
  const clientIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [clientAnnotatedUrl, setClientAnnotatedUrl] = useState<string | null>(null);
  const [isClientStreaming, setIsClientStreaming] = useState<boolean>(false);

  const stopClientStreaming = () => {
    if (clientIntervalRef.current) {
      clearInterval(clientIntervalRef.current);
      clientIntervalRef.current = null;
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
    setIsClientStreaming(false);
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

      ws.onopen = () => {
        setIsClientStreaming(true);
        setStreamError(false);

        clientIntervalRef.current = setInterval(() => {
          if (!clientVideoRef.current || !clientCanvasRef.current || ws.readyState !== WebSocket.OPEN) {
            return;
          }
          const video = clientVideoRef.current;
          const canvas = clientCanvasRef.current;
          if (!video.videoWidth || !video.videoHeight) return;

          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(
              (blob) => {
                if (blob && ws.readyState === WebSocket.OPEN) {
                  ws.send(blob);
                }
              },
              "image/jpeg",
              0.72
            );
          }
        }, 66);
      };

      ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
          const url = URL.createObjectURL(event.data);
          setClientAnnotatedUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return url;
          });
        }
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

  // Initial Data Fetch
  useEffect(() => {
    refreshAllData();
    const interval = setInterval(() => {
      fetchEvents();
      fetchStatus();
    }, 4000);
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
    await Promise.all([fetchStatus(), fetchEnrolled(), fetchEvents()]);
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

        {/* Card 3: Events Logged */}
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Recent Matches
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5">
              {recentEvents.length}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              Realtime detections logged
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-cyan-50 flex items-center justify-center text-cyan-700">
            <Activity className="w-5 h-5" />
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
          <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center text-amber-700">
            <Sparkles className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* NAVIGATION TABS */}
      <div className="flex items-center gap-2 border-b border-slate-200">
        <button
          type="button"
          onClick={() => setActiveTab("stream")}
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer ${
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
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer ${
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
          onClick={() => setActiveTab("scan")}
          className={`px-4 py-2.5 text-xs font-bold rounded-t-xl transition-all flex items-center gap-2 cursor-pointer ${
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
            <div className="bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-lg relative">
              {/* Top Stream Control Bar */}
              <div className="bg-slate-950/80 backdrop-blur-md px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-ping" />
                  <span className="text-xs font-bold font-mono text-emerald-400">BIOMETRIC STREAM</span>
                  <span className="text-xs text-slate-500">|</span>
                  <span className="text-xs text-slate-300 font-mono">{streamSource.toUpperCase()}</span>
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
                </div>
              </div>

              {/* Stream Feed Viewport */}
              <div className="relative aspect-video w-full bg-black flex items-center justify-center overflow-hidden">
                {isStreamPlaying ? (
                  <>
                    {streamSource === "browser" ? (
                      clientAnnotatedUrl ? (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                          src={clientAnnotatedUrl}
                          alt="Live Client Biometric Stream"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="flex flex-col items-center gap-3 text-slate-400 p-6 text-center">
                          <Camera className="w-10 h-10 text-emerald-400 animate-pulse" />
                          <p className="text-sm font-medium text-slate-200">Connecting Local Webcam to AI Engine...</p>
                          <p className="text-xs text-slate-400 max-w-sm">
                            Streaming browser webcam frames to {BACKEND_BASE_URL} for real-time YuNet & SFace biometric identification.
                          </p>
                        </div>
                      )
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
                <div className="relative rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-md">
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
    </div>
  );
}
