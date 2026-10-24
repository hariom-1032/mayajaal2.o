"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import {
  ShieldAlert,
  Shield,
  MapPin,
  Camera as CameraIcon,
  Plus,
  Trash2,
  Check,
  X,
  Play,
  RotateCcw,
  Sliders,
  Eye,
  Crosshair,
  ArrowRight,
  ArrowLeftRight,
  Sparkles,
  Layers,
  Activity,
  AlertTriangle,
  Move,
  MousePointer,
  HelpCircle,
  Maximize2,
  RefreshCw,
  BellRing,
  Sun,
  Flame,
  Grid,
  Radio,
  Video,
  Link2,
  Wifi,
  WifiOff,
  Filter,
  Search,
  Power,
  Zap,
  Cpu,
  CheckCircle2,
  ExternalLink,
} from "lucide-react";
import { useCameras } from "@/lib/camerasStore";
import { useAlerts } from "@/lib/alertsStore";
import { api } from "@/lib/api";
import {
  GeofenceZone,
  DirectionalTripwire,
  TripwireDirection,
  GeofenceSeverity,
  GeofenceBreachEvent,
} from "@/types/geofence";

type ToolMode = "select" | "polygon" | "tripwire" | "simulate";
type FeedViewMode = "optical" | "thermal" | "grid";

const COLOR_PRESETS = [
  { name: "Crimson Red", hex: "#ef4444" },
  { name: "Tactical Amber", hex: "#f59e0b" },
  { name: "Cyber Blue", hex: "#3b82f6" },
  { name: "Emerald Guard", hex: "#10b981" },
  { name: "Hostile Purple", hex: "#8b5cf6" },
  { name: "Zero Line Rose", hex: "#f43f5e" },
];

export default function GeofencesPage() {
  const { cameras } = useCameras();
  const { alerts, addAlert } = useAlerts();

  // Selected Camera
  const [selectedCamId, setSelectedCamId] = useState<string>("bop-jk-01");
  const activeCamera = useMemo(
    () => cameras.find((c) => c.id === selectedCamId) || cameras[0] || null,
    [cameras, selectedCamId]
  );

  // Feed Source: Live RTSP / IP Webcam (Real feed), High-Res Perimeter Snapshot, Live Device Webcam, or Patrol Demo
  type FeedSourceType = "rtsp" | "snapshot" | "webcam" | "video";
  const [feedSource, setFeedSource] = useState<FeedSourceType>("rtsp");

  // RTSP / IP Webcam Feed Configuration
  const [rtspUrl, setRtspUrl] = useState<string>("sample");
  const [rtspInputValue, setRtspInputValue] = useState<string>("sample");
  const [streamKey, setStreamKey] = useState<number>(1);
  const [drawDetectionsOnStream, setDrawDetectionsOnStream] = useState<boolean>(true);
  const [streamError, setStreamError] = useState<boolean>(false);
  const [showStreamConfig, setShowStreamConfig] = useState<boolean>(false);

  // Synchronize RTSP URL when camera selection changes
  useEffect(() => {
    if (activeCamera?.streamUrl) {
      setRtspUrl(activeCamera.streamUrl);
      setRtspInputValue(activeCamera.streamUrl);
      setStreamError(false);
      setStreamKey((k) => k + 1);
    } else {
      setRtspUrl("sample");
      setRtspInputValue("sample");
    }
  }, [selectedCamId, activeCamera]);

  // Session ID for stopping background live inference upon unmount
  const sessionIdRef = useRef(
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2)
  );

  useEffect(() => {
    const sid = sessionIdRef.current;
    return () => {
      fetch(
        `/api/backend/stream/live/stop?session_id=${encodeURIComponent(sid)}`,
        { method: "DELETE", keepalive: true }
      ).catch(() => {});
    };
  }, []);

  // Live MJPEG RTSP Stream URL with explicit camera_id for real-time geofence evaluation
  const activeMjpegUrl = useMemo(() => {
    const params = new URLSearchParams({
      rtsp_url: rtspUrl || "sample",
      camera_id: selectedCamId,
      draw_detections: String(drawDetectionsOnStream),
      enable_face_recognition: String(drawDetectionsOnStream),
      session_id: sessionIdRef.current,
      fps: "25",
      _k: String(streamKey),
    });
    return `/api/backend/stream/live?${params.toString()}`;
  }, [rtspUrl, selectedCamId, drawDetectionsOnStream, streamKey]);

  // Live Device Webcam (WebRTC)
  const webcamVideoRef = useRef<HTMLVideoElement>(null);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);

  useEffect(() => {
    let localStream: MediaStream | null = null;
    if (feedSource === "webcam") {
      navigator.mediaDevices
        ?.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } } })
        .then((stream) => {
          localStream = stream;
          setWebcamStream(stream);
          if (webcamVideoRef.current) {
            webcamVideoRef.current.srcObject = stream;
            webcamVideoRef.current.play().catch(() => {});
          }
        })
        .catch((err) => {
          console.warn("Webcam access error:", err);
        });
    }
    return () => {
      if (localStream) {
        localStream.getTracks().forEach((t) => t.stop());
      }
    };
  }, [feedSource]);

  // Feed View Mode (Optical 4K, Thermal FLIR, Tactical Grid)
  const [feedMode, setFeedMode] = useState<FeedViewMode>("optical");

  // Dynamically resolve authentic CCTV camera snapshot based on selected camera ID
  const cameraSnapshotUrl = useMemo(() => {
    const knownCams = ["bop-jk-01", "bop-jk-02", "bop-jk-03", "bop-jk-04", "bop-jk-05", "bop-jk-06"];
    if (knownCams.includes(selectedCamId)) {
      return `/images/cameras/${selectedCamId}.jpg`;
    }
    return "/images/cameras/bop-jk-01.jpg";
  }, [selectedCamId]);

  // Loaded Geofences (All barriers across all cameras in the network)
  const [allZones, setAllZones] = useState<GeofenceZone[]>([]);
  const [allTripwires, setAllTripwires] = useState<DirectionalTripwire[]>([]);

  // Filtered geofences for the currently focused camera on the live canvas
  const zones = useMemo(() => {
    return allZones.filter((z) => (z.cameraId || "bop-jk-01") === selectedCamId);
  }, [allZones, selectedCamId]);

  const tripwires = useMemo(() => {
    return allTripwires.filter((w) => (w.cameraId || "bop-jk-01") === selectedCamId);
  }, [allTripwires, selectedCamId]);

  // Global Geofence Registry Filter & Search state
  const [globalFilterType, setGlobalFilterType] = useState<
    "all" | "active" | "disabled" | "zones" | "tripwires"
  >("all");
  const [globalFilterCamera, setGlobalFilterCamera] = useState<string>("all");
  const [globalSearchQuery, setGlobalSearchQuery] = useState<string>("");

  const [isLoading, setIsLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  // Active Tool & Canvas Drawing State
  const [toolMode, setToolMode] = useState<ToolMode>("select");
  const [selectedEntityId, setSelectedEntityId] = useState<string | null>(null);
  const [activePoints, setActivePoints] = useState<[number, number][]>([]);
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);

  // Vertex Dragging
  const [draggedVertex, setDraggedVertex] = useState<{
    entityId: string;
    entityType: "zone" | "tripwire";
    index: number;
  } | null>(null);

  // Interactive Target Simulation State
  const [simPos, setSimPos] = useState<[number, number]>([0.5, 0.2]);
  const [simPrevPos, setSimPrevPos] = useState<[number, number]>([0.5, 0.1]);
  const [simBreaches, setSimBreaches] = useState<GeofenceBreachEvent[]>([]);
  const [simIdentity, setSimIdentity] = useState<"suspect" | "unknown" | "authorized">("suspect");
  const [simSilentPassMsg, setSimSilentPassMsg] = useState<string | null>(null);

  // Editor Form Drawer state
  const [editName, setEditName] = useState("");
  const [editSeverity, setEditSeverity] = useState<GeofenceSeverity>("CRITICAL");
  const [editColor, setEditColor] = useState("#ef4444");
  const [editDirection, setEditDirection] = useState<TripwireDirection>("FORWARD");
  const [editClasses, setEditClasses] = useState<string[]>(["person", "suspect"]);
  const [editAccessMode, setEditAccessMode] = useState<string>("AUTHORIZED_PASS");
  const [editCooldown, setEditCooldown] = useState(15);
  const [editDescription, setEditDescription] = useState("");

  const canvasRef = useRef<SVGSVGElement>(null);

  // ── Load All Global Geofences from Backend ──────────────────────────────
  const fetchGeofences = useCallback(async () => {
    setIsLoading(true);
    try {
      // Call with no cameraId filter to load ALL global barriers across all cameras
      const res = await api.getGeofences();
      if (res?.zones) setAllZones(res.zones);
      if (res?.tripwires) setAllTripwires(res.tripwires);
    } catch (err) {
      console.warn("Using local or fallback geofences data:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGeofences();
  }, [fetchGeofences]);

  // Listen for real-time breach alerts dispatched from FastAPI backend / WebSocket
  useEffect(() => {
    if (!alerts || alerts.length === 0) return;
    const latest = alerts[0];
    const isRecent = latest.timestamp ? (Date.now() - latest.timestamp < 6000) : true;
    const isGeofenceAlert = Boolean(
      latest.category?.startsWith("geofence_") ||
      latest.category?.startsWith("tripwire_") ||
      latest.category === "geofence_breach" ||
      latest.category === "tripwire_violation" ||
      latest.category === "tripwire_crossing" ||
      latest.title?.toLowerCase().includes("geofence") ||
      latest.title?.toLowerCase().includes("tripwire") ||
      latest.title?.toLowerCase().includes("breach") ||
      latest.title?.toLowerCase().includes("perimeter") ||
      latest.title?.toLowerCase().includes("intrusion")
    );

    if (isRecent && isGeofenceAlert) {
      const matchedZone = zones.find(
        (z) =>
          latest.location?.toLowerCase().includes(z.name.toLowerCase()) ||
          latest.title?.toLowerCase().includes(z.name.toLowerCase()) ||
          latest.notes?.toLowerCase().includes(z.name.toLowerCase())
      );
      const matchedWire = tripwires.find(
        (w) =>
          latest.location?.toLowerCase().includes(w.name.toLowerCase()) ||
          latest.title?.toLowerCase().includes(w.name.toLowerCase()) ||
          latest.notes?.toLowerCase().includes(w.name.toLowerCase())
      );

      const breachEvt: GeofenceBreachEvent = {
        type: matchedWire ? "tripwire_crossing" : "zone_breach",
        zoneId: matchedZone?.id || zones[0]?.id || "zone-active",
        wireId: matchedWire?.id || tripwires[0]?.id,
        name: matchedZone?.name || matchedWire?.name || latest.title || "Perimeter Breach",
        cameraId: selectedCamId,
        trackId: (latest as any).trackId || 1,
        className: latest.className || "person",
        confidence: latest.confidence || 0.85,
        suspectName: latest.suspectName,
        footpoint: [0.5, 0.5],
        severity: (latest.threatLevel || latest.severity || "CRITICAL") as GeofenceSeverity,
        timestamp: new Date().toISOString(),
      };

      setSimBreaches([breachEvt]);
      const timer = setTimeout(() => setSimBreaches([]), 4500);
      return () => clearTimeout(timer);
    }
  }, [alerts, zones, tripwires, selectedCamId]);

  // Auto-sync any local or newly created zones/wires to the FastAPI backend so stream_generator evaluates them
  useEffect(() => {
    const syncUnsaved = async () => {
      for (const z of allZones) {
        if (z.id.startsWith("zone-local-")) {
          try {
            const res = await api.createGeofenceZone({
              name: z.name,
              cameraId: z.cameraId || selectedCamId,
              color: z.color,
              severity: z.severity,
              polygon: z.polygon,
              targetClasses: z.targetClasses,
              cooldownSeconds: z.cooldownSeconds,
              description: z.description,
            });
            if (res?.zone) {
              setAllZones((prev) => prev.map((item) => (item.id === z.id ? res.zone : item)));
            }
          } catch (err) {
            console.debug("Auto-sync zone retry later:", err);
          }
        }
      }
      for (const w of allTripwires) {
        if (w.id.startsWith("wire-local-")) {
          try {
            const res = await api.createTripwire({
              name: w.name,
              cameraId: w.cameraId || selectedCamId,
              color: w.color,
              severity: w.severity,
              p1: w.p1,
              p2: w.p2,
              direction: w.direction,
              targetClasses: w.targetClasses,
              cooldownSeconds: w.cooldownSeconds,
              description: w.description,
            });
            if (res?.tripwire) {
              setAllTripwires((prev) => prev.map((item) => (item.id === w.id ? res.tripwire : item)));
            }
          } catch (err) {
            console.debug("Auto-sync tripwire retry later:", err);
          }
        }
      }
    };
    syncUnsaved();
  }, [allZones, allTripwires, selectedCamId]);

  // Sync selected entity details into edit form
  useEffect(() => {
    if (!selectedEntityId) return;

    const z = zones.find((item) => item.id === selectedEntityId);
    if (z) {
      setEditName(z.name);
      setEditSeverity(z.severity);
      setEditColor(z.color);
      setEditClasses(z.targetClasses || ["person"]);
      setEditAccessMode(z.accessMode || "AUTHORIZED_PASS");
      setEditCooldown(z.cooldownSeconds || 20);
      setEditDescription(z.description || "");
      return;
    }

    const w = tripwires.find((item) => item.id === selectedEntityId);
    if (w) {
      setEditName(w.name);
      setEditSeverity(w.severity);
      setEditColor(w.color);
      setEditDirection(w.direction);
      setEditClasses(w.targetClasses || ["person"]);
      setEditAccessMode(w.accessMode || "AUTHORIZED_PASS");
      setEditCooldown(w.cooldownSeconds || 15);
      setEditDescription(w.description || "");
    }
  }, [selectedEntityId, zones, tripwires]);

  // ── Coordinate Normalization Helpers ─────────────────────────────────────
  // SVG viewBox is 1920 x 1080 (Exact 16:9 matching container aspect-video)
  const getNormCoords = (e: React.MouseEvent<SVGSVGElement>): [number, number] => {
    if (!canvasRef.current) return [0, 0];
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [parseFloat(x.toFixed(4)), parseFloat(y.toFixed(4))];
  };

  // ── Canvas Interaction Handlers ──────────────────────────────────────────
  const handleCanvasClick = async (e: React.MouseEvent<SVGSVGElement>) => {
    if (draggedVertex) {
      setDraggedVertex(null);
      return;
    }

    const [nx, ny] = getNormCoords(e);

    // 1. Polygon Drawing Mode
    if (toolMode === "polygon") {
      if (activePoints.length >= 3) {
        const [startX, startY] = activePoints[0];
        const dist = Math.hypot(nx - startX, ny - startY);
        // Snapping radius 0.04 to close polygon loop
        if (dist < 0.04) {
          const newZonePayload = {
            name: `Exclusion Zone ${zones.length + 1}`,
            cameraId: selectedCamId,
            color: editColor || "#ef4444",
            severity: editSeverity || "CRITICAL",
            polygon: activePoints,
            targetClasses: editClasses,
            accessMode: (editAccessMode as any) || "AUTHORIZED_PASS",
            cooldownSeconds: editCooldown,
            description: "Custom tactical exclusion zone.",
          };
          try {
            const res = await api.createGeofenceZone(newZonePayload);
            if (res?.zone) {
              setAllZones((prev) => [...prev, res.zone]);
              setSelectedEntityId(res.zone.id);
            }
          } catch {
            const fallbackZone: GeofenceZone = {
              id: `zone-local-${Date.now()}`,
              ...newZonePayload,
              enabled: true,
            };
            setAllZones((prev) => [...prev, fallbackZone]);
            setSelectedEntityId(fallbackZone.id);
          }
          setActivePoints([]);
          setToolMode("select");
          return;
        }
      }
      setActivePoints((prev) => [...prev, [nx, ny]]);
      return;
    }

    // 2. Tripwire Drawing Mode
    if (toolMode === "tripwire") {
      if (activePoints.length === 1) {
        const p1 = activePoints[0];
        const p2: [number, number] = [nx, ny];
        const newWirePayload = {
          name: `Tripwire Barrier ${allTripwires.length + 1}`,
          cameraId: selectedCamId,
          color: editColor || "#dc2626",
          severity: editSeverity || "CRITICAL",
          p1,
          p2,
          direction: editDirection,
          targetClasses: editClasses,
          accessMode: (editAccessMode as any) || "AUTHORIZED_PASS",
          cooldownSeconds: editCooldown,
          description: "High-sensitivity border tripwire barrier.",
        };
        try {
          const res = await api.createTripwire(newWirePayload);
          if (res?.tripwire) {
            setAllTripwires((prev) => [...prev, res.tripwire]);
            setSelectedEntityId(res.tripwire.id);
          }
        } catch {
          const fallbackWire: DirectionalTripwire = {
            id: `wire-local-${Date.now()}`,
            ...newWirePayload,
            enabled: true,
          };
          setAllTripwires((prev) => [...prev, fallbackWire]);
          setSelectedEntityId(fallbackWire.id);
        }
        setActivePoints([]);
        setToolMode("select");
        return;
      }
      setActivePoints([[nx, ny]]);
      return;
    }

    // 3. Target Simulation Mode (Clicking moves the simulated target)
    if (toolMode === "simulate") {
      runSimulationStep([nx, ny]);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const coords = getNormCoords(e);
    setMousePos(coords);

    if (draggedVertex) {
      const [nx, ny] = coords;
      if (draggedVertex.entityType === "zone") {
        setAllZones((prev) =>
          prev.map((z) => {
            if (z.id !== draggedVertex.entityId) return z;
            const updated = [...z.polygon];
            updated[draggedVertex.index] = [nx, ny];
            return { ...z, polygon: updated };
          })
        );
      } else if (draggedVertex.entityType === "tripwire") {
        setAllTripwires((prev) =>
          prev.map((w) => {
            if (w.id !== draggedVertex.entityId) return w;
            if (draggedVertex.index === 0) return { ...w, p1: [nx, ny] };
            return { ...w, p2: [nx, ny] };
          })
        );
      }
    }
  };

  // ── Simulation Engine Step ───────────────────────────────────────────────
  const runSimulationStep = async (newPos: [number, number]) => {
    setSimPrevPos(simPos);
    setSimPos(newPos);

    const isAuth = simIdentity === "authorized";
    const isSusp = simIdentity === "suspect";
    const isUnk = simIdentity === "unknown";

    // Call backend evaluator
    try {
      const evalRes = await api.evaluateGeofences({
        cameraId: selectedCamId,
        tracks: [
          {
            trackId: isAuth ? 801 : isSusp ? 901 : 902,
            box: [
              Math.max(0, newPos[0] - 0.03),
              Math.max(0, newPos[1] - 0.06),
              Math.min(1, newPos[0] + 0.03),
              newPos[1], // bottom footpoint
            ],
            className: isSusp ? "suspect" : "person",
            confidence: isAuth ? 0.99 : 0.92,
            suspectName: isSusp ? "Simulated Suspect #901" : undefined,
            personName: isAuth ? "Hariom (Staff)" : undefined,
            identityStatus: simIdentity,
            isAuthorized: isAuth,
            isSuspect: isSusp,
            isUnknown: isUnk,
          },
        ],
        dispatchAlerts: true,
      });

      if (evalRes?.breaches && evalRes.breaches.length > 0) {
        setSimBreaches(evalRes.breaches);
        setSimSilentPassMsg(null);

        // Visual / Audio feedback
        const primary = evalRes.breaches[0];
        addAlert({
          title:
            primary.alertType === "suspect_breach"
              ? `🚨 Suspect Breach: ${primary.suspectName || "Flagged Target"}`
              : `⚠️ Perimeter Intrusion: ${primary.name}`,
          location: `Camera ${selectedCamId} • Perimeter Fence`,
          severity: primary.severity === "CRITICAL" ? "High" : "Medium",
          cameraId: selectedCamId,
          suspectName: isSusp ? "Simulated Suspect #901" : undefined,
          threatLevel: primary.severity,
          category:
            primary.alertType === "suspect_breach"
              ? "geofence_suspect_breach"
              : "geofence_unknown_intrusion",
          notes: `Tactical simulation triggered ${primary.type} at (${newPos[0].toFixed(2)}, ${newPos[1].toFixed(2)})`,
        });

        setTimeout(() => setSimBreaches([]), 4500);
      } else if (isAuth) {
        // Check if inside any authorized pass zone for instant visual feedback
        const insideAuthZone = zones.find(
          (z) =>
            z.enabled &&
            (z.accessMode || "AUTHORIZED_PASS") === "AUTHORIZED_PASS" &&
            pointInPoly(newPos[0], newPos[1], z.polygon)
        );
        if (insideAuthZone) {
          setSimSilentPassMsg(`✓ Silent Pass: Staff permitted inside ${insideAuthZone.name}`);
          setTimeout(() => setSimSilentPassMsg(null), 3000);
        }
      }
    } catch {
      // Fallback in-browser evaluation
      evaluateClientSimulation(simPos, newPos);
    }
  };

  const evaluateClientSimulation = (pPrev: [number, number], pCurr: [number, number]) => {
    const detected: GeofenceBreachEvent[] = [];
    const isAuth = simIdentity === "authorized";
    const isSusp = simIdentity === "suspect";

    // Check Zones
    zones.forEach((z) => {
      if (!z.enabled) return;
      const mode = z.accessMode || "AUTHORIZED_PASS";
      if (mode === "AUTHORIZED_PASS" && isAuth) return; // Silent pass for authorized staff
      if (mode === "SUSPECT_ONLY" && !isSusp) return;   // Only suspects trigger

      if (pointInPoly(pCurr[0], pCurr[1], z.polygon)) {
        detected.push({
          type: "zone_breach",
          alertType: isSusp ? "suspect_breach" : "unknown_intrusion",
          zoneId: z.id,
          name: z.name,
          cameraId: selectedCamId,
          trackId: isAuth ? 801 : isSusp ? 901 : 902,
          className: isSusp ? "suspect" : "person",
          confidence: isAuth ? 0.99 : 0.92,
          suspectName: isSusp ? "Simulated Suspect #901" : undefined,
          personName: isAuth ? "Hariom (Staff)" : undefined,
          identityStatus: simIdentity,
          footpoint: pCurr,
          severity: isSusp ? "CRITICAL" : z.severity,
          timestamp: new Date().toISOString(),
        });
      }
    });

    if (detected.length > 0) {
      setSimBreaches(detected);
      setSimSilentPassMsg(null);
      setTimeout(() => setSimBreaches([]), 4500);
    } else if (isAuth) {
      const insideAuthZone = zones.find(
        (z) =>
          z.enabled &&
          (z.accessMode || "AUTHORIZED_PASS") === "AUTHORIZED_PASS" &&
          pointInPoly(pCurr[0], pCurr[1], z.polygon)
      );
      if (insideAuthZone) {
        setSimSilentPassMsg(`✓ Silent Pass: Staff permitted inside ${insideAuthZone.name}`);
        setTimeout(() => setSimSilentPassMsg(null), 3000);
      }
    }
  };

  function pointInPoly(x: number, y: number, poly: [number, number][]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0],
        yi = poly[i][1];
      const xj = poly[j][0],
        yj = poly[j][1];
      const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  // ── Save Entity Updates ──────────────────────────────────────────────────
  const handleSaveEntity = async () => {
    if (!selectedEntityId) return;

    const z = allZones.find((item) => item.id === selectedEntityId);
    if (z) {
      const updatedZone = {
        ...z,
        name: editName,
        severity: editSeverity,
        color: editColor,
        targetClasses: editClasses,
        accessMode: (editAccessMode as any) || "AUTHORIZED_PASS",
        cooldownSeconds: editCooldown,
        description: editDescription,
      };
      setAllZones((prev) => prev.map((item) => (item.id === z.id ? updatedZone : item)));
      try {
        await api.updateGeofenceZone(z.id, updatedZone);
        setSaveStatus("Zone saved successfully!");
        setTimeout(() => setSaveStatus(null), 2500);
      } catch (err) {
        console.error("Save error:", err);
      }
      return;
    }

    const w = allTripwires.find((item) => item.id === selectedEntityId);
    if (w) {
      const updatedWire = {
        ...w,
        name: editName,
        severity: editSeverity,
        color: editColor,
        direction: editDirection,
        targetClasses: editClasses,
        accessMode: (editAccessMode as any) || "AUTHORIZED_PASS",
        cooldownSeconds: editCooldown,
        description: editDescription,
      };
      setAllTripwires((prev) => prev.map((item) => (item.id === w.id ? updatedWire : item)));
      try {
        await api.updateTripwire(w.id, updatedWire);
        setSaveStatus("Tripwire saved successfully!");
        setTimeout(() => setSaveStatus(null), 2500);
      } catch (err) {
        console.error("Save error:", err);
      }
    }
  };

  const handleDeleteEntity = async (id: string, type: "zone" | "tripwire") => {
    if (type === "zone") {
      setAllZones((prev) => prev.filter((z) => z.id !== id));
      try {
        await api.deleteGeofenceZone(id);
      } catch {}
    } else {
      setAllTripwires((prev) => prev.filter((w) => w.id !== id));
      try {
        await api.deleteTripwire(id);
      } catch {}
    }
    if (selectedEntityId === id) setSelectedEntityId(null);
  };

  const toggleEntityEnabled = async (id: string, type: "zone" | "tripwire") => {
    if (type === "zone") {
      setAllZones((prev) =>
        prev.map((z) => {
          if (z.id === id) {
            const upd = { ...z, enabled: !z.enabled };
            api.updateGeofenceZone(id, { enabled: upd.enabled }).catch(() => {});
            return upd;
          }
          return z;
        })
      );
    } else {
      setAllTripwires((prev) =>
        prev.map((w) => {
          if (w.id === id) {
            const upd = { ...w, enabled: !w.enabled };
            api.updateTripwire(id, { enabled: upd.enabled }).catch(() => {});
            return upd;
          }
          return w;
        })
      );
    }
  };

  // ── Global Multi-Camera Management Actions ──────────────────────────────
  const handleIsolateCurrentCamera = async () => {
    const otherZones = allZones.filter(
      (z) => (z.cameraId || "bop-jk-01") !== selectedCamId && z.enabled
    );
    const otherWires = allTripwires.filter(
      (w) => (w.cameraId || "bop-jk-01") !== selectedCamId && w.enabled
    );

    setAllZones((prev) =>
      prev.map((z) => {
        if ((z.cameraId || "bop-jk-01") !== selectedCamId) {
          return { ...z, enabled: false };
        }
        return z;
      })
    );
    setAllTripwires((prev) =>
      prev.map((w) => {
        if ((w.cameraId || "bop-jk-01") !== selectedCamId) {
          return { ...w, enabled: false };
        }
        return w;
      })
    );

    for (const z of otherZones) {
      api.updateGeofenceZone(z.id, { enabled: false }).catch(() => {});
    }
    for (const w of otherWires) {
      api.updateTripwire(w.id, { enabled: false }).catch(() => {});
    }
  };

  const handleDisableAllGlobal = async () => {
    const activeZ = allZones.filter((z) => z.enabled);
    const activeW = allTripwires.filter((w) => w.enabled);

    setAllZones((prev) => prev.map((z) => ({ ...z, enabled: false })));
    setAllTripwires((prev) => prev.map((w) => ({ ...w, enabled: false })));

    for (const z of activeZ) {
      api.updateGeofenceZone(z.id, { enabled: false }).catch(() => {});
    }
    for (const w of activeW) {
      api.updateTripwire(w.id, { enabled: false }).catch(() => {});
    }
  };

  const handleEnableAllGlobal = async () => {
    const inactiveZ = allZones.filter((z) => !z.enabled);
    const inactiveW = allTripwires.filter((w) => !w.enabled);

    setAllZones((prev) => prev.map((z) => ({ ...z, enabled: true })));
    setAllTripwires((prev) => prev.map((w) => ({ ...w, enabled: true })));

    for (const z of inactiveZ) {
      api.updateGeofenceZone(z.id, { enabled: true }).catch(() => {});
    }
    for (const w of inactiveW) {
      api.updateTripwire(w.id, { enabled: true }).catch(() => {});
    }
  };

  const handleFocusCamera = (camId: string, entityId: string) => {
    setSelectedCamId(camId);
    setSelectedEntityId(entityId);
    setToolMode("select");
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  // Metrics: active barriers count & estimated load level
  const activeCount = useMemo(() => {
    return allZones.filter((z) => z.enabled).length + allTripwires.filter((w) => w.enabled).length;
  }, [allZones, allTripwires]);

  const loadLevel = useMemo(() => {
    if (activeCount === 0) {
      return { label: "ZERO", color: "text-slate-500 bg-slate-100 border-slate-200" };
    }
    if (activeCount <= 2) {
      return { label: "LOW (Optimal)", color: "text-emerald-700 bg-emerald-50 border-emerald-200" };
    }
    if (activeCount <= 5) {
      return { label: "MODERATE", color: "text-blue-700 bg-blue-50 border-blue-200" };
    }
    return { label: "HIGH (Multi-Camera Load)", color: "text-amber-700 bg-amber-50 border-amber-200" };
  }, [activeCount]);

  // Derived filtered global items for the registry list below
  const filteredGlobalItems = useMemo(() => {
    type UnifiedBarrier =
      | (GeofenceZone & { itemType: "zone" })
      | (DirectionalTripwire & { itemType: "tripwire" });

    const items: UnifiedBarrier[] = [
      ...allZones.map((z) => ({ ...z, itemType: "zone" as const })),
      ...allTripwires.map((w) => ({ ...w, itemType: "tripwire" as const })),
    ];

    return items.filter((item) => {
      if (globalFilterType === "zones" && item.itemType !== "zone") return false;
      if (globalFilterType === "tripwires" && item.itemType !== "tripwire") return false;
      if (globalFilterType === "active" && !item.enabled) return false;
      if (globalFilterType === "disabled" && item.enabled) return false;

      if (globalFilterCamera !== "all" && (item.cameraId || "bop-jk-01") !== globalFilterCamera) {
        return false;
      }

      if (globalSearchQuery.trim()) {
        const q = globalSearchQuery.toLowerCase();
        const matchesName = item.name.toLowerCase().includes(q);
        const matchesCam = (item.cameraId || "").toLowerCase().includes(q);
        const matchesDesc = (item.description || "").toLowerCase().includes(q);
        const matchesSev = (item.severity || "").toLowerCase().includes(q);
        if (!matchesName && !matchesCam && !matchesDesc && !matchesSev) return false;
      }

      return true;
    });
  }, [allZones, allTripwires, globalFilterType, globalFilterCamera, globalSearchQuery]);

  // ── Compute Tripwire Direction Normal Vector and Angle ──────────────────
  const getTripwireGeometry = (p1: [number, number], p2: [number, number], dir: TripwireDirection) => {
    const x1 = p1[0] * 1920;
    const y1 = p1[1] * 1080;
    const x2 = p2[0] * 1920;
    const y2 = p2[1] * 1080;

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;

    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy) || 1;

    // Normal vector pointing to the "Forward" side (right normal)
    let nx = -dy / len;
    let ny = dx / len;

    if (dir === "REVERSE") {
      nx = -nx;
      ny = -ny;
    }

    const angleDeg = (Math.atan2(ny, nx) * 180) / Math.PI;
    const arrowDist = 48;
    const arrowX = midX + nx * arrowDist;
    const arrowY = midY + ny * arrowDist;

    return { x1, y1, x2, y2, midX, midY, arrowX, arrowY, nx, ny, angleDeg };
  };

  return (
    <div className="space-y-6 pb-12">
      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 md:p-6 shadow-xs flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-10 h-10 rounded-xl bg-[#143724] text-white flex items-center justify-center shadow-xs">
              <ShieldAlert className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl md:text-2xl font-black tracking-tight text-slate-900">
                  Virtual Geofencing & Directional Tripwires
                </h1>
                <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 text-[10px] font-bold border border-emerald-200 uppercase tracking-wider">
                  Live Studio
                </span>
              </div>
              <p className="text-xs md:text-sm font-medium text-slate-500 mt-0.5">
                Interactive FOV polygon fences, directional crossing barriers, and automated breach alerts.
              </p>
            </div>
          </div>
        </div>

        {/* Quick Camera Switcher */}
        <div className="flex items-center gap-2.5 flex-wrap">
          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 shadow-2xs">
            <CameraIcon className="w-4 h-4 text-slate-500" />
            <span className="text-xs font-semibold text-slate-700">Camera Feed:</span>
            <select
              value={selectedCamId}
              onChange={(e) => {
                setSelectedCamId(e.target.value);
                setSelectedEntityId(null);
                setActivePoints([]);
              }}
              className="bg-white border border-slate-200 text-xs font-bold text-slate-900 rounded-lg px-2.5 py-1 focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer"
            >
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.id})
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            onClick={fetchGeofences}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-2xs transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-500 ${isLoading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* ── Main Studio Grid ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left / Center: Interactive Canvas Studio (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {/* Tool Control Strip */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-3 shadow-xs flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                type="button"
                onClick={() => {
                  setToolMode("select");
                  setActivePoints([]);
                }}
                className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                  toolMode === "select"
                    ? "bg-slate-900 text-white shadow-xs ring-2 ring-slate-900/20"
                    : "bg-slate-100 hover:bg-slate-200 text-slate-700"
                }`}
              >
                <MousePointer className="w-3.5 h-3.5" />
                <span>Select & Edit</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setToolMode("polygon");
                  setActivePoints([]);
                  setSelectedEntityId(null);
                }}
                className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                  toolMode === "polygon"
                    ? "bg-rose-600 text-white shadow-xs animate-pulse ring-2 ring-rose-500/30"
                    : "bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200"
                }`}
              >
                <Shield className="w-3.5 h-3.5" />
                <span>Draw Polygon Zone</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setToolMode("tripwire");
                  setActivePoints([]);
                  setSelectedEntityId(null);
                }}
                className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                  toolMode === "tripwire"
                    ? "bg-blue-600 text-white shadow-xs animate-pulse ring-2 ring-blue-500/30"
                    : "bg-blue-50 hover:bg-blue-100 text-blue-700 border border-blue-200"
                }`}
              >
                <ArrowRight className="w-3.5 h-3.5" />
                <span>Draw Directional Tripwire</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  setToolMode("simulate");
                  setActivePoints([]);
                }}
                className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                  toolMode === "simulate"
                    ? "bg-emerald-600 text-white shadow-xs animate-pulse ring-2 ring-emerald-500/30"
                    : "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5 text-emerald-500" />
                <span>Test Intrusion Simulator</span>
              </button>
            </div>

            {/* Hint Badge */}
            <div className="hidden sm:inline-flex items-center gap-1 text-[11px] font-medium text-slate-500">
              <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
              {toolMode === "polygon" && "Click to place vertices. Click near the start point to close."}
              {toolMode === "tripwire" && "Click Point A, then Point B to draw the tripwire barrier."}
              {toolMode === "simulate" && "Click anywhere in the video feed to test an intruder footpoint."}
              {toolMode === "select" && "Click any barrier to inspect and drag vertex handles."}
            </div>
          </div>

          {/* RTSP / IP Webcam Live Stream Configuration Strip */}
          {feedSource === "rtsp" && (
            <div className="bg-slate-900 border border-slate-700/80 rounded-2xl p-3 shadow-md text-white space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Radio className="w-4 h-4 text-rose-400 animate-pulse" />
                  <span className="text-xs font-bold font-mono uppercase tracking-wider text-slate-200">
                    Live RTSP / IP Webcam Ingestion
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-950 text-emerald-400 border border-emerald-500/40">
                    STREAM ACTIVE • 25 FPS
                  </span>
                </div>

                {/* Quick Preset Buttons */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[11px] text-slate-400 font-semibold">Presets:</span>
                  <button
                    type="button"
                    onClick={() => {
                      const url = activeCamera?.streamUrl || "sample";
                      setRtspUrl(url);
                      setRtspInputValue(url);
                      setStreamError(false);
                      setStreamKey((k) => k + 1);
                    }}
                    className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-semibold text-slate-300 border border-slate-700 cursor-pointer"
                  >
                    Camera RTSP
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setRtspUrl("sample");
                      setRtspInputValue("sample");
                      setStreamError(false);
                      setStreamKey((k) => k + 1);
                    }}
                    className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-semibold text-slate-300 border border-slate-700 cursor-pointer"
                  >
                    Sample Feed
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const phoneUrl = "http://192.168.1.50:8080/video";
                      setRtspInputValue(phoneUrl);
                      setRtspUrl(phoneUrl);
                      setStreamError(false);
                      setStreamKey((k) => k + 1);
                    }}
                    className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-[11px] font-semibold text-slate-300 border border-slate-700 cursor-pointer"
                  >
                    IP Webcam (Phone)
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Link2 className="absolute left-3 top-2.5 w-3.5 h-3.5 text-slate-500" />
                  <input
                    type="text"
                    value={rtspInputValue}
                    onChange={(e) => setRtspInputValue(e.target.value)}
                    placeholder="rtsp://admin:pass@ip:554/live or http://ip:8080/video or sample"
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-100 font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>

                <button
                  type="button"
                  onClick={() => {
                    const val = rtspInputValue.trim() || "sample";
                    setRtspUrl(val);
                    setStreamError(false);
                    setStreamKey((k) => k + 1);
                  }}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white flex items-center gap-1.5 shadow-sm cursor-pointer transition-all shrink-0"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Connect Stream</span>
                </button>

                <label className="flex items-center gap-1.5 text-[11px] text-slate-300 cursor-pointer select-none shrink-0 px-2.5 py-1.5 bg-slate-800/80 rounded-lg border border-slate-700">
                  <input
                    type="checkbox"
                    checked={drawDetectionsOnStream}
                    onChange={(e) => setDrawDetectionsOnStream(e.target.checked)}
                    className="rounded text-emerald-600 focus:ring-0 cursor-pointer"
                  />
                  <span>AI Detections</span>
                </label>
              </div>
            </div>
          )}

          {/* Canvas Viewport (Fixed 16:9 Aspect Ratio matching 1920x1080 ViewBox) */}
          <div className="relative rounded-2xl overflow-hidden border border-slate-700 bg-slate-950 shadow-xl aspect-video select-none">
            {/* Camera Feed Backdrop (RTSP Live Stream, Webcam, Calibration Snapshot, or Demo) */}
            {feedMode !== "grid" && (
              <>
                {feedSource === "rtsp" ? (
                  <div className="relative w-full h-full">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      key={`rtsp-${streamKey}-${rtspUrl}`}
                      src={activeMjpegUrl}
                      alt="Live RTSP Feed"
                      className={`absolute inset-0 w-full h-full object-cover transition-all duration-300 pointer-events-none ${
                        feedMode === "thermal"
                          ? "contrast-150 saturate-200 hue-rotate-180 brightness-90"
                          : "opacity-95"
                      }`}
                      onError={() => {
                        setStreamError(true);
                      }}
                    />
                    {streamError && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-10 p-6 text-center">
                        <AlertTriangle className="w-8 h-8 text-amber-400 mb-2" />
                        <p className="text-white text-xs font-bold">RTSP Stream Offline or Unreachable</p>
                        <p className="text-slate-400 text-[11px] font-mono mt-1">{rtspUrl}</p>
                      </div>
                    )}
                  </div>
                ) : feedSource === "webcam" ? (
                  <video
                    ref={webcamVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`absolute inset-0 w-full h-full object-cover transition-all duration-300 pointer-events-none ${
                      feedMode === "thermal"
                        ? "contrast-150 saturate-200 hue-rotate-180 brightness-90"
                        : "opacity-95"
                    }`}
                  />
                ) : feedSource === "snapshot" ? (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    key={selectedCamId}
                    src={cameraSnapshotUrl}
                    alt={activeCamera?.name || selectedCamId}
                    className={`absolute inset-0 w-full h-full object-cover transition-all duration-300 pointer-events-none ${
                      feedMode === "thermal"
                        ? "contrast-150 saturate-200 hue-rotate-180 brightness-90"
                        : "opacity-90"
                    }`}
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = "/images/himalayan-border-hero.jpg";
                    }}
                  />
                ) : (
                  <video
                    key="live-border-feed"
                    src="/videos/himalayan-border-animated.mp4"
                    autoPlay
                    loop
                    muted
                    playsInline
                    className={`absolute inset-0 w-full h-full object-cover transition-all duration-300 pointer-events-none ${
                      feedMode === "thermal"
                        ? "contrast-150 saturate-200 hue-rotate-180 brightness-90"
                        : "opacity-85"
                    }`}
                  />
                )}
              </>
            )}

            {/* Tactical Grid Overlay */}
            <div
              className={`absolute inset-0 pointer-events-none ${
                feedMode === "grid" ? "opacity-40" : "opacity-20"
              }`}
              style={{
                backgroundImage: `linear-gradient(to right, #38bdf8 1px, transparent 1px), linear-gradient(to bottom, #38bdf8 1px, transparent 1px)`,
                backgroundSize: "60px 60px",
              }}
            />

            {/* Top HUD Controls & Feed Selector */}
            <div className="absolute top-3 left-3 right-3 z-20 flex items-center justify-between pointer-events-auto">
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/85 backdrop-blur-md border border-slate-700/80 text-white text-xs font-mono shadow-md">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="font-bold">{activeCamera?.name || selectedCamId}</span>
                  <span className="text-slate-400">•</span>
                  <span className="text-slate-300">{activeCamera?.resolution || "3840x2160 (4K)"}</span>
                </div>

                {simBreaches.length > 0 && (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-rose-600 text-white text-xs font-bold shadow-lg animate-bounce border border-rose-400">
                    <BellRing className="w-4 h-4 animate-spin" />
                    <span>BREACH DETECTED: {simBreaches[0].name}</span>
                  </div>
                )}
                {simSilentPassMsg && (
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-600 text-white text-xs font-bold shadow-lg border border-emerald-400 animate-in fade-in">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>{simSilentPassMsg}</span>
                  </div>
                )}

              </div>

              {/* Feed Visual Filter & Source Switcher */}
              <div className="flex items-center gap-2">
                {/* Source Switcher */}
                <div className="bg-slate-900/85 backdrop-blur-md border border-slate-700/80 rounded-lg p-0.5 flex items-center gap-1 shadow-md">
                  <button
                    type="button"
                    onClick={() => setFeedSource("rtsp")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                      feedSource === "rtsp"
                        ? "bg-rose-600 text-white shadow-xs"
                        : "text-slate-400 hover:text-white"
                    }`}
                    title="Live RTSP / IP Webcam Real Camera Stream"
                  >
                    <Radio className="w-3.5 h-3.5 text-white animate-pulse" />
                    <span>RTSP Live</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFeedSource("snapshot")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                      feedSource === "snapshot"
                        ? "bg-slate-700 text-white shadow-xs"
                        : "text-slate-400 hover:text-white"
                    }`}
                    title="Authentic Camera Snapshot (Ideal for Drawing & Calibrating Barriers)"
                  >
                    <CameraIcon className="w-3.5 h-3.5 text-slate-300" />
                    <span>Perimeter Frame</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFeedSource("webcam")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                      feedSource === "webcam"
                        ? "bg-emerald-600 text-white shadow-xs"
                        : "text-slate-400 hover:text-white"
                    }`}
                    title="Live Device / USB Webcam Feed"
                  >
                    <Video className="w-3.5 h-3.5 text-white" />
                    <span>Webcam</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFeedSource("video")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer flex items-center gap-1.5 ${
                      feedSource === "video"
                        ? "bg-blue-600 text-white shadow-xs"
                        : "text-slate-400 hover:text-white"
                    }`}
                    title="Himalayan Patrol Video Simulation"
                  >
                    <Play className="w-3 h-3 fill-current text-white" />
                    <span>Patrol Demo</span>
                  </button>
                </div>

                {/* Filter Switcher */}
                <div className="bg-slate-900/85 backdrop-blur-md border border-slate-700/80 rounded-lg p-0.5 flex items-center gap-1 shadow-md">
                  <button
                    type="button"
                    onClick={() => setFeedMode("optical")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer ${
                      feedMode === "optical" ? "bg-emerald-600 text-white" : "text-slate-400 hover:text-white"
                    }`}
                    title="Optical 4K Stream View"
                  >
                    Optical 4K
                  </button>
                  <button
                    type="button"
                    onClick={() => setFeedMode("thermal")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer ${
                      feedMode === "thermal" ? "bg-amber-600 text-white" : "text-slate-400 hover:text-white"
                    }`}
                    title="Thermal FLIR Ironbow View"
                  >
                    Thermal FLIR
                  </button>
                  <button
                    type="button"
                    onClick={() => setFeedMode("grid")}
                    className={`px-2.5 py-1 rounded text-[11px] font-bold transition-all cursor-pointer ${
                      feedMode === "grid" ? "bg-blue-600 text-white" : "text-slate-400 hover:text-white"
                    }`}
                    title="High-Contrast Radar Grid View"
                  >
                    Radar Grid
                  </button>
                </div>
              </div>
            </div>

            {/* ── Interactive SVG Drawing Canvas (1920 x 1080 Aspect-Perfect ViewBox) ── */}
            <svg
              ref={canvasRef}
              className="absolute inset-0 w-full h-full cursor-crosshair z-10"
              viewBox="0 0 1920 1080"
              preserveAspectRatio="none"
              onClick={handleCanvasClick}
              onMouseMove={handleMouseMove}
              onMouseUp={() => setDraggedVertex(null)}
            >
              <defs>
                <filter id="glow-breach" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="12" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
                <marker
                  id="arrow-marker"
                  viewBox="0 0 12 12"
                  refX="6"
                  refY="6"
                  markerWidth="8"
                  markerHeight="8"
                  orient="auto"
                >
                  <path d="M 1 2 L 10 6 L 1 10 z" fill="#38bdf8" />
                </marker>
              </defs>

              {/* ── 1. Render Existing Polygon Zones ─────────────────────────── */}
              {zones.map((zone) => {
                if (!zone.polygon || zone.polygon.length < 3) return null;
                const isSelected = selectedEntityId === zone.id;
                const isBreached = simBreaches.some((b) => b.zoneId === zone.id);
                const pointsString = zone.polygon
                  .map(([px, py]) => `${px * 1920},${py * 1080}`)
                  .join(" ");

                const avgX =
                  (zone.polygon.reduce((acc, p) => acc + p[0], 0) / zone.polygon.length) * 1920;
                const avgY =
                  (zone.polygon.reduce((acc, p) => acc + p[1], 0) / zone.polygon.length) * 1080;

                return (
                  <g key={zone.id} className="cursor-pointer">
                    <polygon
                      points={pointsString}
                      fill={zone.color || "#ef4444"}
                      fillOpacity={isBreached ? 0.55 : isSelected ? 0.35 : 0.22}
                      stroke={isBreached ? "#ff2222" : zone.color || "#ef4444"}
                      strokeWidth={isSelected ? 5 : 2.5}
                      strokeDasharray={isSelected ? "10 5" : undefined}
                      filter={isBreached ? "url(#glow-breach)" : undefined}
                      className={isBreached ? "animate-pulse" : ""}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedEntityId(zone.id);
                        setToolMode("select");
                      }}
                    />

                    {/* Polygon Centroid Label Pill */}
                    <g transform={`translate(${avgX}, ${avgY})`} className="pointer-events-none">
                      <rect
                        x="-100"
                        y="-18"
                        width="200"
                        height="36"
                        rx="8"
                        fill="#020617"
                        fillOpacity="0.88"
                        stroke={zone.color}
                        strokeWidth="2"
                      />
                      <text
                        x="0"
                        y="5"
                        textAnchor="middle"
                        fill="#ffffff"
                        fontSize="13"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        {zone.name.length > 20 ? zone.name.slice(0, 19) + "…" : zone.name}
                      </text>
                    </g>

                    {/* Draggable Vertex Handles (Circular) */}
                    {isSelected &&
                      zone.polygon.map(([vx, vy], vIdx) => (
                        <circle
                          key={vIdx}
                          cx={vx * 1920}
                          cy={vy * 1080}
                          r={10}
                          fill="#ffffff"
                          stroke={zone.color}
                          strokeWidth={4}
                          className="cursor-move hover:scale-125 transition-transform"
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            setDraggedVertex({
                              entityId: zone.id,
                              entityType: "zone",
                              index: vIdx,
                            });
                          }}
                        />
                      ))}
                  </g>
                );
              })}

              {/* ── 2. Render Existing Directional Tripwires ─────────────────── */}
              {tripwires.map((wire) => {
                if (!wire.p1 || !wire.p2) return null;
                const isSelected = selectedEntityId === wire.id;
                const isBreached = simBreaches.some((b) => b.wireId === wire.id);

                const geom = getTripwireGeometry(wire.p1, wire.p2, wire.direction);

                return (
                  <g key={wire.id} className="cursor-pointer">
                    {/* Breach Glow */}
                    {isBreached && (
                      <line
                        x1={geom.x1}
                        y1={geom.y1}
                        x2={geom.x2}
                        y2={geom.y2}
                        stroke="#f43f5e"
                        strokeWidth={20}
                        strokeOpacity={0.65}
                        filter="url(#glow-breach)"
                        className="animate-pulse"
                      />
                    )}

                    {/* Main Line */}
                    <line
                      x1={geom.x1}
                      y1={geom.y1}
                      x2={geom.x2}
                      y2={geom.y2}
                      stroke={wire.color || "#38bdf8"}
                      strokeWidth={isSelected ? 6 : 3.5}
                      strokeDasharray={wire.direction === "BIDIRECTIONAL" ? "14 8" : undefined}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedEntityId(wire.id);
                        setToolMode("select");
                      }}
                    />

                    {/* Directional Vector Indicator Line */}
                    <line
                      x1={geom.midX}
                      y1={geom.midY}
                      x2={geom.arrowX}
                      y2={geom.arrowY}
                      stroke="#ffffff"
                      strokeWidth={3.5}
                      markerEnd="url(#arrow-marker)"
                    />

                    {/* Midpoint Directional Badge (Arrow Rotates to Exact Crossing Vector) */}
                    <g
                      transform={`translate(${geom.midX}, ${geom.midY})`}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedEntityId(wire.id);
                        setToolMode("select");
                      }}
                    >
                      <circle r={18} fill="#020617" stroke={wire.color} strokeWidth={2.5} />
                      {wire.direction === "BIDIRECTIONAL" ? (
                        <g transform={`rotate(${geom.angleDeg})`}>
                          <path
                            d="M -9 0 L 9 0 M -6 -4 L -9 0 L -6 4 M 6 -4 L 9 0 L 6 4"
                            stroke="#38bdf8"
                            strokeWidth={2.5}
                            fill="none"
                          />
                        </g>
                      ) : (
                        <g transform={`rotate(${geom.angleDeg})`}>
                          <path
                            d="M -7 0 L 7 0 M 3 -4 L 7 0 L 3 4"
                            stroke="#ffffff"
                            strokeWidth={2.5}
                            fill="none"
                          />
                        </g>
                      )}
                    </g>

                    {/* Tripwire Label Pill */}
                    <g
                      transform={`translate(${geom.midX}, ${geom.midY - 32})`}
                      className="pointer-events-none"
                    >
                      <rect
                        x="-105"
                        y="-14"
                        width="210"
                        height="28"
                        rx="6"
                        fill="#020617"
                        fillOpacity="0.9"
                        stroke={wire.color}
                        strokeWidth="1.5"
                      />
                      <text
                        x="0"
                        y="4"
                        textAnchor="middle"
                        fill="#ffffff"
                        fontSize="11"
                        fontWeight="bold"
                        fontFamily="monospace"
                      >
                        {wire.name.length > 22 ? wire.name.slice(0, 21) + "…" : wire.name}
                      </text>
                    </g>

                    {/* Endpoint Handles when Selected */}
                    {isSelected && (
                      <>
                        <circle
                          cx={geom.x1}
                          cy={geom.y1}
                          r={10}
                          fill="#ffffff"
                          stroke={wire.color}
                          strokeWidth={4}
                          className="cursor-move hover:scale-125 transition-transform"
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            setDraggedVertex({
                              entityId: wire.id,
                              entityType: "tripwire",
                              index: 0,
                            });
                          }}
                        />
                        <circle
                          cx={geom.x2}
                          cy={geom.y2}
                          r={10}
                          fill="#ffffff"
                          stroke={wire.color}
                          strokeWidth={4}
                          className="cursor-move hover:scale-125 transition-transform"
                          onMouseDown={(e) => {
                            e.stopPropagation();
                            setDraggedVertex({
                              entityId: wire.id,
                              entityType: "tripwire",
                              index: 1,
                            });
                          }}
                        />
                      </>
                    )}
                  </g>
                );
              })}

              {/* ── 3. Active Drawing In-Progress Overlay ─────────────────────── */}
              {toolMode === "polygon" && activePoints.length > 0 && (
                <g>
                  {activePoints.map(([px, py], i) => (
                    <circle
                      key={i}
                      cx={px * 1920}
                      cy={py * 1080}
                      r={i === 0 ? 12 : 8}
                      fill={i === 0 ? "#10b981" : "#ffffff"}
                      stroke="#ef4444"
                      strokeWidth={3}
                      className={i === 0 ? "animate-pulse" : ""}
                    />
                  ))}
                  <polyline
                    points={activePoints
                      .map(([px, py]) => `${px * 1920},${py * 1080}`)
                      .join(" ")}
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth={3}
                    strokeDasharray="8 5"
                  />
                  {mousePos && (
                    <line
                      x1={activePoints[activePoints.length - 1][0] * 1920}
                      y1={activePoints[activePoints.length - 1][1] * 1080}
                      x2={mousePos[0] * 1920}
                      y2={mousePos[1] * 1080}
                      stroke="#f59e0b"
                      strokeWidth={3}
                      strokeDasharray="6 4"
                    />
                  )}
                </g>
              )}

              {toolMode === "tripwire" && activePoints.length === 1 && mousePos && (
                <g>
                  <circle
                    cx={activePoints[0][0] * 1920}
                    cy={activePoints[0][1] * 1080}
                    r={10}
                    fill="#38bdf8"
                    stroke="#ffffff"
                    strokeWidth={3}
                  />
                  <line
                    x1={activePoints[0][0] * 1920}
                    y1={activePoints[0][1] * 1080}
                    x2={mousePos[0] * 1920}
                    y2={mousePos[1] * 1080}
                    stroke="#38bdf8"
                    strokeWidth={4}
                    strokeDasharray="8 6"
                  />
                  <circle
                    cx={mousePos[0] * 1920}
                    cy={mousePos[1] * 1080}
                    r={8}
                    fill="#ffffff"
                    stroke="#38bdf8"
                    strokeWidth={3}
                  />
                </g>
              )}

              {/* ── 4. Target Intrusion Simulator Beacon ─────────────────────── */}
              {toolMode === "simulate" && (
                <g transform={`translate(${simPos[0] * 1920}, ${simPos[1] * 1080})`}>
                  {/* Trajectory Vector from Previous Position */}
                  <line
                    x1={(simPrevPos[0] - simPos[0]) * 1920}
                    y1={(simPrevPos[1] - simPos[1]) * 1080}
                    x2={0}
                    y2={0}
                    stroke={
                      simIdentity === "authorized"
                        ? "#10b981"
                        : simIdentity === "unknown"
                        ? "#f59e0b"
                        : "#f43f5e"
                    }
                    strokeWidth={4}
                    strokeDasharray="6 4"
                  />
                  <circle
                    r={28}
                    fill={
                      simIdentity === "authorized"
                        ? "#10b981"
                        : simIdentity === "unknown"
                        ? "#f59e0b"
                        : "#f43f5e"
                    }
                    fillOpacity={0.25}
                    className="animate-ping"
                  />
                  <circle
                    r={14}
                    fill={
                      simIdentity === "authorized"
                        ? "#10b981"
                        : simIdentity === "unknown"
                        ? "#f59e0b"
                        : "#f43f5e"
                    }
                    stroke="#ffffff"
                    strokeWidth={3}
                  />
                  <g transform="translate(0, -26)">
                    <rect
                      x="-80"
                      y="-12"
                      width="160"
                      height="24"
                      rx="6"
                      fill="#020617"
                      fillOpacity="0.9"
                      stroke={
                        simIdentity === "authorized"
                          ? "#10b981"
                          : simIdentity === "unknown"
                          ? "#f59e0b"
                          : "#f43f5e"
                      }
                      strokeWidth="1.5"
                    />
                    <text
                      x="0"
                      y="3"
                      textAnchor="middle"
                      fill="#ffffff"
                      fontSize="11"
                      fontWeight="bold"
                      fontFamily="monospace"
                    >
                      {simIdentity === "authorized"
                        ? "STAFF: HARIOM"
                        : simIdentity === "unknown"
                        ? "UNKNOWN INTRUDER"
                        : "SUSPECT #901"}
                    </text>
                  </g>
                </g>
              )}
            </svg>

            {/* Bottom HUD Coordinates Overlay */}
            <div className="absolute bottom-3 left-3 right-3 z-20 flex items-center justify-between pointer-events-none">
              <div className="bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-lg text-white text-xs font-mono border border-slate-700/60 shadow-md">
                Active Fences:{" "}
                <span className="font-bold text-emerald-400">
                  {zones.filter((z) => z.enabled).length} Zones
                </span>{" "}
                •{" "}
                <span className="font-bold text-blue-400">
                  {tripwires.filter((w) => w.enabled).length} Tripwires
                </span>
              </div>

              {mousePos && (
                <div className="bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-lg text-slate-300 text-xs font-mono border border-slate-700/60 shadow-md">
                  FOV Norm: ({mousePos[0].toFixed(3)}, {mousePos[1].toFixed(3)})
                </div>
              )}
            </div>
          </div>

          {/* Quick Simulation Action Bar */}
          <div className="bg-slate-50 border border-slate-200 rounded-2xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-2xs">
            <div>
              <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Intrusion Breach Simulation & Trajectory Testing
              </h4>
              <p className="text-xs text-slate-500 mt-0.5">
                Simulate a real-time target crossing border tripwires or entering exclusion zones.
              </p>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              {/* Identity State Selector */}
              <div className="flex items-center gap-1 bg-white border border-slate-200 p-0.5 rounded-xl text-[11px] font-semibold">
                <button
                  type="button"
                  onClick={() => setSimIdentity("suspect")}
                  className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                    simIdentity === "suspect"
                      ? "bg-rose-600 text-white shadow-2xs font-bold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                  title="Simulate a flagged suspect (triggers CRITICAL breach alert)"
                >
                  🚨 Suspect
                </button>
                <button
                  type="button"
                  onClick={() => setSimIdentity("unknown")}
                  className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                    simIdentity === "unknown"
                      ? "bg-amber-500 text-white shadow-2xs font-bold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                  title="Simulate an unknown/unregistered intruder (triggers intrusion alert)"
                >
                  ⚠️ Unknown
                </button>
                <button
                  type="button"
                  onClick={() => setSimIdentity("authorized")}
                  className={`px-2.5 py-1 rounded-lg transition-colors cursor-pointer ${
                    simIdentity === "authorized"
                      ? "bg-emerald-600 text-white shadow-2xs font-bold"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                  title="Simulate authorized staff (silent pass in Authorized Pass zones)"
                >
                  ✓ Staff (Silent)
                </button>
              </div>

              <button
                type="button"
                onClick={() => {
                  setToolMode("simulate");
                  // Animate simulated dot downward across center
                  let step = 0;
                  const interval = setInterval(() => {
                    step += 1;
                    const nextY = 0.2 + step * 0.08;
                    runSimulationStep([0.5, nextY]);
                    if (nextY >= 0.85) {
                      clearInterval(interval);
                    }
                  }, 400);
                }}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold shadow-xs transition-colors cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 text-emerald-400" />
                <span>Auto-Traverse Center</span>
              </button>

              <button
                type="button"
                onClick={() => {
                  api.resetGeofenceHistory(selectedCamId);
                  setSimBreaches([]);
                  setSimSilentPassMsg(null);
                }}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-slate-200 bg-white hover:bg-slate-100 text-slate-700 text-xs font-semibold shadow-2xs transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
                <span>Reset Cooldowns</span>
              </button>
            </div>
          </div>
        </div>

        {/* Right: Inspector & Configuration Panel (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Properties Form (Active or Selected Entity) */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-emerald-600" />
                <h3 className="text-sm font-bold text-slate-900">
                  {selectedEntityId ? "Edit Barrier Properties" : "Barrier Configuration"}
                </h3>
              </div>
              {selectedEntityId && (
                <span className="text-[11px] font-mono text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                  {selectedEntityId.slice(0, 12)}
                </span>
              )}
            </div>

            {selectedEntityId ? (
              <div className="space-y-3.5 text-xs">
                <div>
                  <label className="block text-slate-600 font-semibold mb-1">Barrier Designation</label>
                  <input
                    type="text"
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-slate-900 font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    placeholder="e.g. RS Pura Forward Exclusion"
                  />
                </div>

                <div className="grid grid-cols-2 gap-2.5">
                  <div>
                    <label className="block text-slate-600 font-semibold mb-1">Threat Severity</label>
                    <select
                      value={editSeverity}
                      onChange={(e) => setEditSeverity(e.target.value as GeofenceSeverity)}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-2 text-slate-900 font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                    >
                      <option value="CRITICAL">CRITICAL</option>
                      <option value="HIGH">HIGH</option>
                      <option value="MEDIUM">MEDIUM</option>
                      <option value="LOW">LOW</option>
                    </select>
                  </div>

                  {tripwires.some((w) => w.id === selectedEntityId) && (
                    <div>
                      <label className="block text-slate-600 font-semibold mb-1">Direction</label>
                      <select
                        value={editDirection}
                        onChange={(e) => setEditDirection(e.target.value as TripwireDirection)}
                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-2 text-slate-900 font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer"
                      >
                        <option value="FORWARD">FORWARD (A➔B)</option>
                        <option value="REVERSE">REVERSE (B➔A)</option>
                        <option value="BIDIRECTIONAL">BIDIRECTIONAL (⇄)</option>
                      </select>
                    </div>
                  )}
                </div>

                {/* Lightweight Access Control Policy */}
                <div>
                  <label className="block text-slate-600 font-semibold mb-1">Access Control</label>
                  <select
                    value={editAccessMode}
                    onChange={(e) => setEditAccessMode(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-2.5 py-2 text-slate-900 font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer text-xs"
                  >
                    <option value="AUTHORIZED_PASS">Authorized Pass (Staff silent, alert suspects & unknowns)</option>
                    <option value="ALL_ALERTS">All Alerts (Standard - alert any person)</option>
                    <option value="SUSPECT_ONLY">Suspect Only (Ignore staff & unknowns)</option>
                  </select>
                  <p className="text-[10px] text-slate-500 mt-1 font-mono">
                    {editAccessMode === "AUTHORIZED_PASS" && "✓ Staff in database enter silently. Suspects & unknowns trigger alert."}
                    {editAccessMode === "ALL_ALERTS" && "⚠ Everyone crossing boundary triggers an alert."}
                    {editAccessMode === "SUSPECT_ONLY" && "⚠ Only flagged database suspects trigger alert."}
                  </p>
                </div>

                <div>
                  <label className="block text-slate-600 font-semibold mb-1">Display Color</label>
                  <div className="flex items-center gap-2 flex-wrap">
                    {COLOR_PRESETS.map((col) => (
                      <button
                        key={col.hex}
                        type="button"
                        onClick={() => setEditColor(col.hex)}
                        className={`w-6 h-6 rounded-full border transition-transform cursor-pointer ${
                          editColor === col.hex
                            ? "scale-125 border-slate-900 ring-2 ring-emerald-400"
                            : "border-transparent"
                        }`}
                        style={{ backgroundColor: col.hex }}
                        title={col.name}
                      />
                    ))}
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between text-slate-600 font-semibold mb-1">
                    <span>Alert Cooldown</span>
                    <span className="font-mono text-emerald-600">{editCooldown}s</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={120}
                    step={5}
                    value={editCooldown}
                    onChange={(e) => setEditCooldown(Number(e.target.value))}
                    className="w-full accent-emerald-600 cursor-pointer"
                  />
                </div>

                <div>
                  <label className="block text-slate-600 font-semibold mb-1">Notes / Mission Directive</label>
                  <textarea
                    rows={2}
                    value={editDescription}
                    onChange={(e) => setEditDescription(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl p-2.5 text-slate-900 font-medium focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    placeholder="Instructions for Quick Reaction Team..."
                  />
                </div>

                {saveStatus && (
                  <div className="p-2 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-semibold flex items-center gap-1.5">
                    <Check className="w-3.5 h-3.5" />
                    <span>{saveStatus}</span>
                  </div>
                )}

                <div className="pt-2 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleSaveEntity}
                    className="flex-1 py-2 px-3 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-xs shadow-xs transition-colors cursor-pointer text-center"
                  >
                    Save Changes
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      const isZone = zones.some((z) => z.id === selectedEntityId);
                      handleDeleteEntity(selectedEntityId, isZone ? "zone" : "tripwire");
                    }}
                    className="p-2 rounded-xl border border-rose-200 bg-rose-50 hover:bg-rose-100 text-rose-600 transition-colors cursor-pointer"
                    title="Delete barrier"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="py-8 text-center text-slate-400 space-y-2.5">
                <MousePointer className="w-8 h-8 mx-auto text-slate-300" />
                <p className="text-xs font-medium">Select a zone or tripwire from the canvas or list below to inspect.</p>
              </div>
            )}
          </div>

          {/* Configured Barriers List */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-5 shadow-xs space-y-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2.5">
              <h3 className="text-sm font-bold text-slate-900">Configured Perimeter Items</h3>
              <span className="text-xs font-semibold text-slate-500">
                {zones.length + tripwires.length} Total
              </span>
            </div>

            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
              {zones.map((zone) => (
                <div
                  key={zone.id}
                  onClick={() => {
                    setSelectedEntityId(zone.id);
                    setToolMode("select");
                  }}
                  className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                    selectedEntityId === zone.id
                      ? "border-emerald-500 bg-emerald-50/60 shadow-xs ring-1 ring-emerald-500"
                      : "border-slate-200/80 bg-slate-50/50 hover:bg-slate-100"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <span
                      className="w-3.5 h-3.5 rounded-full shrink-0 shadow-2xs"
                      style={{ backgroundColor: zone.color }}
                    />
                    <div>
                      <h4 className="text-xs font-bold text-slate-800 leading-tight">{zone.name}</h4>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-[10px] font-semibold text-slate-400">
                          {zone.polygon.length} Vertices
                        </span>
                        <span className="text-slate-300">•</span>
                        <span
                          className={`text-[9px] font-extrabold uppercase px-1.5 py-0.2 rounded ${
                            zone.severity === "CRITICAL"
                              ? "bg-rose-100 text-rose-700"
                              : "bg-amber-100 text-amber-700"
                          }`}
                        >
                          {zone.severity}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleEntityEnabled(zone.id, "zone");
                      }}
                      className={`px-2.5 py-1 rounded text-[10px] font-bold cursor-pointer transition-colors ${
                        zone.enabled
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-slate-200 text-slate-500"
                      }`}
                    >
                      {zone.enabled ? "ACTIVE" : "OFF"}
                    </button>
                  </div>
                </div>
              ))}

              {tripwires.map((wire) => (
                <div
                  key={wire.id}
                  onClick={() => {
                    setSelectedEntityId(wire.id);
                    setToolMode("select");
                  }}
                  className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                    selectedEntityId === wire.id
                      ? "border-blue-500 bg-blue-50/60 shadow-xs ring-1 ring-blue-500"
                      : "border-slate-200/80 bg-slate-50/50 hover:bg-slate-100"
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <div
                      className="w-3.5 h-3.5 rounded-sm shrink-0 flex items-center justify-center text-[9px] text-white font-bold"
                      style={{ backgroundColor: wire.color }}
                    >
                      ➔
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-slate-800 leading-tight">{wire.name}</h4>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-[10px] font-semibold text-slate-400">
                          Tripwire ({wire.direction})
                        </span>
                        <span className="text-slate-300">•</span>
                        <span
                          className={`text-[9px] font-extrabold uppercase px-1.5 py-0.2 rounded ${
                            wire.severity === "CRITICAL"
                              ? "bg-rose-100 text-rose-700"
                              : "bg-blue-100 text-blue-700"
                          }`}
                        >
                          {wire.severity}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleEntityEnabled(wire.id, "tripwire");
                      }}
                      className={`px-2.5 py-1 rounded text-[10px] font-bold cursor-pointer transition-colors ${
                        wire.enabled
                          ? "bg-blue-100 text-blue-800"
                          : "bg-slate-200 text-slate-500"
                      }`}
                    >
                      {wire.enabled ? "ACTIVE" : "OFF"}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── ═══ 4. GLOBAL MULTI-CAMERA GEOFENCE & POLYGON REGISTRY ═══ ── */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 md:p-6 shadow-xs space-y-5">
        {/* Header & Metrics */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-800 flex items-center justify-center">
                <Layers className="w-4 h-4" />
              </div>
              <h2 className="text-base md:text-lg font-black tracking-tight text-slate-900">
                Global Multi-Camera Polygon & Barrier Registry
              </h2>
              <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 text-[10px] font-mono font-bold border border-slate-200">
                {allZones.length + allTripwires.length} Total
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-1 max-w-2xl">
              Active polygons across multiple cameras run continuous inference and increase processing load. Inspect, toggle, or delete them below to focus system resources on specific cameras.
            </p>
          </div>

          {/* Quick Actions & Processing Load */}
          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Processing Load Indicator */}
            <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 shadow-2xs">
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              <div className="text-[11px] font-medium text-slate-600">
                <span>Inference Load: </span>
                <span className={`font-mono font-bold px-1.5 py-0.5 rounded text-[10px] border ${loadLevel.color}`}>
                  {loadLevel.label}
                </span>
              </div>
            </div>

            {/* Isolate Current Camera button */}
            <button
              type="button"
              onClick={handleIsolateCurrentCamera}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 text-white text-xs font-bold shadow-2xs transition-colors cursor-pointer"
              title="Deactivate all polygons on other cameras to focus inference strictly on the selected camera"
            >
              <Crosshair className="w-3.5 h-3.5 text-emerald-400" />
              <span>Isolate Focused Camera</span>
            </button>

            {/* Master Disable All */}
            <button
              type="button"
              onClick={handleDisableAllGlobal}
              disabled={activeCount === 0}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 disabled:opacity-40 text-xs font-semibold shadow-2xs transition-colors cursor-pointer"
              title="Temporarily deactivate all geofences network-wide"
            >
              <Power className="w-3.5 h-3.5 text-amber-500" />
              <span>Disable All</span>
            </button>

            {/* Master Enable All */}
            <button
              type="button"
              onClick={handleEnableAllGlobal}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold shadow-2xs transition-colors cursor-pointer"
              title="Activate all configured geofences"
            >
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Enable All</span>
            </button>
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          {/* Tabs */}
          <div className="inline-flex items-center p-1 rounded-xl bg-slate-100 border border-slate-200 gap-1 flex-wrap">
            <button
              type="button"
              onClick={() => setGlobalFilterType("all")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                globalFilterType === "all"
                  ? "bg-white text-slate-900 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              All ({allZones.length + allTripwires.length})
            </button>
            <button
              type="button"
              onClick={() => setGlobalFilterType("active")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
                globalFilterType === "active"
                  ? "bg-emerald-600 text-white shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Active ({activeCount})
            </button>
            <button
              type="button"
              onClick={() => setGlobalFilterType("disabled")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                globalFilterType === "disabled"
                  ? "bg-slate-800 text-white shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Disabled ({allZones.length + allTripwires.length - activeCount})
            </button>
            <button
              type="button"
              onClick={() => setGlobalFilterType("zones")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                globalFilterType === "zones"
                  ? "bg-rose-600 text-white shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Polygons ({allZones.length})
            </button>
            <button
              type="button"
              onClick={() => setGlobalFilterType("tripwires")}
              className={`px-3 py-1 text-xs font-bold rounded-lg transition-all cursor-pointer ${
                globalFilterType === "tripwires"
                  ? "bg-blue-600 text-white shadow-2xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Tripwires ({allTripwires.length})
            </button>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Camera Filter Selector */}
            <select
              value={globalFilterCamera}
              onChange={(e) => setGlobalFilterCamera(e.target.value)}
              className="bg-slate-50 border border-slate-200 text-xs font-semibold text-slate-800 rounded-xl px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-emerald-500 cursor-pointer"
            >
              <option value="all">All Cameras ({cameras.length})</option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.id})
                </option>
              ))}
            </select>

            {/* Search Input */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-2.5" />
              <input
                type="text"
                value={globalSearchQuery}
                onChange={(e) => setGlobalSearchQuery(e.target.value)}
                placeholder="Filter by name or camera..."
                className="bg-slate-50 border border-slate-200 rounded-xl pl-8 pr-3 py-1.5 text-xs text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-500 w-44 sm:w-56"
              />
            </div>
          </div>
        </div>

        {/* Global Barriers Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
          {filteredGlobalItems.map((item) => {
            const isZone = item.itemType === "zone";
            const camId = item.cameraId || "bop-jk-01";
            const camObj = cameras.find((c) => c.id === camId);
            const camName = camObj?.name || camId;
            const isCurrentCamera = camId === selectedCamId;

            return (
              <div
                key={item.id}
                className={`rounded-xl border p-3.5 transition-all space-y-3 ${
                  item.enabled
                    ? isCurrentCamera
                      ? "bg-emerald-50/40 border-emerald-300/80 shadow-2xs ring-1 ring-emerald-500/30"
                      : "bg-white border-slate-200 shadow-2xs hover:border-slate-300"
                    : "bg-slate-50/80 border-slate-200/70 opacity-75"
                }`}
              >
                {/* Header: Camera Badge & Active Toggle */}
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <CameraIcon className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                    <span
                      className="text-xs font-bold text-slate-800 truncate"
                      title={`${camName} (${camId})`}
                    >
                      {camName}
                    </span>
                    {isCurrentCamera ? (
                      <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-bold bg-emerald-100 text-emerald-800 border border-emerald-200 shrink-0">
                        FOCUSED
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.2 rounded text-[9px] font-mono font-medium text-slate-500 bg-slate-100 shrink-0">
                        {camId}
                      </span>
                    )}
                  </div>

                  {/* Enable / Disable Switch */}
                  <button
                    type="button"
                    onClick={() => toggleEntityEnabled(item.id, isZone ? "zone" : "tripwire")}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-mono font-bold tracking-tight cursor-pointer transition-all flex items-center gap-1 shrink-0 ${
                      item.enabled
                        ? "bg-emerald-600 text-white shadow-2xs hover:bg-emerald-500"
                        : "bg-slate-200 text-slate-600 hover:bg-slate-300"
                    }`}
                    title={item.enabled ? "Click to disable this barrier" : "Click to activate this barrier"}
                  >
                    <Power className={`w-3 h-3 ${item.enabled ? "text-emerald-200" : "text-slate-400"}`} />
                    <span>{item.enabled ? "ACTIVE" : "OFF"}</span>
                  </button>
                </div>

                {/* Name, Type & Color */}
                <div className="flex items-start justify-between gap-2 pt-1 border-t border-slate-100">
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className="w-3.5 h-3.5 rounded-full shrink-0 shadow-2xs"
                      style={{ backgroundColor: item.color || (isZone ? "#ef4444" : "#38bdf8") }}
                    />
                    <div className="min-w-0">
                      <h4 className="text-xs font-bold text-slate-900 truncate" title={item.name}>
                        {item.name}
                      </h4>
                      <div className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
                        {isZone ? (
                          <span>Polygon ({item.polygon.length} pts)</span>
                        ) : (
                          <span>Tripwire ({(item as DirectionalTripwire).direction || "FORWARD"})</span>
                        )}
                        <span>•</span>
                        <span
                          className={`font-bold uppercase ${
                            item.severity === "CRITICAL"
                              ? "text-rose-600"
                              : item.severity === "HIGH"
                              ? "text-orange-600"
                              : "text-amber-600"
                          }`}
                        >
                          {item.severity}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Target Classes Chips & Cooldown */}
                <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 gap-2">
                  <div className="flex items-center gap-1 flex-wrap">
                    <span
                      className={`px-1.5 py-0.2 rounded font-semibold text-[9px] ${
                        item.accessMode === "ALL_ALERTS"
                          ? "bg-rose-50 text-rose-700"
                          : item.accessMode === "SUSPECT_ONLY"
                          ? "bg-amber-50 text-amber-700"
                          : "bg-emerald-50 text-emerald-700"
                      }`}
                    >
                      {item.accessMode === "ALL_ALERTS" ? "ALL ALERTS" : item.accessMode === "SUSPECT_ONLY" ? "SUSPECT ONLY" : "AUTH PASS"}
                    </span>
                    {(item.targetClasses || ["person"]).slice(0, 2).map((cls) => (
                      <span
                        key={cls}
                        className="px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-medium text-[9px]"
                      >
                        {cls}
                      </span>
                    ))}
                    {(item.targetClasses || []).length > 2 && (
                      <span className="text-slate-400 text-[9px]">
                        +{(item.targetClasses || []).length - 2}
                      </span>
                    )}
                  </div>
                  <span className="shrink-0">{item.cooldownSeconds || 15}s cd</span>
                </div>

                {/* Card Action Buttons: Focus Camera + Delete */}
                <div className="pt-2 border-t border-slate-100 flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => handleFocusCamera(camId, item.id)}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 transition-colors cursor-pointer"
                    title="Switch camera and display on canvas above"
                  >
                    <ExternalLink className="w-3 h-3 text-slate-500" />
                    <span>Focus Camera</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleDeleteEntity(item.id, isZone ? "zone" : "tripwire")}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                    title="Delete polygon permanently"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Empty State */}
        {filteredGlobalItems.length === 0 && (
          <div className="py-12 text-center text-slate-400 space-y-2 border border-dashed border-slate-200 rounded-xl">
            <Shield className="w-8 h-8 mx-auto text-slate-300" />
            <p className="text-xs font-semibold text-slate-600">No barriers match your filter.</p>
            <p className="text-[11px] text-slate-400">
              Draw polygons or tripwires on any camera using the Live Studio above.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
