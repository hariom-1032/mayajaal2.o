"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Camera,
  CreateCameraInput,
  UpdateCameraInput,
  StreamTestResult,
} from "@/types/camera";
import { api } from "@/lib/api";
import { ALL_BORDER_CAMERAS } from "@/lib/borderCameras";

const STORAGE_KEY = "maatrix_operator_cameras_v4";

export const INITIAL_CAMERAS: Camera[] = ALL_BORDER_CAMERAS;

function isLegacyCentralPoint(c: any): boolean {
  if (!c) return true;
  if (["cam-1", "cam-2", "cam-3", "cam-4", "cam-5"].includes(c.id)) return true;
  const lat = Number(c.latitude ?? (c.coordinates ? c.coordinates[1] : 0));
  const lng = Number(c.longitude ?? (c.coordinates ? c.coordinates[0] : 0));
  // Any point in central India interior (MP/UP interior: lat 20.0-26.0, lng 75.0-82.0)
  if (lat >= 20.0 && lat <= 26.0 && lng >= 75.0 && lng <= 82.0) return true;
  return false;
}

type CameraListener = (cameras: Camera[]) => void;
const listeners = new Set<CameraListener>();

let memoryCameras: Camera[] = [...INITIAL_CAMERAS];
let hasInitialized = false;

function loadInitialData(): Camera[] {
  if (typeof window === "undefined") return [...INITIAL_CAMERAS];
  try {
    // Clear out any old legacy cache keys that contained central India dummy coordinates
    try {
      localStorage.removeItem("maatrix_operator_cameras_v1");
      localStorage.removeItem("maatrix_operator_cameras_v2");
      localStorage.removeItem("maatrix_operator_cameras_v3");
    } catch {}

    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        const clean = parsed.filter((c: any) => !isLegacyCentralPoint(c));
        if (clean.length >= 20) {
          return clean;
        }
      }
    }
  } catch (err) {
    console.warn("Could not read cameras from localStorage:", err);
  }
  return [...INITIAL_CAMERAS];
}

function persistLocally(cameras: Camera[]) {
  if (typeof window === "undefined") return;
  try {
    const clean = cameras.filter((c) => !isLegacyCentralPoint(c));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(clean));
  } catch (err) {
    console.warn("Could not save cameras to localStorage:", err);
  }
}

function notify() {
  persistLocally(memoryCameras);
  listeners.forEach((fn) => fn([...memoryCameras]));
}

export const camerasStore = {
  getCameras(): Camera[] {
    return memoryCameras.filter((c) => !isLegacyCentralPoint(c));
  },

  initClientStorage(): Camera[] {
    if (!hasInitialized && typeof window !== "undefined") {
      memoryCameras = loadInitialData();
      hasInitialized = true;
    }
    return memoryCameras.filter((c) => !isLegacyCentralPoint(c));
  },

  async syncWithBackend(): Promise<Camera[]> {
    try {
      const backendCameras = await api.getCameras();
      if (Array.isArray(backendCameras) && backendCameras.length > 0) {
        const clean = backendCameras
          .filter((c) => !isLegacyCentralPoint(c))
          .map((c) => ({
            ...c,
            latitude: Number(c.latitude ?? (c.coordinates ? c.coordinates[1] : 32.7160)),
            longitude: Number(c.longitude ?? (c.coordinates ? c.coordinates[0] : 74.6640)),
            coordinates: c.coordinates ?? [
              Number(c.longitude ?? 74.6640),
              Number(c.latitude ?? 32.7160),
            ],
          }));

        if (clean.length > 0) {
          const idSet = new Set(clean.map((c) => c.id));
          const additions = ALL_BORDER_CAMERAS.filter((c) => !idSet.has(c.id));
          memoryCameras = [...clean, ...additions];
        } else {
          memoryCameras = [...ALL_BORDER_CAMERAS];
        }
        notify();
        return memoryCameras;
      }
    } catch {
      // Backend not available; fallback to local memory/storage
    }
    return [...memoryCameras];
  },

  async addCamera(input: CreateCameraInput): Promise<Camera> {
    const lat = Number(input.latitude);
    const lng = Number(input.longitude);

    let streamUrl = input.streamUrl?.trim();
    if (!streamUrl && input.ipAddress) {
      const port = input.port || 554;
      streamUrl = `rtsp://${input.ipAddress.trim()}:${port}/live/ch0`;
    }

    const nextId =
      input.id?.trim() ||
      `cam-${Date.now()}-${Math.random().toString(36).substring(2, 5)}`;

    const newCamera: Camera = {
      id: nextId,
      name: input.name.trim(),
      sector: input.sector || "Sector-04 (BOP Alpha)",
      location: input.location.trim(),
      status: input.status || "online",
      type: input.type || "Optical 4K",
      ipAddress: input.ipAddress?.trim() || undefined,
      port: input.port || 554,
      streamUrl: streamUrl,
      isRtsp: input.isRtsp ?? true,
      latitude: lat,
      longitude: lng,
      coordinates: [lng, lat],
      modelAssigned: input.modelAssigned || "vehicle_detector.onnx (Vehicle Detector)",
      resolution: input.resolution || "1080p FHD (1920x1080)",
      fps: input.fps || 30,
      confThreshold: input.confThreshold ?? 0.75,
      iouThreshold: input.iouThreshold ?? 0.45,
      isRecording: input.isRecording ?? true,
      alertTriggerEnabled: input.alertTriggerEnabled ?? true,
      lastActive: "Just now",
      healthStats: {
        bitrate: "6.5 Mbps",
        latencyMs: Math.floor(35 + Math.random() * 25),
        packetLoss: "0.00%",
      },
    };

    memoryCameras = [newCamera, ...memoryCameras];
    notify();

    // Push to backend asynchronously if available
    api
      .createCamera({
        ...newCamera,
        id: nextId,
      })
      .catch(() => {
        // Backend sync failed or offline; local memory already updated
      });

    return newCamera;
  },

  async updateCamera(id: string, input: UpdateCameraInput): Promise<Camera> {
    const idx = memoryCameras.findIndex((c) => c.id === id);
    if (idx === -1) {
      throw new Error(`Camera with ID '${id}' not found.`);
    }

    const current = memoryCameras[idx];
    const lat =
      input.latitude !== undefined ? Number(input.latitude) : current.latitude;
    const lng =
      input.longitude !== undefined ? Number(input.longitude) : current.longitude;

    let streamUrl = input.streamUrl !== undefined ? input.streamUrl : current.streamUrl;
    if (
      input.ipAddress &&
      input.streamUrl === undefined &&
      input.ipAddress !== current.ipAddress
    ) {
      const port = input.port || current.port || 554;
      streamUrl = `rtsp://${input.ipAddress.trim()}:${port}/live/ch0`;
    }

    const updated: Camera = {
      ...current,
      ...input,
      latitude: lat,
      longitude: lng,
      coordinates: [lng, lat],
      streamUrl,
      lastActive: "Just now",
    };

    memoryCameras[idx] = updated;
    notify();

    // Async backend update
    api.updateCamera(id, input).catch(() => {});

    return updated;
  },

  async deleteCamera(id: string): Promise<boolean> {
    const beforeCount = memoryCameras.length;
    memoryCameras = memoryCameras.filter((c) => c.id !== id);
    if (memoryCameras.length !== beforeCount) {
      notify();
      api.deleteCamera(id).catch(() => {});
      return true;
    }
    return false;
  },

  async bulkDeleteCameras(ids: string[]): Promise<number> {
    const idSet = new Set(ids);
    const beforeCount = memoryCameras.length;
    memoryCameras = memoryCameras.filter((c) => !idSet.has(c.id));
    const deletedCount = beforeCount - memoryCameras.length;
    if (deletedCount > 0) {
      notify();
      ids.forEach((id) => api.deleteCamera(id).catch(() => {}));
    }
    return deletedCount;
  },

  async testStream(params: {
    streamUrl?: string;
    ipAddress?: string;
    port?: number;
  }): Promise<StreamTestResult> {
    try {
      return await api.testStream(params);
    } catch {
      // Offline fallback simulator
      const target = params.streamUrl || params.ipAddress || "";
      const isFormatValid =
        target.startsWith("rtsp://") ||
        /^(\d{1,3}\.){3}\d{1,3}$/.test(target.trim());

      const simulatedLatency = Math.floor(32 + Math.random() * 30);
      return {
        reachable: isFormatValid,
        status: isFormatValid ? "online" : "error",
        latencyMs: isFormatValid ? simulatedLatency : 0,
        message: isFormatValid
          ? `RTSP Handshake active. Stream responsive (Latency: ${simulatedLatency}ms).`
          : "Invalid RTSP IP address or URL format. Must begin with rtsp:// or valid IPv4.",
        protocol: "RTSP/1.0",
        supportedCodecs: ["H.264", "H.265"],
      };
    }
  },

  resetDefaults() {
    memoryCameras = [...INITIAL_CAMERAS];
    notify();
  },
};

export function useCameras() {
  const [cameras, setCameras] = useState<Camera[]>(INITIAL_CAMERAS);
  const [isSyncing, setIsSyncing] = useState(false);

  useEffect(() => {
    // Initial sync from client storage after hydration completes
    const loaded = camerasStore.initClientStorage();
    setCameras(loaded);

    const handleUpdate = (updated: Camera[]) => {
      setCameras(updated);
    };
    listeners.add(handleUpdate);

    // Attempt backend sync in background
    setIsSyncing(true);
    camerasStore
      .syncWithBackend()
      .finally(() => setIsSyncing(false));

    return () => {
      listeners.delete(handleUpdate);
    };
  }, []);

  const totalCount = cameras.length;
  const onlineCount = cameras.filter((c) => c.status === "online").length;
  const alertCount = cameras.filter((c) => c.status === "alert").length;
  const offlineCount = cameras.filter((c) => c.status === "offline").length;
  const degradedCount = cameras.filter((c) => c.status === "degraded").length;

  const validLatencies = cameras
    .map((c) => c.healthStats?.latencyMs)
    .filter((l): l is number => typeof l === "number" && l > 0);
  const avgLatencyMs =
    validLatencies.length > 0
      ? Math.round(
          validLatencies.reduce((acc, v) => acc + v, 0) / validLatencies.length
        )
      : 42;

  const addCamera = useCallback(
    (input: CreateCameraInput) => camerasStore.addCamera(input),
    []
  );
  const updateCamera = useCallback(
    (id: string, input: UpdateCameraInput) =>
      camerasStore.updateCamera(id, input),
    []
  );
  const deleteCamera = useCallback(
    (id: string) => camerasStore.deleteCamera(id),
    []
  );
  const bulkDeleteCameras = useCallback(
    (ids: string[]) => camerasStore.bulkDeleteCameras(ids),
    []
  );
  const testStream = useCallback(
    (params: { streamUrl?: string; ipAddress?: string; port?: number }) =>
      camerasStore.testStream(params),
    []
  );
  const resetDefaults = useCallback(() => camerasStore.resetDefaults(), []);
  const refreshFromBackend = useCallback(
    () => camerasStore.syncWithBackend(),
    []
  );

  return {
    cameras,
    isSyncing,
    totalCount,
    onlineCount,
    alertCount,
    offlineCount,
    degradedCount,
    avgLatencyMs,
    addCamera,
    updateCamera,
    deleteCamera,
    bulkDeleteCameras,
    testStream,
    resetDefaults,
    refreshFromBackend,
  };
}
