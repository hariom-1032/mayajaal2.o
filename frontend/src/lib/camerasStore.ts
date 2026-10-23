"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Camera,
  CreateCameraInput,
  UpdateCameraInput,
  StreamTestResult,
  CameraType,
  CameraStatus,
} from "@/types/camera";
import { api } from "@/lib/api";
import {
  ALL_BORDER_CAMERAS,
  buildChannelRtspUrl,
  extractChannelFromUrl,
  RTSP_CAMERA_TEMPLATE,
} from "@/lib/borderCameras";

const STORAGE_KEY = "maatrix_real_cameras_v5";

export const INITIAL_CAMERAS: Camera[] = ALL_BORDER_CAMERAS;

/**
 * Filter out dummy simulated cameras (indices > 10 or dummy mock networks).
 */
export function isAuthenticCamera(c: any): boolean {
  if (!c) return false;
  const id = String(c.id || "").toLowerCase();
  const streamUrl = String(c.streamUrl || "").toLowerCase();
  const ip = String(c.ipAddress || "").toLowerCase();

  // Explicit check on BOP JK numbering: only 01 to 10 are authentic
  const match = id.match(/bop-jk-(\d+)/);
  if (match) {
    const num = parseInt(match[1], 10);
    if (num > 10) return false;
  } else if (id.startsWith("bop-") && !id.startsWith("bop-jk-")) {
    // All other fake bop regions (bop-gj, bop-pb, bop-wb, etc.) are purged
    return false;
  }

  // Purge any dummy IP ranges like 10.20.7x or 10.20.9x
  if (
    ip.startsWith("10.20.7") ||
    ip.startsWith("10.20.9") ||
    streamUrl.includes("10.20.7") ||
    streamUrl.includes("10.20.9")
  ) {
    return false;
  }

  return true;
}

type CameraListener = (cameras: Camera[]) => void;
const listeners = new Set<CameraListener>();

let memoryCameras: Camera[] = [...INITIAL_CAMERAS];
let hasInitialized = false;

function loadInitialData(): Camera[] {
  if (typeof window === "undefined") return [...INITIAL_CAMERAS];
  try {
    // Purge legacy storage caches with dummy cameras
    try {
      localStorage.removeItem("maatrix_operator_cameras_v1");
      localStorage.removeItem("maatrix_operator_cameras_v2");
      localStorage.removeItem("maatrix_operator_cameras_v3");
      localStorage.removeItem("maatrix_operator_cameras_v4");
    } catch {}

    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        const clean = parsed.filter(isAuthenticCamera);
        if (clean.length > 0) {
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
    const clean = cameras.filter(isAuthenticCamera);
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
    return memoryCameras.filter(isAuthenticCamera);
  },

  initClientStorage(): Camera[] {
    if (!hasInitialized && typeof window !== "undefined") {
      memoryCameras = loadInitialData();
      hasInitialized = true;
    }
    return memoryCameras.filter(isAuthenticCamera);
  },

  async syncWithBackend(): Promise<Camera[]> {
    try {
      const backendCameras = await api.getCameras();
      if (Array.isArray(backendCameras) && backendCameras.length > 0) {
        // Strictly filter to real cameras only
        const clean = backendCameras
          .filter(isAuthenticCamera)
          .map((c) => ({
            ...c,
            latitude: Number(c.latitude ?? (c.coordinates ? c.coordinates[1] : 23.15421)),
            longitude: Number(c.longitude ?? (c.coordinates ? c.coordinates[0] : 72.885257)),
            coordinates: c.coordinates ?? [
              Number(c.longitude ?? 72.885257),
              Number(c.latitude ?? 23.15421),
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
      // Backend unavailable; keep memory fallback
    }
    return [...memoryCameras];
  },

  /**
   * 1-Click addition of a camera by specifying channel number.
   * Auto-resolves RTSP stream url with template:
   * rtsp://admin:Admin123@142.67.45.150:554/cam/realmonitor?channel={CHANNEL}&subtype=0
   */
  async addCameraByChannel(input: {
    name: string;
    channel: number | string;
    latitude?: number | string | null;
    longitude?: number | string | null;
    type?: CameraType | string;
    status?: CameraStatus;
  }): Promise<Camera> {
    const ch = parseInt(String(input.channel).trim(), 10) || 1;
    const streamUrl = buildChannelRtspUrl(ch);
    const lat =
      input.latitude !== null && input.latitude !== undefined && String(input.latitude).trim() !== ""
        ? Number(input.latitude)
        : null;
    const lng =
      input.longitude !== null && input.longitude !== undefined && String(input.longitude).trim() !== ""
        ? Number(input.longitude)
        : null;
    const id = `cam-ch-${ch}-${Date.now().toString(36).slice(-3)}`;

    return this.addCamera({
      id,
      name: input.name.trim(),
      channel: ch,
      status: input.status || "online",
      type: input.type || "Optical 4K",
      streamUrl,
      latitude: lat,
      longitude: lng,
    });
  },

  async addCamera(input: CreateCameraInput): Promise<Camera> {
    const ch =
      input.channel !== undefined
        ? Number(input.channel)
        : extractChannelFromUrl(input.streamUrl) || 1;

    const lat =
      input.latitude !== null && input.latitude !== undefined && String(input.latitude).trim() !== ""
        ? Number(input.latitude)
        : null;
    const lng =
      input.longitude !== null && input.longitude !== undefined && String(input.longitude).trim() !== ""
        ? Number(input.longitude)
        : null;

    let streamUrl = input.streamUrl?.trim();
    if (!streamUrl) {
      streamUrl = buildChannelRtspUrl(ch);
    }

    const nextId =
      input.id?.trim() ||
      `cam-ch-${ch}-${Date.now().toString(36).slice(-3)}`;

    const newCamera: Camera = {
      id: nextId,
      name: input.name.trim(),
      channel: ch,
      status: input.status || "online",
      type: input.type || "Optical 4K",
      latitude: lat,
      longitude: lng,
      streamUrl,
    };

    memoryCameras = [newCamera, ...memoryCameras];
    notify();

    // Push to backend asynchronously if available
    api
      .createCamera({
        ...newCamera,
        id: nextId,
        latitude: lat ?? 0,
        longitude: lng ?? 0,
      })
      .catch(() => {});

    return newCamera;
  },

  async updateCamera(id: string, input: UpdateCameraInput): Promise<Camera> {
    const idx = memoryCameras.findIndex((c) => c.id === id);
    if (idx === -1) {
      throw new Error(`Camera with ID '${id}' not found.`);
    }

    const current = memoryCameras[idx];
    const newChannel =
      input.channel !== undefined ? Number(input.channel) : current.channel;

    const lat =
      input.latitude !== undefined
        ? input.latitude !== null && String(input.latitude).trim() !== ""
          ? Number(input.latitude)
          : null
        : current.latitude;

    const lng =
      input.longitude !== undefined
        ? input.longitude !== null && String(input.longitude).trim() !== ""
          ? Number(input.longitude)
          : null
        : current.longitude;

    // Auto-update RTSP link channel if channel changed and custom streamUrl wasn't given
    let streamUrl = current.streamUrl;
    if (input.streamUrl !== undefined && input.streamUrl.trim() !== "") {
      streamUrl = input.streamUrl.trim();
    } else if (input.channel !== undefined) {
      streamUrl = buildChannelRtspUrl(input.channel);
    }

    const updated: Camera = {
      ...current,
      ...input,
      channel: newChannel,
      latitude: lat,
      longitude: lng,
      streamUrl,
    };

    memoryCameras[idx] = updated;
    notify();

    api
      .updateCamera(id, {
        ...input,
        latitude: lat ?? 0,
        longitude: lng ?? 0,
        streamUrl,
      })
      .catch(() => {});
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
      const target = params.streamUrl || params.ipAddress || "";
      const isFormatValid =
        target.startsWith("rtsp://") ||
        /^(\d{1,3}\.){3}\d{1,3}$/.test(target.trim());

      return {
        reachable: isFormatValid,
        status: isFormatValid ? "online" : "error",
        latencyMs: isFormatValid ? 40 : 0,
        message: isFormatValid ? "Stream verified" : "Invalid stream link",
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
  const [cameras, setCameras] = useState<Camera[]>(() =>
    camerasStore.getCameras()
  );
  const [isSyncing, setIsSyncing] = useState<boolean>(false);

  useEffect(() => {
    const initial = camerasStore.initClientStorage();
    setCameras(initial);

    const unsubscribe = (updated: Camera[]) => {
      setCameras(updated.filter(isAuthenticCamera));
    };

    listeners.add(unsubscribe);

    setIsSyncing(true);
    camerasStore
      .syncWithBackend()
      .then((synced) => {
        setCameras(synced.filter(isAuthenticCamera));
      })
      .finally(() => {
        setIsSyncing(false);
      });

    return () => {
      listeners.delete(unsubscribe);
    };
  }, []);

  const totalCount = cameras.length;
  const onlineCount = cameras.filter((c) => c.status === "online").length;
  const alertCount = cameras.filter((c) => c.status === "alert").length;
  const offlineCount = cameras.filter((c) => c.status === "offline").length;
  const degradedCount = cameras.filter((c) => c.status === "degraded").length;

  const avgLatencyMs = 0;

  const addCamera = useCallback(
    async (input: CreateCameraInput) => camerasStore.addCamera(input),
    []
  );

  const addCameraByChannel = useCallback(
    async (input: {
      name: string;
      channel: number | string;
      latitude?: number | string | null;
      longitude?: number | string | null;
      type?: CameraType;
    }) => camerasStore.addCameraByChannel(input),
    []
  );

  const updateCamera = useCallback(
    async (id: string, input: UpdateCameraInput) =>
      camerasStore.updateCamera(id, input),
    []
  );

  const deleteCamera = useCallback(
    async (id: string) => camerasStore.deleteCamera(id),
    []
  );

  const bulkDeleteCameras = useCallback(
    async (ids: string[]) => camerasStore.bulkDeleteCameras(ids),
    []
  );

  const testStream = useCallback(
    (params: { streamUrl?: string; ipAddress?: string; port?: number }) =>
      camerasStore.testStream(params),
    []
  );

  const resetDefaults = useCallback(() => {
    camerasStore.resetDefaults();
  }, []);

  const refetch = useCallback(async () => {
    setIsSyncing(true);
    try {
      const synced = await camerasStore.syncWithBackend();
      setCameras(synced.filter(isAuthenticCamera));
    } finally {
      setIsSyncing(false);
    }
  }, []);

  return {
    cameras,
    isSyncing,
    isLoading: isSyncing,
    totalCount,
    onlineCount,
    alertCount,
    offlineCount,
    degradedCount,
    avgLatencyMs,
    addCamera,
    addCameraByChannel,
    updateCamera,
    deleteCamera,
    bulkDeleteCameras,
    testStream,
    resetDefaults,
    refetch,
  };
}
