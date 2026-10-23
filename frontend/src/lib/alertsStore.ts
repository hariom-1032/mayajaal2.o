"use client";

import { useState, useEffect } from "react";
import { AlertItem, AlertSeverity } from "@/types/alert";
import { api } from "@/lib/api";

export const INCIDENT_STORAGE_KEY = "maatrix_incident_records";
const MAX_LOCAL_INCIDENTS = 500;

export const SEED_INCIDENTS: AlertItem[] = [];

type AlertListener = (alerts: AlertItem[]) => void;
type ConnectionListener = (connected: boolean) => void;

const listeners = new Set<AlertListener>();
const connectionListeners = new Set<ConnectionListener>();

function loadStoredIncidents(): AlertItem[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(INCIDENT_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.length > 0) {
      const clean = parsed.filter(
        (item: AlertItem) => item && !String(item.id || "").startsWith("incident-demo-")
      );
      return clean;
    }
  } catch (err) {
    console.warn("[IncidentsStore] Failed to parse local incidents:", err);
  }
  return [];
}

function persistIncidents(alerts: AlertItem[]) {
  if (typeof window === "undefined") return;
  try {
    const trimmed = alerts.slice(0, MAX_LOCAL_INCIDENTS);
    localStorage.setItem(INCIDENT_STORAGE_KEY, JSON.stringify(trimmed));
  } catch (err) {
    console.error("[IncidentsStore] Failed to persist incident records to localStorage:", err);
  }
}

let memoryAlerts: AlertItem[] = [];
let wsClient: WebSocket | null = null;
let isConnected = false;
let initialized = false;

function notify(skipPersist = false) {
  if (!skipPersist) {
    persistIncidents(memoryAlerts);
  }
  listeners.forEach((listener) => listener([...memoryAlerts]));
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("maatrix_incidents_updated"));
  }
}

function notifyConnection(status: boolean) {
  isConnected = status;
  connectionListeners.forEach((listener) => listener(status));
}

function getAlertsWsUrl(): string {
  if (typeof window === "undefined") return "ws://localhost:8000/api/alerts/ws";
  const httpUrl =
    process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ||
    `${window.location.protocol}//${window.location.hostname}:8000`;
  const wsProto = httpUrl.startsWith("https") ? "wss" : "ws";
  const host = httpUrl.replace(/^https?:\/\//, "");
  return `${wsProto}://${host}/api/alerts/ws`;
}

function playAlertChime(threatLevel?: string) {
  if (typeof window === "undefined") return;
  try {
    const AudioCtx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    const isUrgent = threatLevel === "CRITICAL" || threatLevel === "HIGH";
    osc.type = isUrgent ? "sawtooth" : "sine";
    osc.frequency.setValueAtTime(isUrgent ? 880 : 587, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(isUrgent ? 440 : 293, ctx.currentTime + 0.28);

    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch {
    // AudioContext blocked until user interaction
  }
}

function initWebSocket() {
  if (typeof window === "undefined") return;
  if (wsClient && (wsClient.readyState === WebSocket.OPEN || wsClient.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    const wsUrl = getAlertsWsUrl();
    wsClient = new WebSocket(wsUrl);

    wsClient.onopen = () => {
      notifyConnection(true);
    };

    wsClient.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        if (payload.type === "NEW_ALERT" && payload.alert) {
          const incoming = payload.alert as AlertItem;
          // Avoid duplicate by ID, or if same suspect + location arrived within 5 seconds
          const existingIdx = memoryAlerts.findIndex(
            (a) =>
              a.id === incoming.id ||
              (Boolean(a.suspectName) &&
                Boolean(incoming.suspectName) &&
                a.suspectName?.toLowerCase() === incoming.suspectName?.toLowerCase() &&
                a.location === incoming.location &&
                Math.abs((a.timestamp || 0) - (incoming.timestamp || Date.now())) < 5000)
          );
          if (existingIdx === -1) {
            memoryAlerts = [incoming, ...memoryAlerts];
            playAlertChime(incoming.threatLevel || (incoming.severity as string));
            notify();
          }
        } else if (payload.type === "ACKNOWLEDGE_ALERT" && payload.alertId) {
          memoryAlerts = memoryAlerts.map((a) =>
            a.id === payload.alertId ? { ...a, acknowledged: true } : a
          );
          notify();
        } else if (payload.type === "CLEAR_ALERTS") {
          // Keep local history unless operator explicitly clears locally
        }
      } catch (err) {
        console.debug("Failed to parse alerts WebSocket payload", err);
      }
    };

    wsClient.onclose = () => {
      notifyConnection(false);
      wsClient = null;
      setTimeout(initWebSocket, 3000);
    };

    wsClient.onerror = () => {
      notifyConnection(false);
    };
  } catch (err) {
    console.debug("Error initializing alerts WebSocket", err);
  }
}

async function syncWithBackend() {
  try {
    const backendAlerts = await api.getAlerts();
    if (Array.isArray(backendAlerts) && backendAlerts.length > 0) {
      // Merge backend incoming alerts with locally stored incidents without erasing local history
      const existingIds = new Set(memoryAlerts.map((a) => a.id));
      const newItems = backendAlerts.filter((a) => !existingIds.has(a.id));
      if (newItems.length > 0) {
        memoryAlerts = [...newItems, ...memoryAlerts];
        notify();
      }
    }
  } catch (err) {
    console.debug("Could not fetch transient alerts from backend, using local incident records:", err);
  }
}

export const alertsStore = {
  getAlerts(): AlertItem[] {
    return memoryAlerts;
  },

  isConnected(): boolean {
    return isConnected;
  },

  async refresh() {
    await syncWithBackend();
  },

  addAlert(alert: Omit<AlertItem, "id" | "time" | "timestamp" | "acknowledged"> & Partial<AlertItem>): AlertItem {
    // Avoid duplicate if identical suspect alert already exists in memory within 5 seconds
    if (alert.suspectName) {
      const existing = memoryAlerts.find(
        (a) =>
          !a.acknowledged &&
          a.suspectName?.toLowerCase() === alert.suspectName?.toLowerCase() &&
          a.location === alert.location &&
          Date.now() - (a.timestamp || 0) < 5000
      );
      if (existing) {
        return existing;
      }
    }

    const newAlert: AlertItem = {
      id: `incident-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
      time: "Just now",
      timestamp: Date.now(),
      acknowledged: false,
      ...alert,
    };

    memoryAlerts = [newAlert, ...memoryAlerts];
    notify();
    return newAlert;
  },

  async acknowledgeAlert(id: string) {
    memoryAlerts = memoryAlerts.map((a) =>
      a.id === id ? { ...a, acknowledged: true } : a
    );
    notify();

    try {
      await api.acknowledgeAlert(id);
    } catch (err) {
      console.debug("Transient backend notification for acknowledge failed (offline mode):", err);
    }
  },

  deleteIncident(id: string) {
    memoryAlerts = memoryAlerts.filter((a) => a.id !== id);
    notify();
  },

  async clearAll() {
    memoryAlerts = [];
    notify();

    try {
      await api.clearAlerts();
    } catch (err) {
      console.debug("Transient backend clear failed (offline mode):", err);
    }
  },

  resetDefaults() {
    memoryAlerts = [...SEED_INCIDENTS];
    notify();
  },

  exportIncidents(format: "json" | "csv" = "json") {
    if (typeof window === "undefined" || memoryAlerts.length === 0) return;

    let blob: Blob;
    let filename: string;
    const nowStr = new Date().toISOString().replace(/[:.]/g, "-");

    if (format === "csv") {
      const headers = ["ID", "Title", "Severity", "ThreatLevel", "Location", "Camera", "Class", "Suspect", "Timestamp", "Acknowledged", "Notes"];
      const rows = memoryAlerts.map((a) => [
        `"${a.id}"`,
        `"${(a.title || "").replace(/"/g, '""')}"`,
        `"${a.severity || ""}"`,
        `"${a.threatLevel || ""}"`,
        `"${(a.location || "").replace(/"/g, '""')}"`,
        `"${a.cameraId || a.cameraName || ""}"`,
        `"${a.className || ""}"`,
        `"${(a.suspectName || "").replace(/"/g, '""')}"`,
        `"${new Date(a.timestamp || Date.now()).toISOString()}"`,
        `"${a.acknowledged ? "YES" : "NO"}"`,
        `"${(a.notes || "").replace(/"/g, '""')}"`,
      ]);
      const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\r\n");
      blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      filename = `maatrix-incident-records-${nowStr}.csv`;
    } else {
      const jsonContent = JSON.stringify(memoryAlerts, null, 2);
      blob = new Blob([jsonContent], { type: "application/json;charset=utf-8;" });
      filename = `maatrix-incident-records-${nowStr}.json`;
    }

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  },

  getStoredCount(): number {
    return memoryAlerts.length;
  },
};

export function useAlerts() {
  const [alerts, setAlerts] = useState<AlertItem[]>(memoryAlerts);

  const [connected, setConnected] = useState<boolean>(isConnected);

  useEffect(() => {
    if (!initialized && typeof window !== "undefined") {
      initialized = true;
      const stored = loadStoredIncidents();
      memoryAlerts = stored;
      notify(true);

      initWebSocket();
      syncWithBackend();

      // Multi-tab synchronization
      const handleStorageEvent = (e: StorageEvent) => {
        if (e.key === INCIDENT_STORAGE_KEY && e.newValue) {
          try {
            const parsed = JSON.parse(e.newValue);
            if (Array.isArray(parsed)) {
              memoryAlerts = parsed;
              setAlerts(parsed);
              listeners.forEach((l) => l([...memoryAlerts]));
            }
          } catch {}
        }
      };

      const handleCustomUpdate = () => {
        setAlerts([...memoryAlerts]);
      };

      window.addEventListener("storage", handleStorageEvent);
      window.addEventListener("maatrix_incidents_updated", handleCustomUpdate);
    }

    const handleUpdate = (updated: AlertItem[]) => setAlerts(updated);
    const handleConn = (status: boolean) => setConnected(status);

    listeners.add(handleUpdate);
    connectionListeners.add(handleConn);

    return () => {
      listeners.delete(handleUpdate);
      connectionListeners.delete(handleConn);
    };
  }, []);

  const unacknowledged = alerts.filter((a) => !a.acknowledged);
  const peopleCount = alerts.filter(
    (a) => a.className?.toLowerCase() === "person" || a.className?.toLowerCase() === "suspect"
  ).length;
  const vehicleCount = alerts.filter((a) =>
    ["car", "truck", "bus", "motorcycle"].includes(
      a.className?.toLowerCase() || ""
    )
  ).length;
  const suspectsCount = alerts.filter(
    (a) => Boolean(a.suspectName) || a.className?.toLowerCase() === "suspect"
  ).length;

  return {
    alerts,
    connected,
    storedLocally: true,
    totalStoredCount: alerts.length,
    unacknowledgedCount: unacknowledged.length,
    highSeverityCount: alerts.filter((a) => a.severity === "High").length,
    peopleCount,
    vehicleCount,
    suspectsCount,
    addAlert: alertsStore.addAlert,
    acknowledgeAlert: alertsStore.acknowledgeAlert,
    deleteIncident: alertsStore.deleteIncident,
    clearAll: alertsStore.clearAll,
    resetDefaults: alertsStore.resetDefaults,
    refresh: alertsStore.refresh,
    exportIncidents: alertsStore.exportIncidents,
  };
}
