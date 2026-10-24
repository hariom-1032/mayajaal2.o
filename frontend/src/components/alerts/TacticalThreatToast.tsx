"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useAlerts } from "@/lib/alertsStore";
import { AlertItem } from "@/types/alert";
import { api } from "@/lib/api";
import { useRouter } from "next/navigation";
import { formatConfidence } from "@/lib/utils";
import {
  ShieldAlert,
  Radio,
  Eye,
  MapPin,
  X,
  Send,
  Check,
} from "lucide-react";

const STORAGE_DISMISSED_IDS_KEY = "maatrix_toast_dismissed_ids";
const STORAGE_DISMISSED_KEYS_KEY = "maatrix_toast_dismissed_keys";
const TOAST_AUTO_DISMISS_MS = 3000; // 3 seconds auto-dismiss
const TICK_INTERVAL_MS = 100;

function getSessionDismissedIds(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = sessionStorage.getItem(STORAGE_DISMISSED_IDS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return new Set(parsed);
    }
  } catch {}
  return new Set();
}

function getSessionDismissedKeys(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = sessionStorage.getItem(STORAGE_DISMISSED_KEYS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return new Set(parsed);
    }
  } catch {}
  return new Set();
}

interface ThreatToastCardProps {
  alert: AlertItem;
  onDismiss: (id: string, title?: string, suspectName?: string) => void;
  onAcknowledge: (id: string, title?: string, suspectName?: string) => void;
}

function ThreatToastCard({ alert, onDismiss, onAcknowledge }: ThreatToastCardProps) {
  const router = useRouter();
  const [isHovered, setIsHovered] = useState(false);
  const [isDispatching, setIsDispatching] = useState(false);
  const [selectedUnit, setSelectedUnit] = useState("QRT Alpha-1");
  const [dispatchSuccess, setDispatchSuccess] = useState(false);
  const hasDismissedRef = useRef(false);

  // Auto-dismiss countdown timer (pauses when hovered or dispatching)
  useEffect(() => {
    if (isHovered || isDispatching || hasDismissedRef.current) return;

    const timer = setTimeout(() => {
      if (!hasDismissedRef.current) {
        hasDismissedRef.current = true;
        onDismiss(alert.id, alert.title, alert.suspectName);
      }
    }, TOAST_AUTO_DISMISS_MS);

    return () => clearTimeout(timer);
  }, [isHovered, isDispatching, alert.id, alert.title, alert.suspectName, onDismiss]);

  const handleQuickDispatch = async (alertId: string) => {
    try {
      await api.dispatchQrt(alertId, selectedUnit, "Urgent intercept response via Tactical HUD");
      setDispatchSuccess(true);
      setTimeout(() => {
        setIsDispatching(false);
        setDispatchSuccess(false);
        onDismiss(alert.id, alert.title, alert.suspectName);
      }, 1500);
    } catch (err) {
      console.error("Failed to dispatch QRT", err);
    }
  };

  const isBreach =
    Boolean(
      alert.category?.startsWith("geofence_") ||
      alert.category?.startsWith("tripwire_") ||
      alert.category === "geofence_breach" ||
      alert.category === "tripwire_violation" ||
      alert.category === "tripwire_crossing" ||
      alert.title?.toLowerCase().includes("geofence") ||
      alert.title?.toLowerCase().includes("tripwire") ||
      alert.title?.toLowerCase().includes("perimeter") ||
      alert.title?.toLowerCase().includes("breach") ||
      alert.title?.toLowerCase().includes("intrusion")
    );

  const backendBase =
    process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";
  const snapshotUrl = alert.snapshotUrl
    ? alert.snapshotUrl.startsWith("http")
      ? alert.snapshotUrl
      : `${backendBase}${alert.snapshotUrl}`
    : null;

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="pointer-events-auto bg-white/98 backdrop-blur-md text-slate-900 rounded-2xl border border-rose-200 shadow-2xl shadow-rose-950/15 p-4 animate-in slide-in-from-right-8 duration-300 ring-1 ring-rose-500/20 relative overflow-hidden transition-all group"
    >
      {/* Tactical top beacon line */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-rose-600 via-rose-500 to-amber-500" />

      {/* Header pill & countdown status */}
      <div className="flex items-center justify-between gap-2 pb-2.5 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-600"></span>
          </span>
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-rose-50 border border-rose-200 text-[10px] font-extrabold uppercase tracking-wider text-rose-700 font-mono">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
            {isBreach ? "PERIMETER BREACH" : "SUSPECT SIGHTING"}
          </span>
          <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white uppercase tracking-wider font-mono">
            {alert.threatLevel || "CRITICAL"}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {/* Auto-dismiss countdown badge */}
          <span
            className="text-[10px] font-mono text-slate-400 font-medium px-1.5 py-0.5 rounded bg-slate-100/90 border border-slate-200/60 select-none"
            title={isHovered ? "Timer paused on hover" : "Auto-dismisses in 3s"}
          >
            {isHovered ? "Paused" : "3s"}
          </span>

          <button
            type="button"
            onClick={() => {
              hasDismissedRef.current = true;
              onDismiss(alert.id, alert.title, alert.suspectName);
            }}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
            title="Dismiss banner"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="py-3 flex items-start gap-3.5">
        {snapshotUrl ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={snapshotUrl}
            alt={alert.suspectName || "Suspect"}
            className="w-14 h-14 rounded-xl object-cover border border-rose-200 shrink-0 bg-slate-100 shadow-2xs"
          />
        ) : (
          <div className="w-14 h-14 rounded-xl bg-rose-50 border border-rose-200 flex items-center justify-center shrink-0 text-rose-600 shadow-2xs">
            <ShieldAlert className="w-7 h-7" />
          </div>
        )}

        <div className="min-w-0 flex-1">
          <h4 className="text-sm font-extrabold text-slate-900 tracking-tight truncate">
            {alert.suspectName || alert.title}
          </h4>
          <p className="text-xs text-slate-600 font-medium truncate mt-0.5 flex items-center gap-1">
            <MapPin className="w-3 h-3 text-slate-400 shrink-0" />
            {alert.location}
          </p>
          <div className="flex flex-wrap items-center gap-1.5 mt-2 text-[11px] font-mono">
            <span className="text-slate-400">{alert.time}</span>
            {alert.confidence && (
              <>
                <span className="text-slate-300">•</span>
                <span className="text-emerald-800 font-bold bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                  {formatConfidence(alert.confidence)} match
                </span>
              </>
            )}
            {alert.category && (
              <>
                <span className="text-slate-300">•</span>
                <span className="text-rose-700 font-semibold bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200 truncate">
                  {alert.category}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Quick Dispatch Drawer if opened */}
      {isDispatching ? (
        <div className="mt-2 pt-2.5 border-t border-slate-100 space-y-2 animate-in fade-in duration-150 bg-rose-50/40 p-2.5 rounded-xl border border-rose-100">
          <div className="text-[11px] font-bold text-rose-800 flex items-center gap-1.5">
            <Radio className="w-3 h-3 text-rose-600 animate-spin" /> Deploy Quick Reaction Team (QRT):
          </div>
          <div className="flex items-center gap-2">
            <select
              value={selectedUnit}
              onChange={(e) => setSelectedUnit(e.target.value)}
              className="bg-white border border-slate-200 text-xs rounded-lg px-2.5 py-1.5 text-slate-800 flex-1 focus:outline-none focus:ring-1 focus:ring-rose-500 shadow-2xs font-medium"
            >
              <option value="QRT Alpha-1">QRT Alpha-1 (Sector BOP Alpha)</option>
              <option value="QRT Bravo-2">QRT Bravo-2 (Octroi Gate)</option>
              <option value="QRT Charlie-3">QRT Charlie-3 (Riverine Island)</option>
            </select>
            <button
              type="button"
              onClick={() => handleQuickDispatch(alert.id)}
              className="px-3 py-1.5 rounded-lg text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
            >
              {dispatchSuccess ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" /> Dispatched!
                </>
              ) : (
                <>
                  <Send className="w-3 h-3" /> Dispatch
                </>
              )}
            </button>
          </div>
        </div>
      ) : null}

      {/* Actions Bar */}
      <div className="pt-2.5 border-t border-slate-100 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => {
              hasDismissedRef.current = true;
              onDismiss(alert.id, alert.title, alert.suspectName);
              router.push(`/live?camera=${alert.cameraId || ""}`);
            }}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Eye className="w-3.5 h-3.5 text-emerald-700" /> Live Feed
          </button>
          <button
            type="button"
            onClick={() => {
              hasDismissedRef.current = true;
              onDismiss(alert.id, alert.title, alert.suspectName);
              router.push(`/gis-map?focus=${alert.cameraId || ""}`);
            }}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <MapPin className="w-3.5 h-3.5 text-rose-600" /> GIS Map
          </button>
          {!isDispatching && (
            <button
              type="button"
              onClick={() => setIsDispatching(true)}
              className="px-3 py-1.5 rounded-lg text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white flex items-center gap-1.5 transition-colors shadow-2xs cursor-pointer"
            >
              <Send className="w-3 h-3" /> QRT
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={() => {
            hasDismissedRef.current = true;
            onAcknowledge(alert.id, alert.title, alert.suspectName);
          }}
          className="px-2.5 py-1.5 rounded-lg text-xs font-semibold text-emerald-800 hover:bg-emerald-50 border border-emerald-200/80 transition-colors flex items-center gap-1 cursor-pointer"
        >
          <Check className="w-3.5 h-3.5" /> Ack
        </button>
      </div>

      {/* Bottom Auto-Dismiss Countdown Progress Bar (Pure CSS Compositor Animation) */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-100 overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-rose-600 via-rose-500 to-amber-500 animate-toast-progress"
          style={{
            animationDuration: `${TOAST_AUTO_DISMISS_MS}ms`,
            animationPlayState: isHovered || isDispatching ? "paused" : "running",
          }}
        />
      </div>
    </div>
  );
}

export function TacticalThreatToast() {
  const { alerts, acknowledgeAlert } = useAlerts();

  // Track toasts dismissed in this browser session
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(getSessionDismissedIds);
  const [dismissedKeys, setDismissedKeys] = useState<Set<string>>(getSessionDismissedKeys);

  const handleDismiss = useCallback(
    (id: string, title?: string, suspectName?: string) => {
      const alertKey = (suspectName || title || id).trim().toLowerCase();

      // Find all IDs in current alerts matching this key, suspect, or title
      const idsToDismiss = new Set<string>([id]);
      alerts.forEach((a) => {
        const aKey = (a.suspectName || a.title || a.id).trim().toLowerCase();
        if (
          aKey === alertKey ||
          (suspectName && a.suspectName?.toLowerCase() === suspectName.toLowerCase()) ||
          (title && a.title?.toLowerCase() === title.toLowerCase())
        ) {
          idsToDismiss.add(a.id);
        }
      });

      setDismissedIds((prev) => {
        const next = new Set(prev);
        idsToDismiss.forEach((item) => next.add(item));
        try {
          sessionStorage.setItem(
            STORAGE_DISMISSED_IDS_KEY,
            JSON.stringify(Array.from(next))
          );
        } catch {}
        return next;
      });

      setDismissedKeys((prev) => {
        const next = new Set(prev);
        next.add(alertKey);
        try {
          sessionStorage.setItem(
            STORAGE_DISMISSED_KEYS_KEY,
            JSON.stringify(Array.from(next))
          );
        } catch {}
        return next;
      });
    },
    [alerts]
  );

  const handleAcknowledge = useCallback(
    (id: string, title?: string, suspectName?: string) => {
      handleDismiss(id, title, suspectName);
      acknowledgeAlert(id);

      // Acknowledge all unacknowledged alerts matching this threat event
      const targetKey = (suspectName || title || id).trim().toLowerCase();
      alerts.forEach((a) => {
        if (!a.acknowledged && a.id !== id) {
          const aKey = (a.suspectName || a.title || a.id).trim().toLowerCase();
          if (
            aKey === targetKey ||
            (suspectName && a.suspectName?.toLowerCase() === suspectName.toLowerCase()) ||
            (title && a.title?.toLowerCase() === title.toLowerCase())
          ) {
            acknowledgeAlert(a.id);
          }
        }
      });
    },
    [alerts, acknowledgeAlert, handleDismiss]
  );

  // Find unacknowledged high-threat alerts that haven't been dismissed in this session
  const activeAlerts: AlertItem[] = [];
  const seenAlertKeys = new Set<string>();

  for (const alert of alerts) {
    const isSuspect = Boolean(alert.suspectName) || alert.className === "suspect";
    const isBreach = Boolean(
      alert.category?.startsWith("geofence_") ||
      alert.category?.startsWith("tripwire_") ||
      alert.category === "geofence_breach" ||
      alert.category === "tripwire_violation" ||
      alert.category === "tripwire_crossing" ||
      alert.title?.toLowerCase().includes("geofence") ||
      alert.title?.toLowerCase().includes("tripwire") ||
      alert.title?.toLowerCase().includes("perimeter") ||
      alert.title?.toLowerCase().includes("breach") ||
      alert.title?.toLowerCase().includes("intrusion")
    );

    const alertKey = (alert.suspectName || alert.title || alert.id).trim().toLowerCase();

    if (
      !alert.acknowledged &&
      (isSuspect || isBreach) &&
      !dismissedIds.has(alert.id) &&
      !dismissedKeys.has(alertKey)
    ) {
      if (!seenAlertKeys.has(alertKey)) {
        seenAlertKeys.add(alertKey);
        activeAlerts.push(alert);
        if (activeAlerts.length >= 2) break; // Show at most 2 distinct banners
      }
    }
  }

  if (activeAlerts.length === 0) return null;

  return (
    <div className="fixed top-20 right-4 sm:right-6 z-50 flex flex-col gap-3 max-w-md w-full pointer-events-none">
      {activeAlerts.map((alert) => (
        <ThreatToastCard
          key={alert.id}
          alert={alert}
          onDismiss={handleDismiss}
          onAcknowledge={handleAcknowledge}
        />
      ))}
    </div>
  );
}

export default TacticalThreatToast;
