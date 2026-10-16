"use client";

import { useState, useEffect } from "react";
import { SuspectTrajectory } from "@/types/alert";
import { api } from "@/lib/api";
import { formatConfidence } from "@/lib/utils";
import {
  MapPin,
  Navigation,
  Clock,
  ShieldAlert,
  Send,
  X,
  Radio,
  ArrowDown,
  Check,
} from "lucide-react";
import { CctvCameraIcon } from "@/components/icons/CctvCameraIcon";

interface SuspectTrajectoryModalProps {
  suspectName: string;
  onClose: () => void;
  onFocusCoordinates?: (lat: number, lng: number) => void;
}

export function SuspectTrajectoryModal({
  suspectName,
  onClose,
  onFocusCoordinates,
}: SuspectTrajectoryModalProps) {
  const [trajectory, setTrajectory] = useState<SuspectTrajectory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // QRT dispatch state
  const [dispatchUnit, setDispatchUnit] = useState("QRT Strike Alpha-1");
  const [dispatching, setDispatching] = useState(false);
  const [dispatchSuccess, setDispatchSuccess] = useState(false);

  useEffect(() => {
    async function loadTrajectory() {
      setLoading(true);
      setError(null);
      try {
        const isTrace = suspectName.startsWith("TRC-");
        const data = isTrace
          ? await api.getGlobalTraceTrajectory(suspectName)
          : await api.getSuspectTrajectory(suspectName);
        setTrajectory(data);
      } catch {
        setError("Failed to load trajectory telemetry from backend.");
      } finally {
        setLoading(false);
      }
    }
    if (suspectName) {
      loadTrajectory();
    }
  }, [suspectName]);

  const handleDispatchQRT = async () => {
    if (!trajectory || trajectory.waypoints.length === 0) return;
    const latestWaypoint = trajectory.waypoints[trajectory.waypoints.length - 1];
    setDispatching(true);
    try {
      await api.dispatchQrt(
        latestWaypoint.alert_id || `trace_${suspectName}`,
        dispatchUnit,
        `Intercept dispatch for suspect ${suspectName} at ${latestWaypoint.location || latestWaypoint.camera_name}`
      );
      setDispatchSuccess(true);
      setTimeout(() => setDispatchSuccess(false), 3000);
    } catch (err) {
      console.error("QRT dispatch failed", err);
    } finally {
      setDispatching(false);
    }
  };

  const backendBase =
    process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

  return (
    <div className="fixed inset-0 z-50 bg-slate-900/60 flex items-center justify-center p-4 backdrop-blur-xs">
      <div className="bg-white border border-slate-200 rounded-2xl max-w-2xl w-full max-h-[90vh] flex flex-col overflow-hidden shadow-2xl text-slate-900 animate-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="p-5 bg-white border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-800 shadow-2xs">
              <Navigation className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold tracking-tight text-slate-900">
                  {suspectName.startsWith("TRC-") ? "Cross-Camera Movement Trajectory" : "Suspect Movement Trajectory"}
                </h3>
                {trajectory?.threat_level && (
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-rose-600 text-white tracking-wider uppercase font-mono">
                    {trajectory.threat_level}
                  </span>
                )}
                {trajectory?.is_cross_camera && (
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-cyan-700 text-white tracking-wider uppercase font-mono">
                    Multi-Camera Re-ID
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500 font-mono mt-0.5">
                {suspectName.startsWith("TRC-") ? (
                  <>
                    Global Trace: <span className="text-cyan-800 font-bold">{suspectName}</span>
                    {trajectory?.suspect_name ? (
                      <> • Linked Suspect: <span className="text-rose-700 font-bold">{trajectory.suspect_name}</span></>
                    ) : ""}
                  </>
                ) : (
                  <>Target: <span className="text-rose-700 font-bold">{suspectName}</span></>
                )}
                {trajectory?.category ? ` • ${trajectory.category}` : ""}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Telemetry Stats Strip */}
        {trajectory && trajectory.found && (
          <div className="grid grid-cols-3 gap-2 px-5 py-3 bg-slate-50 border-b border-slate-200 text-xs">
            <div>
              <span className="text-slate-500 font-mono text-[10px] uppercase block font-semibold">
                Sightings Count
              </span>
              <span className="font-bold text-slate-900 text-sm">
                {trajectory.total_sightings} camera nodes
              </span>
            </div>
            <div>
              <span className="text-slate-500 font-mono text-[10px] uppercase block font-semibold">
                Total Path Distance
              </span>
              <span className="font-bold text-emerald-800 text-sm">
                {trajectory.total_distance_km} km
              </span>
            </div>
            <div>
              <span className="text-slate-500 font-mono text-[10px] uppercase block font-semibold">
                Last Known Outpost
              </span>
              <span className="font-bold text-rose-700 text-sm truncate block">
                {trajectory.last_location}
              </span>
            </div>
          </div>
        )}

        {/* Body / Waypoints Timeline */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4 bg-[#fbfbfa]">
          {loading ? (
            <div className="py-16 text-center text-slate-500 flex flex-col items-center justify-center gap-2">
              <Radio className="w-8 h-8 animate-spin text-emerald-700" />
              <p className="text-xs font-mono font-medium">Reconstructing multi-camera timeline...</p>
            </div>
          ) : error ? (
            <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-700">
              {error}
            </div>
          ) : !trajectory || trajectory.waypoints.length === 0 ? (
            <div className="py-16 text-center text-slate-400">
              <ShieldAlert className="w-10 h-10 mx-auto mb-2 text-slate-300" />
              <p className="text-sm font-semibold text-slate-700">No recorded sightings yet</p>
              <p className="text-xs text-slate-500 mt-1">
                Keep the Continuous Face Scanner active to log suspect sightings across camera feeds.
              </p>
            </div>
          ) : (
            <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-3 before:bottom-3 before:w-0.5 before:bg-slate-200">
              {trajectory.waypoints.map((wp, idx) => {
                const isLast = idx === trajectory.waypoints.length - 1;
                const snapshotUrl = wp.snapshot_url
                  ? wp.snapshot_url.startsWith("http")
                    ? wp.snapshot_url
                    : `${backendBase}${wp.snapshot_url}`
                  : null;

                const dist = wp.delta_distance_km ?? wp.delta_km ?? 0;
                const elapsed = wp.elapsed_time_minutes ?? wp.elapsed_minutes ?? 0;

                return (
                  <div key={wp.alert_id || `${wp.camera_id}-${wp.step}`} className="relative group">
                    {/* Node Dot */}
                    <div
                      className={`absolute -left-[29px] top-1.5 w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border-2 ${
                        isLast
                          ? "bg-rose-600 border-white text-white shadow-md animate-pulse"
                          : "bg-white border-slate-300 text-slate-700 shadow-2xs"
                      }`}
                    >
                      {wp.step}
                    </div>

                    {/* Card */}
                    <div
                      className={`p-4 rounded-xl border transition-all ${
                        isLast
                          ? "bg-rose-50/40 border-rose-200 ring-1 ring-rose-300 shadow-xs"
                          : "bg-white border-slate-200 shadow-2xs hover:border-slate-300"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-[11px] font-bold text-emerald-800 flex items-center gap-1.5">
                              <CctvCameraIcon className="w-3.5 h-3.5 text-emerald-700" />
                              {wp.camera_name}
                            </span>
                            {isLast && (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-600 text-white tracking-wider uppercase font-mono">
                                Current Sighting
                              </span>
                            )}
                          </div>
                          <h4 className="text-sm font-bold text-slate-900 mt-0.5">
                            {wp.location || wp.sector || wp.camera_name}
                          </h4>
                          <p className="text-xs text-slate-500 font-mono mt-1 flex flex-wrap items-center gap-2">
                            <Clock className="w-3 h-3 text-slate-400" />
                            <span>{wp.time_str}</span>
                            <span>•</span>
                            <span className="text-emerald-800 font-bold bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200">
                              Conf: {formatConfidence(wp.confidence)}
                            </span>
                            {wp.reid_score !== undefined && wp.reid_score < 1.0 && (
                              <span className="text-cyan-800 font-bold bg-cyan-50 px-1.5 py-0.2 rounded border border-cyan-200">
                                Re-ID: {Math.round(wp.reid_score * 100)}%
                              </span>
                            )}
                          </p>
                          {dist > 0 && (
                            <p className="text-[11px] text-amber-800 font-mono mt-1 flex items-center gap-1 font-semibold">
                              <ArrowDown className="w-3 h-3 text-amber-600" /> Transited {dist} km in{" "}
                              {elapsed} mins
                            </p>
                          )}
                        </div>

                        {/* Snapshot thumbnail if available */}
                        {snapshotUrl && (
                          /* eslint-disable-next-line @next/next/no-img-element */
                          <img
                            src={snapshotUrl}
                            alt="Waypoint snapshot"
                            className="w-14 h-14 rounded-xl object-cover border border-slate-200 shrink-0 bg-slate-100 shadow-2xs"
                          />
                        )}
                      </div>

                      {/* Waypoint Coordinates Action */}
                      <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-xs">
                        <span className="text-slate-500 font-mono text-[11px]">
                          GPS: {wp.latitude.toFixed(4)}, {wp.longitude.toFixed(4)}
                        </span>
                        {onFocusCoordinates && (
                          <button
                            type="button"
                            onClick={() => onFocusCoordinates(wp.latitude, wp.longitude)}
                            className="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-[#1e4b38] text-slate-700 hover:text-white font-semibold flex items-center gap-1 transition-colors cursor-pointer"
                          >
                            <MapPin className="w-3 h-3 text-rose-600" /> Focus on Map
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer with QRT Deployment Action */}
        <div className="p-4 bg-slate-50 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2 w-full sm:w-auto">
            <span className="text-xs text-slate-600 font-semibold whitespace-nowrap">
              Deploy Intercept Unit:
            </span>
            <select
              value={dispatchUnit}
              onChange={(e) => setDispatchUnit(e.target.value)}
              className="bg-white border border-slate-200 text-xs rounded-xl px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-emerald-700/20 font-medium shadow-2xs"
            >
              <option value="QRT Strike Alpha-1">QRT Strike Alpha-1 (RS Pura Outpost)</option>
              <option value="QRT Strike Bravo-2">QRT Strike Bravo-2 (Suchetgarh Octroi)</option>
              <option value="QRT Charlie-3 Rapid">QRT Charlie-3 (Riverine Island Patrol)</option>
            </select>
          </div>

          <button
            type="button"
            onClick={handleDispatchQRT}
            disabled={dispatching || !trajectory || trajectory.waypoints.length === 0}
            className="w-full sm:w-auto px-4 py-2.5 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white flex items-center justify-center gap-2 transition-all shadow-xs disabled:opacity-40 cursor-pointer"
          >
            {dispatchSuccess ? (
              <>
                <Check className="w-4 h-4 text-emerald-400" /> QRT Intercept Deployed!
              </>
            ) : (
              <>
                <Send className="w-4 h-4" /> Deploy QRT Team Now
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default SuspectTrajectoryModal;
