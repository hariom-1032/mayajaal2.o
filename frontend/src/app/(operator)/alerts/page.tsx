"use client";

import { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useAlerts } from "@/lib/alertsStore";
import { AlertItem, AlertSeverity } from "@/types/alert";
import { api } from "@/lib/api";
import {
  ShieldAlert,
  AlertTriangle,
  CheckCheck,
  Filter,
  Search,
  Check,
  RefreshCw,
  Trash2,
  Eye,
  X,
  Camera,
  Navigation,
  Download,
  RotateCcw,
} from "lucide-react";
import { SuspectTrajectoryModal } from "@/components/map/SuspectTrajectoryModal";
import { formatConfidence } from "@/lib/utils";

function AlertsContent() {
  const {
    alerts,
    connected,
    acknowledgeAlert,
    deleteIncident,
    clearAll,
    resetDefaults,
    refresh,
    exportIncidents,
  } = useAlerts();

  const searchParams = useSearchParams();
  const [severityFilter, setSeverityFilter] = useState<AlertSeverity | "All" | "Suspects">("All");
  const [statusFilter, setStatusFilter] = useState<"All" | "Pending" | "Acknowledged">("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedSnapshot, setSelectedSnapshot] = useState<{ url: string; title: string; camera: string } | null>(null);
  const [selectedSuspectForTrajectory, setSelectedSuspectForTrajectory] = useState<string | null>(null);

  // Sync filters from URL query parameters (e.g., from Dashboard inspect click)
  useEffect(() => {
    const sev = searchParams.get("severity");
    const stat = searchParams.get("status");
    if (sev === "High" || sev === "Medium" || sev === "Low" || sev === "Suspects") {
      setSeverityFilter(sev as AlertSeverity | "All" | "Suspects");
    }
    if (stat === "Pending" || stat === "Acknowledged" || stat === "All") {
      setStatusFilter(stat as "All" | "Pending" | "Acknowledged");
    }
  }, [searchParams]);

  const filteredAlerts = alerts.filter((alert) => {
    if (severityFilter === "Suspects") {
      if (!alert.suspectName && alert.className !== "suspect") return false;
    } else if (severityFilter !== "All" && alert.severity !== severityFilter) {
      return false;
    }
    if (statusFilter === "Pending" && alert.acknowledged) return false;
    if (statusFilter === "Acknowledged" && !alert.acknowledged) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchTitle = alert.title.toLowerCase().includes(q);
      const matchLoc = alert.location.toLowerCase().includes(q);
      const matchClass = alert.className?.toLowerCase().includes(q);
      const matchSuspect = alert.suspectName?.toLowerCase().includes(q);
      if (!matchTitle && !matchLoc && !matchClass && !matchSuspect) return false;
    }
    return true;
  });

  const backendBase =
    process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-slate-200/80">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-[#1c2022] flex items-center gap-2.5">
              <ShieldAlert className="w-6 h-6 text-emerald-800" />
              Tactical Alert Command
            </h1>
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${
                connected
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                  : "bg-slate-100 text-slate-500 border-slate-200"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  connected ? "bg-emerald-500 animate-pulse" : "bg-slate-400"
                }`}
              />
              {connected ? "LIVE WS CONNECTED" : "OFFLINE"}
            </span>
          </div>
          <p className="text-xs md:text-sm font-medium text-slate-500 mt-0.5">
            Suspect facial sightings, automated trajectory tracking, and real-time security alerts.
          </p>
        </div>

        {/* Actions & Export */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Export Incident Log */}
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => exportIncidents("csv")}
              className="px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
              title="Export all incident records as CSV"
            >
              <Download className="w-3.5 h-3.5 text-slate-500" />
              <span>CSV</span>
            </button>
            <button
              type="button"
              onClick={() => exportIncidents("json")}
              className="px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
              title="Export all incident records as JSON"
            >
              <Download className="w-3.5 h-3.5 text-slate-500" />
              <span>JSON</span>
            </button>
          </div>

          <button
            onClick={() => refresh()}
            className="p-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors cursor-pointer"
            title="Refresh Alerts"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => {
              if (confirm("Are you sure you want to clear all incident records from browser storage?")) {
                clearAll();
              }
            }}
            className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-rose-50 hover:text-rose-700 text-slate-600 text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            title="Clear all local incident records"
          >
            <Trash2 className="w-3.5 h-3.5" /> Clear All
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-4 shadow-xs flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Search */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search alerts, suspects, cameras..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-600 focus:border-transparent"
          />
        </div>

        {/* Severity & Threat Filters */}
        <div className="flex flex-wrap items-center gap-2 w-full md:w-auto">
          <span className="text-xs font-medium text-slate-500 flex items-center gap-1 mr-1">
            <Filter className="w-3.5 h-3.5" /> Filter:
          </span>
          {(["All", "Suspects", "High", "Medium", "Low"] as const).map((sev) => (
            <button
              key={sev}
              onClick={() => setSeverityFilter(sev)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                severityFilter === sev
                  ? sev === "Suspects"
                    ? "bg-rose-700 text-white shadow-xs font-bold"
                    : "bg-[#1e4b38] text-white shadow-xs"
                  : sev === "Suspects"
                  ? "bg-rose-50 text-rose-800 border border-rose-200 hover:bg-rose-100 font-semibold"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {sev === "Suspects" ? "🚨 Suspects Only" : sev}
            </button>
          ))}
        </div>

        {/* Status Filter */}
        <div className="flex items-center gap-2 w-full md:w-auto">
          {(["All", "Pending", "Acknowledged"] as const).map((st) => (
            <button
              key={st}
              onClick={() => setStatusFilter(st)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                statusFilter === st
                  ? "bg-slate-800 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Alert List */}
      <div className="bg-white rounded-xl border border-slate-200/80 shadow-xs overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-900">
            Recorded Threat Stream ({filteredAlerts.length})
          </span>
          <span className="text-xs text-slate-400 font-mono">
            Auto-synchronized with Edge Inference & Continuous Biometric Scanner
          </span>
        </div>

        {filteredAlerts.length === 0 ? (
          <div className="py-16 text-center text-slate-500">
            <ShieldAlert className="w-10 h-10 text-slate-300 mx-auto mb-2" />
            <p className="text-sm font-medium">No matching alerts found</p>
            <p className="text-xs text-slate-400 mt-1">
              Adjust filters or keep the Continuous Scanner running to detect suspects across live feeds.
            </p>
            {alerts.length === 0 && (
              <button
                type="button"
                onClick={resetDefaults}
                className="mt-3.5 inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-semibold shadow-xs transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" /> Restore Default Demo Incidents
              </button>
            )}
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {filteredAlerts.map((alert) => {
              const isSuspectAlert = Boolean(alert.suspectName) || alert.className === "suspect";
              const isBreachAlert =
                alert.category === "geofence_breach" ||
                alert.category === "tripwire_violation" ||
                alert.category === "tripwire_crossing" ||
                alert.title?.toLowerCase().includes("geofence") ||
                alert.title?.toLowerCase().includes("tripwire");
              const snapshotFullUrl = alert.snapshotUrl
                ? alert.snapshotUrl.startsWith("http")
                  ? alert.snapshotUrl
                  : `${backendBase}${alert.snapshotUrl}`
                : null;

              return (
                <div
                  key={alert.id}
                  className={`p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-colors ${
                    isSuspectAlert
                      ? alert.acknowledged
                        ? "bg-rose-50/20 border-l-4 border-l-rose-400"
                        : "bg-rose-50/50 border-l-4 border-l-rose-600 hover:bg-rose-50"
                      : isBreachAlert
                      ? alert.acknowledged
                        ? "bg-red-50/20 border-l-4 border-l-red-400"
                        : "bg-red-50/40 border-l-4 border-l-red-600 hover:bg-red-50/60"
                      : alert.acknowledged
                      ? "bg-slate-50/50 opacity-75"
                      : "hover:bg-slate-50"
                  }`}
                >
                  <div className="flex items-start gap-3.5">
                    {/* Snapshot Thumbnail if present */}
                    {snapshotFullUrl && (
                      <button
                        type="button"
                        onClick={() =>
                          setSelectedSnapshot({
                            url: snapshotFullUrl,
                            title: alert.title,
                            camera: alert.location,
                          })
                        }
                        className="relative group shrink-0 w-16 h-16 rounded-lg overflow-hidden border border-slate-300 bg-slate-100 hover:ring-2 hover:ring-rose-500 transition-all cursor-pointer shadow-xs"
                        title="Click to view full forensic snapshot"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={snapshotFullUrl}
                          alt="Alert snapshot"
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                        <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                          <Eye className="w-4 h-4 text-white" />
                        </div>
                      </button>
                    )}

                    <div className="space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        {isSuspectAlert ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[11px] font-bold bg-rose-700 text-white tracking-wide shadow-xs">
                            🚨 SUSPECT SIGHTING
                          </span>
                        ) : isBreachAlert ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded text-[11px] font-bold bg-red-600 text-white tracking-wide shadow-xs">
                            🚨 PERIMETER BREACH
                          </span>
                        ) : (
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold ${
                              alert.severity === "High" || (alert.severity as string) === "CRITICAL"
                                ? "bg-rose-100 text-rose-800 border border-rose-200"
                                : alert.severity === "Medium"
                                ? "bg-amber-100 text-amber-800 border border-amber-200"
                                : "bg-emerald-100 text-emerald-800 border border-emerald-200"
                            }`}
                          >
                            {alert.severity}
                          </span>
                        )}

                        {alert.threatLevel && (
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase border ${
                              alert.threatLevel === "CRITICAL"
                                ? "bg-rose-950 text-rose-200 border-rose-800"
                                : alert.threatLevel === "HIGH"
                                ? "bg-rose-100 text-rose-800 border-rose-300"
                                : "bg-amber-100 text-amber-800 border-amber-300"
                            }`}
                          >
                            THREAT: {alert.threatLevel}
                          </span>
                        )}

                        <h3 className="text-sm font-semibold text-slate-900">
                          {alert.title}
                        </h3>

                        {alert.acknowledged && (
                          <span className="inline-flex items-center gap-1 text-[10px] text-emerald-700 font-medium bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                            <CheckCheck className="w-3 h-3" /> Acknowledged
                          </span>
                        )}
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                        <span className="font-medium text-slate-700 flex items-center gap-1">
                          <Camera className="w-3 h-3 text-slate-400" />
                          {alert.location}
                        </span>
                        <span>•</span>
                        <span className="font-mono text-slate-400">
                          {alert.time}
                        </span>
                        {alert.category && (
                          <>
                            <span>•</span>
                            <span className="font-semibold text-rose-700 bg-rose-50 px-1.5 py-0.5 rounded border border-rose-200">
                              {alert.category}
                            </span>
                          </>
                        )}
                        {alert.confidence && (
                          <>
                            <span>•</span>
                            <span className="text-emerald-700 font-semibold font-mono">
                              {isSuspectAlert ? "Biometric Match: " : "Detection Conf: "}
                              {formatConfidence(alert.confidence)}
                            </span>
                          </>
                        )}
                        {alert.notes && (
                          <>
                            <span>•</span>
                            <span className="italic text-slate-400">
                              {alert.notes}
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 shrink-0">
                    {alert.suspectName && (
                      <button
                        type="button"
                        onClick={() => setSelectedSuspectForTrajectory(alert.suspectName!)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-950/80 hover:bg-rose-900 text-rose-200 border border-rose-700/80 shadow-xs transition-colors"
                        title="View multi-camera movement trajectory"
                      >
                        <Navigation className="w-3.5 h-3.5 text-rose-400" />
                        Trajectory
                      </button>
                    )}
                    {!alert.acknowledged ? (
                      <button
                        onClick={() => acknowledgeAlert(alert.id)}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-700 hover:bg-emerald-800 text-white shadow-xs transition-colors cursor-pointer"
                      >
                        <Check className="w-3.5 h-3.5" />
                        Acknowledge
                      </button>
                    ) : (
                      <span className="text-xs text-slate-400 font-medium">
                        Resolved
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={() => deleteIncident(alert.id)}
                      className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                      title="Delete incident from browser storage"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Snapshot Lightbox Modal */}
      {selectedSnapshot && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4 backdrop-blur-xs">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full overflow-hidden shadow-2xl animate-in fade-in zoom-in duration-150">
            <div className="px-5 py-3.5 bg-slate-950/80 border-b border-slate-800 flex items-center justify-between text-white">
              <div>
                <h3 className="text-sm font-semibold flex items-center gap-2 text-rose-400">
                  <Eye className="w-4 h-4" /> Forensics Snapshot Preview
                </h3>
                <p className="text-xs text-slate-400 font-mono mt-0.5">
                  {selectedSnapshot.camera} • {selectedSnapshot.title}
                </p>
              </div>
              <button
                onClick={() => setSelectedSnapshot(null)}
                className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 bg-black flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={selectedSnapshot.url}
                alt="Enlarged alert snapshot"
                className="max-h-[65vh] w-auto object-contain rounded-lg border border-slate-800"
              />
            </div>
            <div className="px-5 py-3 bg-slate-950/80 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <span>
                {selectedSnapshot.title?.toLowerCase().includes("geofence") ||
                selectedSnapshot.title?.toLowerCase().includes("tripwire") ||
                selectedSnapshot.title?.toLowerCase().includes("breach")
                  ? "Tactical perimeter intrusion forensics • Geofence Engine"
                  : "Automatic facial biometric capture with YuNet & SFace"}
              </span>
              <a
                href={selectedSnapshot.url}
                target="_blank"
                rel="noreferrer"
                download
                className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white font-medium transition-colors"
              >
                Download Snapshot
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Suspect Multi-Camera Trajectory Modal */}
      {selectedSuspectForTrajectory && (
        <SuspectTrajectoryModal
          suspectName={selectedSuspectForTrajectory}
          onClose={() => setSelectedSuspectForTrajectory(null)}
        />
      )}
    </div>
  );
}

export default function AlertsPage() {
  return (
    <Suspense
      fallback={
        <div className="space-y-6 pb-12 animate-pulse">
          <div className="h-10 w-64 bg-slate-200 rounded-xl" />
          <div className="h-16 w-full bg-slate-100 rounded-2xl" />
          <div className="h-96 w-full bg-slate-100 rounded-2xl" />
        </div>
      }
    >
      <AlertsContent />
    </Suspense>
  );
}

