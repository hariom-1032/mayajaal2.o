"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import Link from "next/link";
import {
  Activity,
  Navigation,
  Video,
  RefreshCw,
  Search,
  Filter,
  ShieldAlert,
  Clock,
  ArrowUpRight,
  Trash2,
  Download,
  Upload,
  Play,
  Layers,
  ChevronRight,
  Eye,
  CheckCircle2,
  AlertTriangle,
  Radio,
  Sliders,
  Sparkles,
} from "lucide-react";
import { api } from "@/lib/api";
import { GlobalTraceItem } from "@/types/alert";
import { SuspectTrajectoryModal } from "@/components/map/SuspectTrajectoryModal";
import { TrackingVideoResponse } from "@/types/backend";

export default function TracksPage() {
  const [activeTab, setActiveTab] = useState<"live" | "sandbox">("live");
  const [traces, setTraces] = useState<GlobalTraceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [searchQuery, setSearchQuery] = useState("");
  const [filterType, setFilterType] = useState<"all" | "cross" | "suspects" | "high_activity">("all");
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);

  // Video Tracking Sandbox State
  const [sandboxFile, setSandboxFile] = useState<File | null>(null);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [matchingThreshold, setMatchingThreshold] = useState(0.8);
  const [lostTrackBuffer, setLostTrackBuffer] = useState(30);
  const [maxFrames, setMaxFrames] = useState(150);
  const [sandboxTracking, setSandboxTracking] = useState(false);
  const [sandboxResult, setSandboxResult] = useState<TrackingVideoResponse | null>(null);
  const [sandboxError, setSandboxError] = useState<string | null>(null);

  const fetchTraces = useCallback(async (isManual = false) => {
    if (isManual) setRefreshing(true);
    try {
      const res = await api.getGlobalTraces();
      if (res && Array.isArray(res.traces)) {
        setTraces(res.traces);
        setLastUpdated(new Date());
      }
    } catch (err) {
      console.error("Failed to fetch global traces", err);
    } finally {
      setLoading(false);
      if (isManual) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchTraces();
  }, [fetchTraces]);

  // Periodic polling for real-time tracking updates
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchTraces();
    }, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchTraces]);

  const handleResetTraces = async () => {
    if (!window.confirm("Are you sure you want to reset all active global person traces? This will clear all multi-camera re-ID identities.")) {
      return;
    }
    setResetting(true);
    try {
      await api.resetGlobalTraces();
      await fetchTraces(true);
    } catch (err) {
      console.error("Failed to reset traces", err);
    } finally {
      setResetting(false);
    }
  };

  const handleExportCSV = () => {
    if (traces.length === 0) return;
    const headers = "Trace ID,First Seen,Last Seen,Duration (s),Last Camera ID,Last Camera Name,Total Sightings,Unique Cameras,Cross Camera,Is Suspect,Suspect Name,Threat Level\n";
    const rows = traces
      .map((t) =>
        [
          t.trace_id,
          new Date(t.first_seen * 1000).toISOString(),
          new Date(t.last_seen * 1000).toISOString(),
          t.duration_sec,
          `"${t.last_camera_id}"`,
          `"${t.last_camera_name}"`,
          t.total_sightings,
          t.unique_cameras_count,
          t.is_cross_camera,
          t.is_suspect,
          `"${t.suspect_name || ""}"`,
          `"${t.threat_level || ""}"`,
        ].join(",")
      )
      .join("\n");

    const blob = new Blob([headers + rows], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `border_tracks_telemetry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Run ByteTrack on uploaded video
  const handleRunSandboxTracking = async () => {
    if (!sandboxFile) return;
    setSandboxTracking(true);
    setSandboxError(null);
    setSandboxResult(null);

    try {
      const res = await api.runVideoTracking({
        file: sandboxFile,
        modelName: "best.optimized.onnx",
        confThreshold,
        matchingThreshold,
        lostTrackBuffer,
        maxFrames,
      });
      setSandboxResult(res);
      // Refresh global traces after sandbox run
      fetchTraces();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Video tracking inference failed.";
      setSandboxError(msg);
    } finally {
      setSandboxTracking(false);
    }
  };

  // Filter traces
  const filteredTraces = useMemo(() => {
    return traces.filter((trace) => {
      if (filterType === "cross" && !trace.is_cross_camera) return false;
      if (filterType === "suspects" && !trace.is_suspect) return false;
      if (filterType === "high_activity" && trace.total_sightings < 10) return false;

      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchId = trace.trace_id.toLowerCase().includes(q);
        const matchCam = trace.last_camera_name.toLowerCase().includes(q) || trace.last_camera_id.toLowerCase().includes(q);
        const matchSuspect = trace.suspect_name?.toLowerCase().includes(q);
        if (!matchId && !matchCam && !matchSuspect) return false;
      }
      return true;
    });
  }, [traces, filterType, searchQuery]);

  // Aggregate stats
  const totalCrossCamera = traces.filter((t) => t.is_cross_camera).length;
  const totalSuspects = traces.filter((t) => t.is_suspect).length;
  const totalSightings = traces.reduce((acc, t) => acc + (t.total_sightings || 1), 0);

  const backendBase =
    process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

  return (
    <div className="space-y-6 pb-16">
      {/* Page Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 pb-4 border-b border-slate-200/80">
        <div>
          <div className="flex items-center gap-2.5 mb-1.5">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-[#eaf4ed] text-[#1b5032] border border-[#c4ded0]">
              <span className="w-2 h-2 rounded-full bg-[#16a34a] animate-pulse" />
              ByteTrack + OSNet Re-ID Online
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Synced {lastUpdated.toLocaleTimeString()}
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
            <Activity className="w-7 h-7 text-[#1e4b38]" />
            Tracks & Multi-Camera Trajectories
          </h1>
          <p className="text-xs sm:text-sm font-medium text-slate-500 mt-0.5">
            Real-time identity persistence, border sector transits, and automated cross-camera suspect trajectories.
          </p>
        </div>

        {/* Global Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Tab Selector */}
          <div className="inline-flex p-1 rounded-xl bg-slate-200/70 border border-slate-300/60 shadow-inner">
            <button
              type="button"
              onClick={() => setActiveTab("live")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                activeTab === "live"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Radio className="w-3.5 h-3.5 text-[#1e4b38]" />
              <span>Live Tracks ({traces.length})</span>
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("sandbox")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 ${
                activeTab === "sandbox"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              <Sliders className="w-3.5 h-3.5 text-amber-600" />
              <span>ByteTrack Sandbox</span>
            </button>
          </div>

          {/* Auto Refresh Toggle */}
          <button
            type="button"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition-all cursor-pointer flex items-center gap-1.5 ${
              autoRefresh
                ? "bg-emerald-50 text-emerald-800 border-emerald-300"
                : "bg-slate-100 text-slate-600 border-slate-200"
            }`}
            title={autoRefresh ? "Auto-refresh active (3s)" : "Auto-refresh paused"}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                autoRefresh ? "bg-emerald-500 animate-pulse" : "bg-slate-400"
              }`}
            />
            <span>{autoRefresh ? "Auto (3s)" : "Paused"}</span>
          </button>

          {/* Manual Refresh */}
          <button
            type="button"
            onClick={() => fetchTraces(true)}
            disabled={refreshing}
            className="p-2 rounded-xl bg-white border border-slate-200 hover:border-slate-300 text-slate-700 shadow-2xs hover:bg-slate-50 transition-all cursor-pointer disabled:opacity-50"
            title="Refresh Tracks"
          >
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin text-[#1e4b38]" : ""}`} />
          </button>

          {/* Export CSV */}
          <button
            type="button"
            onClick={handleExportCSV}
            disabled={traces.length === 0}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white border border-slate-200 hover:border-slate-300 text-slate-700 shadow-2xs hover:bg-slate-50 transition-all cursor-pointer disabled:opacity-50"
            title="Export CSV of Tracks"
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Export CSV</span>
          </button>

          {/* Reset Traces */}
          <button
            type="button"
            onClick={handleResetTraces}
            disabled={resetting || traces.length === 0}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-rose-50 border border-rose-200 hover:bg-rose-100 text-rose-700 transition-all cursor-pointer disabled:opacity-50"
            title="Clear and reset all global cross-camera tracking identities"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Reset Traces</span>
          </button>
        </div>
      </div>

      {/* Top 4 Operational Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Active Persons */}
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs">
          <div className="flex items-center justify-between gap-2 text-slate-500 mb-2">
            <span className="text-[11px] font-bold tracking-wider uppercase flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-[#1e4b38]" />
              <span>Active Persons</span>
            </span>
            <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
              Live
            </span>
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
            {traces.length}
          </div>
          <p className="text-[11px] text-slate-400 font-medium mt-1">
            Unique identities persistent across nodes
          </p>
        </div>

        {/* Cross-Camera Infiltrations */}
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs">
          <div className="flex items-center justify-between gap-2 text-slate-500 mb-2">
            <span className="text-[11px] font-bold tracking-wider uppercase flex items-center gap-1.5">
              <Navigation className="w-3.5 h-3.5 text-blue-600" />
              <span>Cross-Border Transits</span>
            </span>
            {totalCrossCamera > 0 && (
              <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-blue-100 text-blue-700 animate-pulse">
                In Transit
              </span>
            )}
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-blue-700 tracking-tight">
            {totalCrossCamera}
          </div>
          <p className="text-[11px] text-slate-400 font-medium mt-1">
            Observed across 2+ distinct border outposts
          </p>
        </div>

        {/* Watchlist Suspect Traces */}
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs">
          <div className="flex items-center justify-between gap-2 text-slate-500 mb-2">
            <span className="text-[11px] font-bold tracking-wider uppercase flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-rose-600" />
              <span>Suspect Sightings</span>
            </span>
            {totalSuspects > 0 && (
              <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-rose-100 text-rose-700">
                Threat Alert
              </span>
            )}
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-rose-600 tracking-tight">
            {totalSuspects}
          </div>
          <p className="text-[11px] text-slate-400 font-medium mt-1">
            Biometric match against watchlist
          </p>
        </div>

        {/* Total Sighting Events */}
        <div className="bg-white rounded-2xl border border-slate-200 p-4 shadow-xs">
          <div className="flex items-center justify-between gap-2 text-slate-500 mb-2">
            <span className="text-[11px] font-bold tracking-wider uppercase flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-emerald-700" />
              <span>Total Detections</span>
            </span>
            <span className="text-[10px] font-bold font-mono px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
              Sightings
            </span>
          </div>
          <div className="text-2xl sm:text-3xl font-extrabold text-[#1e4b38] tracking-tight">
            {totalSightings.toLocaleString()}
          </div>
          <p className="text-[11px] text-slate-400 font-medium mt-1">
            Bounding box detections aggregated
          </p>
        </div>
      </div>

      {activeTab === "live" ? (
        /* LIVE MULTI-CAMERA TRACKS TAB */
        <div className="space-y-4">
          {/* Search & Filter Toolbar */}
          <div className="bg-white rounded-2xl border border-slate-200/90 p-3.5 shadow-xs flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            {/* Search Input */}
            <div className="relative flex-1 max-w-md">
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search by Trace ID (e.g. TRC-0022), Camera, or Suspect..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-xl text-xs border border-slate-200 focus:outline-hidden focus:ring-2 focus:ring-[#1e4b38] focus:border-transparent bg-slate-50/50"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 hover:text-slate-600"
                >
                  Clear
                </button>
              )}
            </div>

            {/* Filter Pills */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs text-slate-400 font-medium mr-1 flex items-center gap-1">
                <Filter className="w-3.5 h-3.5" /> Filter:
              </span>
              <button
                type="button"
                onClick={() => setFilterType("all")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  filterType === "all"
                    ? "bg-[#1e4b38] text-white shadow-2xs"
                    : "bg-slate-100 hover:bg-slate-200 text-slate-700"
                }`}
              >
                All Tracks ({traces.length})
              </button>
              <button
                type="button"
                onClick={() => setFilterType("cross")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  filterType === "cross"
                    ? "bg-blue-600 text-white shadow-2xs"
                    : "bg-blue-50 hover:bg-blue-100 text-blue-700"
                }`}
              >
                Cross-Border ({totalCrossCamera})
              </button>
              <button
                type="button"
                onClick={() => setFilterType("suspects")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  filterType === "suspects"
                    ? "bg-rose-600 text-white shadow-2xs"
                    : "bg-rose-50 hover:bg-rose-100 text-rose-700"
                }`}
              >
                Suspects ({totalSuspects})
              </button>
              <button
                type="button"
                onClick={() => setFilterType("high_activity")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  filterType === "high_activity"
                    ? "bg-emerald-800 text-white shadow-2xs"
                    : "bg-slate-100 hover:bg-slate-200 text-slate-700"
                }`}
              >
                High Activity (≥10)
              </button>
            </div>
          </div>

          {/* Traces List Table / Grid */}
          {loading && traces.length === 0 ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-xs">
              <RefreshCw className="w-8 h-8 text-[#1e4b38] animate-spin mx-auto mb-3" />
              <p className="text-sm font-semibold text-slate-700">Loading active multi-camera tracks...</p>
              <p className="text-xs text-slate-400 mt-1">Connecting to ByteTrack and Global Re-ID engine</p>
            </div>
          ) : filteredTraces.length === 0 ? (
            <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-xs">
              <div className="w-14 h-14 rounded-2xl bg-slate-100 text-slate-400 flex items-center justify-center mx-auto mb-4">
                <Activity className="w-7 h-7" />
              </div>
              <h3 className="text-base font-bold text-slate-900">
                {traces.length === 0
                  ? "No Active Global Tracks in Surveillance Memory"
                  : "No Tracks Match Current Filters"}
              </h3>
              <p className="text-xs text-slate-500 max-w-md mx-auto mt-1 leading-relaxed">
                {traces.length === 0
                  ? "Persons detected on live camera feeds or RTSP streams are automatically tracked, assigned persistent IDs, and correlated across border outposts via OSNet Re-ID."
                  : "Try clearing search keywords or selecting 'All Tracks' to view all monitored subjects."}
              </p>
              <div className="flex items-center justify-center gap-3 mt-6">
                <Link
                  href="/live"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white shadow-xs transition-all cursor-pointer"
                >
                  <Video className="w-4 h-4" />
                  <span>Open Live Surveillance Feeds</span>
                </Link>
                <button
                  type="button"
                  onClick={() => setActiveTab("sandbox")}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold bg-slate-100 hover:bg-slate-200 text-slate-700 transition-all cursor-pointer"
                >
                  <Sliders className="w-4 h-4" />
                  <span>Test ByteTrack Sandbox</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredTraces.map((trace) => {
                const durationMinutes = Math.round(trace.duration_sec / 60);
                const lastSeenRelative = Math.round((Date.now() / 1000 - trace.last_seen));
                const isVeryActive = trace.total_sightings >= 20;

                return (
                  <div
                    key={trace.trace_id}
                    className={`bg-white rounded-2xl border transition-all hover:shadow-md p-5 flex flex-col justify-between group relative ${
                      trace.is_cross_camera
                        ? "border-blue-300 ring-1 ring-blue-100"
                        : trace.is_suspect
                        ? "border-rose-300 ring-1 ring-rose-100"
                        : "border-slate-200 hover:border-[#1e4b38]/50"
                    }`}
                  >
                    <div>
                      {/* Top Badges */}
                      <div className="flex items-center justify-between gap-2 mb-3">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-extrabold px-2.5 py-1 rounded-lg bg-slate-900 text-emerald-400 shadow-2xs">
                            {trace.trace_id}
                          </span>
                          {trace.is_cross_camera && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-extrabold uppercase tracking-wider bg-blue-50 text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full">
                              <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-pulse" />
                              Multi-Outpost ({trace.unique_cameras_count} Cams)
                            </span>
                          )}
                          {trace.is_suspect && (
                            <span className="inline-flex items-center gap-1 text-[10px] font-extrabold uppercase tracking-wider bg-rose-50 text-rose-700 border border-rose-200 px-2 py-0.5 rounded-full">
                              <ShieldAlert className="w-3 h-3 text-rose-600" />
                              {trace.threat_level || "SUSPECT"}
                            </span>
                          )}
                        </div>

                        <span className="text-[11px] text-slate-400 font-mono">
                          {lastSeenRelative <= 5
                            ? "Active Now"
                            : `${lastSeenRelative}s ago`}
                        </span>
                      </div>

                      {/* Subject Classification & Outpost Details */}
                      <div className="mb-4">
                        <div className="flex items-baseline justify-between gap-2">
                          <h4 className="text-base font-bold text-slate-900 group-hover:text-[#1e4b38] transition-colors">
                            {trace.suspect_name || "Unidentified Pedestrian"}
                          </h4>
                          <span className="text-xs font-semibold text-slate-500">
                            {trace.total_sightings} sightings
                          </span>
                        </div>

                        <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-600 font-medium">
                          <Video className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                          <span className="truncate" title={trace.last_camera_name}>
                            {trace.last_camera_name}
                          </span>
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">
                            {trace.last_camera_id}
                          </span>
                        </div>
                      </div>

                      {/* Telemetry Progress & Duration */}
                      <div className="bg-slate-50/80 rounded-xl p-3 border border-slate-100 mb-4 space-y-2">
                        <div className="flex items-center justify-between text-[11px] text-slate-500 font-medium">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-400" />
                            Duration in Perimeter
                          </span>
                          <span className="font-bold text-slate-700 font-mono">
                            {trace.duration_sec.toFixed(1)}s ({durationMinutes}m)
                          </span>
                        </div>

                        {/* Sighting Density Bar */}
                        <div>
                          <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden flex">
                            <div
                              style={{
                                width: `${Math.min(100, (trace.total_sightings / 40) * 100)}%`,
                              }}
                              className={`h-full rounded-full transition-all duration-500 ${
                                trace.is_suspect
                                  ? "bg-rose-500"
                                  : trace.is_cross_camera
                                  ? "bg-blue-600"
                                  : "bg-[#2d6a4f]"
                              }`}
                            />
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                          <span>First: {new Date(trace.first_seen * 1000).toLocaleTimeString()}</span>
                          <span>Last: {new Date(trace.last_seen * 1000).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-center gap-2 pt-1 border-t border-slate-100">
                      <button
                        type="button"
                        onClick={() => setSelectedTraceId(trace.trace_id)}
                        className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white shadow-2xs transition-all cursor-pointer"
                        title="View chronological GPS multi-camera trajectory and QRT dispatch"
                      >
                        <Navigation className="w-3.5 h-3.5" />
                        <span>Inspect Trajectory</span>
                      </button>

                      <Link
                        href={`/live?camera=${encodeURIComponent(trace.last_camera_id)}`}
                        className="inline-flex items-center justify-center p-2 rounded-xl text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 transition-all cursor-pointer"
                        title="Open Live Camera Stream"
                      >
                        <Eye className="w-4 h-4" />
                      </Link>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        /* BYTE TRACK VIDEO SANDBOX TAB */
        <div className="bg-white rounded-2xl border border-slate-200/90 p-6 shadow-xs space-y-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight flex items-center gap-2">
              <Sliders className="w-5 h-5 text-amber-600" />
              ByteTrack Video Inference & Tracking Sandbox
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Upload a surveillance video clip to execute high-speed multi-object tracking with ByteTrack and YOLO ONNX acceleration.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Upload & Controls Panel */}
            <div className="lg:col-span-5 space-y-4">
              {/* File Dropzone */}
              <div>
                <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Surveillance Video File (MP4, AVI, MKV)
                </label>
                <div className="border-2 border-dashed border-slate-300 hover:border-[#1e4b38] rounded-2xl p-6 text-center transition-all bg-slate-50/50 hover:bg-slate-50 cursor-pointer">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setSandboxFile(e.target.files[0]);
                      }
                    }}
                    className="hidden"
                    id="video-sandbox-upload"
                  />
                  <label htmlFor="video-sandbox-upload" className="cursor-pointer block">
                    <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                    {sandboxFile ? (
                      <div>
                        <p className="text-xs font-bold text-[#1e4b38] truncate">{sandboxFile.name}</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">
                          {(sandboxFile.size / (1024 * 1024)).toFixed(2)} MB • Click to change
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-xs font-bold text-slate-700">Choose Video File</p>
                        <p className="text-[10px] text-slate-400 mt-0.5">MP4, MOV, or WEBM up to 100MB</p>
                      </div>
                    )}
                  </label>
                </div>
              </div>

              {/* ByteTracker Tuning Sliders */}
              <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 space-y-3.5">
                <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                  Tracker Hyperparameters
                </h4>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                    <span>Detection Confidence Threshold</span>
                    <span className="font-mono font-bold text-[#1e4b38]">{confThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={confThreshold}
                    onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#1e4b38]"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                    <span>IoU Matching Threshold</span>
                    <span className="font-mono font-bold text-[#1e4b38]">{matchingThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.40"
                    max="0.95"
                    step="0.05"
                    value={matchingThreshold}
                    onChange={(e) => setMatchingThreshold(parseFloat(e.target.value))}
                    className="w-full accent-[#1e4b38]"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                    <span>Lost Track Buffer (Frames)</span>
                    <span className="font-mono font-bold text-[#1e4b38]">{lostTrackBuffer} frames</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="90"
                    step="5"
                    value={lostTrackBuffer}
                    onChange={(e) => setLostTrackBuffer(parseInt(e.target.value))}
                    className="w-full accent-[#1e4b38]"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs font-semibold text-slate-600 mb-1">
                    <span>Max Frames to Process</span>
                    <span className="font-mono font-bold text-[#1e4b38]">{maxFrames} frames</span>
                  </div>
                  <input
                    type="range"
                    min="30"
                    max="300"
                    step="30"
                    value={maxFrames}
                    onChange={(e) => setMaxFrames(parseInt(e.target.value))}
                    className="w-full accent-[#1e4b38]"
                  />
                </div>
              </div>

              {/* Run Button */}
              <button
                type="button"
                onClick={handleRunSandboxTracking}
                disabled={!sandboxFile || sandboxTracking}
                className="w-full inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-xs font-bold bg-[#1e4b38] hover:bg-[#163a2b] text-white shadow-xs transition-all cursor-pointer disabled:opacity-50"
              >
                {sandboxTracking ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Running ByteTrack Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Execute Video Tracking Inference</span>
                  </>
                )}
              </button>

              {sandboxError && (
                <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{sandboxError}</span>
                </div>
              )}
            </div>

            {/* Results & Playback Area */}
            <div className="lg:col-span-7 flex flex-col justify-between">
              {sandboxResult ? (
                <div className="space-y-4">
                  {/* Result Summary Bar */}
                  <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-emerald-900 flex items-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4 text-emerald-700" />
                        Tracking Inference Completed Successfully
                      </h4>
                      <p className="text-[11px] text-emerald-700 mt-0.5 font-medium">
                        Processed {sandboxResult.frames_processed} frames using {sandboxResult.model_name}
                      </p>
                    </div>

                    {sandboxResult.annotated_video_url && (
                      <a
                        href={`${backendBase}${sandboxResult.annotated_video_url}`}
                        target="_blank"
                        rel="noreferrer"
                        download
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold bg-[#1e4b38] text-white shadow-2xs hover:bg-[#163a2b]"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download MP4</span>
                      </a>
                    )}
                  </div>

                  {/* Video Player */}
                  {sandboxResult.annotated_video_url && (
                    <div className="rounded-2xl overflow-hidden border border-slate-200 bg-black aspect-video flex items-center justify-center shadow-inner">
                      <video
                        src={`${backendBase}${sandboxResult.annotated_video_url}`}
                        controls
                        autoPlay
                        loop
                        className="w-full h-full object-contain"
                      />
                    </div>
                  )}

                  {/* Frame Telemetry Table */}
                  <div className="border border-slate-200 rounded-xl overflow-hidden">
                    <div className="bg-slate-100/80 px-4 py-2 border-b border-slate-200 text-xs font-bold text-slate-800">
                      Frame Breakdown (First 10 Sampled Frames)
                    </div>
                    <div className="max-h-56 overflow-y-auto divide-y divide-slate-100 text-xs">
                      {sandboxResult.results.slice(0, 10).map((fr, idx) => (
                        <div key={idx} className="p-3 flex items-center justify-between">
                          <span className="font-mono text-slate-600 font-semibold">Frame #{fr.frame_index}</span>
                          <span className="text-slate-500">
                            {fr.tracked_objects?.length || fr.detections?.length || 0} objects tracked
                          </span>
                          <span className="font-mono text-[10px] text-slate-400">
                            {fr.timestamp_seconds ? `${fr.timestamp_seconds.toFixed(2)}s` : "-"}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full min-h-[300px] rounded-2xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center p-8 text-center text-slate-400 bg-slate-50/30">
                  <Video className="w-12 h-12 mb-3 text-slate-300" />
                  <p className="text-sm font-semibold text-slate-600">Video Tracking Output Preview</p>
                  <p className="text-xs text-slate-400 max-w-sm mt-1">
                    Select a video on the left panel and click 'Execute Video Tracking Inference' to view tracked bounding boxes and persistent identity paths.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Suspect Multi-Camera Trajectory Modal */}
      {selectedTraceId && (
        <SuspectTrajectoryModal
          suspectName={selectedTraceId}
          onClose={() => setSelectedTraceId(null)}
        />
      )}
    </div>
  );
}
