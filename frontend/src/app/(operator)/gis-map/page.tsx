"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";

const BorderMap = dynamic(
  () => import("@/components/map/BorderMap").then((m) => m.BorderMap),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full min-h-[600px] flex items-center justify-center bg-slate-900 rounded-2xl animate-pulse text-emerald-400 font-mono text-sm">
        Initializing Tactical GIS Map Grid...
      </div>
    ),
  }
);
import { RecentAlerts } from "@/components/dashboard/recentalerts";
import { CameraFeedStrip } from "@/components/dashboard/camerafeedstrip";
import {
  Map as MapIcon,
  Layers,
  Crosshair,
  ShieldCheck,
  Maximize2,
  Minimize2,
  Radio,
  Navigation,
  Activity,
} from "lucide-react";
import { useAlerts } from "@/lib/alertsStore";
import { SuspectTrajectoryModal } from "@/components/map/SuspectTrajectoryModal";
import { api } from "@/lib/api";
import { GlobalTraceItem } from "@/types/alert";
import { CctvCameraIcon } from "@/components/icons/CctvCameraIcon";

export default function GisMapPage() {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeLayer, setActiveLayer] = useState<"hybrid" | "satellite" | "terrain" | "roadmap">("hybrid");
  const [selectedTrajectorySuspect, setSelectedTrajectorySuspect] = useState<string | null>(null);
  const [globalTraces, setGlobalTraces] = useState<GlobalTraceItem[]>([]);
  const [showTraceMenu, setShowTraceMenu] = useState(false);

  const { alerts, suspectsCount } = useAlerts();

  useEffect(() => {
    let mounted = true;
    async function fetchTraces() {
      try {
        const res = await api.getGlobalTraces();
        if (mounted && res?.traces) {
          setGlobalTraces(res.traces);
        }
      } catch {
        // silent fallback
      }
    }
    fetchTraces();
    const interval = setInterval(fetchTraces, 4000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // Find suspects with alerts
  const detectedSuspects = Array.from(
    new Set(
      alerts
        .filter((a) => Boolean(a.suspectName))
        .map((a) => a.suspectName!)
    )
  );

  return (
    <div className="space-y-6 pb-12">
      {/* GIS Portal Header */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 md:p-6 shadow-xs flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-[#143724] text-white flex items-center justify-center shadow-xs">
              <MapIcon className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-xl md:text-2xl font-black tracking-tight text-slate-900">
                Tactical GIS Surveillance Map
              </h1>
              <p className="text-xs md:text-sm font-medium text-slate-500 mt-0.5">
                Real-time geospatial camera nodes, perimeter geofences, and alert coordinates.
              </p>
            </div>
          </div>
        </div>

        {/* GIS Controls & Meta */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Geolocation Tag */}
          <div className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-mono font-medium">
            <Crosshair className="w-3.5 h-3.5 text-emerald-600" />
            <span>Camlocations Tactical Grid • 10 Surveillance Nodes</span>
          </div>

          {/* Suspect Trajectory Quick Action */}
          {detectedSuspects.length > 0 && (
            <button
              type="button"
              onClick={() => setSelectedTrajectorySuspect(detectedSuspects[0])}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border border-rose-500/80 bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-xs transition-all animate-pulse cursor-pointer"
            >
              <Navigation className="w-3.5 h-3.5" />
              <span>Suspect Trajectory ({detectedSuspects.length})</span>
            </button>
          )}

          {/* Re-ID Global Traces Quick Action */}
          {globalTraces.length > 0 && (
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowTraceMenu(!showTraceMenu)}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border border-cyan-500/80 bg-cyan-700 hover:bg-cyan-800 text-white text-xs font-bold shadow-xs transition-all cursor-pointer"
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Re-ID Global Traces ({globalTraces.length})</span>
              </button>
              {showTraceMenu && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-white rounded-2xl border border-slate-200 shadow-2xl p-2.5 z-50">
                  <div className="px-2.5 py-1.5 border-b border-slate-100 flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase tracking-wider font-mono">
                    <span>Active Person Traces</span>
                    <span>Last Node</span>
                  </div>
                  <div className="max-h-64 overflow-y-auto divide-y divide-slate-100 mt-1">
                    {globalTraces.map((t) => (
                      <button
                        key={t.trace_id}
                        type="button"
                        onClick={() => {
                          setSelectedTrajectorySuspect(t.trace_id);
                          setShowTraceMenu(false);
                        }}
                        className="w-full text-left px-2.5 py-2 hover:bg-cyan-50/60 rounded-xl flex items-center justify-between transition-colors group cursor-pointer"
                      >
                        <div>
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-xs font-bold text-cyan-900 group-hover:text-cyan-700">
                              {t.trace_id}
                            </span>
                            {t.is_cross_camera && (
                              <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-cyan-100 text-cyan-800 border border-cyan-200">
                                ⇄ Multi-Cam
                              </span>
                            )}
                            {t.is_suspect && (
                              <span className="px-1.5 py-0.2 rounded text-[9px] font-bold bg-rose-100 text-rose-800 border border-rose-200">
                                Suspect
                              </span>
                            )}
                          </div>
                          <p className="text-[10px] text-slate-500 font-mono mt-0.5 flex items-center gap-1">
                            {t.suspect_name ? (
                              <span className="text-rose-700 font-semibold">{t.suspect_name}</span>
                            ) : (
                              <>
                                <span>{t.total_sightings} sightings</span>
                                <span>•</span>
                                <span className="inline-flex items-center gap-0.5">
                                  <CctvCameraIcon className="w-2.5 h-2.5 text-cyan-700" />
                                  {t.unique_cameras_count} camera(s)
                                </span>
                              </>
                            )}
                          </p>
                        </div>
                        <span className="text-[10px] font-mono font-medium text-slate-600 truncate max-w-[110px] text-right bg-slate-100 px-1.5 py-0.5 rounded flex items-center gap-1 shrink-0">
                          <CctvCameraIcon className="w-2.5 h-2.5 text-slate-500 shrink-0" />
                          <span className="truncate">{t.last_camera_name}</span>
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Google Maps Layer Toggle */}
          <div className="inline-flex p-1 rounded-xl bg-slate-100 border border-slate-200">
            <button
              onClick={() => setActiveLayer("hybrid")}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all ${
                activeLayer === "hybrid"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Hybrid
            </button>
            <button
              onClick={() => setActiveLayer("satellite")}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all ${
                activeLayer === "satellite"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Satellite
            </button>
            <button
              onClick={() => setActiveLayer("terrain")}
              className={`px-3 py-1 text-xs font-semibold rounded-lg transition-all ${
                activeLayer === "terrain"
                  ? "bg-white text-slate-900 shadow-xs"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Terrain
            </button>
          </div>

          {/* Focus Map Toggle */}
          <button
            type="button"
            onClick={() => setIsFullscreen(!isFullscreen)}
            className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl border text-xs font-bold transition-all shadow-xs cursor-pointer ${
              isFullscreen
                ? "bg-emerald-600 text-white border-emerald-500 hover:bg-emerald-500"
                : "border-emerald-200 bg-emerald-50 text-emerald-800 hover:bg-emerald-100 hover:border-emerald-300"
            }`}
            title={isFullscreen ? "Exit Maximum Focus Mode (Esc)" : "Open Map to Maximum Screen Space"}
          >
            {isFullscreen ? (
              <>
                <Minimize2 className="w-3.5 h-3.5" />
                <span>Exit Focus</span>
              </>
            ) : (
              <>
                <Maximize2 className="w-3.5 h-3.5 text-emerald-700" />
                <span>Focus Map</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Map + Recent Alerts Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Full Interactive Google BorderMap */}
        <div className="lg:col-span-7 xl:col-span-8 flex flex-col">
          <BorderMap
            initialMapType={activeLayer}
            onMapTypeChange={(layer) => setActiveLayer(layer)}
            isFullscreen={isFullscreen}
            onToggleFullscreen={() => setIsFullscreen(!isFullscreen)}
            height="580px"
          />
        </div>

        {/* Tactical Feed / Recent Alerts */}
        <div className="lg:col-span-5 xl:col-span-4 flex flex-col space-y-6">
          <RecentAlerts />
        </div>
      </div>

      {/* Camera Feed Strip */}
      <CameraFeedStrip />

      {/* Suspect Multi-Camera Trajectory Modal */}
      {selectedTrajectorySuspect && (
        <SuspectTrajectoryModal
          suspectName={selectedTrajectorySuspect}
          onClose={() => setSelectedTrajectorySuspect(null)}
        />
      )}
    </div>
  );
}
