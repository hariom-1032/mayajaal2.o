"use client";

import { useState, useMemo, useEffect, useRef } from "react";
import {
  Camera as CameraIcon,
  Plus,
  Trash2,
  Settings,
  Activity,
  MapPin,
  Wifi,
  WifiOff,
  Video,
  Radio,
  Search,
  Filter,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Copy,
  Check,
  ExternalLink,
  Sliders,
  Shield,
  Layers,
  Eye,
  Maximize2,
  X,
  Crosshair,
  Server,
  Zap,
  Globe,
  SlidersHorizontal,
} from "lucide-react";
import {
  Camera,
  CameraStatus,
  CameraType,
  CreateCameraInput,
  UpdateCameraInput,
  StreamTestResult,
} from "@/types/camera";
import dynamic from "next/dynamic";
import { useCameras } from "@/lib/camerasStore";

const BorderMap = dynamic(
  () => import("@/components/map/BorderMap").then((m) => m.BorderMap),
  {
    ssr: false,
    loading: () => (
      <div className="h-[760px] w-full flex items-center justify-center bg-slate-100 rounded-2xl animate-pulse text-slate-400 text-xs font-mono">
        Loading Tactical Border Map...
      </div>
    ),
  }
);

// Common sector options across India's border sectors
const SECTORS = [
  "Sector-04 (Jammu - RS Pura)",
  "Sector-03 (Jammu - Gajansoo)",
  "Sector-03 (Akhnoor - Chenab)",
  "Sector-02 (Samba - Ramgarh)",
  "Sector-01 (Kathua - Hiranagar)",
  "Sector-06 (Poonch - LoC)",
  "Sector-07 (Baramulla - Uri LoC)",
  "Sector-08 (Kupwara - Kishanganga LoC)",
  "Sector-09 (Punjab - Amritsar)",
  "Sector-10 (Punjab - Gurdaspur)",
  "Sector-11 (Punjab - Ferozepur)",
  "Sector-12 (Punjab - Fazilka)",
  "Sector-13 (Rajasthan - Sri Ganganagar)",
  "Sector-14 (Rajasthan - Bikaner)",
  "Sector-15 (Rajasthan - Jaisalmer)",
  "Sector-16 (Rajasthan - Barmer)",
  "Sector-17 (Gujarat - Sir Creek)",
  "Sector-18 (Gujarat - Khavda Kutch)",
  "Sector-19 (Gujarat - Lakhpat)",
  "Sector-20 (Ladakh - Pangong Sector)",
  "Sector-21 (Ladakh - Sub-Sector North)",
  "Sector-22 (Himachal - Kinnaur LAC)",
  "Sector-23 (Uttarakhand - Chamoli)",
  "Sector-24 (Sikkim - East District)",
  "Sector-25 (Arunachal - Tawang LAC)",
  "Sector-26 (Arunachal - Anjaw)",
  "Sector-27 (West Bengal - North 24 Parganas)",
  "Sector-28 (West Bengal - Dakshin Dinajpur)",
  "Sector-29 (West Bengal - Siliguri)",
  "Sector-30 (Meghalaya - West Jaintia Hills)",
  "Sector-31 (Assam - Barak Valley)",
  "Sector-32 (Manipur - Tengnoupal)",
  "Sector-33 (Mizoram - Champhai)",
  "Sector-34 (Bihar - East Champaran)",
  "Sector-35 (Uttarakhand - Champawat)",
  "Sector-36 (West Bengal - Alipurduar)",
];

// Camera type options
const CAMERA_TYPES: CameraType[] = [
  "Optical 4K",
  "Thermal FLIR",
  "Night Vision / IR",
  "ANPR Dedicated",
  "PTZ 360",
  "Panoramic",
];

// Preset coordinates along border sector
const COORDINATE_PRESETS = [
  { name: "RS Pura BOP Alpha (Jammu IB)", lat: 32.7160, lng: 74.6640 },
  { name: "Suchetgarh JCP Octroi Gate (Jammu)", lat: 32.6840, lng: 74.6720 },
  { name: "Chenab Riverine Bluff (Akhnoor)", lat: 32.7480, lng: 74.6780 },
  { name: "Attari-Wagah Joint Check Post (Punjab)", lat: 31.6045, lng: 74.5750 },
  { name: "Tanot Mata Post (Jaisalmer Thar)", lat: 27.7900, lng: 70.3500 },
  { name: "Sir Creek Harami Nala (Gujarat)", lat: 23.8800, lng: 68.3200 },
  { name: "Pangong Tso LAC (Ladakh)", lat: 33.7200, lng: 78.5000 },
  { name: "Nathu La Pass (Sikkim LAC)", lat: 27.3860, lng: 88.8310 },
  { name: "Petrapole ICP (West Bengal)", lat: 23.0400, lng: 88.8950 },
];

export default function CamerasPage() {
  const {
    cameras,
    isSyncing,
    totalCount,
    onlineCount,
    alertCount,
    offlineCount,
    avgLatencyMs,
    addCamera,
    updateCamera,
    deleteCamera,
    bulkDeleteCameras,
    testStream,
    resetDefaults,
  } = useCameras();

  // View mode
  const [viewMode, setViewMode] = useState<"grid" | "table" | "map">("grid");

  // Filter & Search states
  const [searchQuery, setSearchQuery] = useState("");
  const [sectorFilter, setSectorFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState<CameraStatus | "All">("All");
  const [typeFilter, setTypeFilter] = useState("All");

  // Selection for bulk actions
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // Modals state
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | null>(null);
  const [deletingCamera, setDeletingCamera] = useState<Camera | null>(null);
  const [previewCamera, setPreviewCamera] = useState<Camera | null>(null);

  // Copied indicator
  const [copiedText, setCopiedText] = useState<string | null>(null);

  // Toast message
  const [toastMessage, setToastMessage] = useState<{
    text: string;
    type: "success" | "info" | "error";
  } | null>(null);

  // Active testing camera IDs
  const [testingIds, setTestingIds] = useState<Record<string, StreamTestResult | "testing">>({});

  const showToast = (text: string, type: "success" | "info" | "error" = "success") => {
    setToastMessage({ text, type });
    setTimeout(() => setToastMessage(null), 4000);
  };

  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(text);
    showToast(`Copied ${label} to clipboard!`, "info");
    setTimeout(() => setCopiedText(null), 2000);
  };

  // Filtered cameras list
  const filteredCameras = useMemo(() => {
    return cameras.filter((cam) => {
      if (sectorFilter !== "All" && cam.sector !== sectorFilter) return false;
      if (statusFilter !== "All" && cam.status !== statusFilter) return false;
      if (typeFilter !== "All" && cam.type !== typeFilter) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchName = cam.name.toLowerCase().includes(q);
        const matchId = cam.id.toLowerCase().includes(q);
        const matchIp = cam.ipAddress?.toLowerCase().includes(q) || false;
        const matchLoc = cam.location.toLowerCase().includes(q);
        const matchUrl = cam.streamUrl?.toLowerCase().includes(q) || false;
        if (!matchName && !matchId && !matchIp && !matchLoc && !matchUrl) {
          return false;
        }
      }
      return true;
    });
  }, [cameras, sectorFilter, statusFilter, typeFilter, searchQuery]);

  // Handle single camera test
  const handleTestCamera = async (cam: Camera) => {
    setTestingIds((prev) => ({ ...prev, [cam.id]: "testing" }));
    const res = await testStream({
      streamUrl: cam.streamUrl,
      ipAddress: cam.ipAddress,
      port: cam.port,
    });
    setTestingIds((prev) => ({ ...prev, [cam.id]: res }));
    if (res.reachable) {
      showToast(`${cam.name}: Handshake successful (${res.latencyMs}ms)`, "success");
    } else {
      showToast(`${cam.name}: ${res.message}`, "error");
    }
  };

  // Handle test all streams
  const handleTestAllStreams = async () => {
    showToast("Initiating batch RTSP ping test on all active nodes...", "info");
    for (const cam of cameras) {
      setTestingIds((prev) => ({ ...prev, [cam.id]: "testing" }));
      testStream({
        streamUrl: cam.streamUrl,
        ipAddress: cam.ipAddress,
        port: cam.port,
      }).then((res) => {
        setTestingIds((prev) => ({ ...prev, [cam.id]: res }));
      });
    }
  };

  // Handle delete camera
  const handleDeleteConfirm = async () => {
    if (!deletingCamera) return;
    const success = await deleteCamera(deletingCamera.id);
    if (success) {
      showToast(`Camera '${deletingCamera.name}' removed from registry.`, "info");
      setSelectedIds((prev) => prev.filter((id) => id !== deletingCamera.id));
    }
    setDeletingCamera(null);
  };

  // Bulk delete
  const handleBulkDelete = async () => {
    if (selectedIds.length === 0) return;
    const count = await bulkDeleteCameras(selectedIds);
    showToast(`Deleted ${count} camera node(s) successfully.`, "info");
    setSelectedIds([]);
  };

  return (
    <div className="space-y-6 pb-16">
      {/* Toast Banner */}
      {toastMessage && (
        <div
          className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-xl shadow-lg border text-xs font-semibold flex items-center gap-2.5 transition-all animate-in fade-in slide-in-from-top-2 ${
            toastMessage.type === "success"
              ? "bg-emerald-900 text-emerald-100 border-emerald-700 shadow-emerald-950/20"
              : toastMessage.type === "error"
              ? "bg-rose-900 text-rose-100 border-rose-700 shadow-rose-950/20"
              : "bg-slate-900 text-slate-100 border-slate-700 shadow-slate-950/20"
          }`}
        >
          {toastMessage.type === "success" && <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />}
          {toastMessage.type === "error" && <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />}
          {toastMessage.type === "info" && <Zap className="w-4 h-4 text-amber-400 shrink-0" />}
          <span>{toastMessage.text}</span>
        </div>
      )}

      {/* Header & Tactical Action Bar */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-5 md:p-6 shadow-xs flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#143724] text-white flex items-center justify-center shadow-xs">
              <CameraIcon className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl md:text-2xl font-black tracking-tight text-slate-900">
                  Perimeter Camera Operator
                </h1>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-[11px] font-semibold text-emerald-800">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  {onlineCount}/{totalCount} Online
                </span>
              </div>
              <p className="text-xs md:text-sm font-medium text-slate-500 mt-0.5">
                Register, configure, and calibrate border camera streams via Lat/Long coordinates and RTSP IP.
              </p>
            </div>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleTestAllStreams}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors shadow-xs"
            title="Ping all active RTSP stream endpoints"
          >
            <Activity className="w-3.5 h-3.5 text-emerald-600" />
            <span>Test All Streams</span>
          </button>

          <button
            type="button"
            onClick={() => setIsAddModalOpen(true)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold transition-all shadow-xs hover:shadow-emerald-900/10"
          >
            <Plus className="w-4 h-4 text-emerald-400" />
            <span>Add Camera Node</span>
          </button>
        </div>
      </div>

      {/* Metrics Ribbon */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Configured Nodes
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5">
              {totalCount}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              Perimeter Sector-04 Mesh
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-700">
            <Server className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-emerald-700 uppercase tracking-wider">
              Online Feeds
            </span>
            <div className="text-2xl font-black text-emerald-700 mt-0.5">
              {onlineCount}
            </div>
            <span className="text-[11px] text-emerald-600 font-medium">
              {((onlineCount / (totalCount || 1)) * 100).toFixed(0)}% Network Availability
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
            <Wifi className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-rose-700 uppercase tracking-wider">
              Alerts & Breaches
            </span>
            <div className="text-2xl font-black text-rose-700 mt-0.5">
              {alertCount}
            </div>
            <span className="text-[11px] text-rose-600 font-medium">
              Geofence Detection Active
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-700">
            <AlertTriangle className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Avg RTSP Latency
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5">
              {avgLatencyMs} <span className="text-xs font-semibold text-slate-500">ms</span>
            </div>
            <span className="text-[11px] text-emerald-600 font-medium">
              Ultra-low Edge Stream
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-700">
            <Activity className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Filter and View Toolbar */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Search input & Select Filters */}
        <div className="flex flex-wrap items-center gap-3 flex-1">
          {/* Search box */}
          <div className="relative min-w-[220px] flex-1 max-w-sm">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by name, ID, RTSP IP..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 w-full text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700 transition-all placeholder:text-slate-400"
            />
          </div>

          {/* Sector Filter */}
          <select
            value={sectorFilter}
            onChange={(e) => setSectorFilter(e.target.value)}
            className="px-3 py-1.5 text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none font-medium text-slate-700"
          >
            <option value="All">All Sectors</option>
            {SECTORS.map((sec) => (
              <option key={sec} value={sec}>
                {sec}
              </option>
            ))}
          </select>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as any)}
            className="px-3 py-1.5 text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none font-medium text-slate-700"
          >
            <option value="All">All Statuses</option>
            <option value="online">Online</option>
            <option value="alert">Alert Active</option>
            <option value="degraded">Degraded</option>
            <option value="offline">Offline</option>
          </select>

          {/* Type Filter */}
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="px-3 py-1.5 text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none font-medium text-slate-700"
          >
            <option value="All">All Camera Types</option>
            {CAMERA_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        {/* View Mode Toggle & Bulk actions */}
        <div className="flex items-center gap-2.5">
          {selectedIds.length > 0 && (
            <button
              type="button"
              onClick={handleBulkDelete}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold transition-colors shadow-xs"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete Selected ({selectedIds.length})</span>
            </button>
          )}

          {/* View mode buttons */}
          <div className="inline-flex p-1 rounded-xl bg-slate-100 border border-slate-200 text-xs font-semibold">
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`px-3 py-1 rounded-lg transition-all ${
                viewMode === "grid"
                  ? "bg-white text-slate-900 shadow-xs font-bold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Grid View
            </button>
            <button
              type="button"
              onClick={() => setViewMode("table")}
              className={`px-3 py-1 rounded-lg transition-all ${
                viewMode === "table"
                  ? "bg-white text-slate-900 shadow-xs font-bold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Table View
            </button>
            <button
              type="button"
              onClick={() => setViewMode("map")}
              className={`px-3 py-1 rounded-lg transition-all ${
                viewMode === "map"
                  ? "bg-white text-slate-900 shadow-xs font-bold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Map Split
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area based on View Mode */}
      {viewMode === "grid" && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredCameras.map((camera) => (
            <CameraCardItem
              key={camera.id}
              camera={camera}
              testResult={testingIds[camera.id]}
              onTest={() => handleTestCamera(camera)}
              onEdit={() => setEditingCamera(camera)}
              onDelete={() => setDeletingCamera(camera)}
              onPreview={() => setPreviewCamera(camera)}
              onCopy={handleCopy}
              copiedText={copiedText}
            />
          ))}

          {filteredCameras.length === 0 && (
            <div className="col-span-full bg-white rounded-2xl border border-slate-200/90 p-12 text-center">
              <CameraIcon className="w-10 h-10 text-slate-300 mx-auto mb-3" />
              <h3 className="text-base font-bold text-slate-800">No cameras match filter criteria</h3>
              <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                Adjust search query or sector filters, or register a new camera node.
              </p>
              <button
                type="button"
                onClick={() => setIsAddModalOpen(true)}
                className="mt-4 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#143724] text-white text-xs font-bold"
              >
                <Plus className="w-4 h-4 text-emerald-400" />
                Add Camera Node
              </button>
            </div>
          )}
        </div>
      )}

      {viewMode === "table" && (
        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-500 font-bold uppercase text-[10px] tracking-wider">
                  <th className="p-3.5 pl-4 w-10">
                    <input
                      type="checkbox"
                      checked={
                        filteredCameras.length > 0 &&
                        selectedIds.length === filteredCameras.length
                      }
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedIds(filteredCameras.map((c) => c.id));
                        } else {
                          setSelectedIds([]);
                        }
                      }}
                      className="rounded accent-emerald-700 cursor-pointer"
                    />
                  </th>
                  <th className="p-3.5">Node Name & ID</th>
                  <th className="p-3.5">Sector & Location</th>
                  <th className="p-3.5">Lat / Long Coordinates</th>
                  <th className="p-3.5">RTSP IP & Stream</th>
                  <th className="p-3.5">AI Model / Res</th>
                  <th className="p-3.5">Status & Ping</th>
                  <th className="p-3.5 pr-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700 font-medium">
                {filteredCameras.map((camera) => {
                  const isChecked = selectedIds.includes(camera.id);
                  const isAlert = camera.status === "alert";
                  const isDegraded = camera.status === "degraded";
                  const isOffline = camera.status === "offline";

                  return (
                    <tr
                      key={camera.id}
                      className={`hover:bg-slate-50/70 transition-colors ${
                        isChecked ? "bg-emerald-50/40" : ""
                      }`}
                    >
                      <td className="p-3.5 pl-4">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedIds((prev) => [...prev, camera.id]);
                            } else {
                              setSelectedIds((prev) =>
                                prev.filter((id) => id !== camera.id)
                              );
                            }
                          }}
                          className="rounded accent-emerald-700 cursor-pointer"
                        />
                      </td>
                      <td className="p-3.5">
                        <div className="font-bold text-slate-900">{camera.name}</div>
                        <div className="text-[11px] font-mono text-slate-400 flex items-center gap-1.5 mt-0.5">
                          <span>{camera.id}</span>
                          <span>•</span>
                          <span className="text-emerald-700 font-semibold">{camera.type}</span>
                        </div>
                      </td>
                      <td className="p-3.5">
                        <div className="font-semibold text-slate-800">{camera.sector}</div>
                        <div className="text-[11px] text-slate-500">{camera.location}</div>
                      </td>
                      <td className="p-3.5">
                        <div className="font-mono text-xs font-semibold text-slate-900 flex items-center gap-1">
                          <MapPin className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                          <span>
                            {camera.latitude.toFixed(4)}°N, {camera.longitude.toFixed(4)}°E
                          </span>
                        </div>
                      </td>
                      <td className="p-3.5">
                        <div className="font-mono text-xs text-slate-900 flex items-center gap-1.5">
                          <Wifi className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                          <span>{camera.ipAddress || "RTSP Stream"}</span>
                          {camera.port && (
                            <span className="text-slate-400">:{camera.port}</span>
                          )}
                          <button
                            type="button"
                            onClick={() =>
                              handleCopy(
                                camera.streamUrl || camera.ipAddress || "",
                                "RTSP URL"
                              )
                            }
                            className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-700"
                            title="Copy RTSP Stream"
                          >
                            <Copy className="w-3 h-3" />
                          </button>
                        </div>
                        <div className="text-[10px] font-mono text-slate-400 truncate max-w-[200px]">
                          {camera.streamUrl}
                        </div>
                      </td>
                      <td className="p-3.5">
                        <span className="inline-block px-2 py-0.5 rounded bg-slate-100 text-slate-700 font-mono text-[10px] font-semibold">
                          {camera.modelAssigned || "vehicle_detector.onnx"}
                        </span>
                        <div className="text-[10px] text-slate-500 mt-0.5">
                          {camera.resolution || "1080p"} • {camera.fps || 30} FPS
                        </div>
                      </td>
                      <td className="p-3.5">
                        <div className="flex items-center gap-2">
                          <span
                            className={`w-2 h-2 rounded-full ${
                              isAlert
                                ? "bg-rose-600 animate-pulse"
                                : isDegraded
                                ? "bg-amber-500"
                                : isOffline
                                ? "bg-slate-400"
                                : "bg-emerald-500"
                            }`}
                          />
                          <span
                            className={`font-semibold capitalize text-xs ${
                              isAlert
                                ? "text-rose-700"
                                : isDegraded
                                ? "text-amber-700"
                                : isOffline
                                ? "text-slate-500"
                                : "text-emerald-800"
                            }`}
                          >
                            {camera.status}
                          </span>
                          <span className="text-[10px] font-mono text-slate-400">
                            ({camera.healthStats?.latencyMs || 42}ms)
                          </span>
                        </div>
                      </td>
                      <td className="p-3.5 pr-4 text-right">
                        <div className="inline-flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => handleTestCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-emerald-700"
                            title="Ping RTSP Stream"
                          >
                            <Activity className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setPreviewCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900"
                            title="Live Stream Preview"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900"
                            title="Edit Settings"
                          >
                            <Settings className="w-3.5 h-3.5" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setDeletingCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-500 hover:bg-rose-50 hover:text-rose-600"
                            title="Delete Camera"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {viewMode === "map" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Side List */}
          <div className="lg:col-span-5 space-y-4 max-h-[760px] overflow-y-auto pr-1">
            {filteredCameras.map((camera) => (
              <div
                key={camera.id}
                className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs hover:border-emerald-600/60 transition-all"
              >
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h4 className="text-sm font-bold text-slate-900">{camera.name}</h4>
                    <p className="text-[11px] text-slate-500 font-medium">
                      {camera.sector} • {camera.type}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${
                      camera.status === "alert"
                        ? "bg-rose-100 text-rose-700"
                        : camera.status === "degraded"
                        ? "bg-amber-100 text-amber-800"
                        : camera.status === "offline"
                        ? "bg-slate-100 text-slate-600"
                        : "bg-emerald-100 text-emerald-800"
                    }`}
                  >
                    {camera.status}
                  </span>
                </div>

                <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] bg-slate-50 p-2.5 rounded-xl font-mono">
                  <div>
                    <span className="text-slate-400 block text-[9px] uppercase font-bold">
                      Lat / Long
                    </span>
                    <span className="font-semibold text-slate-800">
                      {camera.latitude.toFixed(3)}°, {camera.longitude.toFixed(3)}°
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[9px] uppercase font-bold">
                      RTSP IP
                    </span>
                    <span className="font-semibold text-slate-800 truncate block">
                      {camera.ipAddress || "RTSP Stream"}
                    </span>
                  </div>
                </div>

                <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                  <span className="text-[10px] font-mono text-slate-400">
                    Model: {camera.modelAssigned || "vehicle_detector.onnx"}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setEditingCamera(camera)}
                      className="px-2.5 py-1 rounded-lg border border-slate-200 hover:bg-slate-50 text-[11px] font-semibold text-slate-700"
                    >
                      Settings
                    </button>
                    <button
                      type="button"
                      onClick={() => setPreviewCamera(camera)}
                      className="px-2.5 py-1 rounded-lg bg-[#143724] text-white text-[11px] font-semibold"
                    >
                      Feed
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Interactive Google Map */}
          <div className="lg:col-span-7">
            <BorderMap height="760px" customCameras={filteredCameras} />
          </div>
        </div>
      )}

      {/* MODAL 1: ADD CAMERA NODE */}
      {isAddModalOpen && (
        <AddCameraModal
          onClose={() => setIsAddModalOpen(false)}
          onAdd={async (input) => {
            const newCam = await addCamera(input);
            showToast(`Camera node '${newCam.name}' registered successfully!`, "success");
            setIsAddModalOpen(false);
          }}
          onTestStream={testStream}
        />
      )}

      {/* MODAL 2: EDIT CAMERA SETTINGS */}
      {editingCamera && (
        <EditCameraModal
          camera={editingCamera}
          onClose={() => setEditingCamera(null)}
          onSave={async (id, updates) => {
            await updateCamera(id, updates);
            showToast(`Camera '${editingCamera.name}' settings updated!`, "success");
            setEditingCamera(null);
          }}
          onTestStream={testStream}
        />
      )}

      {/* MODAL 3: DELETE CONFIRMATION */}
      {deletingCamera && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
          <div className="bg-white rounded-2xl border border-slate-200 max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-rose-50 border border-rose-200 text-rose-600 flex items-center justify-center shrink-0">
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Delete Camera Node?
                </h3>
                <p className="text-xs text-slate-500">
                  This will decommission the node from active border surveillance.
                </p>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs space-y-1.5 font-medium">
              <div>
                <strong className="text-slate-500">Name:</strong>{" "}
                <span className="font-bold text-slate-900">{deletingCamera.name}</span>
              </div>
              <div>
                <strong className="text-slate-500">Node ID:</strong>{" "}
                <span className="font-mono text-slate-700">{deletingCamera.id}</span>
              </div>
              <div>
                <strong className="text-slate-500">Coordinates:</strong>{" "}
                <span className="font-mono text-slate-700">
                  {deletingCamera.latitude.toFixed(4)}°N, {deletingCamera.longitude.toFixed(4)}°E
                </span>
              </div>
              <div>
                <strong className="text-slate-500">RTSP IP:</strong>{" "}
                <span className="font-mono text-slate-700">
                  {deletingCamera.ipAddress || deletingCamera.streamUrl}
                </span>
              </div>
            </div>

            <p className="text-xs text-rose-600 font-medium">
              Warning: AI inference pipeline and geofence alarms linked to this camera will cease monitoring immediately.
            </p>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setDeletingCamera(null)}
                className="px-4 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteConfirm}
                className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-xs transition-colors"
              >
                Confirm Deletion
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 4: LIVE PREVIEW FEED */}
      {previewCamera && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/80 backdrop-blur-sm">
          <div className="bg-slate-950 rounded-2xl border border-slate-800 max-w-4xl w-full p-5 shadow-2xl text-white space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-3">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <span>{previewCamera.name}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
                      LIVE RTSP
                    </span>
                  </h3>
                  <p className="text-[11px] font-mono text-slate-400">
                    {previewCamera.streamUrl || previewCamera.ipAddress} • Sector-04
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setPreviewCamera(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Tactical Feed Simulation Screen */}
            <div className="relative aspect-video w-full bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
              {/* Scanline background effect */}
              <div
                className="absolute inset-0 pointer-events-none opacity-20"
                style={{
                  backgroundImage:
                    "linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))",
                  backgroundSize: "100% 3px, 6px 100%",
                }}
              />

              {/* Bounding box mock for AI detection */}
              <div className="absolute top-1/4 left-1/3 w-32 h-44 border-2 border-emerald-400 rounded-xs bg-emerald-500/10 pointer-events-none">
                <div className="bg-emerald-600 text-white text-[10px] font-mono font-bold px-1.5 py-0.5 absolute -top-5 left-0">
                  person 0.94
                </div>
              </div>

              {/* Bounding box 2 */}
              <div className="absolute top-1/2 right-1/4 w-40 h-28 border-2 border-amber-400 rounded-xs bg-amber-500/10 pointer-events-none">
                <div className="bg-amber-600 text-white text-[10px] font-mono font-bold px-1.5 py-0.5 absolute -top-5 left-0">
                  vehicle 0.88
                </div>
              </div>

              {/* Tactical Crosshair */}
              <Crosshair className="w-12 h-12 text-emerald-400/40 animate-pulse" />

              {/* Live Overlay HUD */}
              <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-xs px-2.5 py-1 rounded-md text-[10px] font-mono text-emerald-400">
                REC • 1080p @ {previewCamera.fps || 30} FPS • {previewCamera.modelAssigned || "vehicle_detector.onnx"}
              </div>

              <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-xs px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-300">
                LAT: {previewCamera.latitude.toFixed(4)}°N | LNG: {previewCamera.longitude.toFixed(4)}°E
              </div>

              <div className="absolute bottom-3 left-3 bg-black/60 backdrop-blur-xs px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-400">
                BITRATE: {previewCamera.healthStats?.bitrate || "6.4 Mbps"} | RTT:{" "}
                {previewCamera.healthStats?.latencyMs || 38}ms
              </div>
            </div>

            <div className="flex items-center justify-between text-xs text-slate-400 pt-2">
              <span>Model Pipeline: YOLOv8 Border Threat ONNX Engine</span>
              <button
                type="button"
                onClick={() => setPreviewCamera(null)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold"
              >
                Close Stream
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// -------------------------------------------------------------
// Component: Camera Card Item (Grid View)
// -------------------------------------------------------------
function CameraCardItem({
  camera,
  testResult,
  onTest,
  onEdit,
  onDelete,
  onPreview,
  onCopy,
  copiedText,
}: {
  camera: Camera;
  testResult?: StreamTestResult | "testing";
  onTest: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onPreview: () => void;
  onCopy: (text: string, label: string) => void;
  copiedText: string | null;
}) {
  const isAlert = camera.status === "alert";
  const isDegraded = camera.status === "degraded";
  const isOffline = camera.status === "offline";

  return (
    <div
      className={`bg-white rounded-2xl border transition-all shadow-xs hover:shadow-md flex flex-col justify-between overflow-hidden ${
        isAlert
          ? "border-rose-300 ring-1 ring-rose-200"
          : "border-slate-200/90 hover:border-emerald-600/50"
      }`}
    >
      <div>
        {/* Card Header */}
        <div className="p-4 pb-3 border-b border-slate-100 flex items-start justify-between gap-2 bg-slate-50/50">
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  isAlert
                    ? "bg-rose-600 animate-pulse"
                    : isDegraded
                    ? "bg-amber-500"
                    : isOffline
                    ? "bg-slate-400"
                    : "bg-emerald-500"
                }`}
              />
              <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                {camera.name}
              </h3>
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5 font-medium">
              {camera.sector}
            </p>
          </div>

          <span
            className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
              isAlert
                ? "bg-rose-100 text-rose-700"
                : isDegraded
                ? "bg-amber-100 text-amber-800"
                : isOffline
                ? "bg-slate-100 text-slate-600"
                : "bg-emerald-100 text-emerald-800"
            }`}
          >
            {camera.status}
          </span>
        </div>

        {/* Video Preview Canvas / Mock Screen */}
        <div
          onClick={onPreview}
          className="relative aspect-16/9 bg-slate-900 cursor-pointer group overflow-hidden flex items-center justify-center"
        >
          {/* Scanline pattern */}
          <div
            className="absolute inset-0 pointer-events-none opacity-25"
            style={{
              backgroundImage:
                "linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%)",
              backgroundSize: "100% 4px",
            }}
          />

          {/* Camera Type & Location overlay */}
          <div className="absolute top-2 left-2.5 z-10 flex items-center gap-1.5">
            <span className="px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-white text-[10px] font-mono font-semibold">
              {camera.type || "Optical"}
            </span>
          </div>

          <div className="absolute top-2 right-2.5 z-10">
            <span className="px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-emerald-400 text-[10px] font-mono">
              {camera.resolution || "1080p"}
            </span>
          </div>

          {/* Center hover play/inspect */}
          <div className="w-10 h-10 rounded-full bg-white/10 group-hover:bg-emerald-500/90 text-white flex items-center justify-center backdrop-blur-xs transition-all transform group-hover:scale-110">
            <Eye className="w-5 h-5" />
          </div>

          {/* Bottom telemetry overlay */}
          <div className="absolute bottom-2 left-2.5 right-2.5 flex items-center justify-between text-[10px] font-mono text-slate-300 bg-black/50 backdrop-blur-xs px-2 py-1 rounded">
            <span>{camera.id}</span>
            <span>{camera.healthStats?.latencyMs || 42}ms</span>
          </div>
        </div>

        {/* Tactical Parameters Body */}
        <div className="p-4 space-y-3">
          {/* Geospatial Lat/Long Badge */}
          <div className="flex items-center justify-between p-2 rounded-xl bg-slate-50 border border-slate-100 text-xs">
            <div className="flex items-center gap-1.5 font-mono text-slate-700">
              <MapPin className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
              <span className="font-semibold">
                {camera.latitude.toFixed(4)}°N, {camera.longitude.toFixed(4)}°E
              </span>
            </div>
            <button
              type="button"
              onClick={() =>
                onCopy(
                  `${camera.latitude}, ${camera.longitude}`,
                  "Coordinates"
                )
              }
              className="text-[11px] font-medium text-emerald-700 hover:text-emerald-800 flex items-center gap-1"
            >
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </button>
          </div>

          {/* RTSP IP & Stream Address */}
          <div className="flex items-center justify-between p-2 rounded-xl bg-slate-50 border border-slate-100 text-xs">
            <div className="flex items-center gap-1.5 font-mono text-slate-700 truncate max-w-[210px]">
              <Wifi className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              <span className="truncate">
                {camera.ipAddress || camera.streamUrl || "RTSP Unconfigured"}
              </span>
            </div>
            <button
              type="button"
              onClick={() =>
                onCopy(camera.streamUrl || camera.ipAddress || "", "RTSP Stream")
              }
              className="text-[11px] font-medium text-emerald-700 hover:text-emerald-800 flex items-center gap-1"
            >
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </button>
          </div>

          {/* AI Model & Thresholds Specs */}
          <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-600">
            <div>
              <span className="text-slate-400 block text-[10px] font-bold uppercase">
                AI Model
              </span>
              <span className="font-semibold text-slate-800 truncate block font-mono">
                {camera.modelAssigned?.split(" ")[0] || "vehicle_detector.onnx"}
              </span>
            </div>
            <div>
              <span className="text-slate-400 block text-[10px] font-bold uppercase">
                Confidence Cutoff
              </span>
              <span className="font-semibold text-slate-800">
                {Math.round((camera.confThreshold ?? 0.75) * 100)}% threshold
              </span>
            </div>
          </div>

          {/* Test Stream Status Pill if tested */}
          {testResult && (
            <div
              className={`p-2 rounded-xl text-xs font-mono flex items-center justify-between ${
                testResult === "testing"
                  ? "bg-slate-100 text-slate-700"
                  : testResult.reachable
                  ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                  : "bg-rose-50 text-rose-800 border border-rose-200"
              }`}
            >
              {testResult === "testing" ? (
                <div className="flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Pinging RTSP port...</span>
                </div>
              ) : (
                <>
                  <span className="truncate">{testResult.message}</span>
                  <span className="font-bold shrink-0">{testResult.latencyMs}ms</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Card Action Footer */}
      <div className="p-3 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={onTest}
          disabled={testResult === "testing"}
          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 hover:bg-white text-slate-700 text-xs font-semibold transition-colors shadow-2xs"
          title="Test RTSP connection"
        >
          <Activity className="w-3.5 h-3.5 text-emerald-700" />
          <span>Ping</span>
        </button>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onEdit}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold transition-colors shadow-2xs"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Settings</span>
          </button>

          <button
            type="button"
            onClick={onDelete}
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
            title="Delete camera node"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Component: Add Camera Modal
// -------------------------------------------------------------
function AddCameraModal({
  onClose,
  onAdd,
  onTestStream,
}: {
  onClose: () => void;
  onAdd: (input: CreateCameraInput) => Promise<void>;
  onTestStream: (p: { streamUrl?: string; ipAddress?: string; port?: number }) => Promise<StreamTestResult>;
}) {
  const [name, setName] = useState("");
  const [sector, setSector] = useState("Sector-04 (BOP Alpha)");
  const [location, setLocation] = useState("");
  const [type, setType] = useState<CameraType>("Optical 4K");

  // RTSP IP / Stream
  const [ipAddress, setIpAddress] = useState("10.20.72.110");
  const [port, setPort] = useState(554);
  const [streamUrl, setStreamUrl] = useState("rtsp://admin:pass@10.20.72.110:554/live/ch0");

  // Lat / Long
  const [latitude, setLatitude] = useState(32.7160);
  const [longitude, setLongitude] = useState(74.6640);

  // Model & Params
  const [modelAssigned, setModelAssigned] = useState("vehicle_detector.onnx (Vehicle Detector)");
  const [resolution, setResolution] = useState("1080p FHD (1920x1080)");
  const [fps, setFps] = useState(30);
  const [confThreshold, setConfThreshold] = useState(0.75);
  const [iouThreshold, setIouThreshold] = useState(0.45);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [testResult, setTestResult] = useState<StreamTestResult | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Auto-update streamUrl when IP or Port changes
  const handleIpChange = (newIp: string) => {
    setIpAddress(newIp);
    setStreamUrl(`rtsp://${newIp.trim()}:${port}/live/ch0`);
  };

  const handlePortChange = (newPort: number) => {
    setPort(newPort);
    setStreamUrl(`rtsp://${ipAddress.trim()}:${newPort}/live/ch0`);
  };

  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await onTestStream({ streamUrl, ipAddress, port });
      setTestResult(res);
    } finally {
      setIsTesting(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setErrorMessage("Please enter a valid Camera Name.");
      return;
    }
    if (!location.trim()) {
      setErrorMessage("Please enter physical outpost location.");
      return;
    }
    if (isNaN(latitude) || latitude < -90 || latitude > 90) {
      setErrorMessage("Latitude must be between -90 and 90 degrees.");
      return;
    }
    if (isNaN(longitude) || longitude < -180 || longitude > 180) {
      setErrorMessage("Longitude must be between -180 and 180 degrees.");
      return;
    }

    setErrorMessage(null);
    setIsSubmitting(true);
    try {
      await onAdd({
        name: name.trim(),
        sector,
        location: location.trim(),
        type,
        ipAddress: ipAddress.trim(),
        port,
        streamUrl: streamUrl.trim(),
        latitude: Number(latitude),
        longitude: Number(longitude),
        modelAssigned,
        resolution,
        fps,
        confThreshold,
        iouThreshold,
        status: "online",
      });
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to add camera node.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs overflow-y-auto">
      <div className="bg-white rounded-2xl border border-slate-200 max-w-2xl w-full p-6 shadow-2xl space-y-5 my-8">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-[#143724] text-white flex items-center justify-center">
              <Plus className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Add Camera Node
              </h3>
              <p className="text-xs text-slate-500">
                Configure stream pipeline via RTSP IP and geospatial coordinates.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {errorMessage && (
          <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Section 1: General Info */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              1. General Details
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Camera Name *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Forward Post Optical 05"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Perimeter Sector *
                </label>
                <select
                  value={sector}
                  onChange={(e) => setSector(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  {SECTORS.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Camera Sensor Type
                </label>
                <select
                  value={type}
                  onChange={(e) => setType(e.target.value as any)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  {CAMERA_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Physical Outpost Location *
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Trench Observation Bunker 18"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700"
                />
              </div>
            </div>
          </div>

          {/* Section 2: RTSP IP & Network Pipeline */}
          <div className="space-y-3 pt-2 border-t border-slate-100">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <Wifi className="w-3.5 h-3.5 text-emerald-700" />
                <span>2. Network & RTSP IP Settings</span>
              </h4>
              <button
                type="button"
                onClick={handleTest}
                disabled={isTesting}
                className="text-xs font-bold text-emerald-700 hover:text-emerald-800 flex items-center gap-1"
              >
                {isTesting ? (
                  <RefreshCw className="w-3 h-3 animate-spin" />
                ) : (
                  <Zap className="w-3 h-3" />
                )}
                <span>Verify RTSP Handshake</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="sm:col-span-2">
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Camera IP Address (RTSP IP)
                </label>
                <input
                  type="text"
                  placeholder="e.g. 10.20.72.110"
                  value={ipAddress}
                  onChange={(e) => handleIpChange(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  RTSP Port
                </label>
                <input
                  type="number"
                  min={1}
                  max={65535}
                  value={port}
                  onChange={(e) => handlePortChange(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Full RTSP Stream URI
              </label>
              <input
                type="text"
                placeholder="rtsp://admin:pass@10.20.72.110:554/live/ch0"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
              />
            </div>

            {testResult && (
              <div
                className={`p-2.5 rounded-xl text-xs font-mono flex items-center justify-between ${
                  testResult.reachable
                    ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                    : "bg-rose-50 text-rose-800 border border-rose-200"
                }`}
              >
                <span>{testResult.message}</span>
                <span className="font-bold">{testResult.latencyMs}ms</span>
              </div>
            )}
          </div>

          {/* Section 3: Geospatial Lat / Long Coordinates */}
          <div className="space-y-3 pt-2 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-emerald-700" />
              <span>3. Geospatial Placement (Lat / Long)</span>
            </h4>

            {/* Presets row */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] text-slate-500 font-semibold mr-1">
                Presets:
              </span>
              {COORDINATE_PRESETS.map((p) => (
                <button
                  key={p.name}
                  type="button"
                  onClick={() => {
                    setLatitude(p.lat);
                    setLongitude(p.lng);
                  }}
                  className="px-2 py-1 rounded-lg bg-slate-100 hover:bg-slate-200 text-[11px] font-mono text-slate-700 transition-colors"
                >
                  {p.name}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Latitude (°N) *
                </label>
                <input
                  type="number"
                  step="0.0001"
                  required
                  value={latitude}
                  onChange={(e) => setLatitude(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Longitude (°E) *
                </label>
                <input
                  type="number"
                  step="0.0001"
                  required
                  value={longitude}
                  onChange={(e) => setLongitude(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Section 4: AI Model & Stream Quality */}
          <div className="space-y-3 pt-2 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-emerald-700" />
              <span>4. AI Detection & Quality Settings</span>
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Assigned ONNX Model
                </label>
                <select
                  value={modelAssigned}
                  onChange={(e) => setModelAssigned(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  <option value="vehicle_detector.onnx (Vehicle Detector)">vehicle_detector.onnx (Vehicle Detector)</option>
                  <option value="yolov8n-border.onnx">yolov8n-border.onnx</option>
                  <option value="anpr_v2.onnx">anpr_v2.onnx</option>
                  <option value="thermal_flir.onnx">thermal_flir.onnx</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Resolution
                </label>
                <select
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  <option value="4K UHD (3840x2160)">4K UHD (3840x2160)</option>
                  <option value="1080p FHD (1920x1080)">1080p FHD (1920x1080)</option>
                  <option value="720p HD (1280x720)">720p HD (1280x720)</option>
                  <option value="1080p Thermal">1080p Thermal</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  FPS (Frames/Sec)
                </label>
                <select
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  <option value={15}>15 FPS (Power Saver)</option>
                  <option value={25}>25 FPS</option>
                  <option value={30}>30 FPS (Standard)</option>
                  <option value={60}>60 FPS (High Speed)</option>
                </select>
              </div>
            </div>

            {/* Threshold Slider */}
            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700">
                  AI Confidence Threshold
                </span>
                <span className="font-mono font-bold text-emerald-800">
                  {Math.round(confThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min={0.3}
                max={0.95}
                step={0.05}
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-700 cursor-pointer"
              />
            </div>
          </div>

          {/* Form Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-5 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold shadow-xs transition-colors flex items-center gap-2"
            >
              {isSubmitting && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
              <span>Register Node</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Component: Edit Camera Settings Modal
// -------------------------------------------------------------
function EditCameraModal({
  camera,
  onClose,
  onSave,
  onTestStream,
}: {
  camera: Camera;
  onClose: () => void;
  onSave: (id: string, updates: UpdateCameraInput) => Promise<void>;
  onTestStream: (p: { streamUrl?: string; ipAddress?: string; port?: number }) => Promise<StreamTestResult>;
}) {
  const [name, setName] = useState(camera.name);
  const [sector, setSector] = useState(camera.sector);
  const [location, setLocation] = useState(camera.location);
  const [type, setType] = useState<CameraType>((camera.type as any) || "Optical 4K");
  const [status, setStatus] = useState<CameraStatus>(camera.status);

  // RTSP IP / Stream
  const [ipAddress, setIpAddress] = useState(camera.ipAddress || "");
  const [port, setPort] = useState(camera.port || 554);
  const [streamUrl, setStreamUrl] = useState(camera.streamUrl || "");

  // Lat / Long
  const [latitude, setLatitude] = useState(camera.latitude);
  const [longitude, setLongitude] = useState(camera.longitude);

  // Model & Params
  const [modelAssigned, setModelAssigned] = useState(camera.modelAssigned || "vehicle_detector.onnx (Vehicle Detector)");
  const [resolution, setResolution] = useState(camera.resolution || "1080p FHD (1920x1080)");
  const [fps, setFps] = useState(camera.fps || 30);
  const [confThreshold, setConfThreshold] = useState(camera.confThreshold ?? 0.75);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [testResult, setTestResult] = useState<StreamTestResult | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await onTestStream({ streamUrl, ipAddress, port });
      setTestResult(res);
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setErrorMessage("Please enter a valid Camera Name.");
      return;
    }
    if (isNaN(latitude) || latitude < -90 || latitude > 90) {
      setErrorMessage("Latitude must be between -90 and 90.");
      return;
    }
    if (isNaN(longitude) || longitude < -180 || longitude > 180) {
      setErrorMessage("Longitude must be between -180 and 180.");
      return;
    }

    setErrorMessage(null);
    setIsSubmitting(true);
    try {
      await onSave(camera.id, {
        name: name.trim(),
        sector,
        location: location.trim(),
        type,
        status,
        ipAddress: ipAddress.trim() || undefined,
        port,
        streamUrl: streamUrl.trim() || undefined,
        latitude: Number(latitude),
        longitude: Number(longitude),
        modelAssigned,
        resolution,
        fps,
        confThreshold,
      });
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to update camera settings.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs overflow-y-auto">
      <div className="bg-white rounded-2xl border border-slate-200 max-w-2xl w-full p-6 shadow-2xl space-y-5 my-8">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-slate-100 text-slate-800 flex items-center justify-center">
              <Settings className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Edit Camera Settings
              </h3>
              <p className="text-xs text-slate-500 font-mono">
                Node ID: {camera.id}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {errorMessage && (
          <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-4">
          {/* General Details */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Camera Name
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
              />
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Operational Status
              </label>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value as any)}
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
              >
                <option value="online">Online</option>
                <option value="alert">Alert Active</option>
                <option value="degraded">Degraded</option>
                <option value="offline">Offline</option>
              </select>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Sector
              </label>
              <select
                value={sector}
                onChange={(e) => setSector(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
              >
                {SECTORS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Physical Location
              </label>
              <input
                type="text"
                required
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
              />
            </div>
          </div>

          {/* RTSP IP & Stream Settings */}
          <div className="space-y-3 pt-2 border-t border-slate-100">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <Wifi className="w-3.5 h-3.5 text-emerald-700" />
                <span>RTSP IP & Stream Settings</span>
              </h4>
              <button
                type="button"
                onClick={handleTest}
                disabled={isTesting}
                className="text-xs font-bold text-emerald-700 hover:text-emerald-800 flex items-center gap-1"
              >
                {isTesting ? (
                  <RefreshCw className="w-3 h-3 animate-spin" />
                ) : (
                  <Zap className="w-3 h-3" />
                )}
                <span>Test Connection</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="sm:col-span-2">
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  RTSP IP Address
                </label>
                <input
                  type="text"
                  value={ipAddress}
                  onChange={(e) => setIpAddress(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Port
                </label>
                <input
                  type="number"
                  value={port}
                  onChange={(e) => setPort(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Full RTSP URL
              </label>
              <input
                type="text"
                value={streamUrl}
                onChange={(e) => setStreamUrl(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
              />
            </div>

            {testResult && (
              <div
                className={`p-2.5 rounded-xl text-xs font-mono flex items-center justify-between ${
                  testResult.reachable
                    ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                    : "bg-rose-50 text-rose-800 border border-rose-200"
                }`}
              >
                <span>{testResult.message}</span>
                <span className="font-bold">{testResult.latencyMs}ms</span>
              </div>
            )}
          </div>

          {/* Lat / Long Coordinates */}
          <div className="space-y-3 pt-2 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-emerald-700" />
              <span>Geospatial Coordinates (Lat / Long)</span>
            </h4>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Latitude (°N)
                </label>
                <input
                  type="number"
                  step="0.0001"
                  required
                  value={latitude}
                  onChange={(e) => setLatitude(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Longitude (°E)
                </label>
                <input
                  type="number"
                  step="0.0001"
                  required
                  value={longitude}
                  onChange={(e) => setLongitude(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* AI Settings */}
          <div className="space-y-3 pt-2 border-t border-slate-100">
            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-emerald-700" />
              <span>AI Detection & Stream Quality</span>
            </h4>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  AI Model
                </label>
                <select
                  value={modelAssigned}
                  onChange={(e) => setModelAssigned(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  <option value="vehicle_detector.onnx (Vehicle Detector)">vehicle_detector.onnx (Vehicle Detector)</option>
                  <option value="yolov8n-border.onnx">yolov8n-border.onnx</option>
                  <option value="anpr_v2.onnx">anpr_v2.onnx</option>
                  <option value="thermal_flir.onnx">thermal_flir.onnx</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Resolution
                </label>
                <select
                  value={resolution}
                  onChange={(e) => setResolution(e.target.value)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  <option value="4K UHD (3840x2160)">4K UHD (3840x2160)</option>
                  <option value="1080p FHD (1920x1080)">1080p FHD (1920x1080)</option>
                  <option value="720p HD (1280x720)">720p HD (1280x720)</option>
                  <option value="1080p Thermal">1080p Thermal</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  FPS
                </label>
                <select
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700"
                >
                  <option value={15}>15 FPS</option>
                  <option value={25}>25 FPS</option>
                  <option value={30}>30 FPS</option>
                  <option value={60}>60 FPS</option>
                </select>
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-semibold text-slate-700">
                  Confidence Threshold Cutoff
                </span>
                <span className="font-mono font-bold text-emerald-800">
                  {Math.round(confThreshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min={0.3}
                max={0.95}
                step={0.05}
                value={confThreshold}
                onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
                className="w-full accent-emerald-700 cursor-pointer"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-5 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold shadow-xs transition-colors flex items-center gap-2"
            >
              {isSubmitting && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
              <span>Save Changes</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
