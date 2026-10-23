"use client";

import { useState, useMemo } from "react";
import Link from "next/link";
import {
  Camera as CameraIcon,
  Plus,
  Trash2,
  Settings,
  Activity,
  MapPin,
  Wifi,
  Search,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Copy,
  Zap,
  Server,
  Eye,
  X,
} from "lucide-react";
import {
  Camera,
  CameraStatus,
  CameraType,
  UpdateCameraInput,
  StreamTestResult,
} from "@/types/camera";
import dynamic from "next/dynamic";
import { useCameras } from "@/lib/camerasStore";
import AddCameraModal from "@/components/cameras/AddCameraModal";
import { buildChannelRtspUrl, updateRtspUrlChannel } from "@/lib/borderCameras";

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

const CAMERA_TYPES: CameraType[] = [
  "Optical 4K",
  "Panoramic",
  "PTZ 360",
  "Thermal FLIR",
  "Night Vision / IR",
  "ANPR Dedicated",
];

export default function CamerasPage() {
  const {
    cameras,
    totalCount,
    onlineCount,
    alertCount,
    updateCamera,
    deleteCamera,
    bulkDeleteCameras,
    testStream,
  } = useCameras();

  // Filter & Search states
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<CameraStatus | "All">("All");
  const [typeFilter, setTypeFilter] = useState("All");
  const [viewMode, setViewMode] = useState<"grid" | "table" | "map">("grid");

  // Selection for bulk actions
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  // Modals state
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | null>(null);
  const [deletingCamera, setDeletingCamera] = useState<Camera | null>(null);

  // Copied indicator
  const [, setCopiedText] = useState<string | null>(null);

  // Toast message
  const [toastMessage, setToastMessage] = useState<{
    text: string;
    type: "success" | "info" | "error";
  } | null>(null);

  // Active testing camera IDs
  const [testingIds, setTestingIds] = useState<Record<string, StreamTestResult | "testing">>({});

  const showToast = (text: string, type: "success" | "info" | "error" = "success") => {
    setToastMessage({ text, type });
    setTimeout(() => setToastMessage(null), 3500);
  };

  const handleCopy = (text: string, label: string) => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopiedText(text);
      showToast(`Copied ${label} to clipboard!`, "info");
      setTimeout(() => setCopiedText(null), 2000);
    }
  };

  // Filtered cameras list
  const filteredCameras = useMemo(() => {
    return cameras.filter((cam) => {
      if (statusFilter !== "All" && cam.status !== statusFilter) return false;
      if (typeFilter !== "All" && cam.type !== typeFilter) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchName = cam.name.toLowerCase().includes(q);
        const matchCh = String(cam.channel ?? "").includes(q);
        const matchUrl = cam.streamUrl?.toLowerCase().includes(q) || false;
        if (!matchName && !matchCh && !matchUrl) {
          return false;
        }
      }
      return true;
    });
  }, [cameras, statusFilter, typeFilter, searchQuery]);

  // Handle single camera test
  const handleTestCamera = async (cam: Camera) => {
    setTestingIds((prev) => ({ ...prev, [cam.id]: "testing" }));
    const res = await testStream({
      streamUrl: cam.streamUrl,
    });
    setTestingIds((prev) => ({ ...prev, [cam.id]: res }));
    if (res.reachable) {
      showToast(`${cam.name}: Stream connection verified`, "success");
    } else {
      showToast(`${cam.name}: ${res.message}`, "error");
    }
  };

  // Handle test all streams
  const handleTestAllStreams = async () => {
    showToast("Pinging RTSP streams on active cameras...", "info");
    for (const cam of cameras) {
      setTestingIds((prev) => ({ ...prev, [cam.id]: "testing" }));
      testStream({
        streamUrl: cam.streamUrl,
      }).then((res: StreamTestResult) => {
        setTestingIds((prev) => ({ ...prev, [cam.id]: res }));
      });
    }
  };

  // Handle delete camera
  const handleDeleteConfirm = async () => {
    if (!deletingCamera) return;
    const success = await deleteCamera(deletingCamera.id);
    if (success) {
      showToast(`Camera '${deletingCamera.name}' removed.`, "info");
      setSelectedIds((prev) => prev.filter((id) => id !== deletingCamera.id));
    }
    setDeletingCamera(null);
  };

  // Bulk delete
  const handleBulkDelete = async () => {
    if (selectedIds.length === 0) return;
    const count = await bulkDeleteCameras(selectedIds);
    showToast(`Deleted ${count} camera(s) successfully.`, "info");
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
                  Perimeter Camera Registry
                </h1>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-[11px] font-semibold text-emerald-800">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  {onlineCount}/{totalCount} Online
                </span>
              </div>
              <p className="text-xs md:text-sm font-medium text-slate-500 mt-0.5">
                Channel-based surveillance nodes. Configured with name, channel, type, status, and GPS coordinates.
              </p>
            </div>
          </div>
        </div>

        {/* Global Action Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleTestAllStreams}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-colors shadow-xs cursor-pointer"
            title="Ping all RTSP streams"
          >
            <Activity className="w-3.5 h-3.5 text-emerald-600" />
            <span>Test All Feeds</span>
          </button>

          <button
            type="button"
            onClick={() => setIsAddModalOpen(true)}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold transition-all shadow-xs cursor-pointer"
          >
            <Plus className="w-4 h-4 text-emerald-400" />
            <span>Add Camera</span>
          </button>
        </div>
      </div>

      {/* Metrics Ribbon */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              Total Cameras
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5">
              {totalCount}
            </div>
            <span className="text-[11px] text-slate-500 font-medium">
              Registered Channels
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
              Operational Streams
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
            <Wifi className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-rose-700 uppercase tracking-wider">
              Active Alerts
            </span>
            <div className="text-2xl font-black text-rose-700 mt-0.5">
              {alertCount}
            </div>
            <span className="text-[11px] text-rose-600 font-medium">
              Flagged Channels
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-rose-50 border border-rose-200 flex items-center justify-center text-rose-700">
            <AlertTriangle className="w-5 h-5" />
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs flex items-center justify-between">
          <div>
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
              GPS Calibrated
            </span>
            <div className="text-2xl font-black text-slate-900 mt-0.5">
              {cameras.filter((c) => c.latitude != null && c.longitude != null).length}
            </div>
            <span className="text-[11px] text-emerald-600 font-medium">
              GIS Positioned
            </span>
          </div>
          <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-700">
            <MapPin className="w-5 h-5" />
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
              placeholder="Search by name or channel number..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 w-full text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-700/20 focus:border-emerald-700 transition-all placeholder:text-slate-400"
            />
          </div>

          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as any)}
            className="px-3 py-1.5 text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none font-medium text-slate-700 cursor-pointer"
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
            className="px-3 py-1.5 text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none font-medium text-slate-700 cursor-pointer"
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
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-700 text-xs font-semibold hover:bg-rose-100 transition-colors cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Delete ({selectedIds.length})</span>
            </button>
          )}

          <div className="flex items-center p-1 rounded-xl bg-slate-100 border border-slate-200 text-xs font-semibold text-slate-600">
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`px-3 py-1 rounded-lg transition-all cursor-pointer ${
                viewMode === "grid"
                  ? "bg-white text-slate-900 shadow-2xs font-bold"
                  : "hover:text-slate-900"
              }`}
            >
              Grid
            </button>
            <button
              type="button"
              onClick={() => setViewMode("table")}
              className={`px-3 py-1 rounded-lg transition-all cursor-pointer ${
                viewMode === "table"
                  ? "bg-white text-slate-900 shadow-2xs font-bold"
                  : "hover:text-slate-900"
              }`}
            >
              Table
            </button>
            <button
              type="button"
              onClick={() => setViewMode("map")}
              className={`px-3 py-1 rounded-lg transition-all cursor-pointer ${
                viewMode === "map"
                  ? "bg-white text-slate-900 shadow-2xs font-bold"
                  : "hover:text-slate-900"
              }`}
            >
              GIS Map
            </button>
          </div>
        </div>
      </div>

      {/* Grid View */}
      {viewMode === "grid" && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {filteredCameras.map((camera) => (
            <CameraCardItem
              key={camera.id}
              camera={camera}
              testResult={testingIds[camera.id]}
              onTest={() => handleTestCamera(camera)}
              onEdit={() => setEditingCamera(camera)}
              onDelete={() => setDeletingCamera(camera)}
              onCopy={handleCopy}
            />
          ))}

          {filteredCameras.length === 0 && (
            <div className="col-span-full bg-white rounded-2xl border border-slate-200/90 p-12 text-center">
              <CameraIcon className="w-10 h-10 text-slate-300 mx-auto mb-3" />
              <h3 className="text-base font-bold text-slate-800">No cameras match filter criteria</h3>
              <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
                Adjust search query or register a new camera channel.
              </p>
              <button
                type="button"
                onClick={() => setIsAddModalOpen(true)}
                className="mt-4 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-[#143724] text-white text-xs font-bold cursor-pointer"
              >
                <Plus className="w-4 h-4 text-emerald-400" />
                Add Camera
              </button>
            </div>
          )}
        </div>
      )}

      {/* Table View */}
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
                  <th className="p-3.5">Channel</th>
                  <th className="p-3.5">Camera Name</th>
                  <th className="p-3.5">Sensor Type</th>
                  <th className="p-3.5">Status</th>
                  <th className="p-3.5">GPS Coordinates</th>
                  <th className="p-3.5">RTSP Link</th>
                  <th className="p-3.5 pr-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-slate-700 font-medium">
                {filteredCameras.map((camera) => {
                  const isChecked = selectedIds.includes(camera.id);
                  const ch = camera.channel ?? "—";
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
                        <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-800">
                          CH {String(ch).padStart(2, "0")}
                        </span>
                      </td>
                      <td className="p-3.5">
                        <div className="font-bold text-slate-900">{camera.name}</div>
                      </td>
                      <td className="p-3.5">
                        <span className="text-slate-600 font-medium">{camera.type || "Optical 4K"}</span>
                      </td>
                      <td className="p-3.5">
                        <span
                          className={`inline-block px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider ${
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
                      </td>
                      <td className="p-3.5">
                        {camera.latitude != null && camera.longitude != null ? (
                          <div className="font-mono text-xs font-semibold text-slate-800 flex items-center gap-1">
                            <MapPin className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                            <span>
                              {camera.latitude.toFixed(4)}°N, {camera.longitude.toFixed(4)}°E
                            </span>
                          </div>
                        ) : (
                          <span className="text-slate-400 font-mono text-xs">—</span>
                        )}
                      </td>
                      <td className="p-3.5">
                        <div className="flex items-center gap-1.5 font-mono text-[11px] text-slate-600 max-w-[220px]">
                          <span className="truncate">{camera.streamUrl}</span>
                          <button
                            type="button"
                            onClick={() => handleCopy(camera.streamUrl || "", "RTSP Link")}
                            className="p-1 hover:bg-slate-100 rounded text-slate-400 hover:text-slate-700 shrink-0 cursor-pointer"
                            title="Copy RTSP Stream"
                          >
                            <Copy className="w-3 h-3" />
                          </button>
                        </div>
                      </td>
                      <td className="p-3.5 pr-4 text-right">
                        <div className="inline-flex items-center gap-1.5">
                          <Link
                            href={`/live?channel=${ch}`}
                            className="p-1.5 rounded-lg text-slate-500 hover:text-emerald-700 hover:bg-emerald-50 transition-colors"
                            title="View Live Stream"
                          >
                            <Eye className="w-4 h-4" />
                          </Link>
                          <button
                            type="button"
                            onClick={() => handleTestCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors cursor-pointer"
                            title="Ping RTSP Feed"
                          >
                            <Activity className="w-4 h-4 text-emerald-700" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-800 hover:bg-slate-100 transition-colors cursor-pointer"
                            title="Edit Camera"
                          >
                            <Settings className="w-4 h-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => setDeletingCamera(camera)}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
                            title="Delete Camera"
                          >
                            <Trash2 className="w-4 h-4" />
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

      {/* Map View */}
      {viewMode === "map" && (
        <div className="bg-white rounded-2xl border border-slate-200/90 p-4 shadow-xs">
          <BorderMap height="720px" customCameras={filteredCameras} />
        </div>
      )}

      {/* ADD CAMERA MODAL */}
      <AddCameraModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAdded={() => {
          showToast("Camera node registered successfully!", "success");
          setIsAddModalOpen(false);
        }}
      />

      {/* EDIT CAMERA MODAL */}
      {editingCamera && (
        <EditCameraModal
          camera={editingCamera}
          onClose={() => setEditingCamera(null)}
          onSave={async (id, updates) => {
            await updateCamera(id, updates);
            showToast(`Camera '${editingCamera.name}' updated!`, "success");
            setEditingCamera(null);
          }}
          onTestStream={testStream}
        />
      )}

      {/* DELETE CONFIRMATION MODAL */}
      {deletingCamera && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
          <div className="bg-white rounded-2xl border border-slate-200 max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-rose-100 text-rose-600 flex items-center justify-center shrink-0">
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-slate-900">
                  Delete Camera?
                </h3>
                <p className="text-xs text-slate-500">
                  Remove this camera from surveillance grid.
                </p>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs space-y-1.5 font-medium">
              <div>
                <strong className="text-slate-500">Name:</strong>{" "}
                <span className="font-bold text-slate-900">{deletingCamera.name}</span>
              </div>
              <div>
                <strong className="text-slate-500">Channel:</strong>{" "}
                <span className="font-mono text-slate-700">CH {deletingCamera.channel ?? "—"}</span>
              </div>
              <div>
                <strong className="text-slate-500">RTSP Link:</strong>{" "}
                <span className="font-mono text-slate-700 truncate block">
                  {deletingCamera.streamUrl}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setDeletingCamera(null)}
                className="px-4 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteConfirm}
                className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-xs transition-colors cursor-pointer"
              >
                Confirm Deletion
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// -------------------------------------------------------------
// Component: Camera Card Item
// -------------------------------------------------------------
function CameraCardItem({
  camera,
  testResult,
  onTest,
  onEdit,
  onDelete,
  onCopy,
}: {
  camera: Camera;
  testResult?: StreamTestResult | "testing";
  onTest: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onCopy: (text: string, label: string) => void;
}) {
  const isAlert = camera.status === "alert";
  const isDegraded = camera.status === "degraded";
  const isOffline = camera.status === "offline";
  const ch = camera.channel ?? 1;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/90 shadow-xs overflow-hidden flex flex-col justify-between hover:border-slate-300 transition-all">
      {/* Top Details */}
      <div className="p-4 space-y-3">
        {/* Header: Name + Channel + Status */}
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[11px] font-bold px-2 py-0.5 rounded bg-slate-100 text-slate-800">
                CH {String(ch).padStart(2, "0")}
              </span>
              <h3 className="text-sm font-bold text-slate-900 tracking-tight">
                {camera.name}
              </h3>
            </div>
            <p className="text-[11px] text-slate-500 mt-1 font-medium">
              {camera.type || "Optical 4K"}
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

        {/* GPS Coordinates */}
        <div className="flex items-center justify-between p-2 rounded-xl bg-slate-50 border border-slate-100 text-xs">
          <div className="flex items-center gap-1.5 font-mono text-slate-700">
            <MapPin className="w-3.5 h-3.5 text-emerald-700 shrink-0" />
            <span className="font-semibold text-[11px]">
              {camera.latitude != null && camera.longitude != null
                ? `${camera.latitude.toFixed(4)}°N, ${camera.longitude.toFixed(4)}°E`
                : "No GPS Coordinates"}
            </span>
          </div>
          {camera.latitude != null && camera.longitude != null && (
            <button
              type="button"
              onClick={() =>
                onCopy(`${camera.latitude}, ${camera.longitude}`, "Coordinates")
              }
              className="text-[11px] font-medium text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
            >
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </button>
          )}
        </div>

        {/* RTSP Stream Address */}
        <div className="flex items-center justify-between p-2 rounded-xl bg-slate-50 border border-slate-100 text-xs">
          <div className="flex items-center gap-1.5 font-mono text-slate-700 truncate max-w-[210px]">
            <Wifi className="w-3.5 h-3.5 text-slate-400 shrink-0" />
            <span className="truncate text-[11px]">
              {camera.streamUrl || "RTSP Unconfigured"}
            </span>
          </div>
          <button
            type="button"
            onClick={() => onCopy(camera.streamUrl || "", "RTSP Stream")}
            className="text-[11px] font-medium text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
          >
            <Copy className="w-3 h-3" />
            <span>Copy</span>
          </button>
        </div>

        {/* Test Result Status if tested */}
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

      {/* Card Action Footer */}
      <div className="p-3 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between gap-2">
        <Link
          href={`/live?channel=${ch}`}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-800 text-white text-xs font-bold transition-colors shadow-2xs"
          title="Open Live Feed"
        >
          <Eye className="w-3.5 h-3.5" />
          <span>Live Feed</span>
        </Link>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={onTest}
            disabled={testResult === "testing"}
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 hover:bg-white text-slate-700 text-xs font-semibold transition-colors shadow-2xs cursor-pointer"
            title="Ping RTSP connection"
          >
            <Activity className="w-3.5 h-3.5 text-emerald-700" />
            <span>Ping</span>
          </button>

          <button
            type="button"
            onClick={onEdit}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-bold transition-colors shadow-2xs cursor-pointer"
          >
            <Settings className="w-3.5 h-3.5" />
            <span>Settings</span>
          </button>

          <button
            type="button"
            onClick={onDelete}
            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors cursor-pointer"
            title="Delete camera"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
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
  const [channel, setChannel] = useState(String(camera.channel ?? 1));
  const [type, setType] = useState<CameraType>((camera.type as any) || "Optical 4K");
  const [status, setStatus] = useState<CameraStatus>(camera.status || "online");
  const [latitude, setLatitude] = useState(camera.latitude != null ? String(camera.latitude) : "");
  const [longitude, setLongitude] = useState(camera.longitude != null ? String(camera.longitude) : "");
  const [streamUrl, setStreamUrl] = useState(
    camera.streamUrl || buildChannelRtspUrl(camera.channel ?? 1)
  );

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [testResult, setTestResult] = useState<StreamTestResult | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleChannelChange = (val: string) => {
    setChannel(val);
    const chNum = parseInt(val.trim(), 10);
    if (!isNaN(chNum) && chNum >= 1) {
      setStreamUrl(updateRtspUrlChannel(streamUrl, chNum));
    }
  };

  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await onTestStream({ streamUrl });
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
    const chNum = parseInt(channel.trim(), 10);
    if (isNaN(chNum) || chNum < 1) {
      setErrorMessage("Please enter a valid channel number (e.g. 1 to 64).");
      return;
    }

    const latStr = latitude.trim();
    const lngStr = longitude.trim();
    let latNum: number | null = null;
    let lngNum: number | null = null;

    if (latStr !== "") {
      latNum = parseFloat(latStr);
      if (isNaN(latNum) || latNum < -90 || latNum > 90) {
        setErrorMessage("Latitude must be a valid number between -90 and 90, or left blank.");
        return;
      }
    }
    if (lngStr !== "") {
      lngNum = parseFloat(lngStr);
      if (isNaN(lngNum) || lngNum < -180 || lngNum > 180) {
        setErrorMessage("Longitude must be a valid number between -180 and 180, or left blank.");
        return;
      }
    }

    setErrorMessage(null);
    setIsSubmitting(true);
    try {
      await onSave(camera.id, {
        name: name.trim(),
        channel: chNum,
        type,
        status,
        latitude: latNum,
        longitude: lngNum,
        streamUrl: streamUrl.trim() || buildChannelRtspUrl(chNum),
      });
      onClose();
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to update camera settings.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs overflow-y-auto">
      <div className="bg-white rounded-2xl border border-slate-200 max-w-lg w-full p-6 shadow-2xl space-y-5 my-8">
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-slate-100 text-slate-800 flex items-center justify-center">
              <Settings className="w-5 h-5 text-emerald-700" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Camera Settings
              </h3>
              <p className="text-xs text-slate-500 font-mono">
                Channel {camera.channel ?? "—"} • {camera.name}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 cursor-pointer"
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
          {/* General Details (5 Fields) */}
          <div className="space-y-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Camera Name *
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. CSD or MAIN OFFICE OUTDOOR"
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Channel Number *
                </label>
                <input
                  type="number"
                  min="1"
                  max="64"
                  required
                  value={channel}
                  onChange={(e) => handleChannelChange(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Operational Status
                </label>
                <select
                  value={status}
                  onChange={(e) => setStatus(e.target.value as any)}
                  className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700 cursor-pointer"
                >
                  <option value="online">Online</option>
                  <option value="alert">Alert Active</option>
                  <option value="degraded">Degraded</option>
                  <option value="offline">Offline</option>
                </select>
              </div>
            </div>

            <div>
              <label className="text-xs font-semibold text-slate-700 block mb-1">
                Sensor Type
              </label>
              <select
                value={type}
                onChange={(e) => setType(e.target.value as any)}
                className="w-full px-3 py-2 text-xs rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none font-medium text-slate-700 cursor-pointer"
              >
                {CAMERA_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Latitude (Optional)
                </label>
                <input
                  type="text"
                  value={latitude}
                  onChange={(e) => setLatitude(e.target.value)}
                  placeholder="e.g. 23.1542"
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>

              <div>
                <label className="text-xs font-semibold text-slate-700 block mb-1">
                  Longitude (Optional)
                </label>
                <input
                  type="text"
                  value={longitude}
                  onChange={(e) => setLongitude(e.target.value)}
                  placeholder="e.g. 72.8852"
                  className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* RTSP Stream Link Section */}
          <div className="space-y-2 pt-3 border-t border-slate-100">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-700 block">
                RTSP Stream Link (Auto-updated with Channel)
              </label>
              <button
                type="button"
                onClick={handleTest}
                disabled={isTesting}
                className="text-xs font-bold text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
              >
                {isTesting ? (
                  <RefreshCw className="w-3 h-3 animate-spin" />
                ) : (
                  <Zap className="w-3 h-3" />
                )}
                <span>Ping Feed</span>
              </button>
            </div>
            <input
              type="text"
              value={streamUrl}
              onChange={(e) => setStreamUrl(e.target.value)}
              className="w-full px-3 py-2 text-xs font-mono rounded-xl border border-slate-200 bg-slate-50/50 focus:bg-white focus:outline-none"
            />
            {testResult && (
              <div
                className={`p-2 rounded-xl text-xs font-mono flex items-center justify-between ${
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

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-slate-100">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-xl border border-slate-200 text-xs font-semibold text-slate-700 hover:bg-slate-50 cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-5 py-2 rounded-xl bg-[#143724] hover:bg-[#1a472f] text-white text-xs font-bold shadow-xs transition-colors flex items-center gap-2 cursor-pointer"
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
