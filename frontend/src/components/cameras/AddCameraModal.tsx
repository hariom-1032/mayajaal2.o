"use client";

import { useState } from "react";
import { Plus, X } from "lucide-react";
import { buildChannelRtspUrl, RTSP_CAMERA_TEMPLATE } from "@/lib/borderCameras";
import { camerasStore } from "@/lib/camerasStore";
import { CameraType } from "@/types/camera";

interface AddCameraModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdded?: (newCamId: string) => void;
}

export function AddCameraModal({ isOpen, onClose, onAdded }: AddCameraModalProps) {
  const [name, setName] = useState("");
  const [channel, setChannel] = useState("11");
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [cameraType, setCameraType] = useState<CameraType>("Optical 4K");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const resolvedRtspUrl = buildChannelRtspUrl(channel || "1");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Please enter a camera name.");
      return;
    }
    const chNum = parseInt(channel.trim(), 10);
    if (isNaN(chNum) || chNum < 1) {
      setError("Please enter a valid channel number (e.g. 1 to 64).");
      return;
    }

    // Lat/lng are optional — parse only if provided
    const latStr = latitude.trim();
    const lngStr = longitude.trim();
    let latVal: number | null = null;
    let lngVal: number | null = null;

    if (latStr !== "") {
      latVal = parseFloat(latStr);
      if (isNaN(latVal)) {
        setError("Latitude must be a valid decimal number or left blank.");
        return;
      }
    }
    if (lngStr !== "") {
      lngVal = parseFloat(lngStr);
      if (isNaN(lngVal)) {
        setError("Longitude must be a valid decimal number or left blank.");
        return;
      }
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const created = await camerasStore.addCameraByChannel({
        name: name.trim(),
        channel: chNum,
        latitude: latVal,
        longitude: lngVal,
        type: cameraType,
      });

      setName("");
      setChannel("11");
      setLatitude("");
      setLongitude("");
      onAdded?.(created.id);
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to add camera";
      setError(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl max-w-md w-full shadow-2xl overflow-hidden text-slate-200">
        {/* Modal Header */}
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-md bg-emerald-950 text-emerald-400 border border-emerald-800/50">
              <Plus className="w-4 h-4" />
            </span>
            <h3 className="text-sm font-bold text-white tracking-tight">
              Add Camera
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-md hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Form */}
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          {error && (
            <div className="p-2.5 rounded-lg bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs font-mono">
              {error}
            </div>
          )}

          {/* Camera Name — REQUIRED */}
          <div>
            <label className="block text-xs font-mono font-semibold text-slate-300 mb-1.5">
              CAMERA NAME *
            </label>
            <input
              type="text"
              required
              placeholder="e.g. Main Gate Entrance"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 text-xs rounded-lg bg-slate-950 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-sans"
            />
          </div>

          {/* Channel Number — REQUIRED */}
          <div>
            <label className="block text-xs font-mono font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
              <span>NVR CHANNEL NUMBER *</span>
              <span className="text-[10px] text-emerald-400 font-normal">Host: 142.67.45.150</span>
            </label>
            <input
              type="number"
              min="1"
              max="128"
              required
              placeholder="e.g. 11"
              value={channel}
              onChange={(e) => setChannel(e.target.value)}
              className="w-full px-3 py-2 text-xs font-mono rounded-lg bg-slate-950 border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
            {/* Live RTSP Preview */}
            <div className="mt-1.5 p-2 rounded bg-slate-950/80 border border-slate-800 text-[10px] font-mono text-slate-400 break-all">
              <span className="text-emerald-400 font-bold">RTSP: </span>
              {resolvedRtspUrl}
            </div>
          </div>

          {/* Camera Type */}
          <div>
            <label className="block text-xs font-mono font-semibold text-slate-300 mb-1.5">
              SENSOR TYPE
            </label>
            <select
              value={cameraType}
              onChange={(e) => setCameraType(e.target.value as CameraType)}
              className="w-full px-3 py-2 text-xs rounded-lg bg-slate-950 border border-slate-700 text-white focus:outline-none focus:border-emerald-500 cursor-pointer"
            >
              <option value="Optical 4K">Optical 4K UHD</option>
              <option value="Thermal FLIR">Thermal FLIR IR</option>
              <option value="PTZ 360">PTZ 360 Tracking</option>
              <option value="ANPR Dedicated">ANPR Dedicated Plate Scanner</option>
              <option value="Panoramic">Panoramic Wide Angle</option>
              <option value="Night Vision / IR">Night Vision / Low Light</option>
            </select>
          </div>

          {/* Coordinates — OPTIONAL */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-mono font-semibold text-slate-300 mb-1.5">
                LATITUDE <span className="text-slate-500 font-normal">(optional)</span>
              </label>
              <input
                type="text"
                placeholder="e.g. 23.15421"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono rounded-lg bg-slate-950 border border-slate-700 text-white focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label className="block text-xs font-mono font-semibold text-slate-300 mb-1.5">
                LONGITUDE <span className="text-slate-500 font-normal">(optional)</span>
              </label>
              <input
                type="text"
                placeholder="e.g. 72.885257"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                className="w-full px-3 py-2 text-xs font-mono rounded-lg bg-slate-950 border border-slate-700 text-white focus:outline-none focus:border-emerald-500"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="pt-2 flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs font-medium rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-4 py-2 text-xs font-bold rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors cursor-pointer disabled:opacity-50 flex items-center gap-1.5 shadow-sm"
            >
              {isSubmitting ? "Adding..." : "Add Camera"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default AddCameraModal;
