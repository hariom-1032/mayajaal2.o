"use client";

import { useMemo } from "react";
import { ShieldAlert } from "lucide-react";
import { extractChannelFromUrl } from "@/lib/borderCameras";

export interface CameraCardProps {
  id?: string;
  name: string;
  type?: string;
  status?: string;
  isOnline?: boolean;
  streamUrl?: string;
  isSelected?: boolean;
  hasSuspectAlert?: boolean;
  suspectName?: string;
  threatLevel?: string;
  onClick?: () => void;
}

export function CameraCard({
  name,
  type,
  isOnline = true,
  streamUrl,
  isSelected = false,
  hasSuspectAlert = false,
  onClick,
}: CameraCardProps) {
  const channel = useMemo(() => {
    return extractChannelFromUrl(streamUrl);
  }, [streamUrl]);

  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left rounded-lg p-3.5 transition-colors cursor-pointer select-none border focus:outline-none focus:ring-1 ${
        hasSuspectAlert
          ? "bg-rose-950/20 border-rose-600/80 text-rose-100 hover:bg-rose-950/30 focus:ring-rose-500"
          : isSelected
          ? "bg-slate-900 border-emerald-500/80 text-white shadow-sm focus:ring-emerald-400"
          : "bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/90 text-slate-300 focus:ring-slate-600"
      }`}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider shrink-0 ${
              isSelected
                ? "bg-emerald-600 text-white"
                : "bg-slate-800 text-slate-400 border border-slate-700/60"
            }`}
          >
            {channel ? `CH ${String(channel).padStart(2, "0")}` : "CAM"}
          </span>
          <span className="font-semibold text-xs tracking-tight truncate text-white">
            {name}
          </span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          {hasSuspectAlert ? (
            <span className="flex items-center gap-1 text-[10px] font-mono font-bold text-rose-400 bg-rose-950/80 px-1.5 py-0.5 rounded border border-rose-700/50">
              <ShieldAlert className="w-3 h-3 text-rose-400" />
              ALERT
            </span>
          ) : isOnline ? (
            <span className="flex items-center gap-1 text-[10px] font-mono text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              LIVE
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] font-mono text-slate-500">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
              IDLE
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between text-[11px] font-mono text-slate-500">
        <span className="truncate pr-2">{type || "Camera"}</span>
        {isSelected && (
          <span className="text-[10px] font-bold text-emerald-400 tracking-wider">
            ● VIEWING
          </span>
        )}
      </div>
    </button>
  );
}

export default CameraCard;
