"use client";

import Link from "next/link";
import { ArrowRight, Video } from "lucide-react";
import { useCameras } from "@/lib/camerasStore";

export function CameraOutpostGrid() {
  const { cameras } = useCameras();
  const displayCameras = cameras.slice(0, 4);

  return (
    <div className="bg-white rounded-2xl border border-[#dce5df] p-6 shadow-xs flex flex-col justify-between h-full">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-5">
        <div>
          <h2 className="text-base font-bold text-slate-900 tracking-tight">
            Camera Outpost Status
          </h2>
          <p className="text-xs text-slate-500 mt-0.5 font-medium">
            Live operational state across key perimeter zones.
          </p>
        </div>
        <Link
          href="/cameras"
          className="text-xs font-semibold text-[#1e4b38] hover:text-[#123621] hover:underline flex items-center gap-1 shrink-0"
        >
          <span>All Cameras</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* 2x2 Grid with Clickable Camera Outposts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {displayCameras.map((cam, idx) => {
          const ch = cam.channel ?? idx + 1;
          const isAlert = cam.status === "alert";
          const isOffline = cam.status === "offline";

          return (
            <Link
              key={cam.id}
              href={`/live?channel=${ch}`}
              title={`Click to open live video stream for ${cam.name}`}
              className={`p-4 rounded-xl border transition-all flex items-center justify-between group cursor-pointer ${
                isAlert
                  ? "bg-rose-50/25 border-rose-200/70 hover:border-rose-400 hover:shadow-md hover:bg-rose-50/50"
                  : "bg-slate-50/70 border-slate-100 hover:border-[#1e4b38]/40 hover:shadow-md hover:bg-white"
              }`}
            >
              <div className="flex items-center gap-3">
                {/* Index Pill */}
                <span
                  className={`w-9 h-9 rounded-lg flex items-center justify-center text-xs font-mono font-bold shrink-0 transition-transform group-hover:scale-105 ${
                    isAlert
                      ? "bg-rose-600 text-white shadow-xs"
                      : "bg-slate-200/80 text-slate-700 group-hover:bg-[#1e4b38] group-hover:text-white"
                  }`}
                >
                  {String(ch).padStart(2, "0")}
                </span>

                {/* Details */}
                <div>
                  <h4 className="text-xs sm:text-sm font-bold text-slate-900 group-hover:text-[#1e4b38] transition-colors flex items-center gap-1.5">
                    <span>{cam.name}</span>
                    <Video className="w-3.5 h-3.5 text-slate-400 group-hover:text-[#1e4b38] opacity-0 group-hover:opacity-100 transition-opacity" />
                  </h4>
                  <p
                    className={`text-[11px] font-medium mt-0.5 ${
                      isAlert ? "text-rose-600 font-semibold" : "text-slate-500"
                    }`}
                  >
                    Channel {ch} • {cam.type || "Optical 4K"}
                  </p>
                </div>
              </div>

              {/* Status Badge */}
              <div>
                {isAlert ? (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-[11px] font-semibold bg-rose-50 text-rose-700 border border-rose-200/80 group-hover:bg-rose-100">
                    Alert
                  </span>
                ) : isOffline ? (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-[11px] font-semibold bg-slate-100 text-slate-600 border border-slate-200 group-hover:bg-slate-200">
                    Offline
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2.5 py-1 rounded-md text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/60 group-hover:bg-emerald-100">
                    Normal
                  </span>
                )}
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
