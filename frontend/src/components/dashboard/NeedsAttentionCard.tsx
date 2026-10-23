"use client";

import Link from "next/link";
import { ArrowUpRight, Video, ShieldCheck, AlertTriangle } from "lucide-react";
import { useAlerts } from "@/lib/alertsStore";

export function NeedsAttentionCard() {
  const { alerts } = useAlerts();
  const pendingAlerts = alerts.filter((a) => !a.acknowledged);
  const displayAlerts = pendingAlerts.slice(0, 2);

  return (
    <div className="bg-white rounded-2xl border border-[#dce5df] p-6 shadow-xs flex flex-col justify-between h-full">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-base font-bold text-slate-900 tracking-tight">
          What Needs Attention
        </h2>
        <p className="text-xs text-slate-500 mt-0.5 font-medium">
          High priority highlights for current watch. Click to inspect live feed.
        </p>
      </div>

      {/* Content */}
      <div className="space-y-3.5">
        {displayAlerts.length > 0 ? (
          displayAlerts.map((alert) => (
            <Link
              key={alert.id}
              href="/live"
              className="block p-4 rounded-xl border border-rose-200/80 bg-rose-50/20 hover:bg-rose-50/70 hover:border-rose-300 hover:shadow-sm transition-all group cursor-pointer"
              title={`Inspect: ${alert.title}`}
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <span className="text-xs sm:text-sm font-bold text-slate-900 group-hover:text-rose-700 transition-colors flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-rose-500" />
                  <span>{alert.title}</span>
                </span>
                <span className="text-[10px] font-extrabold tracking-wider uppercase text-rose-600 bg-rose-100/60 px-2 py-0.5 rounded flex items-center gap-1">
                  <span>{alert.threatLevel || "ALERT"}</span>
                  <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
                </span>
              </div>
              <p className="text-xs text-slate-600 font-normal leading-relaxed">
                {alert.location || alert.notes || "Real-time surveillance event requiring review."}
              </p>
            </Link>
          ))
        ) : (
          <div className="p-6 rounded-xl border border-emerald-200/80 bg-emerald-50/20 text-center space-y-2">
            <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-800 mx-auto flex items-center justify-center">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <h4 className="text-xs font-bold text-slate-900">
              All Perimeter Sectors Nominal
            </h4>
            <p className="text-[11px] text-slate-500 max-w-xs mx-auto">
              Zero active alerts or unauthorized tripwire breaches detected. All perimeter channels operating normally.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
