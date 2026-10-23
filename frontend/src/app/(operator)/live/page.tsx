import { ConnectionStatus } from "@/components/status/connectionstatus";
import { LiveSurveillanceConsole } from "@/components/surveillance/LiveSurveillanceConsole";

export default function LivePage() {
  return (
    <div className="space-y-4 pb-8">
      {/* Header bar with Title, Status, and ConnectionStatus */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pb-3 border-b border-slate-200/80">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900">
            CCTV Surveillance Console
          </h1>
          <p className="text-xs font-mono text-slate-500 mt-0.5">
            Primary NVR Terminal • 10 Active Real-time Channels (142.67.45.150)
          </p>
        </div>

        {/* Connection Status indicator near the title */}
        <div>
          <ConnectionStatus />
        </div>
      </div>

      {/* Industrial Minimalist Surveillance Console */}
      <LiveSurveillanceConsole />
    </div>
  );
}
