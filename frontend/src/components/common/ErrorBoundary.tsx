"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Resilient Error Boundary for Surveillance & AI HUD Components.
 * Prevents memory-spike or rendering crashes from taking down the entire application.
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("[SurveillanceErrorBoundary] Caught unhandled component error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-xl border border-rose-300/80 bg-rose-50/70 p-6 text-slate-900 shadow-sm space-y-4 font-sans">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-rose-100 text-rose-700">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-rose-950">
                {this.props.fallbackTitle || "Surveillance Stream Component Suspended"}
              </h3>
              <p className="text-xs text-rose-700 mt-0.5">
                The component encountered a high-load execution interruption. All other system subsystems remain operational.
              </p>
            </div>
          </div>

          {this.state.error && (
            <div className="p-3 rounded-lg bg-white/90 border border-rose-200 text-[11px] font-mono text-rose-800 break-words max-h-32 overflow-y-auto">
              {this.state.error.message || String(this.state.error)}
            </div>
          )}

          <div className="flex items-center gap-2 pt-1">
            <button
              type="button"
              onClick={this.handleReset}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-700 hover:bg-rose-800 text-white shadow-2xs transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Recover Component</span>
            </button>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-rose-300 bg-white hover:bg-rose-50 text-rose-900 transition-colors cursor-pointer"
            >
              <span>Reload Workspace</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
