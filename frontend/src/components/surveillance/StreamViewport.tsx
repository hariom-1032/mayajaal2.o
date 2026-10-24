"use client";

import { useRef, useEffect, useCallback, memo } from "react";

interface StreamViewportProps {
  /** Backend MJPEG stream URL (proxied via Next.js → FastAPI). */
  src: string | null;
  /** Optional label shown top-left (camera name). */
  label?: string;
  /** Unique key to force-remount the <img> element. */
  streamKey?: number;
  /** Real-time output FPS measured from stream pipeline. */
  fps?: number | null;
  /** Callback when image fails to load. */
  onError?: () => void;
  /** Callback when image loads successfully. */
  onLoad?: () => void;
}

/**
 * Ultra-lightweight MJPEG stream viewport.
 *
 * Design principles:
 *  - Zero React re-renders for overlay updates (time/fps use refs + DOM mutations).
 *  - `contain: strict` isolates paint/layout from parent.
 *  - `will-change: transform` promotes to GPU layer for compositing.
 *  - No decorative overlays, canvas, SVG, blur, or animations.
 *  - Only shows: current Time (HH:MM:SS) and FPS counter.
 */
function StreamViewportInner({
  src,
  label,
  streamKey = 0,
  fps,
  onError,
  onLoad,
}: StreamViewportProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const timeRef = useRef<HTMLSpanElement>(null);
  const fpsRef = useRef<HTMLSpanElement>(null);

  // ── Time overlay: update every second via direct DOM write ──
  useEffect(() => {
    const updateTime = () => {
      if (timeRef.current) {
        const now = new Date();
        timeRef.current.textContent =
          now.getHours().toString().padStart(2, "0") +
          ":" +
          now.getMinutes().toString().padStart(2, "0") +
          ":" +
          now.getSeconds().toString().padStart(2, "0");
      }
    };
    updateTime();
    const id = setInterval(updateTime, 1000);
    return () => clearInterval(id);
  }, []);

  // ── Clean up MJPEG stream image on unmount or src change to release Chromium HTTP sockets ──
  useEffect(() => {
    const imgEl = imgRef.current;
    return () => {
      if (imgEl) {
        // Explicitly clearing src terminates the browser's persistent multipart HTTP/1.1 TCP connection
        imgEl.src = "";
      }
    };
  }, [src, streamKey]);

  // ── FPS counter: display pipeline/render FPS ──
  useEffect(() => {
    if (fpsRef.current) {
      if (typeof fps === "number" && fps > 0) {
        fpsRef.current.textContent = `${fps.toFixed(1)} FPS`;
      } else if (!src) {
        fpsRef.current.textContent = `-- FPS`;
      }
    }
  }, [fps, src]);

  const handleError = useCallback(() => {
    onError?.();
  }, [onError]);

  const handleLoad = useCallback(() => {
    onLoad?.();
  }, [onLoad]);

  if (!src) {
    return (
      <div
        className="w-full h-full flex items-center justify-center bg-slate-950 text-slate-500 text-xs font-mono select-none"
        style={{ contain: "strict" }}
      >
        Select a camera to begin streaming
      </div>
    );
  }

  return (
    <div
      className="relative w-full h-full bg-black overflow-hidden select-none"
      style={{
        contain: "strict",
        willChange: "transform",
      }}
    >
      {/* MJPEG stream image — fills container */}
      <img
        ref={imgRef}
        key={streamKey}
        src={src}
        alt={label || "Live stream"}
        onError={handleError}
        onLoad={handleLoad}
        draggable={false}
        className="absolute inset-0 w-full h-full object-contain"
        style={{
          transform: "translateZ(0)",
          imageRendering: "auto",
        }}
      />

      {/* Minimal overlay: Time + FPS — GPU-composited text */}
      <div
        className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 py-1.5 pointer-events-none"
        style={{ transform: "translateZ(0)" }}
      >
        {/* Camera label + time */}
        <div className="flex items-center gap-2">
          {label && (
            <span className="text-[11px] font-mono font-bold text-white/90 drop-shadow-sm tracking-wide uppercase">
              {label}
            </span>
          )}
          <span
            ref={timeRef}
            className="text-[11px] font-mono text-white/70 drop-shadow-sm tabular-nums"
          >
            --:--:--
          </span>
        </div>

        {/* FPS counter */}
        <span
          ref={fpsRef}
          className="text-[11px] font-mono text-emerald-400/80 drop-shadow-sm tabular-nums"
        >
          -- FPS
        </span>
      </div>
    </div>
  );
}

export const StreamViewport = memo(StreamViewportInner);
export default StreamViewport;
