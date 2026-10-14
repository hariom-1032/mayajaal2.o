"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  LayoutDashboard,
  Map as MapIcon,
  Video,
  AlertTriangle,
  Camera,
  Activity,
  Globe,
  Car,
  Settings,
  Search,
  Bell,
  Cpu,
  LogOut,
  Pin,
  PinOff,
  PanelLeft,
  Maximize2,
  Minimize2,
  ScanFace,
  ShieldCheck,
  ShieldAlert,
  Clock,
  ChevronDown,
  Lock,
} from "lucide-react";
import { useAlerts } from "@/lib/alertsStore";
import { useAuth, getRemainingShiftTime, isSessionExpired } from "@/lib/authStore";
import { TacticalThreatToast } from "@/components/alerts/TacticalThreatToast";

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number | string;
  badgeColor?: string;
}

function HeaderClock() {
  const [timestamp, setTimestamp] = useState<{ date: string; time: string }>({
    date: "09 Sep 2026",
    time: "10:24:00",
  });

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const date = now.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: "Asia/Kolkata",
      });
      const time = now.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
        timeZone: "Asia/Kolkata",
      });
      setTimestamp((prev) => {
        if (prev.date === date && prev.time === time) return prev;
        return { date, time };
      });
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className="hidden md:flex flex-col text-xs font-mono text-slate-500 flex-shrink-0"
      suppressHydrationWarning
    >
      <span suppressHydrationWarning>{timestamp.date}</span>
      <span className="text-[10px] text-slate-400" suppressHydrationWarning>
        {timestamp.time} IST
      </span>
    </div>
  );
}

function DutyShiftCountdown({ expiresAt }: { expiresAt?: string }) {
  const [shiftRemaining, setShiftRemaining] = useState<string>("");

  useEffect(() => {
    if (!expiresAt) return;
    const updateShift = () => {
      const rem = getRemainingShiftTime(expiresAt);
      setShiftRemaining((prev) => (prev === rem.formatted ? prev : rem.formatted));
    };
    updateShift();
    const interval = setInterval(updateShift, 1000);
    return () => clearInterval(interval);
  }, [expiresAt]);

  return (
    <span className="font-mono text-[11px] font-bold text-emerald-900">
      {shiftRemaining || "Active"}
    </span>
  );
}

export default function OperatorLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();
  const { operator, isAuthenticated, isLoading, logout } = useAuth();
  const { unacknowledgedCount } = useAlerts();
  const [searchQuery, setSearchQuery] = useState("");
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement | null>(null);

  // Client-side authentication guard: immediately redirect unauthenticated users
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.replace(`/login?redirect=${encodeURIComponent(pathname)}`);
    }
  }, [isLoading, isAuthenticated, pathname, router]);

  // Periodic duty shift session expiration monitor (runs every 10s without forced layout re-renders)
  useEffect(() => {
    if (!operator) return;

    const checkExpiry = () => {
      if (isSessionExpired(operator)) {
        logout();
        router.replace(
          `/login?reason=expired&redirect=${encodeURIComponent(pathname)}`
        );
      }
    };

    checkExpiry();
    const interval = setInterval(checkExpiry, 10000);
    return () => clearInterval(interval);
  }, [operator?.badgeNumber, operator?.expiresAt, logout, pathname, router]);

  // Close profile dropdown on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (
        profileMenuRef.current &&
        !profileMenuRef.current.contains(e.target as Node)
      ) {
        setShowProfileMenu(false);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, []);

  // Sidebar modes:
  // "rail" = compact icon rail (68px) that expands to 256px on hover
  // "hidden" = completely off-screen (0px), expands to 256px on left-edge / trigger hover
  // "pinned" = locked open at 256px
  const [sidebarMode, setSidebarMode] = useState<"rail" | "hidden" | "pinned">("rail");
  const [isHovered, setIsHovered] = useState(false);
  const leaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Restore saved sidebar preference
  useEffect(() => {
    try {
      const saved = localStorage.getItem("maatrix_sidebar_mode");
      if (saved === "rail" || saved === "hidden" || saved === "pinned") {
        setSidebarMode(saved);
      }
    } catch {
      // Ignore localStorage access errors
    }
  }, []);

  const handleModeChange = (newMode: "rail" | "hidden" | "pinned") => {
    setSidebarMode(newMode);
    try {
      localStorage.setItem("maatrix_sidebar_mode", newMode);
    } catch {}
  };

  const handleMouseEnter = () => {
    if (leaveTimeoutRef.current) {
      clearTimeout(leaveTimeoutRef.current);
      leaveTimeoutRef.current = null;
    }
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    if (leaveTimeoutRef.current) {
      clearTimeout(leaveTimeoutRef.current);
    }
    leaveTimeoutRef.current = setTimeout(() => {
      setIsHovered(false);
    }, 250);
  };

  // Whether the sidebar is currently showing its expanded view
  const isExpanded = sidebarMode === "pinned" || isHovered;

  // Seamless navigation handler that preserves Next.js client-side prefetching and instantly closes hover sidebar
  const handleNavClick = () => {
    // Immediately dismiss floating hover menu so operator gets instant feedback
    if (sidebarMode !== "pinned") {
      if (leaveTimeoutRef.current) {
        clearTimeout(leaveTimeoutRef.current);
        leaveTimeoutRef.current = null;
      }
      setIsHovered(false);
    }
  };

  const mainNavItems: NavItem[] = [
    {
      label: "Dashboard",
      href: "/dashboard",
      icon: LayoutDashboard,
    },
    {
      label: "GIS Map",
      href: "/gis-map",
      icon: MapIcon,
    },
    {
      label: "Live Surveillance",
      href: "/live",
      icon: Video,
    },
    {
      label: "Alerts",
      href: "/alerts",
      icon: AlertTriangle,
      badge: unacknowledgedCount > 0 ? unacknowledgedCount : 8,
      badgeColor: "bg-rose-600 text-white",
    },
    {
      label: "Cameras",
      href: "/cameras",
      icon: Camera,
    },
    {
      label: "Tracks & Trajectories",
      href: "/tracks",
      icon: Activity,
    },
    {
      label: "Geofences",
      href: "/geofences",
      icon: Globe,
    },
    {
      label: "ANPR Vehicles",
      href: "/anpr",
      icon: Car,
    },
    {
      label: "Facial Recognition",
      href: "/facial-recognition",
      icon: ScanFace,
    },
  ];

  const systemNavItems: NavItem[] = [
    {
      label: "Settings",
      href: "/settings",
      icon: Settings,
    },
    {
      label: "Diagnostics",
      href: "/diagnostics",
      icon: Cpu,
    },
  ];

  // Render high-security clearance loading gate while determining session state
  if (isLoading) {
    return (
      <div className="fixed inset-0 z-50 bg-[#edf3ef] flex flex-col items-center justify-center select-none text-slate-800">
        <div className="relative mb-6">
          <div className="w-20 h-20 rounded-2xl bg-white border border-emerald-200/80 shadow-xl flex items-center justify-center p-3.5">
            <img
              src="/images/logo/maatrix-emblem.png"
              alt="MAATRIX Logo"
              className="w-full h-full object-contain animate-pulse"
            />
          </div>
          <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-emerald-700 text-white flex items-center justify-center shadow-md">
            <ShieldCheck className="w-3.5 h-3.5" />
          </div>
        </div>
        <div className="text-center space-y-2">
          <div className="text-sm font-black tracking-widest text-[#1e4b38] uppercase">
            MAATRIX DEFENSE GRID
          </div>
          <div className="text-xs text-slate-600 font-mono flex items-center justify-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-600 animate-ping" />
            <span>Verifying Operator Clearance & Duty Session...</span>
          </div>
        </div>
      </div>
    );
  }

  // Block rendering of sensitive operator interface if not authenticated
  if (!isAuthenticated || !operator) {
    return null;
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#f4f7f5] relative">
      {/* Edge Hover Sensor for Pure Fullscreen Mode */}
      {sidebarMode === "hidden" && !isHovered && (
        <div
          onMouseEnter={handleMouseEnter}
          className="fixed top-0 bottom-0 left-0 w-3 z-30 group cursor-pointer flex items-center"
          title="Move cursor here to reveal navigation menu"
        >
          <div className="w-1.5 h-16 rounded-r-full bg-[#1e4b38]/50 group-hover:bg-[#1e4b38] group-hover:w-2.5 transition-all shadow-sm ml-0" />
        </div>
      )}

      {/* Spacer in document flow to keep layout stable and maximize screen width */}
      <div
        className={`flex-shrink-0 transition-all duration-300 ease-in-out ${
          sidebarMode === "pinned"
            ? "w-64"
            : sidebarMode === "rail"
            ? "w-[68px]"
            : "w-0"
        }`}
        aria-hidden="true"
      />

      {/* Subtle backdrop overlay when expanded in floating hover mode */}
      {sidebarMode !== "pinned" && isExpanded && (
        <div
          onClick={() => {
            if (leaveTimeoutRef.current) {
              clearTimeout(leaveTimeoutRef.current);
              leaveTimeoutRef.current = null;
            }
            setIsHovered(false);
          }}
          className="fixed inset-0 bg-slate-900/20 backdrop-blur-[0.5px] z-30 transition-opacity duration-300 cursor-pointer"
        />
      )}

      {/* Light Government Green Left Sidebar (Auto-expand on hover, auto-collapse on mouse leave) */}
      <aside
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className={`fixed top-0 bottom-0 left-0 z-40 bg-[#edf3ef] text-[#1c3829] flex flex-col border-r border-[#d2dfd6] select-none transition-all duration-300 ease-in-out ${
          sidebarMode === "hidden"
            ? isExpanded
              ? "w-64 translate-x-0 shadow-xl shadow-slate-900/10 ring-1 ring-[#1e4b38]/10"
              : "w-64 -translate-x-full shadow-none"
            : isExpanded
            ? "w-64 shadow-xl shadow-slate-900/10 ring-1 ring-[#1e4b38]/10"
            : "w-[68px] shadow-xs"
        }`}
      >
        {/* Top Header: MAATRIX Logo, Status & Controls */}
        <div className="p-3.5 border-b border-[#d2dfd6] bg-[#e4ede6]/60">
          <div className="flex items-center justify-between gap-2">
            <Link
              href="/dashboard"
              prefetch={true}
              onClick={handleNavClick}
              className="flex items-center gap-3 group min-w-0 cursor-pointer"
              title="MAATRIX - Border Video Analytics"
            >
              {/* Official MAATRIX Brand Emblem */}
              <div className="w-10 h-10 rounded-xl bg-white border border-[#bcd3c4] flex items-center justify-center flex-shrink-0 shadow-2xs group-hover:border-[#1e4b38]/60 transition-all p-1.5 overflow-hidden">
                <img
                  src="/images/logo/maatrix-emblem.png"
                  alt="MAATRIX Logo"
                  className="w-full h-full object-contain filter drop-shadow-xs"
                />
              </div>

              {isExpanded && (
                <div className="overflow-hidden min-w-0 animate-in fade-in duration-200">
                  <h1 className="text-base font-extrabold tracking-wider text-[#143924] uppercase truncate leading-tight">
                    MAATRIX
                  </h1>
                  <p className="text-[11px] font-semibold text-[#2b593d] truncate">
                    Border Video Analytics
                  </p>
                </div>
              )}
            </Link>

            {/* Quick Action Controls in Expanded Mode */}
            {isExpanded && (
              <div className="flex items-center gap-1 animate-in fade-in duration-150">
                {/* Switch between Compact Rail and Pure Fullscreen */}
                <button
                  type="button"
                  onClick={() =>
                    handleModeChange(sidebarMode === "hidden" ? "rail" : "hidden")
                  }
                  className={`p-1.5 rounded-lg text-[#325b42] hover:text-[#143924] hover:bg-[#d8e6db] transition-colors cursor-pointer ${
                    sidebarMode === "hidden" ? "text-[#143924] bg-[#d8e6db]" : ""
                  }`}
                  title={
                    sidebarMode === "hidden"
                      ? "Switch to Compact Rail mode"
                      : "Switch to 100% Pure Fullscreen mode (hide completely)"
                  }
                >
                  {sidebarMode === "hidden" ? (
                    <Minimize2 className="w-3.5 h-3.5" />
                  ) : (
                    <Maximize2 className="w-3.5 h-3.5" />
                  )}
                </button>

                {/* Pin / Unpin Button */}
                <button
                  type="button"
                  onClick={() =>
                    handleModeChange(sidebarMode === "pinned" ? "rail" : "pinned")
                  }
                  className={`p-1.5 rounded-lg text-[#325b42] hover:text-[#143924] hover:bg-[#d8e6db] transition-colors cursor-pointer ${
                    sidebarMode === "pinned"
                      ? "text-[#143924] bg-[#d3e3d7] ring-1 ring-[#1e4b38]/20"
                      : ""
                  }`}
                  title={
                    sidebarMode === "pinned"
                      ? "Unpin sidebar (enable auto-collapse on hover)"
                      : "Pin sidebar permanently open"
                  }
                >
                  {sidebarMode === "pinned" ? (
                    <Pin className="w-3.5 h-3.5 rotate-45" />
                  ) : (
                    <PinOff className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            )}
          </div>

        </div>

        {/* Navigation Items */}
        <div className="flex-1 px-2.5 py-4 space-y-5 overflow-y-auto custom-scrollbar overflow-x-hidden">
          {/* Main Operational Links */}
          <nav className="space-y-1">
            {mainNavItems.map((item) => {
              const isActive =
                pathname === item.href ||
                (item.href === "/dashboard" && pathname === "/");
              const Icon = item.icon;

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  prefetch={true}
                  onClick={handleNavClick}
                  title={!isExpanded ? item.label : undefined}
                  className={`w-full h-11 flex items-center rounded-xl text-xs font-semibold relative select-none cursor-pointer transition-colors active:scale-[0.98] px-3 ${
                    isActive
                      ? "bg-[#d5e6db] text-[#123621] font-bold shadow-2xs border border-[#b8d1c1]"
                      : "text-[#2e543d] hover:bg-[#e0ede4] hover:text-[#123621]"
                  }`}
                >
                  <div className="w-5 h-5 flex items-center justify-center flex-shrink-0 pointer-events-none">
                    <Icon
                      className={`w-5 h-5 flex-shrink-0 pointer-events-none transition-colors ${
                        isActive
                          ? "text-[#18492d]"
                          : "text-[#3d654f] group-hover:text-[#18492d]"
                      }`}
                    />
                  </div>

                  <span
                    className={`ml-3 text-xs font-semibold truncate whitespace-nowrap pointer-events-none transition-all duration-200 ${
                      isExpanded
                        ? "opacity-100 translate-x-0 max-w-[130px]"
                        : "opacity-0 -translate-x-2 max-w-0 overflow-hidden"
                    }`}
                  >
                    {item.label}
                  </span>

                  {item.badge !== undefined && (
                    <span
                      className={`pointer-events-none transition-all duration-200 ${
                        isExpanded
                          ? "ml-auto px-2 py-0.5 text-[10px] font-bold rounded-full flex-shrink-0"
                          : "absolute top-1.5 right-2 flex h-4 min-w-4 px-1 items-center justify-center text-[9px] font-bold rounded-full ring-2 ring-[#edf3ef]"
                      } ${item.badgeColor || "bg-[#1e4b38] text-white"}`}
                    >
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* SYSTEM section */}
          <div>
            {isExpanded ? (
              <div className="px-3 mb-2 text-[10px] font-bold tracking-wider uppercase text-[#476e57] animate-in fade-in duration-150">
                SYSTEM
              </div>
            ) : (
              <div className="w-8 mx-auto h-px bg-[#cddcd2] my-3" />
            )}
            <nav className="space-y-1">
              {systemNavItems.map((item) => {
                const isActive = pathname === item.href;
                const Icon = item.icon;

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    prefetch={true}
                    onClick={handleNavClick}
                    title={!isExpanded ? item.label : undefined}
                    className={`w-full h-11 flex items-center rounded-xl text-xs font-semibold relative select-none cursor-pointer transition-colors active:scale-[0.98] px-3 ${
                      isActive
                        ? "bg-[#d5e6db] text-[#123621] font-bold shadow-2xs border border-[#b8d1c1]"
                        : "text-[#2e543d] hover:bg-[#e0ede4] hover:text-[#123621]"
                    }`}
                  >
                    <div className="w-5 h-5 flex items-center justify-center flex-shrink-0 pointer-events-none">
                      <Icon
                        className={`w-5 h-5 flex-shrink-0 pointer-events-none transition-colors ${
                          isActive
                            ? "text-[#18492d]"
                            : "text-[#3d654f] group-hover:text-[#18492d]"
                        }`}
                      />
                    </div>

                    <span
                      className={`ml-3 text-xs font-semibold truncate whitespace-nowrap pointer-events-none transition-all duration-200 ${
                        isExpanded
                          ? "opacity-100 translate-x-0 max-w-[130px]"
                          : "opacity-0 -translate-x-2 max-w-0 overflow-hidden"
                      }`}
                    >
                      {item.label}
                    </span>
                  </Link>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div
          className={`border-t border-[#d2dfd6] bg-[#e4ede6]/40 transition-all ${
            isExpanded
              ? "px-4 py-3.5 flex items-center justify-between text-xs text-[#2e543d]"
              : "py-3 flex flex-col items-center justify-center"
          }`}
        >
          {isExpanded ? (
            <>
              <div className="overflow-hidden">
                <p className="font-bold text-[#143924] text-[11px] truncate">
                  Sector HQ Command
                </p>
                <p className="text-[10px] text-[#476e57] truncate">Edge Inference v2.1</p>
              </div>
              <span className="inline-block w-2 h-2 rounded-full bg-[#16a34a] flex-shrink-0" />
            </>
          ) : (
            <div
              className="w-8 h-8 rounded-full bg-[#dbe8df] flex items-center justify-center cursor-help"
              title="Sector HQ Command • Edge Inference v2.1"
            >
              <span className="w-2 h-2 rounded-full bg-[#16a34a]" />
            </div>
          )}
        </div>
      </aside>

      {/* Main Area: Top Bar + Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden min-w-0">
        {/* Global Top Surveillance Bar */}
        <header className="h-16 flex-shrink-0 bg-white border-b border-[#dce5df] px-4 sm:px-6 lg:px-8 flex items-center justify-between gap-3 sm:gap-4 z-20">
          {/* Left Outpost & Mesh Status + Sidebar Quick Toggle */}
          <div className="flex items-center gap-3 sm:gap-4 text-xs sm:text-sm min-w-0">
            {/* Sidebar quick mode toggle button */}
            <button
              type="button"
              onClick={() => {
                if (sidebarMode === "rail") handleModeChange("hidden");
                else if (sidebarMode === "hidden") handleModeChange("pinned");
                else handleModeChange("rail");
              }}
              className="p-2 rounded-xl text-slate-600 hover:text-[#18492d] hover:bg-[#ebf2ed] border border-slate-200 transition-all flex items-center gap-1.5 group cursor-pointer shadow-2xs flex-shrink-0"
              title={`Sidebar Mode: ${
                sidebarMode === "rail"
                  ? "Hover Rail (Click for 100% Pure Fullscreen)"
                  : sidebarMode === "hidden"
                  ? "100% Fullscreen (Click to Pin Open)"
                  : "Pinned Open (Click for Hover Rail)"
              }`}
            >
              <PanelLeft className="w-4 h-4 text-[#1e4b38] group-hover:scale-105 transition-transform" />
              <span className="hidden xl:inline text-[11px] font-semibold text-slate-600 group-hover:text-[#18492d]">
                {sidebarMode === "rail"
                  ? "Hover Rail"
                  : sidebarMode === "hidden"
                  ? "100% Fullscreen"
                  : "Pinned"}
              </span>
            </button>

            <div className="hidden md:block w-px h-4 bg-slate-200 flex-shrink-0" />

            {/* Time Stamp */}
            <HeaderClock />

          </div>

          {/* Right Search, Alerts, & Commander Profile */}
          <div className="flex items-center gap-3 sm:gap-4 flex-shrink-0">
            {/* Global Search Input */}
            <div className="relative hidden sm:block">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search cameras, zones, alert IDs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-8 pr-3 py-1.5 w-44 lg:w-64 text-xs rounded-xl border border-slate-200 bg-slate-50/70 focus:bg-white focus:outline-none focus:ring-2 focus:ring-[#1e4b38]/20 focus:border-[#1e4b38] transition-all placeholder:text-slate-400"
              />
            </div>

            {/* Notifications Bell */}
            <Link
              href="/alerts"
              className="relative p-2 rounded-xl text-slate-600 hover:bg-slate-100 transition-colors"
              title="Tactical Alerts"
            >
              <Bell className="w-4 h-4" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-rose-600 ring-2 ring-white" />
            </Link>

            <div className="w-px h-6 bg-slate-200" />

            {/* Authenticated Officer Profile & Active Duty Shift Status */}
            <div className="relative" ref={profileMenuRef}>
              <button
                type="button"
                onClick={() => setShowProfileMenu((prev) => !prev)}
                className="flex items-center gap-2.5 p-1 rounded-xl hover:bg-slate-100/80 transition-colors text-left cursor-pointer"
                title="Click to view Duty Shift details or Switch Operator"
              >
                <div className="text-right hidden sm:block">
                  <div className="text-xs font-bold text-slate-900 leading-tight">
                    {operator?.name || "Admin"}
                  </div>
                  <div className="text-[10px] font-medium text-emerald-700 flex items-center justify-end gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse" />
                    <span>{operator?.role || "Shift Commander"}</span>
                  </div>
                </div>

                <div className="relative">
                  <div
                    className="w-8 h-8 rounded-full bg-[#1e4b38] text-white flex items-center justify-center text-xs font-bold ring-2 ring-[#1e4b38]/20 shadow-xs select-none"
                  >
                    {operator?.avatarInitials || "AD"}
                  </div>
                  <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-500 ring-2 ring-white" />
                </div>

                <ChevronDown className="w-3 h-3 text-slate-400 hidden sm:block" />
              </button>

              {/* Interactive Tactical Profile Card Dropdown */}
              {showProfileMenu && (
                <div className="absolute right-0 top-full mt-2 w-72 bg-white rounded-2xl shadow-2xl border border-slate-200/80 p-3.5 z-50 text-slate-900 animate-in fade-in slide-in-from-top-2">
                  {/* Officer Header */}
                  <div className="flex items-center gap-3 pb-3 border-b border-slate-100">
                    <div className="w-10 h-10 rounded-xl bg-[#1e4b38] text-white flex items-center justify-center text-sm font-bold shadow-xs">
                      {operator?.avatarInitials || "AD"}
                    </div>
                    <div className="overflow-hidden">
                      <div className="text-xs font-bold text-slate-950 truncate">
                        {operator?.name || "Admin"}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {operator?.rank || "System Administrator"}
                      </div>
                      <div className="text-[9px] font-mono text-emerald-800 font-semibold truncate">
                        {operator?.serviceBranch}
                      </div>
                    </div>
                  </div>

                  {/* Security Clearance & Duty Sector Details */}
                  <div className="py-2.5 space-y-1.5 text-[11px] border-b border-slate-100">
                    <div className="flex items-center justify-between text-slate-600">
                      <span>Badge ID:</span>
                      <span className="font-mono font-bold text-slate-900">
                        {operator.badgeNumber}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-slate-600">
                      <span>Assigned Sector:</span>
                      <span className="font-medium text-slate-900 truncate max-w-[140px]">
                        {operator.sector}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-slate-600">
                      <span>Clearance:</span>
                      <span className="text-[10px] font-semibold text-emerald-800 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200 truncate max-w-[150px]">
                        {operator.clearanceLevel}
                      </span>
                    </div>
                  </div>

                  {/* Duty Shift Timer */}
                  <div className="py-2 px-2.5 my-2 bg-slate-50 rounded-xl border border-slate-100 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1.5 text-slate-600">
                      <Clock className="w-3.5 h-3.5 text-emerald-700" />
                      <span className="text-[11px] font-medium">Duty Shift:</span>
                    </div>
                    <DutyShiftCountdown expiresAt={operator?.expiresAt} />
                  </div>

                  {/* Quick Tactical Session Actions */}
                  <div className="space-y-1 pt-1">
                    <button
                      type="button"
                      onClick={() => {
                        logout();
                        router.push(`/login?redirect=${encodeURIComponent(pathname)}`);
                      }}
                      className="w-full py-2 px-2.5 rounded-xl hover:bg-slate-100 text-slate-700 text-xs font-semibold flex items-center justify-between transition-colors cursor-pointer"
                    >
                      <span className="flex items-center gap-2">
                        <Lock className="w-3.5 h-3.5 text-slate-500" />
                        <span>Lock Console</span>
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">Shift Pause</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => {
                        logout();
                        router.push("/login");
                      }}
                      className="w-full py-2 px-2.5 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 text-xs font-semibold flex items-center justify-between transition-colors cursor-pointer"
                    >
                      <span className="flex items-center gap-2">
                        <LogOut className="w-3.5 h-3.5 text-rose-600" />
                        <span>Sign Out / End Shift</span>
                      </span>
                      <span className="text-[10px] text-rose-500 font-mono">Terminates Session</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Scrollable Page Body - Full screen utilization */}
        <main className="flex-1 overflow-y-auto bg-[#f4f7f5] p-4 sm:p-6 lg:p-8 relative">
          <TacticalThreatToast />
          <div className="w-full max-w-[1920px] mx-auto">{children}</div>
        </main>
      </div>
    </div>
  );
}
