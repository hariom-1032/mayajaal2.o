"use client";

import { useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  User,
  Lock,
  ArrowRight,
  Eye,
  EyeOff,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Clock,
} from "lucide-react";
import { EmblemIndia } from "@/components/landing/EmblemIndia";
import {
  useAuth,
  validateCredentials,
  OperatorUser,
} from "@/lib/authStore";

function LoginFormCard() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const rawRedirect = searchParams.get("redirect");
  const redirectTarget = !rawRedirect || rawRedirect === "/gis-map" || rawRedirect === "/login" ? "/dashboard" : rawRedirect;
  const sessionReason = searchParams.get("reason");

  const { login, isAuthenticated } = useAuth();

  // If already logged in, navigate straight to destination
  useEffect(() => {
    if (isAuthenticated) {
      router.replace(redirectTarget);
    }
  }, [isAuthenticated, redirectTarget, router]);

  // Credentials state
  const [username, setUsername] = useState("BSF-9482-KR");
  const [password, setPassword] = useState("MAATRIX2026");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const executeLogin = (user: OperatorUser) => {
    setIsLoading(true);
    setErrorMessage(null);

    // Call login with rememberMe configuration
    login(user, rememberMe);

    // Small delay to ensure cookies are written before navigating
    setTimeout(() => {
      window.location.href = redirectTarget;
    }, 150);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    const validation = validateCredentials(username, password);
    if (!validation.success || !validation.user) {
      setErrorMessage(
        validation.error || "Authentication failed. Please verify Badge ID and Password."
      );
      return;
    }

    executeLogin(validation.user);
  };


  return (
    <div className="w-full max-w-[440px] mx-auto">
      {/* Clean White Card matching reference design */}
      <div className="bg-white/95 backdrop-blur-md rounded-[32px] p-8 sm:p-10 shadow-2xl shadow-black/30 border border-white/60 text-slate-900 transition-all">
        {/* MAATRIX Emblem & Title */}
        <div className="flex flex-col items-center mb-5">
          <div className="w-16 h-16 rounded-2xl bg-emerald-50/80 border border-emerald-200/80 p-2.5 shadow-sm flex items-center justify-center mb-3 group">
            <img
              src="/images/logo/maatrix-emblem.png"
              alt="MAATRIX Logo"
              className="w-full h-full object-contain"
            />
          </div>
          <h1 className="text-2xl font-black text-slate-900 tracking-wider">
            MAATRIX
          </h1>
          <p className="text-[11px] font-bold text-[#1a4a32] tracking-[0.18em] uppercase mt-0.5">
            Border Video Analytics
          </p>
          <span className="text-xs text-slate-500 mt-1">
            Enforced Security Clearance Portal
          </span>
        </div>

        {/* Expired Session Notice */}
        {sessionReason === "expired" && (
          <div className="mb-4 p-3 rounded-2xl bg-amber-50 border border-amber-200/80 text-amber-900 text-xs flex items-start gap-2.5 shadow-xs animate-in fade-in slide-in-from-top-2">
            <Clock className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Duty Shift Expired: </span>
              Your operational session timed out. Re-authenticate to access the tactical surveillance console.
            </div>
          </div>
        )}

        {/* Authentication Error Banner */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-2xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5 shadow-xs animate-in fade-in slide-in-from-top-2">
            <ShieldAlert className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
            <div>
              <span className="font-bold">Access Denied: </span>
              {errorMessage}
            </div>
          </div>
        )}



        {/* Credential Form */}
        <form onSubmit={handleSubmit} className="space-y-3.5">
          {/* Badge ID Field */}
          <div>
            <label className="block text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-1 px-1">
              Security Badge ID / Officer Code
            </label>
            <div className="relative">
              <User className="w-4 h-4 text-slate-500 absolute left-4 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setErrorMessage(null);
                }}
                placeholder="e.g. BSF-9482-KR or ITBP-4108-AS"
                required
                className="w-full pl-11 pr-4 py-2.5 bg-[#f3f4f6] hover:bg-[#ebeef1] focus:bg-white rounded-2xl border border-transparent focus:border-emerald-700/30 focus:ring-2 focus:ring-emerald-800/15 text-slate-900 text-sm font-medium placeholder:text-slate-400 focus:outline-none transition-all"
              />
            </div>
          </div>

          {/* Password Field */}
          <div>
            <label className="block text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-1 px-1">
              Clearance Code / Password
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-500 absolute left-4 top-1/2 -translate-y-1/2" />
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setErrorMessage(null);
                }}
                placeholder="Clearance Code"
                required
                className="w-full pl-11 pr-11 py-2.5 bg-[#f3f4f6] hover:bg-[#ebeef1] focus:bg-white rounded-2xl border border-transparent focus:border-emerald-700/30 focus:ring-2 focus:ring-emerald-800/15 text-slate-900 text-sm font-medium placeholder:text-slate-400 focus:outline-none transition-all"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-900 transition-colors"
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? (
                  <EyeOff className="w-4 h-4" />
                ) : (
                  <Eye className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>

          {/* Login Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full mt-2 py-3.5 px-6 rounded-2xl bg-[#1e4b38] hover:bg-[#163a2b] active:scale-[0.99] text-white text-sm font-semibold tracking-wide shadow-md shadow-emerald-950/20 transition-all duration-150 flex items-center justify-center gap-2 cursor-pointer disabled:opacity-70"
          >
            {isLoading ? (
              <span className="w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                <ShieldCheck className="w-4 h-4 text-emerald-300" />
                <span>Verify & Enter Console</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>

          {/* Shift Duration & Reset Options */}
          <div className="flex items-center justify-between pt-2 text-xs">
            <label className="flex items-center gap-2 text-slate-600 hover:text-slate-800 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="w-4 h-4 rounded border-slate-300 text-[#1c3a28] focus:ring-[#1c3a28]"
              />
              <span>Keep Shift Active (24h)</span>
            </label>

            <button
              type="button"
              onClick={() => {
                setUsername("BSF-9482-KR");
                setPassword("MAATRIX2026");
                setErrorMessage(null);
              }}
              className="text-slate-600 hover:text-slate-900 underline transition-colors"
            >
              Reset to Admin
            </button>
          </div>
        </form>

        {/* Security Notice Footer */}
        <div className="mt-5 pt-3.5 border-t border-slate-100 flex items-center justify-center gap-2 text-[10px] text-slate-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <span>Restricted Military & Border Surveillance Network</span>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <div className="relative min-h-screen w-full flex flex-col justify-between overflow-hidden select-none">
      {/* 1. Full Himalayan Mountain Background */}
      <div className="absolute inset-0 z-0">
        <img
          src="/images/himalayan-border-hero.jpg"
          alt="Himalayan Border Background"
          className="w-full h-full object-cover object-[center_35%]"
        />
        {/* Soft atmospheric gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/25 via-black/15 to-black/50" />
      </div>

      {/* 2. Top Header with Lion Emblem & Government Title */}
      <header className="relative z-10 pt-8 sm:pt-10 px-6 flex flex-col items-center text-center">
        <Link href="/" className="flex flex-col items-center group">
          {/* Emblem of India */}
          <EmblemIndia className="w-12 h-14 text-slate-900 drop-shadow-xs" />
          <span className="text-xs font-semibold text-slate-900 tracking-wide mt-2">
            Government of India
          </span>
          <span className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
            Border Surveillance Portal
          </span>
        </Link>
      </header>

      {/* 3. Center Login Card */}
      <main className="relative z-10 px-4 py-6 flex items-center justify-center">
        <Suspense
          fallback={
            <div className="w-full max-w-[440px] h-[360px] bg-white/90 rounded-[32px] animate-pulse" />
          }
        >
          <LoginFormCard />
        </Suspense>
      </main>

      {/* 4. Bottom Motto & Indian Tricolor Bar */}
      <footer className="relative z-10 pb-6 sm:pb-8 px-8 sm:px-14 flex flex-col items-start">
        <div className="space-y-0.5">
          <div className="text-[11px] sm:text-xs font-bold tracking-[0.22em] text-white/95 uppercase drop-shadow-md">
            SECURE BORDERS
          </div>
          <div className="text-[11px] sm:text-xs font-bold tracking-[0.22em] text-white/95 uppercase drop-shadow-md">
            SAFER TOMORROW
          </div>
        </div>

        {/* Indian Tricolor Bar (Saffron, White, Green) */}
        <div
          className="h-[3px] w-40 mt-2 rounded-full shadow-sm"
          style={{
            background: "linear-gradient(to right, #ff9933 0%, #ffffff 50%, #138808 100%)",
          }}
        />
      </footer>
    </div>
  );
}
