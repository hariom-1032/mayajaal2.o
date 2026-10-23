import type { NextConfig } from "next";
import path from "path";

const backendUrl =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  "http://10.20.72.81:8000";

const nextConfig: NextConfig = {
  outputFileTracingRoot: path.resolve(__dirname),
  devIndicators: false,
  experimental: {
    devtoolSegmentExplorer: false,
  },
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
