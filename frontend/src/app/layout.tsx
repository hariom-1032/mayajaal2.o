import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MAATRIX — Border Video Analytics",
  description: "Intelligent Border Video Analytics Platform • Real-time surveillance for a secure tomorrow",
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/images/logo/favicon-32x32.png", sizes: "32x32", type: "image/png" },
      { url: "/images/logo/favicon-64x64.png", sizes: "64x64", type: "image/png" },
    ],
    apple: [
      { url: "/images/logo/apple-touch-icon.png", sizes: "192x192", type: "image/png" },
    ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">{children}</body>
    </html>
  );
}
