import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DeepReads — Explainable AI Recommendations",
  description:
    "Production-grade explainable multi-stage e-commerce recommendation system using Amazon Reviews.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="min-h-screen bg-bg-base text-text-primary font-body antialiased">
        {children}
      </body>
    </html>
  );
}
