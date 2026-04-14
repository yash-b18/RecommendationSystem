/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        // Proxy /api/* → FastAPI backend.
        // INTERNAL_API_URL is used in single-Space mode (Next.js and FastAPI co-located).
        // Falls back to localhost:8001 for local development.
        source: '/api/:path*',
        destination: `${process.env.INTERNAL_API_URL || 'http://localhost:8001'}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
