/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Large video uploads and occupancy jobs go through the /api/v1 proxy below.
  experimental: { proxyTimeout: 600_000 },
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://127.0.0.1:8000/api/v1/:path*',
      },
      {
        source: '/health/:path*',
        destination: 'http://127.0.0.1:8000/health/:path*',
      },
    ];
  },
};

export default nextConfig;
