/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        command: {
          bg: "#0B0F14",
          surface: "#111822",
          elevated: "#182230",
          border: "#1F2E40",
          subtle: "#33455E",
          text: "#F1F5F9",
          muted: "#94A3B8",
        },
        radar: {
          green: "#10B981",
          bright: "#22C55E",
          dim: "#065F46",
          glow: "rgba(16, 185, 129, 0.25)",
        },
        accent: {
          amber: "#F59E0B",
          red: "#EF4444",
          cyan: "#06B6D4",
          purple: "#8B5CF6",
        }
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
        sans: ["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"],
      },
      boxShadow: {
        radar: "0 0 15px rgba(16, 185, 129, 0.2)",
        subtle: "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
      }
    },
  },
  plugins: [],
};
