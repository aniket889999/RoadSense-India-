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
        // Neutral near-black surfaces (no blue tint), inspired by a bold editorial dark UI.
        command: {
          bg: "#111111",
          surface: "#161616",
          elevated: "#1E1E1E",
          border: "#2A2A2A",
          subtle: "#3D3D3D",
          text: "#FFFFFF",
          muted: "#A1A1A1",
        },
        // Brand accent for chrome only (header, navigation, primary actions, hero).
        brand: {
          DEFAULT: "#ED4439",
          bright: "#FF5A4E",
          deep: "#C93329",
          ink: "#161616",
          soft: "rgba(237, 68, 57, 0.14)",
        },
        // Status colors keep their meaning (ready/good, vacant bays, etc.).
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
        sans: ["\"SF Pro Display\"", "\"SF Pro Text\"", "system-ui", "-apple-system", "BlinkMacSystemFont", "\"Segoe UI\"", "Roboto", "\"Helvetica Neue\"", "sans-serif"],
      },
      borderRadius: {
        "4xl": "2rem",
      },
      boxShadow: {
        radar: "0 0 15px rgba(16, 185, 129, 0.2)",
        subtle: "0 4px 20px -2px rgba(0, 0, 0, 0.5)",
        brand: "0 12px 40px -12px rgba(237, 68, 57, 0.55)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(14px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        rise: "rise 700ms cubic-bezier(0.22, 1, 0.36, 1) both",
      },
    },
  },
  plugins: [],
};
