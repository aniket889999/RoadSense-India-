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
        // Clean light surfaces: cool near-white page, white cards, slate text.
        command: {
          bg: "#F6F7F9",
          surface: "#FFFFFF",
          elevated: "#F1F3F6",
          border: "#E4E7EC",
          subtle: "#C9CED6",
          text: "#0F172A",
          muted: "#64748B",
        },
        // Brand accent (indigo) for chrome only: header, navigation, primary actions, hero.
        brand: {
          DEFAULT: "#4F46E5",
          bright: "#6366F1",
          deep: "#4338CA",
          ink: "#1E1B4B",
          soft: "rgba(79, 70, 229, 0.08)",
        },
        // Status colors keep their meaning, darkened for contrast on white.
        radar: {
          green: "#059669",
          bright: "#16A34A",
          dim: "#BBF7D0",
          glow: "rgba(22, 163, 74, 0.18)",
        },
        accent: {
          amber: "#D97706",
          red: "#DC2626",
          cyan: "#0891B2",
          purple: "#7C3AED",
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
        radar: "0 1px 2px rgba(22, 163, 74, 0.25)",
        subtle: "0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px -12px rgba(15, 23, 42, 0.12)",
        brand: "0 8px 24px -10px rgba(79, 70, 229, 0.55)",
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
