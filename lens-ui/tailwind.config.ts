import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0A0A0F",
        surface: "#12121A",
        border: "#1E1E2E",
        forensic: {
          blue: "#3B82F6",
          amber: "#F59E0B",
          danger: "#EF4444",
          green: "#10B981",
          deepred: "#991B1B",
        }
      },
      fontFamily: {
        sans: ["var(--font-inter)", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      }
    },
  },
  plugins: [],
};
export default config;
