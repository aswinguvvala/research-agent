/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        'dark-blue': {
          900: '#0a0f1f',
          800: '#121832',
          700: '#1a2248',
          600: '#222c5d',
          500: '#3b4c8a',
          400: '#546ac5',
          300: '#7d91e1',
          200: '#a6b8fa',
          100: '#d0d9ff',
        },
        'light-blue': {
          100: '#f0f5ff',
          200: '#dbe7ff',
          300: '#c6d9ff',
          400: '#b2cbff',
          500: '#9ebcff',
          600: '#8aaeff',
          700: '#75a0ff',
          800: '#6192ff',
          900: '#4c84ff',
        },
        'accent-purple': '#9b59b6',
        'accent-pink': '#e91e63',
        'accent-teal': '#1abc9c',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'soft-glow': '0 0 20px rgba(125, 145, 225, 0.1)',
        'hard-glow': '0 0 25px rgba(176, 136, 255, 0.3)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out forwards',
        'slide-up': 'slideUp 0.6s ease-out forwards',
        'subtle-pulse': 'subtlePulse 3s ease-in-out infinite',
        'gradient-shift': 'gradientShift 15s ease infinite',
        'particle-float': 'particleFloat 20s linear infinite',
        'network-pulse': 'networkPulse 8s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        slideUp: {
          '0%': { opacity: 0, transform: 'translateY(20px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        subtlePulse: {
          '0%, 100%': { opacity: 1, transform: 'scale(1)' },
          '50%': { opacity: 0.95, transform: 'scale(1.02)' },
        },
        gradientShift: {
          '0%, 100%': { 'background-position': '0% 50%' },
          '50%': { 'background-position': '100% 50%' },
        },
        particleFloat: {
          '0%': { transform: 'translate(0, 0) rotate(0deg)' },
          '33%': { transform: 'translate(30px, -30px) rotate(120deg)' },
          '66%': { transform: 'translate(-20px, 20px) rotate(240deg)' },
          '100%': { transform: 'translate(0, 0) rotate(360deg)' },
        },
        networkPulse: {
          '0%, 100%': { opacity: 0.05 },
          '50%': { opacity: 0.15 },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          'from': { 'box-shadow': '0 0 20px rgba(129, 140, 248, 0.3)' },
          'to': { 'box-shadow': '0 0 30px rgba(129, 140, 248, 0.6)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
