/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'table-empty': '#10B981',
        'table-seated': '#3B82F6',
        'table-ordering': '#F59E0B',
        'table-waiting': '#8B5CF6',
        'table-served': '#06B6D4',
        'table-paying': '#EF4444',
      },
    },
  },
  plugins: [],
}
