import type { Config } from "tailwindcss";

export default {

	darkMode: ["class"]
		content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			fontFamily: {
				'vazir': ['Vazirmatn', 'Tahoma', 'Arial', 'sans-serif'],
				'display': ['Vazirmatn', 'Tahoma', 'Arial', 'sans-serif'],
},
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',


















