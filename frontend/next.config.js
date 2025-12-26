/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // pdfjs ESM(pdf.mjs) 깨지는 문제 방지: legacy로 강제
    config.resolve.alias["pdfjs-dist/build/pdf.mjs"] = "pdfjs-dist/legacy/build/pdf.js";
    return config;
  },
};

module.exports = nextConfig;
