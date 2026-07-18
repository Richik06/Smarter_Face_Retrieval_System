import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/app": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/process-event": "http://127.0.0.1:8000",
      "/process-event-from-drive": "http://127.0.0.1:8000",
      "/search-face": "http://127.0.0.1:8000",
      "/get-embedding": "http://127.0.0.1:8000",
      "/recluster-event": "http://127.0.0.1:8000",
      "/event": "http://127.0.0.1:8000",
      "/images": "http://127.0.0.1:8000",
    }
  }
});
