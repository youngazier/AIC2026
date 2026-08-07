import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api sang backend FastAPI: nhờ vậy mọi URL trong code là đường dẫn TƯƠNG ĐỐI
// ('/api/v1/...'), giống hệt lúc chạy production khi frontend được serve cùng origin.
// Không có proxy thì ảnh keyframe và video phải gắn host tuyệt đối vào từng URL, và
// đổi máy chạy backend là phải sửa code.
//
//   BACKEND_URL=http://192.168.1.50:8000 npm run dev   # backend chạy máy khác
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,          // cho phép truy cập từ máy khác trong mạng LAN lúc thi
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.BACKEND_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
