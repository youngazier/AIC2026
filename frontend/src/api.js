// Lớp gọi API duy nhất. Mọi component gọi backend qua đây, không component nào tự
// fetch() — nhờ vậy đổi đường dẫn hay thêm header chỉ phải sửa một chỗ.
//
// Đường dẫn TƯƠNG ĐỐI: lúc dev thì vite.config.js proxy /api sang FastAPI, lúc deploy
// thì frontend nằm cùng origin với backend. Cả hai trường hợp đều không phải sửa gì.
const BASE = '/api/v1'

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
      ...options,
    })
  } catch (cause) {
    // fetch() chỉ ném khi không nối được tới server — phân biệt rõ với lỗi 4xx/5xx để
    // người dùng biết là phải bật backend chứ không phải sửa câu truy vấn.
    throw new ApiError('Không kết nối được backend. Kiểm tra đã chạy ./scripts/run_backend.sh chưa.', 0)
  }

  if (!response.ok) {
    let detail = `Lỗi ${response.status}`
    try {
      const body = await response.json()
      // FastAPI trả detail là chuỗi (HTTPException) hoặc mảng (lỗi validate Pydantic).
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((e) => e.msg).join('; ')
    } catch {
      /* body không phải JSON -> giữ nguyên thông báo mặc định */
    }
    throw new ApiError(detail, response.status)
  }
  return response.json()
}

const post = (path, body) => request(path, { method: 'POST', body: JSON.stringify(body) })

export const api = {
  health: () => request('/health'),
  models: () => request('/models'),
  playlists: () => request('/playlists'),

  searchKis: (payload) => post('/search/kis', payload),
  searchSimilar: (payload) => post('/search/similar', payload),

  videoInfo: (videoId) => request(`/videos/${videoId}`),

  createSubmission: (payload) => post('/submissions', payload),
  listSubmissions: () => request('/submissions'),
}

export const videoStreamUrl = (videoId) => `${BASE}/videos/${videoId}/stream`
export { ApiError }
