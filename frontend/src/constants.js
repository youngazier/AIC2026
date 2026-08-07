// Ba dạng truy vấn của vòng sơ tuyển (docs/Thong-tin-vong-So-tuyen-AIC2026.pdf mục 1).
export const QUERY_TYPES = [
  {
    id: 'kis',
    label: 'KIS',
    title: 'Textual Known-Item Search',
    hint: 'Mô tả một sự kiện, tìm đúng khung hình của nó. Nộp: video_id, frame_id',
  },
  {
    id: 'trake',
    label: 'TRAKE',
    title: 'Temporal Retrieval and Alignment of Key Events',
    hint: 'Mô tả chuỗi sự kiện, ngăn cách bằng dấu chấm. Nộp: video_id, frame_1, ..., frame_n',
  },
  {
    id: 'qna',
    label: 'Q&A',
    title: 'Hỏi — Đáp trên nội dung video',
    hint: 'Tìm khoảnh khắc chứa câu trả lời. Nộp: video_id, frame_id, answer',
  },
]

// Lựa chọn model trên UI. 'hybrid' KHÔNG phải một model — nó gửi nhiều model lên
// backend rồi backend trộn kết quả bằng RRF, nên `models` ở đây là một mảng.
export const MODEL_CHOICES = [
  { id: 'siglip2', label: 'SigLIP 2', models: ['siglip2'], hint: 'Text encoder mạnh, mặc định cho KIS' },
  { id: 'beit3', label: 'BEiT-3', models: ['beit3'], hint: 'Text encoder thứ hai, khác góc nhìn với SigLIP 2' },
  { id: 'hybrid', label: 'Hybrid', models: ['siglip2', 'beit3'], hint: 'Trộn cả hai model bằng Reciprocal Rank Fusion' },
]

// Model dùng cho tìm-ảnh-giống-ảnh. DINOv3 là model thị giác tự giám sát, không encode
// được text nhưng so ảnh với ảnh thì mạnh nhất trong ba model.
export const SIMILARITY_MODEL = 'dinov3'

// Trần cứng của BTC: mỗi truy vấn nộp tối đa 100 câu trả lời.
export const MAX_SUBMISSION_ROWS = 100

export const frameKey = (videoId, frameIdx) => `${videoId}#${frameIdx}`

/** Tách câu truy vấn TRAKE thành danh sách sự kiện theo dấu chấm. */
export const splitTrakeEvents = (text) =>
  text
    .split('.')
    .map((part) => part.trim())
    .filter(Boolean)

/** Giây -> "mm:ss" (hoặc "h:mm:ss" với video dài). */
export function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '--:--'
  const total = Math.floor(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(s).padStart(2, '0')
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`
}
