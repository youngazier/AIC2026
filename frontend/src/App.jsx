import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import { MAX_SUBMISSION_ROWS, MODEL_CHOICES, SIMILARITY_MODEL, frameKey } from './constants.js'
import SearchPanel from './components/SearchPanel.jsx'
import ResultGrid from './components/ResultGrid.jsx'
import SimilarPanel from './components/SimilarPanel.jsx'
import SubmissionPanel from './components/SubmissionPanel.jsx'
import VideoModal from './components/VideoModal.jsx'
import ImageLightbox from './components/ImageLightbox.jsx'

const EMPTY_SEARCH = { results: [], meta: null, loading: false, error: null }
const EMPTY_SIMILAR = { source: null, results: [], loading: false, error: null }

export default function App() {
  // --- Thông tin backend: model nào đã nạp, playlist nào có index ---
  const [backend, setBackend] = useState({ models: [], playlists: [], health: null, error: null })

  // --- Form tìm kiếm ---
  const [queryType, setQueryType] = useState('kis')
  const [modelChoice, setModelChoice] = useState('siglip2')
  const [textQuery, setTextQuery] = useState('')
  const [answer, setAnswer] = useState('')
  const [topk, setTopk] = useState(100)
  const [playlist, setPlaylist] = useState('')

  const [search, setSearch] = useState(EMPTY_SEARCH)
  const [similar, setSimilar] = useState(EMPTY_SIMILAR)
  const [videoTarget, setVideoTarget] = useState(null)

  // Lightbox giữ CẢ danh sách đang xem + vị trí hiện tại, không chỉ một ảnh, để lướt
  // ←/→ qua các kết quả ở cỡ lớn. `null` là đang đóng.
  const [zoom, setZoom] = useState(null)
  const openZoom = useCallback((items, index) => setZoom({ items, index }), [])

  // --- Danh sách nộp. THỨ TỰ TRONG MẢNG LÀ THỨ HẠNG NỘP ---
  // Final Score = trung bình R@k với k ∈ {1,5,20,50,100}, nên frame tự tin nhất phải
  // nằm ở đầu mảng. Vì vậy mọi thao tác thêm đều nối vào CUỐI (thứ tự người dùng tick)
  // và người dùng chỉnh lại thứ hạng bằng nút lên/xuống ở bảng nộp.
  const [rows, setRows] = useState([])
  const [notice, setNotice] = useState(null)
  const rowId = useRef(0)

  useEffect(() => {
    Promise.all([api.models(), api.playlists(), api.health()])
      .then(([models, playlists, health]) => setBackend({ models, playlists, health, error: null }))
      .catch((err) => setBackend((prev) => ({ ...prev, error: err.message })))
  }, [])

  // --- Model nào thực sự dùng được ------------------------------------------ //
  // Đọc từ backend chứ không hard-code: BEiT-3 chỉ chọn được sau khi đã build index
  // và nạp encoder cho nó. Hiện một lựa chọn rồi để nó báo lỗi khi bấm là tệ hơn nhiều
  // so với việc làm mờ nó kèm lý do.
  const availability = useMemo(() => {
    const textModels = backend.models.filter((m) => m.loaded && m.supports_text).map((m) => m.name)
    return Object.fromEntries(
      MODEL_CHOICES.map((choice) => [choice.id, choice.models.every((m) => textModels.includes(m))]),
    )
  }, [backend.models])

  const similarityReady = backend.models.some((m) => m.name === SIMILARITY_MODEL && m.loaded)

  // Nếu lựa chọn hiện tại không dùng được, tự nhảy sang lựa chọn đầu tiên dùng được.
  useEffect(() => {
    if (backend.models.length === 0 || availability[modelChoice]) return
    const fallback = MODEL_CHOICES.find((choice) => availability[choice.id])
    if (fallback) setModelChoice(fallback.id)
  }, [availability, backend.models.length, modelChoice])

  // --- Tập frame đang được chọn (suy ra từ `rows`, không giữ state song song) --- //
  // Một nguồn sự thật duy nhất: bỏ tick ở lưới kết quả và xóa dòng ở bảng nộp là cùng
  // một hành động, nên không thể lệch nhau.
  const selectedKeys = useMemo(() => {
    const keys = new Set()
    rows.forEach((row) => row.frames.forEach((frame) => keys.add(frameKey(row.video_id, frame))))
    return keys
  }, [rows])

  // --- Tìm kiếm ------------------------------------------------------------- //
  const runSearch = useCallback(
    async (query) => {
      const trimmed = (query ?? '').trim()
      if (!trimmed) {
        setSearch({ ...EMPTY_SEARCH, error: 'Chưa nhập câu truy vấn.' })
        return
      }
      const choice = MODEL_CHOICES.find((c) => c.id === modelChoice)
      setSearch((prev) => ({ ...prev, loading: true, error: null }))
      try {
        const data = await api.searchKis({
          text_query: trimmed,
          models: choice.models,
          topk: Number(topk) || 100,
          playlist: playlist || null,
        })
        setSearch({ results: data.results, meta: data, loading: false, error: null })
      } catch (err) {
        setSearch({ results: [], meta: null, loading: false, error: err.message })
      }
    },
    [modelChoice, topk, playlist],
  )

  const runSimilar = useCallback(
    async (result) => {
      setSimilar({ source: result, results: [], loading: true, error: null })
      try {
        const data = await api.searchSimilar({
          video_id: result.video_id,
          keyframe_name: result.keyframe_name,
          models: [SIMILARITY_MODEL],
          topk: Number(topk) || 100,
          playlist: playlist || null,
        })
        setSimilar({ source: result, results: data.results, loading: false, error: null })
      } catch (err) {
        setSimilar({ source: result, results: [], loading: false, error: err.message })
      }
    },
    [topk, playlist],
  )

  // --- Chọn / bỏ chọn frame cho danh sách nộp ------------------------------- //
  const toggleFrame = useCallback(
    (videoId, frameIdx) => {
      setNotice(null)
      setRows((prev) => {
        const isSelected = prev.some(
          (row) => row.video_id === videoId && row.frames.includes(frameIdx),
        )
        if (isSelected) {
          return prev
            .map((row) =>
              row.video_id === videoId
                ? { ...row, frames: row.frames.filter((f) => f !== frameIdx) }
                : row,
            )
            .filter((row) => row.frames.length > 0)
        }

        // TRAKE: mỗi dòng là MỘT video kèm chuỗi frame theo thứ tự sự kiện, nên frame
        // mới của video đã có sẽ nối vào đúng dòng đó thay vì tạo dòng mới.
        if (queryType === 'trake') {
          const existing = prev.find((row) => row.video_id === videoId)
          if (existing) {
            return prev.map((row) =>
              row === existing ? { ...row, frames: [...row.frames, frameIdx] } : row,
            )
          }
        }

        if (prev.length >= MAX_SUBMISSION_ROWS) {
          setNotice(
            `Đã đủ ${MAX_SUBMISSION_ROWS} dòng — trần cứng của BTC. Bỏ bớt dòng kém tự tin ở cuối danh sách trước khi thêm mới.`,
          )
          return prev
        }
        rowId.current += 1
        return [...prev, { id: rowId.current, video_id: videoId, frames: [frameIdx], answer: null }]
      })
    },
    [queryType],
  )

  // Đổi dạng truy vấn -> cấu trúc dòng nộp khác hẳn (số frame mỗi dòng, có/không có
  // answer), nên danh sách cũ không còn nghĩa. Hỏi trước khi xóa vì đó là công sức thật.
  const changeQueryType = useCallback(
    (next) => {
      if (next === queryType) return
      if (rows.length > 0) {
        const ok = window.confirm(
          `Đổi sang ${next.toUpperCase()} sẽ xóa ${rows.length} dòng đang có trong danh sách nộp (cấu trúc dòng khác nhau). Tiếp tục?`,
        )
        if (!ok) return
        setRows([])
      }
      setNotice(null)
      setQueryType(next)
    },
    [queryType, rows.length],
  )

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <strong>AIC 2026</strong>
          <span>Video Retrieval</span>
        </div>
        <BackendStatus backend={backend} />
      </header>

      <SearchPanel
        queryType={queryType}
        onQueryType={changeQueryType}
        modelChoice={modelChoice}
        onModelChoice={setModelChoice}
        availability={availability}
        textQuery={textQuery}
        onTextQuery={setTextQuery}
        answer={answer}
        onAnswer={setAnswer}
        topk={topk}
        onTopk={setTopk}
        playlist={playlist}
        onPlaylist={setPlaylist}
        playlists={backend.playlists}
        loading={search.loading}
        onSearch={runSearch}
      />

      <main className="workspace">
        <ResultGrid
          state={search}
          selectedKeys={selectedKeys}
          onToggle={toggleFrame}
          onSimilar={runSimilar}
          onViewVideo={setVideoTarget}
          onZoom={openZoom}
          similarityReady={similarityReady}
          activeSimilarKey={
            similar.source ? frameKey(similar.source.video_id, similar.source.frame_idx) : null
          }
        />
        {similar.source && (
          <SimilarPanel
            state={similar}
            selectedKeys={selectedKeys}
            onToggle={toggleFrame}
            onViewVideo={setVideoTarget}
            onZoom={openZoom}
            onClose={() => setSimilar(EMPTY_SIMILAR)}
          />
        )}
      </main>

      <SubmissionPanel
        queryType={queryType}
        rows={rows}
        setRows={setRows}
        globalAnswer={answer}
        notice={notice}
        setNotice={setNotice}
        onViewVideo={setVideoTarget}
      />

      {videoTarget && (
        // `key` theo target: mở video/frame khác là dựng lại modal từ đầu, nên state
        // "đã tua tới frame đích chưa" không sót lại từ lần xem trước.
        <VideoModal
          key={`${videoTarget.video_id}#${videoTarget.frame_idx}`}
          target={videoTarget}
          selectedKeys={selectedKeys}
          onToggle={toggleFrame}
          onClose={() => setVideoTarget(null)}
        />
      )}

      {zoom && (
        <ImageLightbox
          items={zoom.items.map((r) => ({
            src: r.image_url,
            videoId: r.video_id,
            frameIdx: r.frame_idx,
          }))}
          index={zoom.index}
          onIndex={(index) => setZoom((prev) => ({ ...prev, index }))}
          isSelected={selectedKeys.has(
            frameKey(zoom.items[zoom.index].video_id, zoom.items[zoom.index].frame_idx),
          )}
          onToggle={toggleFrame}
          onClose={() => setZoom(null)}
        />
      )}
    </div>
  )
}

function BackendStatus({ backend }) {
  if (backend.error) return <div className="status status-bad">⚠ {backend.error}</div>
  if (!backend.health) return <div className="status">Đang kết nối backend…</div>

  const { status, n_videos: videos, n_keyframes: keyframes } = backend.health
  const loaded = backend.models.filter((m) => m.loaded).map((m) => m.name)
  return (
    <div className={`status ${status === 'ok' ? 'status-ok' : 'status-warn'}`}>
      <span>{status === 'ok' ? '● sẵn sàng' : '● thiếu tài nguyên'}</span>
      <span>{videos.toLocaleString()} video</span>
      <span>{keyframes.toLocaleString()} keyframe</span>
      <span>{loaded.length ? loaded.join(' + ') : 'chưa nạp model'}</span>
    </div>
  )
}
