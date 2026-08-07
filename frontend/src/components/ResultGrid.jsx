import { frameKey } from '../constants.js'
import ResultCard from './ResultCard.jsx'

export default function ResultGrid({
  state, selectedKeys, onToggle, onSimilar, onViewVideo, onZoom, similarityReady, activeSimilarKey,
}) {
  const { results, meta, loading, error } = state

  return (
    <section className="results">
      <div className="section-head">
        <h2>Kết quả tìm kiếm</h2>
        {meta && (
          <span className="section-meta">
            {meta.total} frame · {meta.elapsed_ms} ms · {meta.models.join(' + ')}
            {meta.playlist ? ` · ${meta.playlist}` : ''}
          </span>
        )}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {loading && <div className="placeholder">Đang tìm kiếm…</div>}

      {!loading && !error && results.length === 0 && (
        <div className="placeholder">
          Nhập mô tả phân cảnh rồi bấm <b>Tìm kiếm</b>. Tick vào ảnh để đưa frame đó vào danh sách nộp.
        </div>
      )}

      <div className="grid">
        {results.map((result, i) => {
          const key = frameKey(result.video_id, result.frame_idx)
          return (
            <ResultCard
              key={`${result.video_id}/${result.keyframe_name}`}
              result={result}
              selected={selectedKeys.has(key)}
              isSimilarSource={activeSimilarKey === key}
              onToggle={onToggle}
              onSimilar={onSimilar}
              onViewVideo={onViewVideo}
              // Truyền CẢ danh sách chứ không chỉ một ảnh: mở lightbox xong còn lướt
              // ←/→ sang kết quả kế tiếp mà không phải đóng ra bấm lại.
              onZoom={onZoom && (() => onZoom(results, i))}
              similarityReady={similarityReady}
            />
          )
        })}
      </div>
    </section>
  )
}
