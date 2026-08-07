import { SIMILARITY_MODEL, frameKey } from '../constants.js'
import ResultCard from './ResultCard.jsx'

/** Cột kết quả tìm-ảnh-giống-ảnh, mở ra bên phải khi bấm "Tương tự" ở một frame. */
export default function SimilarPanel({
  state, selectedKeys, onToggle, onViewVideo, onZoom, onClose,
}) {
  const { source, results, loading, error } = state

  return (
    <aside className="similar">
      <div className="section-head">
        <h2>
          Tương tự <span className="tag">{SIMILARITY_MODEL}</span>
        </h2>
        <button type="button" className="btn btn-mini" onClick={onClose} title="Đóng cột này">
          ✕
        </button>
      </div>

      <div className="similar-source">
        <img src={source.image_url} alt={`Frame nguồn ${source.video_id}`} />
        <div>
          <b>{source.video_id}</b>
          <div className="card-meta">frame {source.frame_idx}</div>
        </div>
      </div>

      {error && <div className="alert alert-error">{error}</div>}
      {loading && <div className="placeholder">Đang tìm frame tương tự…</div>}

      <div className="grid grid-narrow">
        {results.map((result, i) => (
          <ResultCard
            key={`${result.video_id}/${result.keyframe_name}`}
            result={result}
            compact
            selected={selectedKeys.has(frameKey(result.video_id, result.frame_idx))}
            onToggle={onToggle}
            onViewVideo={onViewVideo}
            onZoom={onZoom && (() => onZoom(results, i))}
          />
        ))}
      </div>
    </aside>
  )
}
