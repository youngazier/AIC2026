import { formatTime } from '../constants.js'

/**
 * Một ô kết quả. Dùng chung cho lưới kết quả chính và cột "tương tự" bên phải để hai
 * nơi không bao giờ lệch nhau về cách tick chọn hay cách hiển thị frame_idx.
 */
export default function ResultCard({
  result, selected, onToggle, onSimilar, onViewVideo, onZoom,
  similarityReady = false, isSimilarSource = false, compact = false,
}) {
  const { video_id: videoId, keyframe_name: keyframeName, frame_idx: frameIdx } = result

  return (
    <figure
      className={`card${selected ? ' selected' : ''}${isSimilarSource ? ' source' : ''}${compact ? ' compact' : ''}`}
    >
      <div className="card-thumb">
        {/* Bấm vào ảnh = tick chọn: thao tác nhiều nhất lúc thi, phải là vùng bấm to nhất. */}
        {/* fetchpriority="low": trình duyệt chỉ mở 6 kết nối cho mỗi origin, và một lượt
            tìm kiếm đổ về hàng chục thumbnail. Không hạ ưu tiên thì request /stream của
            video xếp hàng SAU đám ảnh này — đúng triệu chứng "bấm ▶ Video mà phải đợi
            ảnh load xong mới thấy gì". Hạ ưu tiên xong, video được cấp socket trước.

            `fetchpriority` viết THƯỜNG: React 18 chưa biết prop `fetchPriority` nên nó
            vứt thuộc tính đi kèm một cảnh báo console — tức là việc hạ ưu tiên không hề
            xảy ra. Dạng viết thường thì React cho đi thẳng ra DOM. */}
        <img
          src={result.image_url}
          alt={`${videoId} / frame ${frameIdx}`}
          loading="lazy"
          decoding="async"
          fetchpriority="low"
          onClick={() => onToggle(videoId, frameIdx)}
          // Nháy đúp = phóng to. Hai cú nhấp đơn vẫn bắn ra trước, nhưng chúng tick rồi
          // bỏ tick chính frame đó nên tổng lại không đổi gì — an toàn để chồng lên.
          onDoubleClick={onZoom}
        />
        <input
          type="checkbox"
          className="card-check"
          checked={selected}
          onChange={() => onToggle(videoId, frameIdx)}
          title="Chọn frame này để đưa vào file nộp"
          aria-label={`Chọn ${videoId} frame ${frameIdx}`}
        />
        {onZoom && (
          <button
            type="button"
            className="card-zoom"
            title="Phóng to để soi chi tiết (hoặc nháy đúp vào ảnh)"
            aria-label={`Phóng to ${videoId} frame ${frameIdx}`}
            onClick={onZoom}
          >
            ⛶
          </button>
        )}
        <span className="card-rank">#{result.rank}</span>
        <span className="card-score">{result.score.toFixed(3)}</span>
      </div>

      <figcaption className="card-body">
        <div className="card-id" title={result.title || videoId}>
          <b>{videoId}</b>
          <span className="card-frame">frame {frameIdx}</span>
        </div>
        <div className="card-meta">
          <span>{formatTime(result.timestamp)}</span>
          <span className="card-kf">kf {keyframeName}</span>
        </div>
        <div className="card-actions">
          {onSimilar && (
            <button
              type="button"
              className="btn btn-mini"
              disabled={!similarityReady}
              title={
                similarityReady
                  ? 'Tìm các frame giống frame này (DINOv3)'
                  : 'Chưa nạp index DINOv3 nên không tìm ảnh giống ảnh được'
              }
              onClick={() => onSimilar(result)}
            >
              ◎ Tương tự
            </button>
          )}
          <button
            type="button"
            className="btn btn-mini"
            title="Xem video đúng tại frame này"
            onClick={() => onViewVideo(result)}
          >
            ▶ Video
          </button>
        </div>
      </figcaption>
    </figure>
  )
}
