import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Xem ảnh phóng to để soi chi tiết nhỏ (biển số, chữ trên bảng hiệu, số áo…).
 *
 * Ô keyframe trong lưới rộng ~210px trong khi ảnh gốc là 1280px — tức đang xem ở 16%
 * kích thước thật. Rất nhiều manh mối phân biệt hai cảnh gần giống nhau chỉ đọc được ở
 * 100% trở lên, nên đây không phải tiện ích trang trí mà là công cụ chấm bài.
 *
 * Điều hướng ←/→ đi thẳng sang kết quả kế tiếp mà KHÔNG đóng lightbox: lúc thi việc
 * cần làm là lướt nhanh hai chục ứng viên ở cỡ lớn, đóng-mở từng cái là quá chậm.
 */
export default function ImageLightbox({ items, index, onIndex, onClose, isSelected, onToggle }) {
  const item = items[index]

  // scale/tx/ty gộp chung một state: mỗi lần zoom phải sửa cả ba cùng lúc (xem
  // `zoomAt`), tách ra là React dựng khung trung gian với scale mới nhưng offset cũ —
  // ảnh giật nảy một nhịp.
  const [view, setView] = useState({ scale: 1, tx: 0, ty: 0 })
  const stageRef = useRef(null)
  const dragRef = useRef(null)

  const reset = useCallback(() => setView({ scale: 1, tx: 0, ty: 0 }), [])

  // Đổi ảnh thì về lại khung nhìn mặc định — giữ nguyên mức zoom của ảnh trước sẽ ném
  // người dùng vào một góc ngẫu nhiên của ảnh mới.
  useEffect(reset, [index, reset])

  /**
   * Phóng to quanh một điểm neo, giữ đúng điểm dưới con trỏ đứng yên.
   *
   * Với transform-origin ở tâm: screen = x * s + t (x là toạ độ trong ảnh so với tâm).
   * Muốn điểm `p` không dịch chuyển thì x phải không đổi, suy ra:
   *     t' = p − (p − t) · s'/s
   * Không có bù trừ này thì cuộn để zoom sẽ trôi về tâm ảnh, và soi một góc là bất khả.
   */
  const zoomAt = useCallback((factor, px, py) => {
    setView((prev) => {
      const scale = Math.min(12, Math.max(1, prev.scale * factor))
      if (scale === prev.scale) return prev
      // Về đúng 100% thì kéo ảnh lại giữa khung. Không có nhánh này, ảnh giữ nguyên độ
      // lệch tích luỹ lúc phóng to và nằm chệch hẳn ra rìa dù đã thu nhỏ hết cỡ.
      if (scale === 1) return { scale: 1, tx: 0, ty: 0 }
      const ratio = scale / prev.scale
      return {
        scale,
        tx: px - (px - prev.tx) * ratio,
        ty: py - (py - prev.ty) * ratio,
      }
    })
  }, [])

  // Đăng ký `wheel` bằng tay với { passive: false }: React gắn onWheel ở chế độ passive
  // nên preventDefault() trong đó không có tác dụng, và trang sẽ cuộn theo sau lưng.
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return undefined
    const onWheel = (event) => {
      event.preventDefault()
      const box = stage.getBoundingClientRect()
      zoomAt(
        event.deltaY < 0 ? 1.25 : 1 / 1.25,
        event.clientX - box.left - box.width / 2,
        event.clientY - box.top - box.height / 2,
      )
    }
    stage.addEventListener('wheel', onWheel, { passive: false })
    return () => stage.removeEventListener('wheel', onWheel)
  }, [zoomAt])

  const onPointerDown = (event) => {
    if (view.scale === 1) return
    event.currentTarget.setPointerCapture(event.pointerId)
    dragRef.current = { x: event.clientX - view.tx, y: event.clientY - view.ty }
  }

  const onPointerMove = (event) => {
    if (!dragRef.current) return
    setView((prev) => ({
      ...prev,
      tx: event.clientX - dragRef.current.x,
      ty: event.clientY - dragRef.current.y,
    }))
  }

  const endDrag = () => {
    dragRef.current = null
  }

  useEffect(() => {
    const onKeyDown = (event) => {
      switch (event.key) {
        case 'Escape':
          return onClose()
        case 'ArrowLeft':
          event.preventDefault()
          return onIndex && onIndex((index - 1 + items.length) % items.length)
        case 'ArrowRight':
          event.preventDefault()
          return onIndex && onIndex((index + 1) % items.length)
        case '+':
        case '=':
          event.preventDefault()
          return zoomAt(1.4, 0, 0)
        case '-':
        case '_':
          event.preventDefault()
          return zoomAt(1 / 1.4, 0, 0)
        case '0':
          event.preventDefault()
          return reset()
        case ' ':
          if (!onToggle || !item) return undefined
          event.preventDefault()
          return onToggle(item.videoId, item.frameIdx)
        default:
          return undefined
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [index, items.length, onClose, onIndex, onToggle, zoomAt, reset, item])

  if (!item) return null

  const zoomed = view.scale > 1

  return (
    // stopPropagation: lightbox có thể được mở TỪ BÊN TRONG VideoModal, mà nền của modal
    // đó cũng đóng-khi-bấm. Không chặn thì một cú bấm ra nền đóng luôn cả hai lớp, người
    // dùng mất chỗ đang xem trong video.
    <div
      className="lightbox-backdrop"
      onClick={(event) => {
        event.stopPropagation()
        onClose()
      }}
    >
      <div className="lightbox" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-head">
          <div className="lightbox-id">
            <b>{item.videoId}</b>
            <span className="card-frame">frame {item.frameIdx}</span>
            {item.caption && <span className="modal-title">{item.caption}</span>}
            {items.length > 1 && (
              <span className="lightbox-count">
                {index + 1}/{items.length}
              </span>
            )}
          </div>
          <div className="lightbox-tools">
            <span className="lightbox-scale">{Math.round(view.scale * 100)}%</span>
            <button type="button" className="btn btn-mini" onClick={() => zoomAt(1 / 1.4, 0, 0)}>
              −
            </button>
            <button type="button" className="btn btn-mini" onClick={() => zoomAt(1.4, 0, 0)}>
              +
            </button>
            <button type="button" className="btn btn-mini" onClick={reset} disabled={!zoomed}>
              vừa khung
            </button>
            {onToggle && (
              <button
                type="button"
                className={isSelected ? 'btn btn-mini picked' : 'btn btn-mini'}
                onClick={() => onToggle(item.videoId, item.frameIdx)}
              >
                {isSelected ? '✓ đã lấy' : '+ lấy frame'}
              </button>
            )}
            <button type="button" className="btn btn-mini" onClick={onClose}>
              ✕
            </button>
          </div>
        </div>

        <div
          ref={stageRef}
          className={`lightbox-stage${zoomed ? ' zoomed' : ''}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onDoubleClick={() => (zoomed ? reset() : zoomAt(3, 0, 0))}
        >
          <img
            src={item.src}
            alt={`${item.videoId} frame ${item.frameIdx}`}
            draggable={false}
            style={{
              transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`,
            }}
          />
        </div>

        <div className="hint-row">
          Lăn chuột phóng to tại con trỏ · kéo để di chuyển · nháy đúp 3× · +/− · 0 vừa khung
          {items.length > 1 && ' · ← → ảnh trước/sau'}
          {onToggle && ' · Space lấy frame'} · Esc đóng
        </div>
      </div>
    </div>
  )
}
