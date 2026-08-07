import { useCallback, useEffect, useRef, useState } from 'react'
import { api, videoStreamUrl } from '../api.js'
import { formatTime, frameKey } from '../constants.js'
import ImageLightbox from './ImageLightbox.jsx'

/**
 * Xem lại video đúng tại frame kết quả, và lấy frame bất kỳ tại vị trí con trỏ.
 *
 * QUY ƯỚC FRAME <-> THỜI GIAN (phải khớp tuyệt đối với pipeline, sai là mất điểm):
 * pipeline ghi `timestamp = frame_idx / fps`, nên frame f chiếm khoảng thời gian
 * [f/fps, (f+1)/fps). Từ đó:
 *   - đọc ngược:  frame = floor(t * fps)
 *   - tua tới f:  t = (f + 0.5) / fps  — nhắm GIỮA frame, không nhắm mép.
 * Nhắm đúng mép f/fps là canh bạc với sai số dấu phẩy động: lệch xuống một chút là
 * trình duyệt hiển thị frame f-1, và ta nộp nhầm số.
 */
// Số lần thử tua tối đa trước khi báo cho người dùng biết là không tua được. 12 lần x
// 300ms ~ 3.6 giây, đủ cho một file lớn đọc xong `moov` ở cuối trên mạng nội bộ.
const MAX_SEEK_TRIES = 12

export default function VideoModal({ target, selectedKeys, onToggle, onClose }) {
  const videoRef = useRef(null)
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)
  const [metadataLoaded, setMetadataLoaded] = useState(false)
  const [currentTime, setCurrentTime] = useState(target.timestamp ?? 0)
  const [zoomShot, setZoomShot] = useState(null)
  // 'seeking' đang cố tua tới frame kết quả | 'ok' đã tới | 'giveup' chịu thua, báo ra
  const [seekPhase, setSeekPhase] = useState('seeking')
  // Người dùng đã tự tua tay chưa. Khi rồi thì vòng tua tự động phải im ngay, không
  // được giật họ ngược về frame kết quả giữa lúc họ đang dò.
  const userMoved = useRef(false)
  // Phân biệt lệnh tua của ta với lệnh tua của người dùng, vì cả hai bắn cùng sự kiện.
  const ourSeek = useRef(false)

  // Nhúng ĐIỂM BẮT ĐẦU thẳng vào URL bằng media fragment `#t=`.
  //
  // Trước đây modal chỉ tua bằng JS sau khi `loadedmetadata` bắn VÀ API /videos trả fps
  // về. Hai điều kiện đó tới muộn và không theo thứ tự, nên trình duyệt đã kịp dựng
  // khung hình đầu tiên ở giây 0 — bấm play trước lúc tua kịp là xem lại từ đầu video.
  // Với `#t=`, trình duyệt biết mốc cần tới NGAY từ request đầu tiên, trước cả khi
  // JavaScript chạy: nó tải thẳng đoạn đó và poster hiện luôn đúng cảnh.
  //
  // Dùng `target.timestamp` vì kết quả tìm kiếm đã kèm sẵn — không phải đợi API. Sai số
  // tối đa nửa frame so với mốc chuẩn, và effect bên dưới tinh chỉnh nốt khi có fps.
  //
  // `useRef` giữ URL CỐ ĐỊNH suốt vòng đời modal: đổi `src` là trình duyệt vứt bộ đệm
  // và tải lại từ đầu, nên tuyệt đối không được tính lại nó khi fps về.
  const srcRef = useRef(
    `${videoStreamUrl(target.video_id)}#t=${Math.max(0, target.timestamp ?? 0).toFixed(3)}`,
  )

  const fps = info?.fps || null
  const toFrame = useCallback((time) => (fps ? Math.floor(time * fps + 1e-6) : null), [fps])
  const toTime = useCallback((frame) => (fps ? (frame + 0.5) / fps : 0), [fps])
  const currentFrame = toFrame(currentTime)

  // Mốc cần tới, tính bằng giây. Có fps thì nhắm GIỮA frame kết quả; chưa có thì tạm
  // dùng `timestamp` mà kết quả tìm kiếm đã kèm sẵn — nhờ vậy việc tua bắt đầu ngay,
  // không phải đợi API /videos trả về.
  const targetTime =
    fps && Number.isFinite(target.frame_idx) ? toTime(target.frame_idx) : (target.timestamp ?? 0)

  useEffect(() => {
    let alive = true
    setInfo(null)
    setError(null)
    api
      .videoInfo(target.video_id)
      .then((data) => {
        if (!alive) return
        setInfo(data)
        if (!data.has_file) setError(`Không tìm thấy file ${target.video_id}.mp4 trên máy chủ.`)
      })
      .catch((err) => alive && setError(err.message))
    return () => {
      alive = false
    }
  }, [target.video_id])

  // Bám sát thời gian thực của video. requestVideoFrameCallback báo `mediaTime` của
  // ĐÚNG khung hình đang hiển thị, còn `timeupdate` chỉ bắn ~4 lần/giây và trả về thời
  // gian đã trôi qua — chênh nhau vài frame, đủ để nộp sai.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return undefined

    if (typeof video.requestVideoFrameCallback !== 'function') {
      const onTimeUpdate = () => setCurrentTime(video.currentTime)
      video.addEventListener('timeupdate', onTimeUpdate)
      video.addEventListener('seeked', onTimeUpdate)
      return () => {
        video.removeEventListener('timeupdate', onTimeUpdate)
        video.removeEventListener('seeked', onTimeUpdate)
      }
    }

    let handle = null
    let cancelled = false
    const tick = (_now, metadata) => {
      if (cancelled) return
      setCurrentTime(metadata.mediaTime)
      handle = video.requestVideoFrameCallback(tick)
    }
    handle = video.requestVideoFrameCallback(tick)
    return () => {
      cancelled = true
      if (handle !== null && typeof video.cancelVideoFrameCallback === 'function') {
        video.cancelVideoFrameCallback(handle)
      }
    }
  }, [info])

  const seekToFrame = useCallback(
    (frame) => {
      const video = videoRef.current
      if (!video || !fps) return
      const clamped = Math.max(0, frame)
      video.currentTime = toTime(clamped)
      setCurrentTime(toTime(clamped))
    },
    [fps, toTime],
  )

  // TUA TỚI FRAME KẾT QUẢ — VÀ KIỂM CHỨNG LÀ ĐÃ THẬT SỰ TỚI.
  //
  // Gán `video.currentTime` một lần là KHÔNG đủ. Trình duyệt lặng lẽ bỏ qua lệnh tua
  // khi mốc cần tới chưa nằm trong `video.seekable`, mà các file .mp4 ở đây đặt atom
  // `moov` ở CUỐI file (không mux faststart) nên vùng tua được chỉ nở dần theo dữ liệu
  // về. Lệnh bị bỏ qua không ném lỗi, không bắn sự kiện nào: video cứ nằm im ở giây 0.
  // Đó đúng là triệu chứng "phát từ đầu thì được, nhảy tới frame thì không".
  //
  // Nên: tua → đợi → so lại vị trí → còn lệch thì tua tiếp, có trần số lần, và báo ra
  // màn hình nếu chịu thua thay vì im lặng hiển thị nhầm frame.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !metadataLoaded || userMoved.current) return undefined
    if (!Number.isFinite(targetTime) || targetTime <= 0) {
      setSeekPhase('ok')
      return undefined
    }

    // Sai số cho phép: dưới một frame. Đủ rộng để lần tinh chỉnh khi fps về (lệch đúng
    // nửa frame so với `#t=`) không tốn thêm một vòng tua nữa.
    const tolerance = fps ? 0.75 / fps : 0.04
    const arrived = () => Math.abs(video.currentTime - targetTime) <= tolerance

    let tries = 0
    let timer = null
    let cancelled = false

    const attempt = () => {
      if (cancelled || userMoved.current) return
      if (arrived()) return setSeekPhase('ok')
      if (tries >= MAX_SEEK_TRIES) return setSeekPhase('giveup')
      tries += 1
      setSeekPhase('seeking')
      ourSeek.current = true
      video.currentTime = targetTime
      // Không chỉ dựa vào sự kiện `seeked`: lệnh tua bị bỏ qua thì nó không bao giờ bắn,
      // và ta sẽ đợi mãi. Hẹn giờ thử lại là cách duy nhất thoát được trường hợp đó.
      timer = setTimeout(attempt, 300)
    }

    const onSeeked = () => {
      ourSeek.current = false
      if (!arrived()) return
      clearTimeout(timer)
      setSeekPhase('ok')
    }

    // Người dùng tự kéo thanh trượt hoặc bấm ±1f -> nhường quyền điều khiển, dừng hẳn.
    const onSeeking = () => {
      if (ourSeek.current) return
      userMoved.current = true
      cancelled = true
      clearTimeout(timer)
      setSeekPhase('ok')
    }

    video.addEventListener('seeked', onSeeked)
    video.addEventListener('seeking', onSeeking)
    attempt()
    return () => {
      cancelled = true
      clearTimeout(timer)
      video.removeEventListener('seeked', onSeeked)
      video.removeEventListener('seeking', onSeeking)
    }
  }, [metadataLoaded, targetTime, fps])

  // Đọc frame hiện tại thẳng từ thẻ <video> chứ không từ state: state được cập nhật
  // bất đồng bộ qua requestVideoFrameCallback, nên bấm "−1f" năm lần thật nhanh sẽ chỉ
  // lùi được một hai frame nếu tính dựa trên state.
  const step = useCallback(
    (deltaFrames) => {
      const video = videoRef.current
      if (!video || !fps) return
      seekToFrame(toFrame(video.currentTime) + deltaFrames)
    },
    [fps, seekToFrame, toFrame],
  )

  /**
   * Chụp khung hình ĐANG hiển thị ra ảnh để soi ở cỡ lớn.
   *
   * Vẽ ra canvas theo `videoWidth/videoHeight` — kích thước GỐC của luồng video, không
   * phải kích thước thẻ <video> đang bị thu nhỏ trong modal. Nhờ vậy ảnh chụp giữ trọn
   * độ phân giải để phóng to đọc chi tiết, thay vì phóng to một bản đã mất nét.
   *
   * Ảnh và video cùng origin với trang (đều qua proxy /api) nên canvas không bị "vấy
   * bẩn" và toDataURL() chạy được; nếu sau này tách origin thì phải thêm CORS.
   */
  const captureFrame = useCallback(() => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    video.pause()
    setZoomShot({
      src: canvas.toDataURL('image/png'),
      videoId: target.video_id,
      frameIdx: toFrame(video.currentTime),
      caption: 'khung hình đang xem',
    })
  }, [target.video_id, toFrame])

  useEffect(() => {
    const onKeyDown = (event) => {
      // Lightbox đang mở thì nó tự lo phím của nó — không để hai lớp cùng ăn một phím.
      if (zoomShot) return
      if (event.key === 'Escape') return onClose()
      const video = videoRef.current
      if (!video) return
      if (event.key === 'f' || event.key === 'F') {
        event.preventDefault()
        return captureFrame()
      }
      if (event.key === ' ') {
        event.preventDefault()
        video.paused ? video.play() : video.pause()
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        video.pause()
        step(event.shiftKey ? -10 : -1)
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        video.pause()
        step(event.shiftKey ? 10 : 1)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, step, captureFrame, zoomShot])

  const picked = currentFrame !== null && selectedKeys.has(frameKey(target.video_id, currentFrame))

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <b>{target.video_id}</b>
            {info?.title && <span className="modal-title"> — {info.title}</span>}
          </div>
          <button type="button" className="btn btn-mini" onClick={onClose}>
            ✕
          </button>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <video
          ref={videoRef}
          className="video"
          src={srcRef.current}
          controls
          preload="metadata"
          onLoadedMetadata={() => setMetadataLoaded(true)}
          onError={() => setError('Trình duyệt không phát được file video này.')}
        />

        <div className="video-controls">
          <div className="frame-readout">
            <span className="frame-now">frame {currentFrame ?? '—'}</span>
            <span className="frame-time">
              {formatTime(currentTime)} / {formatTime(info?.duration ?? 0)}
              {fps ? ` · ${fps} fps` : ' · chưa biết fps'}
            </span>
            <span className="frame-origin">kết quả: frame {target.frame_idx}</span>
            {seekPhase === 'seeking' && (
              <span className="seek-note">đang tua tới frame {target.frame_idx}…</span>
            )}
            {seekPhase === 'giveup' && (
              <span className="seek-note seek-bad">
                ⚠ chưa tua tới được frame {target.frame_idx} — bấm ⟲ để thử lại
              </span>
            )}
          </div>

          <div className="stepper">
            <button type="button" className="btn btn-mini" onClick={() => step(-10)} disabled={!fps}>
              −10f
            </button>
            <button type="button" className="btn btn-mini" onClick={() => step(-1)} disabled={!fps}>
              −1f
            </button>
            <button type="button" className="btn btn-mini" onClick={() => step(1)} disabled={!fps}>
              +1f
            </button>
            <button type="button" className="btn btn-mini" onClick={() => step(10)} disabled={!fps}>
              +10f
            </button>
            <button
              type="button"
              className="btn btn-mini"
              onClick={() => seekToFrame(target.frame_idx)}
              disabled={!fps}
              title="Quay lại frame của kết quả tìm kiếm"
            >
              ⟲ về frame gốc
            </button>
            <button
              type="button"
              className="btn btn-mini"
              onClick={captureFrame}
              title="Phóng to khung hình đang xem để soi chi tiết (phím F)"
            >
              ⛶ phóng to
            </button>
          </div>

          <button
            type="button"
            className={picked ? 'btn btn-primary picked' : 'btn btn-primary'}
            disabled={!fps || currentFrame === null}
            title="Đưa frame đang hiển thị vào danh sách nộp"
            onClick={() => onToggle(target.video_id, currentFrame)}
          >
            {picked ? `✓ Đã lấy frame ${currentFrame}` : `+ Lấy frame ${currentFrame ?? '—'}`}
          </button>
        </div>

        <div className="hint-row">
          Space phát/dừng · ← → lùi/tiến 1 frame · Shift + ← → 10 frame · F phóng to · Esc đóng
        </div>
      </div>

      {zoomShot && (
        <ImageLightbox
          items={[zoomShot]}
          index={0}
          isSelected={selectedKeys.has(frameKey(target.video_id, zoomShot.frameIdx))}
          onToggle={onToggle}
          onClose={() => setZoomShot(null)}
        />
      )}
    </div>
  )
}
