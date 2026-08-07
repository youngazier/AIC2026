import { useMemo, useState } from 'react'
import { api } from '../api.js'
import { MAX_SUBMISSION_ROWS } from '../constants.js'

/** Escape đúng chuẩn CSV: chỉ câu trả lời Q&A mới có thể chứa dấu phẩy hoặc nháy kép. */
function csvCell(value) {
  const text = String(value ?? '')
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

/** Dựng đúng nội dung file mà backend sẽ ghi — dùng cho ô xem trước và nút tải về. */
function buildCsv(queryType, rows, globalAnswer) {
  return rows
    .map((row) => {
      const cells = [row.video_id, ...row.frames]
      if (queryType === 'qna') cells.push(row.answer ?? globalAnswer ?? '')
      return cells.map(csvCell).join(',')
    })
    .join('\n')
}

export default function SubmissionPanel({
  queryType, rows, setRows, globalAnswer, notice, setNotice, onViewVideo,
}) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showPreview, setShowPreview] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  const effectiveAnswer = (row) => row.answer ?? globalAnswer ?? ''
  const csv = useMemo(
    () => buildCsv(queryType, rows, globalAnswer),
    [queryType, rows, globalAnswer],
  )

  // Kiểm tra tại chỗ, trước khi gửi: báo lỗi ngay lúc người dùng còn nhìn vào danh sách
  // vẫn tốt hơn là để backend trả 400 sau khi họ đã bấm nút và chờ.
  const problems = useMemo(() => {
    const found = []
    if (rows.length === 0) found.push('Danh sách nộp đang trống.')
    if (rows.length > MAX_SUBMISSION_ROWS) found.push(`Vượt quá ${MAX_SUBMISSION_ROWS} dòng.`)
    rows.forEach((row, index) => {
      if (queryType !== 'trake' && row.frames.length !== 1) {
        found.push(`Dòng ${index + 1}: ${queryType.toUpperCase()} chỉ nhận 1 frame, đang có ${row.frames.length}.`)
      }
      if (queryType === 'qna' && !effectiveAnswer(row).trim()) {
        found.push(`Dòng ${index + 1}: thiếu câu trả lời.`)
      }
    })
    return found
  }, [rows, queryType, globalAnswer])

  const update = (id, patch) =>
    setRows((prev) => prev.map((row) => (row.id === id ? { ...row, ...patch } : row)))

  const move = (index, delta) =>
    setRows((prev) => {
      const next = [...prev]
      const target = index + delta
      if (target < 0 || target >= next.length) return prev
      ;[next[index], next[target]] = [next[target], next[index]]
      return next
    })

  const removeRow = (id) => setRows((prev) => prev.filter((row) => row.id !== id))

  const removeFrame = (id, frame) =>
    setRows((prev) =>
      prev
        .map((row) => (row.id === id ? { ...row, frames: row.frames.filter((f) => f !== frame) } : row))
        .filter((row) => row.frames.length > 0),
    )

  const submit = async () => {
    setError(null)
    setResult(null)
    setNotice(null)
    if (problems.length > 0) {
      setError(problems[0])
      return
    }
    if (!name.trim()) {
      setError('Nhập tên file trước khi tạo.')
      return
    }
    setBusy(true)
    try {
      const payload = {
        name: name.trim(),
        query_type: queryType,
        rows: rows.map((row) => ({
          video_id: row.video_id,
          frames: row.frames,
          answer: queryType === 'qna' ? effectiveAnswer(row) : null,
        })),
      }
      setResult(await api.createSubmission(payload))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  // Tải thẳng về máy: dùng khi trình duyệt không cùng máy với backend, hoặc muốn giữ
  // một bản trước khi ghi đè file trên máy chủ.
  const download = () => {
    const blob = new Blob([csv + '\n'], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${name.trim() || 'submission'}.csv`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <section className={`submission${collapsed ? ' collapsed' : ''}`}>
      <div className="section-head submission-head">
        <h2>
          Danh sách nộp
          <span className={rows.length >= MAX_SUBMISSION_ROWS ? 'counter full' : 'counter'}>
            {rows.length}/{MAX_SUBMISSION_ROWS}
          </span>
          <span className="tag">{queryType}</span>
        </h2>

        <div className="submission-tools">
          <input
            className="name-input"
            value={name}
            placeholder="tên file (VD: query-1-kis)"
            onChange={(e) => setName(e.target.value)}
          />
          <button type="button" className="btn btn-primary" onClick={submit} disabled={busy}>
            {busy ? 'Đang ghi…' : 'Tạo file CSV'}
          </button>
          <button type="button" className="btn" onClick={download} disabled={rows.length === 0}>
            ⭳ Tải về
          </button>
          <button type="button" className="btn btn-mini" onClick={() => setShowPreview((v) => !v)}>
            {showPreview ? 'Ẩn CSV' : 'Xem CSV'}
          </button>
          <button
            type="button"
            className="btn btn-mini"
            onClick={() => rows.length && window.confirm(`Xóa cả ${rows.length} dòng?`) && setRows([])}
          >
            Xóa hết
          </button>
          <button type="button" className="btn btn-mini" onClick={() => setCollapsed((v) => !v)}>
            {collapsed ? '▲' : '▼'}
          </button>
        </div>
      </div>

      {notice && <div className="alert alert-warn">{notice}</div>}
      {error && <div className="alert alert-error">{error}</div>}
      {result && (
        <div className="alert alert-ok">
          Đã ghi <b>{result.name}</b> ({result.n_rows} dòng) vào <code>{result.path}</code>
          {result.warnings.length > 0 && (
            <ul className="warn-list">
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {!collapsed && (
        <>
          {rows.length === 0 ? (
            <div className="placeholder small">
              Tick vào ảnh kết quả, hoặc mở video rồi bấm <b>Lấy frame</b>, để thêm dòng vào đây.
              Thứ tự dòng chính là thứ hạng nộp — dòng đầu tiên ăn điểm cao nhất.
            </div>
          ) : (
            <ol className="rows">
              {rows.map((row, index) => (
                <li key={row.id} className="row">
                  <span className="row-rank">{index + 1}</span>
                  <b className="row-video">{row.video_id}</b>

                  <span className="row-frames">
                    {row.frames.map((frame) => (
                      <button
                        key={frame}
                        type="button"
                        className="chip chip-frame"
                        title="Bỏ frame này khỏi dòng"
                        onClick={() => removeFrame(row.id, frame)}
                      >
                        {frame} ✕
                      </button>
                    ))}
                    {queryType === 'trake' && row.frames.length > 1 && (
                      <button
                        type="button"
                        className="btn btn-mini"
                        title="Sắp xếp frame tăng dần (thứ tự thời gian của chuỗi sự kiện)"
                        onClick={() => update(row.id, { frames: [...row.frames].sort((a, b) => a - b) })}
                      >
                        ↕ sắp xếp
                      </button>
                    )}
                  </span>

                  {queryType === 'qna' && (
                    <input
                      className="row-answer"
                      value={effectiveAnswer(row)}
                      placeholder="câu trả lời"
                      onChange={(e) => update(row.id, { answer: e.target.value })}
                    />
                  )}

                  <span className="row-actions">
                    <button
                      type="button"
                      className="btn btn-mini"
                      title="Xem video tại frame đầu của dòng"
                      onClick={() =>
                        // Không bịa `timestamp`: ở đây chưa biết fps của video. Đưa 0
                        // vào sẽ khiến modal nhắm mốc bắt đầu là giây 0 rồi mới tua
                        // lại — bỏ trống thì modal tự tính mốc đúng khi có fps.
                        onViewVideo({
                          video_id: row.video_id,
                          frame_idx: row.frames[0],
                          keyframe_name: '',
                        })
                      }
                    >
                      ▶
                    </button>
                    <button type="button" className="btn btn-mini" onClick={() => move(index, -1)} disabled={index === 0}>
                      ↑
                    </button>
                    <button
                      type="button"
                      className="btn btn-mini"
                      onClick={() => move(index, 1)}
                      disabled={index === rows.length - 1}
                    >
                      ↓
                    </button>
                    <button type="button" className="btn btn-mini" onClick={() => removeRow(row.id)}>
                      ✕
                    </button>
                  </span>
                </li>
              ))}
            </ol>
          )}

          {showPreview && <pre className="csv-preview">{csv || '(trống)'}</pre>}
        </>
      )}
    </section>
  )
}
