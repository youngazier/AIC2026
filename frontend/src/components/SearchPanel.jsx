import { useMemo } from 'react'
import { MODEL_CHOICES, QUERY_TYPES, splitTrakeEvents } from '../constants.js'

export default function SearchPanel({
  queryType, onQueryType,
  modelChoice, onModelChoice, availability,
  textQuery, onTextQuery,
  answer, onAnswer,
  topk, onTopk,
  playlist, onPlaylist, playlists,
  loading, onSearch,
}) {
  const events = useMemo(
    () => (queryType === 'trake' ? splitTrakeEvents(textQuery) : []),
    [queryType, textQuery],
  )
  const activeType = QUERY_TYPES.find((t) => t.id === queryType)

  const submit = (event) => {
    event.preventDefault()
    onSearch(textQuery)
  }

  return (
    <form className="search-panel" onSubmit={submit}>
      <div className="control-row">
        <div className="segmented" role="tablist" aria-label="Dạng truy vấn">
          {QUERY_TYPES.map((type) => (
            <button
              key={type.id}
              type="button"
              role="tab"
              aria-selected={queryType === type.id}
              className={queryType === type.id ? 'seg active' : 'seg'}
              title={type.title}
              onClick={() => onQueryType(type.id)}
            >
              {type.label}
            </button>
          ))}
        </div>

        <div className="segmented" role="group" aria-label="Model tìm kiếm">
          {MODEL_CHOICES.map((choice) => {
            const enabled = availability[choice.id]
            return (
              <button
                key={choice.id}
                type="button"
                disabled={!enabled}
                className={modelChoice === choice.id ? 'seg active' : 'seg'}
                title={
                  enabled
                    ? choice.hint
                    : `Chưa dùng được: cần index + text encoder cho ${choice.models.join(' và ')}`
                }
                onClick={() => onModelChoice(choice.id)}
              >
                {choice.label}
              </button>
            )
          })}
        </div>

        <label className="field field-topk">
          <span>top-k</span>
          <input
            type="number"
            min="1"
            max="500"
            value={topk}
            onChange={(e) => onTopk(e.target.value)}
          />
        </label>

        <label className="field">
          <span>playlist</span>
          <select value={playlist} onChange={(e) => onPlaylist(e.target.value)}>
            <option value="">Tất cả</option>
            {playlists.map((item) => (
              <option key={item.name} value={item.name} disabled={!item.has_index}>
                {item.name} ({item.n_videos}){item.has_index ? '' : ' — chưa có index'}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="query-row">
        <textarea
          className="query-input"
          rows={2}
          value={textQuery}
          placeholder={
            queryType === 'trake'
              ? 'Mô tả chuỗi sự kiện, mỗi sự kiện kết thúc bằng dấu chấm. VD: Vận động viên chạy đà. Giậm nhảy rời mặt đất. Bay qua xà. Tiếp đất.'
              : 'Mô tả phân cảnh cần tìm. Text encoder mạnh tiếng Anh hơn tiếng Việt.'
          }
          onChange={(e) => onTextQuery(e.target.value)}
          onKeyDown={(e) => {
            // Enter tìm kiếm, Shift+Enter xuống dòng: câu truy vấn TRAKE thường dài
            // nhiều dòng nên vẫn phải xuống dòng được.
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSearch(textQuery)
            }
          }}
        />
        <button className="btn btn-primary btn-search" type="submit" disabled={loading}>
          {loading ? 'Đang tìm…' : 'Tìm kiếm'}
        </button>
      </div>

      {queryType === 'qna' && (
        <div className="query-row">
          <input
            className="answer-input"
            value={answer}
            placeholder='Câu trả lời cho truy vấn Q&A — sẽ nằm ở cột thứ 3 của file CSV (VD: "6")'
            onChange={(e) => onAnswer(e.target.value)}
          />
        </div>
      )}

      {queryType === 'trake' && (
        <div className="events-row">
          {events.length === 0 ? (
            <span className="hint">Chưa tách được sự kiện nào — ngăn cách các sự kiện bằng dấu chấm.</span>
          ) : (
            <>
              <span className="hint">
                {events.length} sự kiện — bấm vào một sự kiện để tìm riêng khoảnh khắc của nó:
              </span>
              {events.map((event, index) => (
                <button
                  key={`${index}-${event}`}
                  type="button"
                  className="chip"
                  title="Tìm kiếm riêng sự kiện này"
                  onClick={() => onSearch(event)}
                >
                  <b>{index + 1}</b> {event}
                </button>
              ))}
            </>
          )}
        </div>
      )}

      <div className="hint-row">{activeType.hint}</div>
    </form>
  )
}
