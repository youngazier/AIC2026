# AI Video Retrieval System — Project Brief

## 1. Purpose

Build a local, single-user application for finding relevant moments in a video collection with natural-language queries. The system detects shots, selects one keyframe per shot, encodes keyframes with CLIP, and searches their embeddings with FAISS. Search results link back to the source video and its timestamp.

The project is motivated by video retrieval tasks in the **HCM AI Challenge**. The initial collection is approximately 100 videos; the design should remain practical as the collection grows toward 1,000 videos.

## 2. Scope and current status

The intended end-to-end workflow is:

```text
Video → shot detection → keyframe extraction → CLIP embeddings
      → FAISS index → text search → ranked keyframes → source playback
```

**Implemented now:** command-line video processing with TransNetV2, one middle keyframe per shot, CLIP image embeddings, source-linked metadata artifacts, FAISS index construction, and a SQLite catalog with migrations, validated atomic imports, and typed queries.

**Application work still planned:** text-query retrieval, a FastAPI service, a React interface, and browser playback. The processing pipeline uses validated JSON artifacts and local files; the implemented SQLite catalog is synchronized through a separate import CLI.

This brief describes the intended application scope without treating planned components as already delivered. [Video processing](03_VIDEO_PROCESSING.md), [feature extraction](04_FEATURE_EXTRACTION.md), and the [implementation checklist](TODOS.md) describe the current commands and progress.

## 3. User workflow

The initial user is one developer or researcher working with a local video collection. The application should let that user:

1. Add videos and see their imported metadata.
2. Run shot detection, keyframe extraction, embedding, and indexing through the existing CLIs.
3. Search indexed keyframes with a natural-language description.
4. Browse ranked keyframes with video names, shot indices, timestamps, and similarity scores.
5. Open a source video near a selected keyframe's timestamp.
6. Inspect CLI processing failures and rerun the affected work.

For example, a query such as “a person holding an umbrella while crossing a street” should return relevant keyframes. Similarity scores express ranking, not certainty that every detail in a query is present.

## 4. Processing and retrieval requirements

### Video processing

- Probe each video for duration, resolution, frame rate, and decoded timestamps.
- Detect shot boundaries with TransNetV2 and select one representative middle keyframe per shot.
- Preserve the video name, shot index, source frame index, and presentation timestamp for every keyframe.
- Keep original videos and extracted keyframe images on the local filesystem.
- Run long processing tasks through the existing CLIs, outside interactive search requests.
- Keep valid artifacts on failure and allow safe reruns without creating duplicate outputs.

### Embeddings and search

- Encode keyframes and text queries with compatible CLIP model settings.
- Store per-video image embeddings as NumPy files with source mappings and model/configuration provenance.
- Build a rebuildable FAISS `IndexFlatIP` index from compatible image embeddings.
- Return ranked keyframes with their source video, shot, frame index, timestamp, thumbnail reference, and similarity score.
- Reject missing or incompatible index and embedding artifacts with clear errors.
- Load media and result images as needed; measure memory and query latency as the collection grows.

### Application data

- Use the implemented SQLite catalog through SQLAlchemy for videos, shots, and keyframes; integrate its typed queries when the API is built.
- Keep raw videos, keyframe images, and embedding files on the local filesystem. The FAISS index remains a rebuildable search artifact.
- Preserve stable video names and processing/model versions so search results can be traced to the files and configuration that produced them.
- Migrate or import existing validated JSON processing artifacts into application records without losing source references.

## 5. Technology stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | React + TypeScript + Vite | Web application |
| Styling | Tailwind CSS | UI styling |
| Backend | FastAPI + Pydantic | REST API |
| ORM | SQLAlchemy 2.x | Metadata access |
| Metadata DB | SQLite initially | Videos, shots, keyframes |
| Shot detection | TransNetV2 | Detect video shot boundaries |
| Video decoding | FFmpeg / FFprobe | Extract frames and metadata |
| Embedding | CLIP (PyTorch + Transformers) | Encode images and text |
| Vector search | FAISS (`IndexFlatIP`) | Similarity search |
| File storage | Local filesystem | Raw videos, keyframes |
| Containers | Docker Compose (optional) | Reproducible deployment |
| Testing | Pytest + API tests | Verify correctness |

The current Python pipeline probes video metadata with FFprobe and decodes frames with PyAV/FFmpeg. The table lists the target application stack; the status in Section 2 distinguishes completed work from planned work.

## 6. Quality and acceptance criteria

The end-to-end application is complete when:

| Area | Acceptance criterion |
| --- | --- |
| Ingestion | A video keeps its filename stem as `video_name`, with a source reference, metadata, and visible processing status. |
| Processing | Each detected shot has a representative keyframe with a valid source timestamp. |
| Indexing | Compatible CLIP embeddings are stored with source mappings and included in a FAISS index. |
| Search | A text query returns ranked keyframes with IDs, timestamps, thumbnails, and scores. |
| Playback | A selected result opens the source video near its recorded timestamp. |
| Recovery | Failed CLI runs can be retried while prior valid artifacts remain usable. |
| Reproducibility | Results can be traced to the source video, processing run, model, and configuration. |

Pytest coverage should verify processing and index invariants, and API tests should verify search, metadata, status, and media responses once those endpoints exist. Retrieval quality and latency should be checked with representative queries and the actual collection.
