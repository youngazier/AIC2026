# Database and FAISS Index

Status: **M3 implemented; retrieval and API integration pending**.
SQLAlchemy models, SQLite engine/session factories, pre-write artifact
validation, the initial Alembic migration, atomic import, typed queries, and the
standalone import CLI are implemented. Phase 1 and Phase 2 write validated
JSON, JPEG, and NumPy artifacts; the CLI builds global and optional playlist
FAISS indexes. This document defines the schema and storage boundaries.

## 1. Storage roles and layout

| Store | Purpose | Authority |
| --- | --- | --- |
| `data/` | Organizer videos and metadata | Original inputs; never rewritten by the application |
| Phase 1 JSON and keyframes | Shot boundaries, source timestamps, image paths, and processing settings | Validated source for catalog imports |
| Phase 2 JSON and `.npy` | Per-video vector rows and model identity | Validated source for index rebuilds |
| SQLite | API catalog and shot/keyframe lookup | Queryable application view, synchronized from completed Phase 1 artifacts |
| FAISS index plus row metadata and manifest | Similarity search and row-to-source resolution | Rebuildable search snapshot |

```text
data/
├── videos/<batch>/<video_name>.<extension>
└── media-info/<video_name>.json

data_processed/
├── transnetv2/<batch>/<video_name>/
│   ├── <video_name>.predictions.txt
│   ├── <video_name>.scenes.txt
│   └── shots.json
├── keyframes/<batch>/<video_name>/*.jpg
├── features/<model_key>/<video_name>.{npy,json}
├── indexes/                                # index, row mapping, manifest
└── database/catalog.sqlite3                # migrated local catalog
```

Store paths as relative keys. Resolve `video_path` from the project root, then
require its resolved target to stay under the approved video root. Resolve
generated artifact keys under `data_processed/`; reject absolute paths,
traversal, and symlinks escaping those roots. Keep media bytes, embeddings, and
FAISS files outside SQLite. Do not store absolute machine paths or vector blobs
in catalog rows.

## 2. Identity and source contracts

Use `(batch, video_name)` as the stable video key. `video_name` is the source
filename without its extension: `L21_V001` for `L21_V001.mp4`. A batch avoids collisions across
inputs. A shot is `(batch, video_name, shot_index)`. The current Phase 1
pipeline creates exactly one middle keyframe per shot. The keyframe reuses the
shot's `(batch, video_name, shot_index)` key and adds its frame index, timestamp,
and image path. Consequently, the catalog relationship from a shot to its
keyframe is one-to-zero-or-one: completed imports have one, while a shot cannot
have multiple keyframe rows. These keys remain stable across FAISS rebuilds; a
FAISS row number does not.

The current Phase 2 filename and optional playlist mapping use video name
alone. Until they are changed to include batch, the importer must reject a
second batch with the same video name before indexing can overwrite or
misattribute features.

Import only a published Phase 1 result after validating `shots.json`
(including its video metadata and settings) and readable keyframe images. Preserve the stored
presentation timestamps in seconds. Never derive playback time from frame index
or average FPS. Phase 2 sidecars and `.npy` files stay outside SQLite; validate
them against the current Phase 1 `shots.json` when loading the search index.

The JSON artifacts remain the reproducible source for catalog re-import and
index rebuild. SQLite becomes the API's catalog once an import commits. If a
Phase 1 artifact changes, synchronize its SQLite rows before reporting them as
current. Processing stays in the existing CLI.

The existing catalog column names `manifest_path` and `manifest_mtime_ns` now
refer to Phase 1's `shots.json` and its modification time.

## 3. SQLite schema

`backend/app/models.py` defines the three models and `ix_videos_video_name`.
The initial migration is `backend/migrations/versions/0001_create_video_shot_and_keyframe_catalog.py`.
Run these commands from the project root:

```bash
uv run alembic upgrade head
uv run alembic current
uv run alembic check

# Use a separate catalog file when needed.
uv run alembic -x database_path=data_processed/database/test.sqlite3 upgrade head
```

Migration commands use the same SQLite connection settings as the application.
`upgrade head` is safe to repeat. Downgrading to `base` deletes the catalog tables
and their contents; reserve that operation for disposable databases.
Alembic autogeneration checks model/schema drift but does not detect every CHECK
constraint change, so review future migrations against this contract.

`backend/app/database.py` provides `create_catalog_engine()` and
`create_session_factory(engine)`. Relative database paths resolve from the
project root; the default is `data_processed/database/catalog.sqlite3`.
The engine creates the parent directory and enables foreign keys and a
5,000 ms busy timeout on every connection. Importing the module does not open
SQLite or create tables. Schema creation remains the migration's responsibility.

```python
from backend.app.database import create_catalog_engine, create_session_factory

engine = create_catalog_engine()
SessionLocal = create_session_factory(engine)

# Prepare and validate artifacts before entering this block.
with SessionLocal.begin() as session:
    # Keep database writes here; the importer provides this workflow.
    pass
```

The session block commits on success, rolls back on failure, and closes the
session. Short transactions depend on callers keeping artifact I/O outside it.

Implement these logical tables with SQLAlchemy 2.x and an Alembic initial
migration. The following SQLite DDL specifies keys, relationships, and the
fields the API needs; model and migration code should enforce the same contract.
Enable `PRAGMA foreign_keys = ON` on every connection. Use UTC ISO-8601 text for
import times and finite `REAL` seconds for media timestamps.

```sql
CREATE TABLE videos (
    batch TEXT NOT NULL,
    video_name TEXT NOT NULL,
    video_path TEXT NOT NULL UNIQUE,
    manifest_path TEXT NOT NULL,
    manifest_mtime_ns INTEGER NOT NULL,
    stream_index INTEGER NOT NULL,
    codec TEXT,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    duration_seconds REAL NOT NULL CHECK (duration_seconds > 0),
    average_fps REAL,
    timeline_origin_seconds REAL NOT NULL,
    frame_count INTEGER NOT NULL CHECK (frame_count > 0),
    shot_count INTEGER NOT NULL CHECK (shot_count > 0),
    imported_at TEXT NOT NULL,
    PRIMARY KEY (batch, video_name)
);

CREATE TABLE shots (
    batch TEXT NOT NULL,
    video_name TEXT NOT NULL,
    shot_index INTEGER NOT NULL CHECK (shot_index >= 0),
    start_frame INTEGER NOT NULL CHECK (start_frame >= 0),
    end_frame INTEGER NOT NULL CHECK (end_frame >= start_frame),
    start_timestamp REAL NOT NULL CHECK (start_timestamp >= 0),
    end_timestamp REAL NOT NULL CHECK (end_timestamp >= start_timestamp),
    PRIMARY KEY (batch, video_name, shot_index),
    FOREIGN KEY (batch, video_name) REFERENCES videos(batch, video_name)
        ON DELETE CASCADE
);

CREATE TABLE keyframes (
    batch TEXT NOT NULL,
    video_name TEXT NOT NULL,
    shot_index INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role = 'middle'),
    frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
    timestamp REAL NOT NULL CHECK (timestamp >= 0),
    image_path TEXT NOT NULL,
    PRIMARY KEY (batch, video_name, shot_index),
    UNIQUE (image_path),
    FOREIGN KEY (batch, video_name, shot_index)
        REFERENCES shots(batch, video_name, shot_index) ON DELETE CASCADE
);

CREATE INDEX ix_videos_video_name ON videos(video_name);
```

A source video with no completed Phase 1 result has no `videos` row yet. For
video routes, use the composite key internally.

Do not copy FAISS row numbers into these tables. A search hit is resolved through
the loaded index's row metadata, then joined to SQLite by video key and shot
index. The stored keyframe `image_path` and timestamp must agree with the row
metadata; reject or withhold mismatched hits. The Phase 2 sidecar remains the
feature provenance and full row record.

## 4. Import, updates, and recovery

The validation boundary is `backend.app.catalog.validation.prepare_catalog_video`:

```python
from backend.app.catalog.validation import prepare_catalog_video

prepared = prepare_catalog_video(
    "data_processed/transnetv2/Videos_L21/L21_V001/shots.json"
)
```

It returns immutable video, shot, and keyframe records without opening SQLite.
Paths remain relative keys, and recorded presentation timestamps are preserved.
It reuses Phase 1 artifact validation and checks catalog metadata, processing
provenance, source size, identity, and file containment. Invalid inputs
raise `CatalogValidationError`. The importer calls it before beginning its
write transaction and supplies a UTC `imported_at` value on successful import.
`import_video()` holds the existing Phase 1 `.processing.lock` during validation
and through commit, so publication and import cannot overlap. A busy lock fails
with a clear error; rerun after Phase 1 completes. File signatures also detect
changes made during validation outside that locking protocol.

The validator requires the source video to exist and match the recorded size.
It checks the detector artifacts and fully decodes each
JPEG using the existing Phase 1 validator; it does not rerun video decoding or
shot detection. Prepared records omit `imported_at`, which belongs to the
successful import. Duplicate names across batches require a database lookup
inside the importer and are not checked by this validation function.

Initialize and import from the project root:

```bash
uv run alembic upgrade head
uv run python -m backend.app.catalog.cli import --input data_processed/transnetv2

# A single video or batch can also be imported.
uv run python -m backend.app.catalog.cli import \
  --input data_processed/transnetv2/Videos_L21/L21_V001/shots.json

# An alternate catalog must be migrated separately first.
uv run alembic -x database_path=data_processed/database/test.sqlite3 upgrade head
uv run python -m backend.app.catalog.cli import \
  --database-path data_processed/database/test.sqlite3
```

The CLI scans published `shots.json` files in sorted order and excludes hidden
staging/backup directories. It reports errors per artifact, continues to other
videos, and exits with status 1 on any failure or when no inputs are found.
An unmigrated database is reported as an import failure; the CLI does not create
schema. Optional `--project-root` redirects the artifact roots and default
catalog to another project directory.

`backend.app.catalog.importer.import_video(manifest_path, sessions)` revalidates
on every run and upserts the video, deletes its previous shots (cascading to
keyframes), and inserts the current shot/keyframe rows in one transaction.
A no-op UPDATE reserves SQLite's writer before duplicate-name checks, preventing
concurrent writers from both accepting the same video name in different batches.
Validation and image I/O occur before that short transaction. Repeated imports
produce the same source rows without duplicates; successful imports refresh
`imported_at`. Validation or write failure leaves the last committed rows intact.
Missing artifacts do not cause existing catalog rows to be deleted. Failed
refreshes preserve a previous snapshot rather than prove its freshness; M4 must
check catalog/index provenance before serving search.

Typed queries return immutable records detached from their SQLAlchemy sessions:

```python
from backend.app.database import create_catalog_engine, create_session_factory
from backend.app.catalog.queries import CatalogQueries

engine = create_catalog_engine()
queries = CatalogQueries(create_session_factory(engine))
videos = queries.list_videos(limit=50, offset=0)
video = queries.get_video("Videos_L21", "L21_V001")
keyframes = queries.list_keyframes("Videos_L21", "L21_V001", limit=50, offset=0)
engine.dispose()
```

Pages expose `items`, `total`, `limit`, and `offset`. Videos sort by
`(batch, video_name)` and keyframes by `shot_index`. Limits must be integers
between 1 and 1000, and offsets nonnegative integers. Detail and keyframe queries
return `None` for a missing video; an offset past the end returns an empty page.
Stored timestamps are passed through without deriving times from FPS.

Run the catalog tests with:

```bash
uv run pytest tests/test_catalog_validation.py tests/test_catalog_migrations.py \
  tests/test_catalog_import.py -q
```

The implemented import/recovery workflow is:

1. Scan Phase 1 `shots.json` files under `transnetv2/`; validate each result
   with the existing pipeline validators before writing SQLite.
2. In one transaction per video, upsert `videos` and replace its `shots` and
   `keyframes`. Preserve the last committed rows if validation or import fails.
3. Re-run the importer after CLI processing or at controlled API startup.
   Report catalog readiness separately from search readiness. Log invalid
   artifacts with their video path rather than partially importing them.

SQLite writes need a single owner at a time. Use short transactions and a busy
timeout. The catalog can be recreated from validated Phase 1 artifacts.

## 5. FAISS index contract

The current builder concatenates per-video `.npy` files in stable filename
order. Rows within a file follow `embedding_index`. It creates an exact
`IndexFlatIP` over L2-normalized float32 image vectors. For the default CLIP
ViT-B/32 model the dimension is 512; inner product then equals cosine
similarity. The score ranks similarity and is not an object-presence probability.

For `model_key=clip`, the global published set is:

```text
data_processed/indexes/
├── clip_keyframe_embedding.index          # FAISS vectors
├── clip_keyframe_metadata.json            # ordered row records
└── clip_keyframe_index.json               # index manifest
```

Each row record includes `index`, `batch`, `video_name`,
`embedding_index`, `keyframe_name`, `shot_index`, `frame_index`, `timestamp`,
and `image_path`. The manifest includes `index_version`, `metric`,
`normalized_features`, `vector_count`, `dimension`, and source file names,
feature sizes, and sidecar SHA-256 hashes. Optional playlist
indexes live under `indexes/playlist/` and have their own independent row
numbers, metadata, and manifest.

At API startup, load the matching index, row metadata, and manifest as one
snapshot. Verify index type/metric, dimension, vector count, contiguous row
indices, finite values, model and representation compatibility with the text
encoder. Check each source sidecar's `input_shots_sha256` against its
current Phase 1 `shots.json`. Check every row's video/shot/keyframe
against the synchronized catalog. A missing or inconsistent set makes search
unready; the catalog and media routes can remain available.

The builder compares source file names, feature sizes, and sidecar SHA-256 hashes before
skipping a rebuild and publishes the manifest last. Validate the loaded index
and row metadata before serving search. Keep the CLI indexer as the rebuild
path. After new features, rebuild the full index and restart the API to load
the new snapshot; live swapping is a later concern.

For 512 float32 values, vectors alone use about 2 KiB per keyframe, before row
metadata and model memory. Measure actual corpus memory and latency before
changing the index type.

## 6. Implementation order

1. **Complete (M3):** SQLite models/migration, validated atomic importer, typed
   queries, and import CLI.
2. **Next (M4):** Harden index publication and loading, then add reusable text-to-keyframe
   retrieval over the published FAISS snapshot.
3. **Planned (M5):** Build FastAPI catalog, search, health, and media routes on those services.

See [backend architecture](05_BACKEND_ARCHITECTURE.md) for routes and
[implementation TODOs](TODOS.md) for concrete tasks.
