"""SQLite engine and sessions for the local catalog.

Schema creation belongs to Alembic; importing this module does not create tables.
"""

from pathlib import Path
import sqlite3

from sqlalchemy import Engine, URL, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data_processed/database/catalog.sqlite3"
BUSY_TIMEOUT_MS = 5_000


def create_catalog_engine(database_path: str | Path = DEFAULT_DATABASE_PATH) -> Engine:
    """Open a catalog file, resolving relative paths from the project root."""
    path = Path(database_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        URL.create("sqlite+pysqlite", database=str(path)),
        connect_args={"timeout": BUSY_TIMEOUT_MS / 1_000, "autocommit": False},
    )

    @event.listens_for(engine, "connect")
    def configure_connection(connection: sqlite3.Connection, _record: object) -> None:
        # SQLite ignores foreign_keys changes inside a transaction.
        previous_autocommit = connection.autocommit
        connection.autocommit = True
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        finally:
            cursor.close()
            connection.autocommit = previous_autocommit

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create sessions; use ``with factory.begin()`` for short atomic writes.

    Validate artifacts before entering that block. It commits on success and
    rolls back on failure, then closes the session.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)
