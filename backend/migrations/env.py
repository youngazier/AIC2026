"""Migrate the local catalog using the application's SQLite configuration."""

from alembic import context
from sqlalchemy import URL

from backend.app.database import DEFAULT_DATABASE_PATH, PROJECT_ROOT, create_catalog_engine
from backend.app.models import Base


def database_path():
    from pathlib import Path

    path = Path(context.get_x_argument(as_dictionary=True).get('database_path', DEFAULT_DATABASE_PATH))
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_migrations_offline():
    context.configure(
        url=URL.create('sqlite+pysqlite', database=str(database_path())),
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={'paramstyle': 'named'},
        transactional_ddl=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    engine = create_catalog_engine(database_path())
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=Base.metadata,
                transactional_ddl=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
