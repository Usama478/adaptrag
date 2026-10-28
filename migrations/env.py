from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from adaptrag.config import get_settings
from adaptrag.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def get_database_url() -> str:
    """Use a URL passed in by code (the tests), otherwise the app's own setting."""
    url = config.attributes.get("database_url")
    if isinstance(url, str):
        return url
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Print the SQL instead of running it (used with `alembic upgrade head --sql`)."""
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the database and apply the migrations."""
    engine = create_engine(get_database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
