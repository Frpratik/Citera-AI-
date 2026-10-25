from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


def create_pool(database_url: str, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """Create a connection pool whose connections understand the `vector` type.

    Run migrations BEFORE calling this: register_vector needs the extension to exist.
    Rows come back as dicts. `pool.connection()` commits on success, rolls back on error.
    """
    return ConnectionPool(
        database_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        configure=register_vector,
        # Test a connection before handing it out, so a database restart or a dropped
        # connection is replaced quietly instead of failing the next request.
        check=ConnectionPool.check_connection,
        open=True,
    )
