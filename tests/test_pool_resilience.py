from app.db.pool import create_pool


def test_the_pool_replaces_a_connection_the_database_dropped(db_pool):
    """Simulates a database restart or a dropped connection: kill the connection from outside,
    then use the pool again. It must hand out a working connection, not the dead one."""
    pool = create_pool(db_pool.conninfo, min_size=1, max_size=1)
    try:
        with pool.connection() as conn:
            pid = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
        with db_pool.connection() as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))

        with pool.connection() as conn:  # without the pool's connection check this would raise
            assert conn.execute("SELECT 1 AS ok").fetchone()["ok"] == 1
            assert conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"] != pid
    finally:
        pool.close()
