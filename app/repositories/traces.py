from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


class TraceRepository:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def add(self, owner_id: str, request_id: str | None, route: str | None, outcome: str, summary: dict) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO chat_traces (owner_id, request_id, route, outcome, total_ms, llm_calls, "
                "input_tokens, output_tokens, fallback_used, spans, events) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    owner_id, request_id, route, outcome, summary["total_ms"], summary["llm_calls"],
                    summary["input_tokens"], summary["output_tokens"], summary["fallback_used"],
                    Jsonb(summary["spans"]), Jsonb(summary["events"]),
                ),
            )

    def usage(self, owner_id: str, hours: int) -> dict:
        """Totals for one user over the last `hours` hours."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) AS questions, "
                "       coalesce(sum(llm_calls), 0) AS llm_calls, "
                "       coalesce(sum(input_tokens), 0) AS input_tokens, "
                "       coalesce(sum(output_tokens), 0) AS output_tokens, "
                "       coalesce(round(avg(total_ms)), 0) AS avg_response_ms, "
                "       count(*) FILTER (WHERE fallback_used) AS fallbacks "
                "FROM chat_traces WHERE owner_id = %s AND created_at > now() - make_interval(hours => %s)",
                (owner_id, hours),
            ).fetchone()
        return {key: int(value) for key, value in row.items()}

    def recent(self, owner_id: str, limit: int = 20) -> list[dict]:
        with self._pool.connection() as conn:
            return conn.execute(
                "SELECT created_at, route, outcome, total_ms, llm_calls, input_tokens, output_tokens, "
                "fallback_used, spans FROM chat_traces WHERE owner_id = %s ORDER BY id DESC LIMIT %s",
                (owner_id, limit),
            ).fetchall()
