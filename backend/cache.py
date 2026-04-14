"""
=============================================================================
cache.py — SQLite-backed result cache with TTL
=============================================================================

Replaces the ad-hoc JSON file cache in b2b_engine. Results are stored in
a single SQLite DB, keyed by product_id (extracted from URL). The cache
TTL is configurable (default 24 hours).

Usage:
    cache = AnalysisCache(ttl_hours=24)
    result = cache.get("1955625")
    cache.set("1955625", result_dict)
    cache.invalidate("1955625")
    cache.cleanup_expired()
=============================================================================
"""

import json
import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default cache location — in project data dir
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "b2b_cache" / "analysis_cache.db"


class AnalysisCache:
    """
    SQLite-backed cache for product analysis results.

    Schema:
        product_id  TEXT PRIMARY KEY
        result_json TEXT
        created_at  TEXT   (ISO-8601)
        expires_at  TEXT   (ISO-8601)
        hit_count   INTEGER
    """

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH, ttl_hours: int = 24):
        self.db_path = Path(db_path)
        self.ttl = timedelta(hours=ttl_hours)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    product_id  TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    expires_at  TEXT NOT NULL,
                    hit_count   INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON analysis_cache(expires_at)
            """)
            conn.commit()
        logger.info(f"[cache] SQLite cache initialised at {self.db_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Return cached result or None if missing/expired."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json, expires_at, hit_count FROM analysis_cache WHERE product_id = ?",
                (product_id,)
            ).fetchone()

            if row is None:
                logger.debug(f"[cache] MISS  {product_id}")
                return None

            if row["expires_at"] < now:
                logger.info(f"[cache] EXPIRED {product_id}")
                conn.execute("DELETE FROM analysis_cache WHERE product_id = ?", (product_id,))
                conn.commit()
                return None

            # Update hit counter
            conn.execute(
                "UPDATE analysis_cache SET hit_count = hit_count + 1 WHERE product_id = ?",
                (product_id,)
            )
            conn.commit()

        logger.info(f"[cache] HIT    {product_id}  (hits: {row['hit_count'] + 1})")
        result = json.loads(row["result_json"])
        result["from_cache"] = True
        result["cache_expires_at"] = row["expires_at"]
        return result

    def set(self, product_id: str, result: Dict[str, Any]) -> None:
        """Store result in cache. Overwrites any existing entry."""
        now = datetime.utcnow()
        expires = (now + self.ttl).isoformat()
        # Strip transient fields before storing
        clean = {k: v for k, v in result.items() if k not in ("from_cache", "cache_expires_at")}
        payload = json.dumps(clean, ensure_ascii=False, default=str)

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO analysis_cache (product_id, result_json, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(product_id) DO UPDATE SET
                    result_json = excluded.result_json,
                    created_at  = excluded.created_at,
                    expires_at  = excluded.expires_at,
                    hit_count   = 0
            """, (product_id, payload, now.isoformat(), expires))
            conn.commit()
        logger.info(f"[cache] SET    {product_id}  (expires {expires})")

    def invalidate(self, product_id: str) -> bool:
        """Delete a specific entry. Returns True if it existed."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM analysis_cache WHERE product_id = ?", (product_id,)
            )
            conn.commit()
        existed = cur.rowcount > 0
        if existed:
            logger.info(f"[cache] INVALIDATED {product_id}")
        return existed

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count deleted."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM analysis_cache WHERE expires_at < ?", (now,)
            )
            conn.commit()
        count = cur.rowcount
        if count:
            logger.info(f"[cache] CLEANUP: removed {count} expired entries")
        return count

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics (for /api/health or admin endpoints)."""
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM analysis_cache").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM analysis_cache WHERE expires_at > ?", (now,)
            ).fetchone()[0]
            total_hits = conn.execute(
                "SELECT COALESCE(SUM(hit_count), 0) FROM analysis_cache"
            ).fetchone()[0]
        return {
            "total_entries": total,
            "active_entries": active,
            "expired_entries": total - active,
            "total_hits": total_hits,
            "db_path": str(self.db_path),
            "ttl_hours": self.ttl.total_seconds() / 3600,
        }


def extract_product_id(url_or_id: str) -> str:
    """
    Extract a stable cache key from a product URL or bare ID.

    Examples:
        "https://uzum.uz/uz/product/smartfon-honor-x9c-1955625/reviews" → "1955625"
        "1955625" → "1955625"
        "https://uzum.uz/ru/product/tovar-12345" → "12345"
    """
    import re
    # Try to extract trailing digits from URL path segment
    match = re.search(r"-(\d{5,})(?:/|$|\?)", url_or_id)
    if match:
        return match.group(1)
    # Fallback: extract any long digit sequence
    match = re.search(r"\b(\d{5,})\b", url_or_id)
    if match:
        return match.group(1)
    # Last resort: hash the whole string
    return hashlib.sha256(url_or_id.encode()).hexdigest()[:16]
