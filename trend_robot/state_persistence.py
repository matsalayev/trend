"""
Trend Robot - State Persistence

Pozitsiya holatini saqlash va qayta yuklash.
Bot restart bo'lganda individual grid levels va lot sizes saqlanadi.

Backend tanlash:
    STATE_BACKEND=file (default) — JSON fayllar /data/state/ papkasida
    STATE_BACKEND=redis — Redis da saqlash (STATE_REDIS_URL env var bilan)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Backend konfiguratsiyasi
STATE_BACKEND = os.getenv("STATE_BACKEND", "file").lower()  # "file" yoki "redis"
STATE_REDIS_URL = os.getenv("STATE_REDIS_URL", "redis://localhost:6379/0")


@dataclass
class PersistedPosition:
    """Saqlanuvchi pozitsiya ma'lumotlari"""
    id: str
    side: str  # "buy" or "sell"
    entry_price: float
    lot: float
    level: int
    opened_at: str  # ISO format


class StatePersistence:
    """
    Pozitsiya holatini faylga saqlash va yuklash.

    Bot restart bo'lganda exchange API faqat aggregated pozitsiya qaytaradi.
    Bu klass individual pozitsiyalar (grid levels, lot sizes) ni saqlaydi.

    Usage:
        persistence = StatePersistence(user_id="123", symbol="BTCUSDT")

        # Pozitsiya ochilganda
        persistence.save_position({
            "id": "order123",
            "side": "buy",
            "entry_price": 50000.0,
            "lot": 0.01,
            "level": 0
        })

        # Pozitsiya yopilganda
        persistence.remove_position("order123")

        # Bot restart bo'lganda
        positions = persistence.load_positions()
    """

    def __init__(self, user_id: str, symbol: str, state_dir: str = "/data/state"):
        """
        Args:
            user_id: Foydalanuvchi ID
            symbol: Trading symbol (e.g., BTCUSDT)
            state_dir: State fayllar papkasi (default: /data/state for Docker compatibility)
        """
        self.user_id = user_id
        self.symbol = symbol
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / f"{user_id}_{symbol}.json"
        self._enabled = True

        # Papkani yaratish (graceful handling for permission errors)
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning(f"Cannot create state directory {state_dir}, state persistence disabled")
            self._enabled = False
        except Exception as e:
            logger.warning(f"Failed to create state directory: {e}, state persistence disabled")
            self._enabled = False

    def save_position(self, position: Dict[str, Any]) -> bool:
        """
        Yangi pozitsiyani saqlash.

        Args:
            position: Pozitsiya ma'lumotlari (id, side, entry_price, lot, level)

        Returns:
            True agar muvaffaqiyatli
        """
        if not self._enabled:
            return False
        try:
            state = self._load_state()

            # Mavjud pozitsiyani yangilash yoki yangi qo'shish
            positions = state.get("positions", [])
            position_ids = {p["id"] for p in positions}

            if position.get("id") in position_ids:
                # Mavjud pozitsiyani yangilash
                positions = [
                    position if p["id"] == position["id"] else p
                    for p in positions
                ]
            else:
                # Yangi pozitsiya qo'shish
                if "opened_at" not in position:
                    position["opened_at"] = datetime.utcnow().isoformat()
                positions.append(position)

            state["positions"] = positions
            state["updated_at"] = datetime.utcnow().isoformat()

            self._save_state(state)
            logger.debug(f"Position saved: {position.get('id')} (level={position.get('level')})")
            return True

        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            return False

    def remove_position(self, position_id: str) -> bool:
        """
        Pozitsiyani o'chirish (yopilganda).

        Args:
            position_id: O'chiriladigan pozitsiya ID

        Returns:
            True agar muvaffaqiyatli
        """
        if not self._enabled:
            return False
        try:
            state = self._load_state()
            positions = state.get("positions", [])

            # Pozitsiyani filtrlash
            new_positions = [p for p in positions if p.get("id") != position_id]

            if len(new_positions) < len(positions):
                state["positions"] = new_positions
                state["updated_at"] = datetime.utcnow().isoformat()
                self._save_state(state)
                logger.debug(f"Position removed: {position_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to remove position: {e}")
            return False

    def load_positions(self) -> List[Dict[str, Any]]:
        """
        Saqlangan pozitsiyalarni yuklash.

        Returns:
            Pozitsiyalar ro'yxati
        """
        if not self._enabled:
            return []
        try:
            state = self._load_state()
            positions = state.get("positions", [])
            logger.info(f"Loaded {len(positions)} positions from state file")
            return positions

        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            return []

    def clear_positions(self) -> bool:
        """
        Barcha pozitsiyalarni o'chirish.

        Returns:
            True agar muvaffaqiyatli
        """
        if not self._enabled:
            return False
        try:
            state = self._load_state()
            state["positions"] = []
            state["updated_at"] = datetime.utcnow().isoformat()
            self._save_state(state)
            logger.info("All positions cleared from state file")
            return True

        except Exception as e:
            logger.error(f"Failed to clear positions: {e}")
            return False

    def save_stats(self, stats: Dict[str, Any]) -> bool:
        """
        Statistika ma'lumotlarini saqlash.

        Args:
            stats: Statistika (total_trades, winning_trades, etc.)

        Returns:
            True agar muvaffaqiyatli
        """
        if not self._enabled:
            return False
        try:
            state = self._load_state()
            state["stats"] = stats
            state["updated_at"] = datetime.utcnow().isoformat()
            self._save_state(state)
            return True

        except Exception as e:
            logger.error(f"Failed to save stats: {e}")
            return False

    def load_stats(self) -> Dict[str, Any]:
        """
        Statistika ma'lumotlarini yuklash.

        Returns:
            Statistika dict
        """
        if not self._enabled:
            return {}
        try:
            state = self._load_state()
            return state.get("stats", {})

        except Exception as e:
            logger.error(f"Failed to load stats: {e}")
            return {}

    def get_state_file_path(self) -> str:
        """State fayl yo'lini olish"""
        return str(self.state_file)

    def _load_state(self) -> Dict[str, Any]:
        """State faylni o'qish"""
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in state file, creating new")
                return {"positions": [], "stats": {}}
        return {"positions": [], "stats": {}}

    def _save_state(self, state: Dict[str, Any]):
        """State faylni yozish"""
        self.state_file.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def close(self):
        """Resurslarni tozalash (file backend uchun hech narsa qilmaydi)"""
        pass


class RedisStatePersistence(StatePersistence):
    """
    Redis asosida state saqlash.

    StatePersistence ning file I/O metodlarini Redis bilan almashtiradi.
    Barcha biznes logika (save_position, remove_position, etc.) ota klassdan meros.

    Redis key formati: trend:state:{user_id}:{symbol}
    Qiymat: JSON string (positions + stats + updated_at)

    Usage:
        persistence = RedisStatePersistence(user_id="123", symbol="BTCUSDT")
        persistence.save_position({...})  # Redis ga yozadi
    """

    def __init__(self, user_id: str, symbol: str, redis_url: str = None):
        """
        Args:
            user_id: Foydalanuvchi ID
            symbol: Trading symbol (e.g., BTCUSDT)
            redis_url: Redis ulanish URL (default: STATE_REDIS_URL env var)
        """
        # Ota klassning __init__ ni chaqirmaymiz — file yaratish kerak emas
        self.user_id = user_id
        self.symbol = symbol
        self._key = f"trend:state:{user_id}:{symbol}"
        self._enabled = True
        self._redis = None

        try:
            import redis as sync_redis
            self._redis = sync_redis.from_url(
                redis_url or STATE_REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._redis.ping()
            logger.info(f"Redis state persistence ulandi: {user_id}:{symbol}")
        except ImportError:
            logger.warning(
                "redis kutubxonasi o'rnatilmagan — pip install redis. "
                "File backend ishlatiladi."
            )
            self._enabled = False
        except Exception as e:
            logger.warning(f"Redis ulanish xatosi: {e}, state persistence o'chirildi")
            self._enabled = False

    def _load_state(self) -> Dict[str, Any]:
        """Redis dan state o'qish"""
        try:
            data = self._redis.get(self._key)
            if data:
                return json.loads(data)
        except json.JSONDecodeError:
            logger.warning(f"Redis da noto'g'ri JSON: {self._key}, yangi state yaratilmoqda")
        except Exception as e:
            logger.error(f"Redis o'qish xatosi: {e}")
        return {"positions": [], "stats": {}}

    def _save_state(self, state: Dict[str, Any]):
        """Redis ga state yozish"""
        try:
            self._redis.set(
                self._key,
                json.dumps(state, indent=2, ensure_ascii=False)
            )
        except Exception as e:
            logger.error(f"Redis yozish xatosi: {e}")

    def get_state_file_path(self) -> str:
        """State saqlash joyini ko'rsatish (Redis key)"""
        return f"redis://{self._key}"

    def close(self):
        """Redis ulanishni yopish"""
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None


def create_state_persistence(
    user_id: str,
    symbol: str,
    state_dir: str = "/data/state",
) -> StatePersistence:
    """
    State persistence backend yaratish (factory function).

    STATE_BACKEND env var asosida Redis yoki file backend qaytaradi.
    Redis ishlamasa, avtomatik file backend ga o'tadi (fallback).

    Args:
        user_id: Foydalanuvchi ID
        symbol: Trading symbol (e.g., BTCUSDT)
        state_dir: File backend uchun papka (default: /data/state)

    Returns:
        StatePersistence yoki RedisStatePersistence instance
    """
    if STATE_BACKEND == "redis":
        persistence = RedisStatePersistence(user_id, symbol)
        if persistence._enabled:
            return persistence
        logger.warning(
            f"Redis backend ishlamadi ({user_id}:{symbol}), "
            f"file backend ga o'tilmoqda"
        )

    return StatePersistence(user_id=user_id, symbol=symbol, state_dir=state_dir)
