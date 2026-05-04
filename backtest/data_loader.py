"""
Historical candle data loader for backtesting.

Bitget public API dan candle'larni oladi va lokal CSV ga keshlaydi.
API key kerak emas — market endpoint ochiq.
"""

import csv
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class Candle:
    timestamp: int      # ms
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc)


# Bitget granularity mapping (public API — capital H/D)
GRANULARITY_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
}

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1H": 3_600_000,
    "4H": 14_400_000,
    "1D": 86_400_000,
}


def _fetch_bitget_candles(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    product_type: str = "USDT-FUTURES",
) -> List[Candle]:
    """
    Bitget public market endpoint — 1000 candles max per request.
    Avval `mix/market/candles` (yangi data), keyin `history-candles` (eski) try qiladi.
    """
    granularity = GRANULARITY_MAP[timeframe]
    step_ms = TIMEFRAME_MS[timeframe]

    all_candles: List[Candle] = []

    # Ikki endpoint try qilamiz: avval recent (limit 1000), keyin history (limit 200)
    endpoints = [
        ("https://api.bitget.com/api/v2/mix/market/candles", 1000),
        ("https://api.bitget.com/api/v2/mix/market/history-candles", 200),
    ]

    for url, max_per_req in endpoints:
        current = start_ms
        endpoint_candles: List[Candle] = []
        empty_in_a_row = 0

        while current < end_ms:
            batch_end = min(current + step_ms * max_per_req, end_ms)
            params = {
                "symbol": symbol,
                "productType": product_type,
                "granularity": granularity,
                "startTime": str(current),
                "endTime": str(batch_end),
                "limit": str(max_per_req),
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 400:
                    logger.debug(f"Bitget 400 for {symbol} ({url}) — keyingi endpoint sinab ko'ramiz")
                    break  # bu endpoint ishlamadi, history-candles'ga o'tamiz
                r.raise_for_status()
                data = r.json()
                if data.get("code") != "00000":
                    logger.error(f"Bitget xato ({url}): {data}")
                    break
                raw = data.get("data", [])
                if not raw:
                    empty_in_a_row += 1
                    if empty_in_a_row >= 3:
                        # Bu endpoint shu davr uchun ma'lumot bera olmadi
                        break
                    current = batch_end
                    time.sleep(0.05)
                    continue
                empty_in_a_row = 0
                for item in raw:
                    ts = int(item[0])
                    endpoint_candles.append(Candle(
                        timestamp=ts,
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]) if len(item) > 5 else 0.0,
                    ))
                last_ts = int(raw[-1][0])
                if last_ts <= current:
                    break
                current = last_ts + step_ms
                time.sleep(0.1)
            except requests.RequestException as e:
                logger.error(f"Request xato: {e}")
                time.sleep(2)
                continue

        all_candles.extend(endpoint_candles)
        # Agar yetarli ma'lumot olingan bo'lsa (90% qoplagan) — keyingi endpoint kerak emas
        expected_bars = (end_ms - start_ms) // step_ms
        if expected_bars > 0 and len(endpoint_candles) >= expected_bars * 0.9:
            break

    # Sort + deduplicate
    seen = set()
    unique = []
    for c in sorted(all_candles, key=lambda x: x.timestamp):
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)

    return unique


def _cache_path(symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    name = f"{symbol}_{timeframe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    return CACHE_DIR / name


def _save_cache(candles: List[Candle], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            w.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])


def _load_cache(path: Path) -> List[Candle]:
    candles: List[Candle] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            candles.append(Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return candles


def load_candles(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    use_cache: bool = True,
) -> List[Candle]:
    """
    Tarixiy candle'larni olish. Agar kesh bor bo'lsa — undan, aks holda Bitget'dan.

    Args:
        symbol: 'BTCUSDT', 'BNBUSDT', 'SOLUSDT' va h.k.
        timeframe: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
        start, end: UTC datetime
        use_cache: Mavjud keshdan foydalanish

    Returns:
        Candle ro'yxati (vaqt bo'yicha tartiblangan)
    """
    if timeframe not in GRANULARITY_MAP:
        raise ValueError(f"Noto'g'ri timeframe: {timeframe}")

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    path = _cache_path(symbol, timeframe, start, end)
    if use_cache and path.exists():
        logger.info(f"Kesh ishlatilmoqda: {path.name}")
        return _load_cache(path)

    logger.info(f"Bitget'dan yuklanmoqda: {symbol} {timeframe} {start.date()} -> {end.date()}")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    candles = _fetch_bitget_candles(symbol, timeframe, start_ms, end_ms)

    if candles:
        _save_cache(candles, path)
        logger.info(f"Yuklandi: {len(candles)} candle, kesh: {path.name}")
    else:
        logger.warning(f"Candle topilmadi: {symbol} {timeframe}")

    return candles


if __name__ == "__main__":
    # Sinov
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start = datetime.now(timezone.utc) - timedelta(days=7)
    end = datetime.now(timezone.utc)
    cs = load_candles("BTCUSDT", "15m", start, end)
    print(f"Yuklandi: {len(cs)} candle")
    if cs:
        print(f"Birinchi: {cs[0].dt}  close=${cs[0].close:.2f}")
        print(f"Oxirgi:   {cs[-1].dt}  close=${cs[-1].close:.2f}")
