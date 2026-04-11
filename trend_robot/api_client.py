"""
Trend Robot - Bitget API Client

REST API va order management
"""

import hmac
import hashlib
import base64
import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from urllib.parse import urlencode

import aiohttp

from .config import APIConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#                               CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreakerState:
    """Circuit breaker holatlari"""
    CLOSED = "closed"      # Normal ishlash
    OPEN = "open"          # Barcha so'rovlar bloklangan
    HALF_OPEN = "half_open"  # Test so'rov ruxsat berilgan


class CircuitBreaker:
    """
    API xatolari uchun circuit breaker pattern.

    Agar API ko'p marta fail bo'lsa, ma'lum vaqtga barcha so'rovlar bloklanadi.
    Bu server va API'ni ortiqcha yukdan himoya qiladi.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 1
    ):
        """
        Args:
            failure_threshold: Nechta xato bo'lganda circuit ochiladi
            recovery_timeout: Circuit ochiq bo'lgan vaqt (sekund)
            half_open_requests: Half-open holatda ruxsat berilgan so'rovlar soni
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._state = CircuitBreakerState.CLOSED
        self._failures = 0
        self._last_failure_time: float = 0
        self._half_open_successes = 0

    @property
    def state(self) -> str:
        """Joriy holat"""
        # Open holatdan half-open ga avtomatik o'tish
        if self._state == CircuitBreakerState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_successes = 0
                logger.info("Circuit breaker: OPEN -> HALF_OPEN")
        return self._state

    def is_available(self) -> bool:
        """So'rov yuborish mumkinmi?"""
        state = self.state  # Property chaqiruvi

        if state == CircuitBreakerState.CLOSED:
            return True
        elif state == CircuitBreakerState.OPEN:
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Muvaffaqiyatli so'rovni qayd qilish"""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_requests:
                self._state = CircuitBreakerState.CLOSED
                self._failures = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
        elif self._state == CircuitBreakerState.CLOSED:
            # Har 10 ta success da failure counter reset
            if self._failures > 0:
                self._failures = max(0, self._failures - 1)

    def record_failure(self):
        """Xatolik qayd qilish"""
        self._failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Half-open da xato - qaytadan open
            self._state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN (fail)")

        elif self._state == CircuitBreakerState.CLOSED:
            if self._failures >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker: CLOSED -> OPEN "
                    f"({self._failures} failures, waiting {self.recovery_timeout}s)"
                )


class CircuitBreakerError(Exception):
    """Circuit breaker ochiq - so'rov bloklandi"""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#                               EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class BitgetAPIError(Exception):
    """Bitget API xatosi"""
    def __init__(self, code: str, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"[{code}] {message}")


class BitgetAuthError(BitgetAPIError):
    """Autentifikatsiya xatosi"""
    pass


class BitgetRateLimitError(BitgetAPIError):
    """Rate limit xatosi"""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#                               DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Pozitsiya ma'lumotlari"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    liquidation_price: float
    leverage: int
    margin_mode: str


@dataclass
class Order:
    """Order ma'lumotlari"""
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    order_type: str
    status: str
    filled_size: float
    avg_fill_price: float
    create_time: int


# ═══════════════════════════════════════════════════════════════════════════════
#                               BITGET CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class BitgetClient:
    """
    Bitget Futures API Client

    REST API orqali trading operatsiyalari.
    Async context manager sifatida ishlatilishi mumkin:

    async with BitgetClient(config) as client:
        await client.get_price("BTCUSDT")
    """

    def __init__(self, config: APIConfig):
        """
        Args:
            config: API konfiguratsiyasi
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_requests=2
        )
        # Symbol info cache (pricePlace, volumePlace, etc.)
        self._symbol_cache: Dict[str, Dict] = {}
        self._closed = False

    # ─────────────────────────────────────────────────────────────────────────
    #                           CONTEXT MANAGER
    # ─────────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "BitgetClient":
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures session cleanup"""
        await self.close()

    def __del__(self):
        """Destructor - warn if session not closed"""
        if self._session and not self._session.closed and not self._closed:
            logger.warning(
                "BitgetClient session was not properly closed! "
                "Use 'await client.close()' or async context manager."
            )

    # ─────────────────────────────────────────────────────────────────────────
    #                           SESSION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        """HTTP session olish"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Sessionni yopish"""
        self._closed = True
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def get_circuit_breaker_status(self) -> dict:
        """Circuit breaker holatini olish"""
        return {
            "state": self._circuit_breaker.state.value if hasattr(self._circuit_breaker.state, 'value') else str(self._circuit_breaker.state),
            "isAvailable": self._circuit_breaker.is_available(),
            "failures": self._circuit_breaker._failures,
            "recoveryTimeout": self._circuit_breaker.recovery_timeout
        }

    # ─────────────────────────────────────────────────────────────────────────
    #                           AUTHENTICATION
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_signature(self, timestamp: str, method: str,
                            path: str, body: str = "") -> str:
        """
        HMAC-SHA256 imzo yaratish

        Bitget imzo formati:
        sign = base64(hmac_sha256(secret, timestamp + method + path + body))
        """
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.config.SECRET_KEY.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(signature.digest()).decode('utf-8')

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict:
        """API headers yaratish"""
        timestamp = str(int(time.time() * 1000))
        sign = self._generate_signature(timestamp, method, path, body)

        headers = {
            "ACCESS-KEY": self.config.API_KEY,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.config.PASSPHRASE,
            "Content-Type": "application/json",
            "locale": "en-US"
        }

        # Demo mode uchun maxsus header
        # Market data endpoint'lari (candles, ticker) public — paptrading header
        # demo muhitda flat/noto'g'ri narx qaytaradi, shuning uchun skip qilamiz
        base_path = path.split("?")[0]
        if self.config.DEMO_MODE and not base_path.startswith("/api/v2/mix/market/"):
            headers["paptrading"] = "1"

        return headers

    # ─────────────────────────────────────────────────────────────────────────
    #                           REQUEST METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def _request(self, method: str, path: str,
                       params: Optional[Dict] = None,
                       body: Optional[Dict] = None) -> Dict:
        """
        API so'rov yuborish

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API endpoint
            params: Query parameters
            body: Request body

        Returns:
            API javobi
        """
        # Circuit breaker tekshirish
        if not self._circuit_breaker.is_available():
            raise CircuitBreakerError(
                f"Circuit breaker OPEN - API so'rovlar bloklangan. "
                f"{self._circuit_breaker.recovery_timeout}s kutilmoqda."
            )

        session = await self._get_session()

        url = self.config.BASE_URL + path
        request_path = path

        # Query string qo'shish (SORTED - imzo uchun muhim!)
        if params:
            query = urlencode(sorted(params.items()))
            request_path = f"{path}?{query}"
            url = f"{url}?{query}"

        # Body
        body_str = json.dumps(body) if body else ""

        # Headers (request_path ishlatiladi, path emas)
        headers = self._get_headers(method, request_path, body_str)

        # Request
        for attempt in range(self.config.MAX_RETRIES):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=body_str if body else None
                ) as response:

                    text = await response.text()

                    # Rate limit
                    if response.status == 429:
                        wait_time = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limit! {wait_time}s kutilmoqda...")
                        self._circuit_breaker.record_failure()
                        await asyncio.sleep(wait_time)
                        continue

                    # Cloudflare 5xx xatolar - retry qilish kerak
                    if response.status >= 500:
                        is_cloudflare = "cloudflare" in text.lower() or "<!DOCTYPE" in text
                        if is_cloudflare:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning(f"Cloudflare {response.status} xatosi (attempt {attempt + 1}), {wait_time}s kutilmoqda...")
                            self._circuit_breaker.record_failure()
                            if attempt < self.config.MAX_RETRIES - 1:
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                raise BitgetAPIError("CLOUDFLARE_ERROR", f"Cloudflare xatosi: {response.status}")

                    # Parse response
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        # HTML javob - bu odatda Cloudflare xatosi
                        if "<!DOCTYPE" in text or "<html" in text.lower():
                            wait_time = 2 ** attempt
                            logger.warning(f"HTML javob olindi (Cloudflare?), attempt {attempt + 1}, {wait_time}s kutilmoqda...")
                            self._circuit_breaker.record_failure()
                            if attempt < self.config.MAX_RETRIES - 1:
                                await asyncio.sleep(wait_time)
                                continue
                        raise BitgetAPIError("PARSE_ERROR", f"JSON parse error: {text[:500]}")

                    # Error check
                    if data.get("code") != "00000":
                        code = data.get("code", "UNKNOWN")
                        msg = data.get("msg", "Unknown error")

                        if "signature" in msg.lower() or "auth" in msg.lower():
                            raise BitgetAuthError(code, msg)
                        if "rate" in msg.lower() or "limit" in msg.lower():
                            self._circuit_breaker.record_failure()
                            raise BitgetRateLimitError(code, msg)

                        raise BitgetAPIError(code, msg, data)

                    # Muvaffaqiyatli so'rov
                    self._circuit_breaker.record_success()
                    return data.get("data", data)

            except aiohttp.ClientError as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                self._circuit_breaker.record_failure()
                if attempt < self.config.MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise BitgetAPIError("NETWORK_ERROR", str(e))

        self._circuit_breaker.record_failure()
        raise BitgetAPIError("MAX_RETRIES", "Maksimal urinishlar soni oshdi")

    async def get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """GET so'rov"""
        return await self._request("GET", path, params=params)

    async def post(self, path: str, body: Optional[Dict] = None) -> Dict:
        """POST so'rov"""
        return await self._request("POST", path, body=body)

    async def delete(self, path: str, body: Optional[Dict] = None) -> Dict:
        """DELETE so'rov"""
        return await self._request("DELETE", path, body=body)

    # ─────────────────────────────────────────────────────────────────────────
    #                           MARKET INFO
    # ─────────────────────────────────────────────────────────────────────────

    async def get_contracts(self, product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Barcha mavjud kontraktlarni olish

        Args:
            product_type: Product type (USDT-FUTURES, COIN-FUTURES, etc.)

        Returns:
            Kontraktlar ro'yxati
        """
        params = {"productType": product_type}
        return await self.get("/api/v2/mix/market/contracts", params)

    async def get_usdt_futures_symbols(self) -> List[str]:
        """
        Barcha USDT-M Futures symbollarini olish

        Returns:
            Symbol nomlari ro'yxati (masalan: ['BTCUSDT', 'ETHUSDT', ...])
        """
        try:
            contracts = await self.get_contracts("USDT-FUTURES")
            symbols = []
            for contract in contracts:
                symbol = contract.get("symbol", "")
                # Faqat USDT bilan tugaydigan va trading uchun ochiq bo'lganlarni olish
                if symbol.endswith("USDT") and contract.get("symbolStatus") == "normal":
                    symbols.append(symbol)
            # Alifbo tartibida, lekin BTC va ETH birinchi
            priority = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
            sorted_symbols = [s for s in priority if s in symbols]
            sorted_symbols += sorted([s for s in symbols if s not in priority])
            return sorted_symbols
        except Exception as e:
            logger.error(f"Symbollarni olishda xatolik: {e}")
            # Fallback - asosiy symbollar
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]

    async def get_symbol_info(self, symbol: str, product_type: str = "USDT-FUTURES") -> Optional[Dict]:
        """
        Symbol uchun trading parametrlarini olish

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            product_type: Product type

        Returns:
            Symbol info dict with:
            - minTradeNum: Minimum order size (in base currency)
            - maxTradeNum: Maximum order size
            - sizeMultiplier: Size step/precision
            - volumePlace: Volume decimal places
            - pricePlace: Price decimal places
            - priceEndStep: Price step
        """
        try:
            contracts = await self.get_contracts(product_type)
            for contract in contracts:
                if contract.get("symbol") == symbol:
                    return {
                        "symbol": symbol,
                        "minTradeNum": float(contract.get("minTradeNum", 0.001)),
                        "maxTradeNum": float(contract.get("maxTradeNum", 10000)),
                        "sizeMultiplier": float(contract.get("sizeMultiplier", 0.001)),
                        "volumePlace": int(contract.get("volumePlace", 3)),
                        "pricePlace": int(contract.get("pricePlace", 2)),
                        "priceEndStep": float(contract.get("priceEndStep", 0.1)),
                        "maxLever": int(contract.get("maxLever", 125)),
                    }
            logger.warning(f"Symbol {symbol} not found in contracts")
            return None
        except Exception as e:
            logger.error(f"Symbol info olishda xatolik: {e}")
            return None

    async def get_price_place(self, symbol: str, product_type: str = "USDT-FUTURES") -> int:
        """
        Symbol uchun narx decimal places olish (cached)

        Args:
            symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
            product_type: Product type

        Returns:
            Price decimal places (default: 2)
        """
        # Cache'dan tekshirish
        cache_key = f"{symbol}_{product_type}"
        if cache_key in self._symbol_cache:
            return self._symbol_cache[cache_key].get("pricePlace", 2)

        # Cache'da yo'q - fetch qilish
        try:
            symbol_info = await self.get_symbol_info(symbol, product_type)
            if symbol_info:
                self._symbol_cache[cache_key] = symbol_info
                return symbol_info.get("pricePlace", 2)
        except Exception as e:
            logger.warning(f"Symbol info olishda xatolik: {e}, default pricePlace=2 ishlatiladi")

        return 2  # Default

    def calculate_lot_size(self, trade_amount_usdt: float, current_price: float,
                           min_trade_num: float, size_multiplier: float,
                           volume_place: int = 3) -> float:
        """
        USDT miqdoridan lot size hisoblash

        Args:
            trade_amount_usdt: Trade amount in USDT
            current_price: Current price of the symbol
            min_trade_num: Minimum order size from exchange
            size_multiplier: Size step/precision from exchange
            volume_place: Decimal places for volume

        Returns:
            Calculated lot size (in base currency)
        """
        if current_price <= 0:
            return min_trade_num

        # USDT dan base currency ga o'tkazish
        raw_lot = trade_amount_usdt / current_price

        # Size multiplier ga moslashtirish (truncate, not round)
        if size_multiplier > 0:
            lot = (raw_lot // size_multiplier) * size_multiplier
        else:
            lot = raw_lot

        # Decimal places ga moslashtirish
        lot = round(lot, volume_place)

        # Minimum bilan tekshirish
        if lot < min_trade_num:
            lot = min_trade_num
            logger.warning(f"Calculated lot {raw_lot:.6f} is below minimum {min_trade_num}, using minimum")

        return lot

    # ─────────────────────────────────────────────────────────────────────────
    #                           ACCOUNT METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def get_account(self, product_type: str = "USDT-FUTURES") -> Dict:
        """Hisob ma'lumotlarini olish"""
        params = {"productType": product_type}
        return await self.get("/api/v2/mix/account/accounts", params)

    async def get_balance(self, product_type: str = "USDT-FUTURES",
                          margin_coin: str = "USDT") -> float:
        """Balansni olish (available — erkin margin)"""
        data = await self.get_account(product_type)
        if isinstance(data, list):
            for account in data:
                if account.get("marginCoin") == margin_coin:
                    return float(account.get("available", 0))
        return 0.0

    async def get_equity(self, product_type: str = "USDT-FUTURES",
                         margin_coin: str = "USDT") -> float:
        """Equity olish (balance + unrealized PnL — drawdown uchun)

        Cross margin da 'available' unrealized PnL ni o'z ichiga olmaydi.
        Drawdown to'g'ri ishlashi uchun equity kerak:
        equity = accountEquity yoki available + unrealizedPL
        """
        data = await self.get_account(product_type)
        if isinstance(data, list):
            for account in data:
                if account.get("marginCoin") == margin_coin:
                    # accountEquity = available + frozen + unrealizedPL
                    equity = account.get("accountEquity")
                    if equity is not None:
                        return float(equity)
                    # Fallback: available + unrealizedPL
                    available = float(account.get("available", 0))
                    unrealized = float(account.get("unrealizedPL", 0))
                    return available + unrealized
        return 0.0

    async def set_leverage(self, symbol: str, leverage: int,
                          product_type: str = "USDT-FUTURES",
                          margin_coin: str = "USDT") -> Dict:
        """Leverage o'rnatish"""
        body = {
            "symbol": symbol,
            "productType": product_type,
            "marginCoin": margin_coin,
            "leverage": str(leverage)
        }
        return await self.post("/api/v2/mix/account/set-leverage", body)

    # ─────────────────────────────────────────────────────────────────────────
    #                           MARKET DATA
    # ─────────────────────────────────────────────────────────────────────────

    async def get_ticker(self, symbol: str,
                         product_type: str = "USDT-FUTURES") -> Dict:
        """Ticker (narx) olish"""
        params = {"symbol": symbol, "productType": product_type}
        data = await self.get("/api/v2/mix/market/ticker", params)
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data

    async def get_price(self, symbol: str,
                        product_type: str = "USDT-FUTURES") -> float:
        """Joriy narxni olish"""
        ticker = await self.get_ticker(symbol, product_type)
        return float(ticker.get("lastPr", ticker.get("last", 0)))

    async def get_candles(self, symbol: str, granularity: str = "1H",
                          limit: int = 100,
                          product_type: str = "USDT-FUTURES") -> List[Dict]:
        """
        Candlestick ma'lumotlarini olish

        Args:
            symbol: Trading pair
            granularity: Timeframe (1m, 5m, 15m, 30m, 1H, 4H, 1D)
            limit: Candlelar soni
            product_type: Product type

        Returns:
            Candle ro'yxati
        """
        params = {
            "symbol": symbol,
            "productType": product_type,
            "granularity": granularity,
            "limit": str(limit)
        }
        return await self.get("/api/v2/mix/market/candles", params)

    # ─────────────────────────────────────────────────────────────────────────
    #                           POSITION METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def get_positions(self, symbol: str = None,
                            product_type: str = "USDT-FUTURES",
                            margin_coin: str = "USDT") -> List[Position]:
        """Pozitsiyalarni olish (symbol bo'yicha filtrlangan)"""
        params = {"productType": product_type, "marginCoin": margin_coin}
        if symbol:
            params["symbol"] = symbol

        data = await self.get("/api/v2/mix/position/all-position", params)

        positions = []
        if data:
            for item in data:
                if float(item.get("total", 0)) > 0:
                    item_symbol = item.get("symbol", "")

                    # FIX: API barcha pozitsiyalarni qaytarishi mumkin —
                    # faqat so'ralgan symbol'ni qabul qilish
                    if symbol and item_symbol != symbol:
                        # FIX v2: Log spamni kamaytirish — debug darajasida
                        logger.debug(
                            f"Skipping position for {item_symbol} "
                            f"(requested {symbol})"
                        )
                        continue

                    positions.append(Position(
                        symbol=item_symbol,
                        side=item.get("holdSide", ""),
                        size=float(item.get("total", 0)),
                        entry_price=float(item.get("openPriceAvg", 0)),
                        mark_price=float(item.get("markPrice", 0)),
                        unrealized_pnl=float(item.get("unrealizedPL", 0)),
                        liquidation_price=float(item.get("liquidationPrice", 0)),
                        leverage=int(item.get("leverage", 1)),
                        margin_mode=item.get("marginMode", "")
                    ))

        return positions

    async def get_position(self, symbol: str, side: str,
                           product_type: str = "USDT-FUTURES") -> Optional[Position]:
        """Bitta pozitsiyani olish"""
        positions = await self.get_positions(symbol, product_type)
        for pos in positions:
            if pos.side == side:
                return pos
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #                           ORDER METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def set_position_mode(self, hedge_mode: bool = True,
                                product_type: str = "USDT-FUTURES") -> Dict:
        """
        Position mode o'rnatish.

        Args:
            hedge_mode: True = hedge mode (double_hold) - ikkala tomonga pozitsiya mumkin
                       False = one-way mode (single_hold) - faqat bitta tomon

        RSI strategiya uchun HEDGE MODE kerak!
        Aks holda long ochganda short yopiladi va aksincha.
        """
        hold_mode = "double_hold" if hedge_mode else "single_hold"
        body = {
            "productType": product_type,
            "holdMode": hold_mode
        }
        logger.info(f"Setting position mode: {hold_mode}")
        try:
            return await self.post("/api/v2/mix/account/set-position-mode", body)
        except BitgetAPIError as e:
            # "Already set" xatosini ignore qilish
            error_str = str(e).lower()
            if any(k in error_str for k in ("already", "same", "40756")) or "400172" in str(e.code):
                logger.info(f"Position mode allaqachon {hold_mode} holatida")
                return {"success": True, "note": "already set"}
            raise

    async def set_margin_mode(self, symbol: str, margin_mode: str = "isolated",
                             product_type: str = "USDT-FUTURES",
                             margin_coin: str = "USDT") -> Dict:
        """
        Margin rejimini o'rnatish (isolated yoki crossed)

        Args:
            symbol: Trading pair
            margin_mode: 'isolated' yoki 'crossed'
            product_type: Product type
            margin_coin: Margin valyutasi

        Returns:
            API javobi
        """
        body = {
            "symbol": symbol,
            "productType": product_type,
            "marginCoin": margin_coin,
            "marginMode": margin_mode
        }
        return await self.post("/api/v2/mix/account/set-margin-mode", body)

    async def place_order(self, symbol: str, side: str, trade_side: str,
                          size: float, order_type: str = "market",
                          price: Optional[float] = None,
                          product_type: str = "USDT-FUTURES",
                          margin_coin: str = "USDT",
                          tp_price: Optional[float] = None,
                          sl_price: Optional[float] = None,
                          margin_mode: str = "crossed") -> Dict:
        """
        Order yaratish

        Args:
            symbol: Trading pair
            side: 'buy' yoki 'sell'
            trade_side: 'open' (ochish) yoki 'close' (yopish)
            size: Hajm
            order_type: 'market' yoki 'limit'
            price: Limit order uchun narx
            product_type: Product type
            margin_coin: Margin valyutasi
            tp_price: Take Profit narx
            sl_price: Stop Loss narx
            margin_mode: 'crossed' yoki 'isolated'

        Returns:
            Order javobi
        """
        body = {
            "symbol": symbol,
            "productType": product_type,
            "marginMode": margin_mode,
            "marginCoin": margin_coin,
            "side": side,
            "tradeSide": trade_side,
            "orderType": order_type,
            "size": str(size),
            "force": "GTC"
        }

        if order_type == "limit" and price:
            body["price"] = str(price)

        # TP/SL
        if tp_price:
            body["presetStopSurplusPrice"] = str(tp_price)
        if sl_price:
            body["presetStopLossPrice"] = str(sl_price)

        return await self.post("/api/v2/mix/order/place-order", body)

    async def cancel_order(self, symbol: str, order_id: str,
                           product_type: str = "USDT-FUTURES") -> Dict:
        """Order bekor qilish"""
        body = {
            "symbol": symbol,
            "productType": product_type,
            "orderId": order_id
        }
        return await self.post("/api/v2/mix/order/cancel-order", body)

    async def get_open_orders(self, symbol: str = None,
                              product_type: str = "USDT-FUTURES") -> List[Dict]:
        """Ochiq orderlarni olish"""
        params = {"productType": product_type}
        if symbol:
            params["symbol"] = symbol

        return await self.get("/api/v2/mix/order/orders-pending", params)

    async def cancel_all_orders(self, symbol: str,
                                product_type: str = "USDT-FUTURES") -> Dict:
        """Barcha orderlarni bekor qilish"""
        body = {
            "symbol": symbol,
            "productType": product_type
        }
        return await self.post("/api/v2/mix/order/cancel-all-orders", body)

    # ─────────────────────────────────────────────────────────────────────────
    #                           SHORTCUT METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def open_long(self, symbol: str, size: float,
                        tp_price: Optional[float] = None,
                        sl_price: Optional[float] = None,
                        margin_mode: str = "crossed") -> Dict:
        """LONG pozitsiya ochish (market order)"""
        return await self.place_order(
            symbol=symbol,
            side="buy",
            trade_side="open",
            size=size,
            order_type="market",
            tp_price=tp_price,
            sl_price=sl_price,
            margin_mode=margin_mode
        )

    async def open_short(self, symbol: str, size: float,
                         tp_price: Optional[float] = None,
                         sl_price: Optional[float] = None,
                         margin_mode: str = "crossed") -> Dict:
        """SHORT pozitsiya ochish (market order)"""
        return await self.place_order(
            symbol=symbol,
            side="sell",
            trade_side="open",
            size=size,
            order_type="market",
            tp_price=tp_price,
            sl_price=sl_price,
            margin_mode=margin_mode
        )

    async def close_long(self, symbol: str, size: float,
                         margin_mode: str = "crossed") -> Dict:
        """LONG pozitsiyani yopish"""
        return await self.place_order(
            symbol=symbol,
            side="sell",
            trade_side="close",
            size=size,
            order_type="market",
            margin_mode=margin_mode
        )

    async def close_short(self, symbol: str, size: float,
                          margin_mode: str = "crossed") -> Dict:
        """SHORT pozitsiyani yopish"""
        return await self.place_order(
            symbol=symbol,
            side="buy",
            trade_side="close",
            size=size,
            order_type="market",
            margin_mode=margin_mode
        )

    async def close_all_positions(self, symbol: str,
                                  product_type: str = "USDT-FUTURES") -> Dict:
        """Barcha pozitsiyalarni yopish"""
        body = {
            "symbol": symbol,
            "productType": product_type
        }
        return await self.post("/api/v2/mix/order/close-positions", body)

    # ─────────────────────────────────────────────────────────────────────────
    #                           TP/SL METHODS
    # ─────────────────────────────────────────────────────────────────────────

    async def modify_tpsl(self, symbol: str, side: str,
                          tp_price: Optional[float] = None,
                          sl_price: Optional[float] = None,
                          product_type: str = "USDT-FUTURES",
                          margin_coin: str = "USDT") -> Dict:
        """
        Pozitsiya TP/SL ni o'rnatish (Bitget API v2)

        Endpoint: POST /api/v2/mix/order/place-pos-tpsl
        Docs: https://www.bitget.com/api-doc/contract/plan/Place-Pos-Tpsl-Order

        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            side: 'long' yoki 'short'
            tp_price: Take Profit trigger narx
            sl_price: Stop Loss trigger narx
            product_type: Product type (default: USDT-FUTURES)
            margin_coin: Margin coin (default: USDT)

        Returns:
            API response with orderId, stopSurplusClientOid, stopLossClientOid
        """
        if not tp_price and not sl_price:
            logger.debug("modify_tpsl: TP va SL berilmadi, skip")
            return {"success": True, "note": "No TP/SL provided"}

        # FIX: Narxlarni symbol pricePlace ga muvofiq yaxlitlash
        # Bitget API checkBDScale error oldini olish uchun
        price_place = await self.get_price_place(symbol, product_type)

        if tp_price:
            tp_price = round(tp_price, price_place)
        if sl_price:
            sl_price = round(sl_price, price_place)

        # Bitget API v2 uchun productType kichik harfda bo'lishi kerak
        product_type_lower = product_type.lower().replace("_", "-")

        body: Dict[str, Any] = {
            "symbol": symbol,
            "productType": product_type_lower,
            "marginCoin": margin_coin,
            "holdSide": side
        }

        # Take Profit
        if tp_price and tp_price > 0:
            body["stopSurplusTriggerPrice"] = str(tp_price)
            body["stopSurplusTriggerType"] = "mark_price"
            # Execute price = trigger price (market order)
            body["stopSurplusExecutePrice"] = str(tp_price)

        # Stop Loss
        if sl_price and sl_price > 0:
            body["stopLossTriggerPrice"] = str(sl_price)
            body["stopLossTriggerType"] = "mark_price"
            body["stopLossExecutePrice"] = str(sl_price)

        try:
            result = await self.post("/api/v2/mix/order/place-pos-tpsl", body)
            logger.info(f"TP/SL o'rnatildi: {side}, TP={tp_price}, SL={sl_price}")
            return result
        except BitgetAPIError as e:
            # Agar pozitsiya topilmasa yoki boshqa xato bo'lsa, local TP/SL ishlatiladi
            logger.warning(f"Exchange TP/SL xatosi ({e}), local monitoring ishlatiladi")
            return {"success": False, "error": str(e), "note": "Local TP/SL monitoring used"}

    # ─────────────────────────────────────────────────────────────────────────
    #                           ORDER WITH FILL VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    async def get_order(self, symbol: str, order_id: str,
                        product_type: str = "USDT-FUTURES") -> Optional[Dict]:
        """Order ma'lumotlarini olish"""
        params = {
            "symbol": symbol,
            "productType": product_type,
            "orderId": order_id
        }
        try:
            return await self.get("/api/v2/mix/order/detail", params)
        except BitgetAPIError:
            return None

    async def open_long_with_fill(
        self,
        symbol: str,
        size: float,
        expected_price: float,
        max_slippage_percent: float = 0.5,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        margin_mode: str = "crossed"
    ) -> Dict:
        """
        LONG pozitsiya ochish - fill narxini tekshirish bilan.

        Args:
            symbol: Trading pair
            size: Lot hajmi
            expected_price: Kutilgan narx
            max_slippage_percent: Maksimal slippage foizi
            tp_price: Take Profit narx
            sl_price: Stop Loss narx
            margin_mode: Margin mode

        Returns:
            Order javobi + fillPrice va slippage ma'lumotlari
        """
        result = await self.open_long(symbol, size, tp_price, sl_price, margin_mode=margin_mode)

        if result.get('orderId'):
            order_id = result['orderId']

            # Fill narxini olish
            order_details = await self.get_order(symbol, order_id)

            if order_details:
                fill_price = float(order_details.get('priceAvg', expected_price) or expected_price)

                # Slippage hisoblash
                if expected_price > 0:
                    slippage = abs(fill_price - expected_price) / expected_price * 100
                else:
                    slippage = 0

                if slippage > max_slippage_percent:
                    logger.warning(
                        f"Yuqori slippage aniqlandi: {slippage:.2f}% "
                        f"(kutilgan: {expected_price}, haqiqiy: {fill_price})"
                    )

                result['fillPrice'] = fill_price
                result['slippage'] = slippage
            else:
                result['fillPrice'] = expected_price
                result['slippage'] = 0

        return result

    async def open_short_with_fill(
        self,
        symbol: str,
        size: float,
        expected_price: float,
        max_slippage_percent: float = 0.5,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        margin_mode: str = "crossed"
    ) -> Dict:
        """
        SHORT pozitsiya ochish - fill narxini tekshirish bilan.

        Args:
            symbol: Trading pair
            size: Lot hajmi
            expected_price: Kutilgan narx
            max_slippage_percent: Maksimal slippage foizi
            tp_price: Take Profit narx
            sl_price: Stop Loss narx
            margin_mode: Margin mode

        Returns:
            Order javobi + fillPrice va slippage ma'lumotlari
        """
        result = await self.open_short(symbol, size, tp_price, sl_price, margin_mode=margin_mode)

        if result.get('orderId'):
            order_id = result['orderId']

            # Fill narxini olish
            order_details = await self.get_order(symbol, order_id)

            if order_details:
                fill_price = float(order_details.get('priceAvg', expected_price) or expected_price)

                # Slippage hisoblash
                if expected_price > 0:
                    slippage = abs(fill_price - expected_price) / expected_price * 100
                else:
                    slippage = 0

                if slippage > max_slippage_percent:
                    logger.warning(
                        f"Yuqori slippage aniqlandi: {slippage:.2f}% "
                        f"(kutilgan: {expected_price}, haqiqiy: {fill_price})"
                    )

                result['fillPrice'] = fill_price
                result['slippage'] = slippage
            else:
                result['fillPrice'] = expected_price
                result['slippage'] = 0

        return result
