"""
Webhook Client - HEMA platformasiga trade eventlarini yuborish

Trade ochilganda, yopilganda, TP/SL bo'lganda HEMA ga webhook yuboriladi
HEMA kutgan formatda yuborish
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from urllib.parse import urlparse
import os

logger = logging.getLogger(__name__)

# R5: Production mode - HTTPS majburiy
PRODUCTION_MODE = os.getenv("PRODUCTION_MODE", "false").lower() == "true"

# Queue limits
MAX_QUEUE_SIZE = 1000
QUEUE_WARNING_THRESHOLD = 800

# Bitget fee rate per side (NOT roundtrip):
#   Demo: 0.1% taker → 0.001
#   Real: 0.06% taker → 0.0006
DEFAULT_TAKER_FEE_RATE = 0.001  # 0.1% per side (demo default)


@dataclass
class WebhookConfig:
    """Webhook sozlamalari"""
    url: str
    secret: str
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    user_id: str = ''
    user_bot_id: str = ''
    taker_fee_rate: float = 0.001


class WebhookClient:
    """
    HEMA platformasiga webhook yuboruvchi client

    HEMA kutgan legacy formatda yuboradi:
    {
        "event": "trade_opened",
        "timestamp": "2024-01-01T00:00:00Z",
        "data": {
            "userId": "...",
            "userBotId": "...",
            "trade": {...}
        }
    }

    Async context manager sifatida ishlatilishi mumkin:

    async with WebhookClient(config) as webhook:
        await webhook.send_trade_opened(...)
    """

    def __init__(self, config: WebhookConfig, taker_fee_rate: float = DEFAULT_TAKER_FEE_RATE):
        self.config = config
        self.taker_fee_rate = taker_fee_rate
        self._session: Optional[aiohttp.ClientSession] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self._worker_task: Optional[asyncio.Task] = None
        self._user_id: Optional[str] = None
        self._dropped_events: int = 0
        self._stopped = False

    async def __aenter__(self) -> "WebhookClient":
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - ensures cleanup"""
        await self.stop()

    def __del__(self):
        """Destructor - warn if not properly stopped"""
        if self._session and not self._session.closed and not self._stopped:
            logger.warning(
                "WebhookClient session was not properly stopped! "
                "Use 'await client.stop()' or async context manager."
            )

    def set_user_id(self, user_id: str):
        """Set user ID for webhook events"""
        self._user_id = user_id

    async def start(self):
        """Webhook worker'ni ishga tushirish"""
        # R5: HTTPS URL tekshiruvi
        parsed_url = urlparse(self.config.url)

        if PRODUCTION_MODE:
            if parsed_url.scheme != "https":
                raise ValueError(
                    f"Production mode da faqat HTTPS webhook URL qabul qilinadi. "
                    f"Berilgan: {self.config.url}"
                )
        else:
            # Development mode - ogohlantirish
            if parsed_url.scheme != "https":
                logger.warning(
                    f"⚠️  Webhook URL HTTPS emas: {self.config.url}. "
                    f"Production uchun HTTPS ishlatish tavsiya qilinadi!"
                )

        # URL validatsiya
        if not parsed_url.netloc:
            raise ValueError(f"Noto'g'ri webhook URL: {self.config.url}")

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info(f"Webhook client started: {self.config.url}")

    async def stop(self):
        """Webhook worker'ni to'xtatish"""
        self._stopped = True

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Webhook client stopped")

    def _generate_signature(self, timestamp: str, payload: str) -> str:
        """HMAC-SHA256 signature yaratish (HEMA formati)"""
        message = f"{timestamp}.{payload}"
        return hmac.new(
            self.config.secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    # O9: Muhim eventlar (trade) - yo'qolmasligi kerak
    PRIORITY_EVENTS = {"trade_opened", "trade_closed", "error_occurred", "balance_warning", "global_limit_hit", "positions_synced", "martingale_explosion"}

    async def _send_event(
        self,
        event_type: str,
        user_bot_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        HEMA formatida event yuborish

        Args:
            event_type: Event turi (lowercase: trade_opened, trade_closed, etc.)
            user_bot_id: HEMA dagi UserBot ID
            data: Event ma'lumotlari
        """
        event = {
            "event": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": {
                "userId": self._user_id or "",
                "userBotId": user_bot_id,
                **data
            }
        }

        is_priority = event_type in self.PRIORITY_EVENTS

        # O9: Muhim eventlar uchun - queue to'lsa ham qo'shish
        if is_priority:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue to'ldi - eng eski eventni olib tashlab, muhim eventni qo'shish
                try:
                    self._queue.get_nowait()  # Eng eski eventni olib tashlash
                    self._dropped_events += 1
                    self._queue.put_nowait(event)
                    logger.warning(
                        f"Queue to'ldi - eski event o'chirildi, {event_type} qo'shildi"
                    )
                except asyncio.QueueEmpty:
                    pass
            return True

        # Oddiy eventlar (status_update) uchun - queue to'lsa skip
        try:
            await asyncio.wait_for(
                self._queue.put(event),
                timeout=0.5
            )
        except asyncio.TimeoutError:
            self._dropped_events += 1
            # Status update tashlab yuborildi - bu normal
            if event_type != "status_update":
                logger.warning(
                    f"Webhook queue to'ldi, event tashlab yuborildi: {event_type}. "
                    f"Jami tashlab yuborilgan: {self._dropped_events}"
                )
            return False

        # Queue to'lib qolgani haqida ogohlantirish
        queue_size = self._queue.qsize()
        if queue_size > QUEUE_WARNING_THRESHOLD:
            logger.warning(f"Webhook queue yuqori: {queue_size}/{MAX_QUEUE_SIZE}")

        return True

    async def send_trade_opened(
        self,
        user_bot_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        order_id: str,
        leverage: int = 1,
        margin_mode: str = "CROSS"
    ):
        """Trade ochildi eventi"""
        cost = price * quantity
        fee = round(cost * self.taker_fee_rate, 8)
        margin = round(cost / leverage, 4) if leverage > 0 else cost

        await self._send_event(
            "trade_opened",
            user_bot_id,
            {
                "trade": {
                    "id": order_id,
                    "exchangeOrderId": order_id,
                    "pair": symbol,
                    "side": side.upper(),
                    "type": "MARKET",
                    "amount": quantity,
                    "price": price,
                    "cost": cost,
                    "fee": fee,
                    "feeRate": self.taker_fee_rate,
                    "feeCurrency": "USDT",
                    "leverage": leverage,
                    "marginMode": margin_mode,
                    "margin": margin,
                    "openedAt": datetime.utcnow().isoformat() + "Z"
                }
            }
        )

    async def send_trade_closed(
        self,
        user_bot_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        reason: str = "MANUAL"
    ):
        """Trade yopildi eventi - pnl parametri GROSS, ichida NET hisoblanadi"""
        gross_pnl = pnl

        # Fee hisoblash (round-trip: entry + exit)
        entry_fee = round(entry_price * quantity * self.taker_fee_rate, 8)
        exit_fee = round(exit_price * quantity * self.taker_fee_rate, 8)
        total_fees = round(entry_fee + exit_fee, 8)
        net_pnl = round(gross_pnl - total_fees, 8)

        # PnL percent hisoblash
        cost_basis = entry_price * quantity
        if cost_basis > 0:
            gross_pnl_percent = round(gross_pnl / cost_basis * 100, 4)
            net_pnl_percent = round(net_pnl / cost_basis * 100, 4)
        else:
            gross_pnl_percent = 0.0
            net_pnl_percent = 0.0

        cost = exit_price * quantity

        await self._send_event(
            "trade_closed",
            user_bot_id,
            {
                "trade": {
                    "id": f"close-{int(time.time()*1000)}",
                    "pair": symbol,
                    "side": side.upper(),
                    "type": "MARKET",
                    "amount": quantity,
                    "price": exit_price,
                    "entryPrice": entry_price,
                    "cost": cost,
                    "fee": total_fees,
                    "entryFee": entry_fee,
                    "exitFee": exit_fee,
                    "feeRate": self.taker_fee_rate,
                    "feeCurrency": "USDT",
                    "grossPnl": gross_pnl,
                    "pnl": net_pnl,
                    "grossPnlPercent": gross_pnl_percent,
                    "pnlPercent": net_pnl_percent,
                    "closedAt": datetime.utcnow().isoformat() + "Z",
                    "reason": reason
                }
            }
        )

    async def send_tp_hit(
        self,
        user_bot_id: str,
        symbol: str,
        side: str,
        pnl: float,
        positions_closed: int,
        entry_price: float = 0,
        exit_price: float = 0,
        quantity: float = 0
    ):
        """Take Profit eventi - trade_closed sifatida yuboriladi"""
        gross_pnl = pnl

        # Use quantity if provided, otherwise use positions_closed as fallback
        amount = quantity if quantity > 0 else positions_closed

        # Fee hisoblash (round-trip)
        entry_fee = round(entry_price * amount * self.taker_fee_rate, 8) if entry_price else 0
        exit_fee = round(exit_price * amount * self.taker_fee_rate, 8) if exit_price else 0
        total_fees = round(entry_fee + exit_fee, 8)
        net_pnl = round(gross_pnl - total_fees, 8)

        # PnL percent
        cost_basis = entry_price * amount if entry_price else 0
        if cost_basis > 0:
            gross_pnl_percent = round(gross_pnl / cost_basis * 100, 4)
            net_pnl_percent = round(net_pnl / cost_basis * 100, 4)
        else:
            gross_pnl_percent = 0.0
            net_pnl_percent = 0.0

        cost = exit_price * amount if exit_price else 0

        await self._send_event(
            "trade_closed",
            user_bot_id,
            {
                "trade": {
                    "id": f"tp-{int(time.time()*1000)}",
                    "pair": symbol,
                    "side": side.upper(),
                    "type": "TAKE_PROFIT",
                    "amount": amount,
                    "price": exit_price,
                    "entryPrice": entry_price,
                    "cost": cost,
                    "fee": total_fees,
                    "entryFee": entry_fee,
                    "exitFee": exit_fee,
                    "feeRate": self.taker_fee_rate,
                    "feeCurrency": "USDT",
                    "grossPnl": gross_pnl,
                    "pnl": net_pnl,
                    "grossPnlPercent": gross_pnl_percent,
                    "pnlPercent": net_pnl_percent,
                    "closedAt": datetime.utcnow().isoformat() + "Z"
                },
                "reason": "TP_HIT",
                "positionsClosed": positions_closed
            }
        )

    async def send_sl_hit(
        self,
        user_bot_id: str,
        symbol: str,
        side: str,
        pnl: float,
        positions_closed: int,
        entry_price: float = 0,
        exit_price: float = 0,
        quantity: float = 0
    ):
        """Stop Loss eventi - trade_closed sifatida yuboriladi"""
        gross_pnl = pnl

        # Use quantity if provided, otherwise use positions_closed as fallback
        amount = quantity if quantity > 0 else positions_closed

        # Fee hisoblash (round-trip)
        entry_fee = round(entry_price * amount * self.taker_fee_rate, 8) if entry_price else 0
        exit_fee = round(exit_price * amount * self.taker_fee_rate, 8) if exit_price else 0
        total_fees = round(entry_fee + exit_fee, 8)
        net_pnl = round(gross_pnl - total_fees, 8)

        # PnL percent
        cost_basis = entry_price * amount if entry_price else 0
        if cost_basis > 0:
            gross_pnl_percent = round(gross_pnl / cost_basis * 100, 4)
            net_pnl_percent = round(net_pnl / cost_basis * 100, 4)
        else:
            gross_pnl_percent = 0.0
            net_pnl_percent = 0.0

        cost = exit_price * amount if exit_price else 0

        await self._send_event(
            "trade_closed",
            user_bot_id,
            {
                "trade": {
                    "id": f"sl-{int(time.time()*1000)}",
                    "pair": symbol,
                    "side": side.upper(),
                    "type": "STOP_LOSS",
                    "amount": amount,
                    "price": exit_price,
                    "entryPrice": entry_price,
                    "cost": cost,
                    "fee": total_fees,
                    "entryFee": entry_fee,
                    "exitFee": exit_fee,
                    "feeRate": self.taker_fee_rate,
                    "feeCurrency": "USDT",
                    "grossPnl": gross_pnl,
                    "pnl": net_pnl,
                    "grossPnlPercent": gross_pnl_percent,
                    "pnlPercent": net_pnl_percent,
                    "closedAt": datetime.utcnow().isoformat() + "Z"
                },
                "reason": "SL_HIT",
                "positionsClosed": positions_closed
            }
        )

    async def send_status_changed(
        self,
        user_bot_id: str,
        status: str,
        message: str = ""
    ):
        """Status o'zgarishi eventi"""
        await self._send_event(
            "status_changed",
            user_bot_id,
            {
                "previousStatus": "",
                "newStatus": status.lower(),
                "reason": message
            }
        )

    async def send_error(
        self,
        user_bot_id: str,
        error_code: str,
        error_message: str
    ):
        """Xatolik eventi"""
        await self._send_event(
            "error_occurred",
            user_bot_id,
            {
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "severity": "medium"
                }
            }
        )

    async def send_balance_warning(
        self,
        user_bot_id: str,
        current_balance: float,
        required_balance: float,
        message: str = ""
    ):
        """Balance ogohlantirishi"""
        await self._send_event(
            "balance_warning",
            user_bot_id,
            {
                "currentBalance": current_balance,
                "requiredBalance": required_balance,
                "message": message or f"Balance is low: ${current_balance}"
            }
        )

    async def send_global_limit_hit(
        self,
        user_bot_id: str,
        symbol: str,
        limit_type: str,
        current_loss_percent: float,
        limit_value: float,
        current_balance: float,
        initial_balance: float
    ):
        """
        Global zarar limiti oshdi eventi.

        Args:
            user_bot_id: HEMA dagi UserBot ID
            symbol: Trading symbol
            limit_type: "DAILY_LOSS" yoki "TOTAL_LOSS"
            current_loss_percent: Hozirgi zarar foizi
            limit_value: Limit qiymati (foiz)
            current_balance: Joriy balans
            initial_balance: Boshlang'ich balans
        """
        await self._send_event(
            "global_limit_hit",
            user_bot_id,
            {
                "symbol": symbol,
                "limitType": limit_type,
                "currentLossPercent": round(current_loss_percent, 2),
                "limitValue": limit_value,
                "currentBalance": round(current_balance, 2),
                "initialBalance": round(initial_balance, 2),
                "message": f"{limit_type} limit reached: {current_loss_percent:.2f}% >= {limit_value}%"
            }
        )

    async def send_martingale_explosion(
        self,
        user_bot_id: str,
        side: str,
        calculated_lot: float,
        max_lot: float,
        cooldown_seconds: int
    ):
        """Martingale explosion xabarnomasi"""
        await self._send_event(
            "martingale_explosion",
            user_bot_id,
            {
                "side": side,
                "calculatedLot": calculated_lot,
                "maxLot": max_lot,
                "cooldownSeconds": cooldown_seconds,
                "message": (
                    f"Martingale {side.upper()} lot={calculated_lot:.4f} "
                    f"MAX_LOT={max_lot:.4f} dan oshdi. "
                    f"{cooldown_seconds}s cooldown boshlandi."
                )
            }
        )

    async def send_positions_synced(
        self,
        user_bot_id: str,
        symbol: str,
        positions: list
    ):
        """
        Bot startup da exchange pozitsiyalar ro'yxatini HEMA ga yuborish.

        HEMA bu event orqali o'z DB sidagi OPEN tradelarni exchange bilan solishtiradi.
        Exchange da yo'q lekin HEMA da OPEN bo'lgan tradelar avtomatik yopiladi (RECONCILED).

        Args:
            user_bot_id: HEMA dagi UserBot ID
            symbol: Trading pair
            positions: Exchange dagi haqiqiy pozitsiyalar ro'yxati
        """
        await self._send_event(
            "positions_synced",
            user_bot_id,
            {
                "symbol": symbol,
                "syncedPositions": positions,
                "syncedAt": datetime.utcnow().isoformat() + "Z"
            }
        )

    async def send_status_update(
        self,
        user_bot_id: str,
        symbol: str,
        current_price: float,
        rsi: float,
        rsi_prev: float,
        balance: float,
        buy_positions: list,
        sell_positions: list,
        stats: dict,
        settings: dict,
        runtime: dict = None
    ):
        """
        Real-time status update - UI da ko'rsatish uchun

        Har bir tick da yuboriladi (yangilangan format - HEMA bilan to'liq integratsiya)
        """
        # Trend bot: RSI not used, always NEUTRAL (rsi/rsi_prev are always 0.0 from caller)
        # RSI fields kept in payload — HEMA frontend expects them
        rsi_signal = "NEUTRAL"
        if rsi < settings.get("rsiLevelBuy", 30):
            rsi_signal = "OVERSOLD"
        elif rsi > settings.get("rsiLevelSell", 70):
            rsi_signal = "OVERBOUGHT"

        # Settings dan kerakli qiymatlarni olish
        leverage = settings.get("leverage", 10)
        tp_percent = settings.get("takeProfitPercent", 1.0)
        sl_percent = settings.get("stopLossPercent", 5.0)
        min_step = settings.get("minStepPercent", 0.5)
        k_lot = settings.get("kLot", 1.5)
        base_lot = settings.get("baseLot", 0.001)

        # Har bir pozitsiya uchun PnL hisoblash (gross + net with fees)
        # R7: lot - bu Bitget'dagi haqiqiy pozitsiya hajmi (BTC).
        # Leverage allaqachon pozitsiya ochilganda hisobga olingan.
        # PnL = (narx_o'zgarishi) * lot_hajmi
        # pnlPercent = narx foiz o'zgarishi (leverage HISOBLANMAGAN!)
        # HEMA frontend o'zi leverage ga ko'paytiradi → ROE
        fee_rate = self.taker_fee_rate

        def calc_position_pnl(pos, side):
            entry = pos.get("price", 0)
            lot = pos.get("lot", 0)
            if entry <= 0 or lot <= 0:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            if side == "buy":
                gross_pnl = (current_price - entry) * lot
                gross_pnl_percent = (current_price - entry) / entry * 100
            else:
                gross_pnl = (entry - current_price) * lot
                gross_pnl_percent = (entry - current_price) / entry * 100

            # Fee estimation (round-trip: entry + estimated exit at current price)
            entry_fee = entry * lot * fee_rate
            estimated_exit_fee = current_price * lot * fee_rate
            estimated_total_fee = round(entry_fee + estimated_exit_fee, 8)

            net_pnl = gross_pnl - estimated_total_fee
            cost_basis = entry * lot
            net_pnl_percent = (net_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

            return (round(gross_pnl, 4), round(gross_pnl_percent, 4),
                    round(net_pnl, 4), round(net_pnl_percent, 4),
                    round(estimated_total_fee, 8))

        # BUY pozitsiyalarni PnL bilan tayyorlash
        buy_with_pnl = []
        for p in buy_positions:
            gross_pnl, gross_pct, net_pnl, net_pct, est_fee = calc_position_pnl(p, "buy")
            buy_with_pnl.append({
                "price": p.get("price", 0),
                "lot": p.get("lot", 0),
                "orderId": p.get("order_id", ""),
                "pnl": net_pnl,
                "grossPnl": gross_pnl,
                "pnlPercent": net_pct,
                "grossPnlPercent": gross_pct,
                "estimatedFee": est_fee,
                "openedAt": p.get("opened_at", "")
            })

        # SELL pozitsiyalarni PnL bilan tayyorlash
        sell_with_pnl = []
        for p in sell_positions:
            gross_pnl, gross_pct, net_pnl, net_pct, est_fee = calc_position_pnl(p, "sell")
            sell_with_pnl.append({
                "price": p.get("price", 0),
                "lot": p.get("lot", 0),
                "orderId": p.get("order_id", ""),
                "pnl": net_pnl,
                "grossPnl": gross_pnl,
                "pnlPercent": net_pct,
                "grossPnlPercent": gross_pct,
                "estimatedFee": est_fee,
                "openedAt": p.get("opened_at", "")
            })

        # O'rtacha narxlarni hisoblash
        avg_buy = 0.0
        avg_sell = 0.0
        if buy_with_pnl:
            total_value = sum(p["price"] * p["lot"] for p in buy_with_pnl)
            total_lot = sum(p["lot"] for p in buy_with_pnl)
            avg_buy = total_value / total_lot if total_lot > 0 else 0
        if sell_with_pnl:
            total_value = sum(p["price"] * p["lot"] for p in sell_with_pnl)
            total_lot = sum(p["lot"] for p in sell_with_pnl)
            avg_sell = total_value / total_lot if total_lot > 0 else 0

        # TP narxlarni hisoblash (FIX #3: fixed-reference, strategy bilan bir xil)
        # tp_distance = first_entry * TP_PERCENT (doimiy masofa)
        if buy_with_pnl and avg_buy > 0:
            first_entry_buy = max(p["price"] for p in buy_with_pnl)
            tp_distance_buy = first_entry_buy * (tp_percent / 100)
            buy_tp = avg_buy + tp_distance_buy
        else:
            buy_tp = 0
        if sell_with_pnl and avg_sell > 0:
            first_entry_sell = min(p["price"] for p in sell_with_pnl)
            tp_distance_sell = first_entry_sell * (tp_percent / 100)
            sell_tp = avg_sell - tp_distance_sell
        else:
            sell_tp = 0

        # Keyingi order narxlarini hisoblash
        if buy_with_pnl:
            min_buy_price = min(p["price"] for p in buy_with_pnl)
            next_buy_price = min_buy_price * (1 - min_step / 100)
            next_buy_lot = base_lot * (k_lot ** len(buy_with_pnl))
        else:
            next_buy_price = current_price * (1 - min_step / 100)
            next_buy_lot = base_lot

        if sell_with_pnl:
            max_sell_price = max(p["price"] for p in sell_with_pnl)
            next_sell_price = max_sell_price * (1 + min_step / 100)
            next_sell_lot = base_lot * (k_lot ** len(sell_with_pnl))
        else:
            next_sell_price = current_price * (1 + min_step / 100)
            next_sell_lot = base_lot

        # Performance hisoblash
        total_trades = stats.get("trades", {}).get("total", 0)
        winning_trades = stats.get("trades", {}).get("winning", 0)
        losing_trades = total_trades - winning_trades

        # Net unrealized PnL hisoblash (open pozitsiyalar uchun)
        all_positions_with_pnl = buy_with_pnl + sell_with_pnl
        gross_unrealized_from_positions = sum(p["grossPnl"] for p in all_positions_with_pnl)
        total_estimated_open_fees = sum(p["estimatedFee"] for p in all_positions_with_pnl)
        net_unrealized_from_positions = gross_unrealized_from_positions - total_estimated_open_fees

        gross_unrealized_pnl = round(stats.get("unrealized_pnl", 0), 4)

        await self._send_event(
            "status_update",
            user_bot_id,
            {
                "symbol": symbol,
                "currentPrice": current_price,
                "rsi": {
                    "value": round(rsi, 1),
                    "previous": round(rsi_prev, 1),
                    "signal": rsi_signal,
                    "levelBuy": settings.get("rsiLevelBuy", 30),
                    "levelSell": settings.get("rsiLevelSell", 70)
                },
                "balance": round(balance, 2),
                "positions": {
                    "buy": buy_with_pnl,
                    "sell": sell_with_pnl,
                    "buyCount": len(buy_with_pnl),
                    "sellCount": len(sell_with_pnl)
                },
                "averagePrice": {
                    "buy": round(avg_buy, 2),
                    "sell": round(avg_sell, 2)
                },
                "tpsl": {
                    "buyTp": round(buy_tp, 2),
                    "buyTpPercent": tp_percent,
                    "sellTp": round(sell_tp, 2),
                    "sellTpPercent": tp_percent,
                    "slPercent": sl_percent
                },
                "nextOrder": {
                    "buyPrice": round(next_buy_price, 2),
                    "buyLot": round(next_buy_lot, 5),
                    "sellPrice": round(next_sell_price, 2),
                    "sellLot": round(next_sell_lot, 5)
                },
                "performance": {
                    "totalTrades": total_trades,
                    "winningTrades": winning_trades,
                    "losingTrades": losing_trades,
                    "winRate": stats.get("trades", {}).get("win_rate", 0),
                    "totalPnL": round(stats.get("profit", 0), 4),
                    "unrealizedPnL": gross_unrealized_pnl,
                    "grossUnrealizedPnL": gross_unrealized_pnl,
                    "netUnrealizedPnL": round(net_unrealized_from_positions, 4),
                    "estimatedOpenFees": round(total_estimated_open_fees, 4),
                    "feeRate": fee_rate,
                    "peakBalance": round(stats.get("peak_balance", 0), 4),
                    "maxDrawdown": round(stats.get("max_drawdown", 0), 4),
                    "maxDrawdownPercent": round(stats.get("max_drawdown_percent", 0), 2)
                },
                "settings": {
                    "leverage": leverage,
                    "timeframe": settings.get("timeframe", "1H"),
                    "baseLot": base_lot,
                    "kLot": k_lot,
                    "minStepPercent": min_step,
                    "takeProfitPercent": tp_percent,
                    "stopLossPercent": sl_percent,
                    "maxOrders": settings.get("maxOrders", 10),
                    "feeRate": fee_rate
                },
                "runtime": runtime or {
                    "tick": stats.get("tick", 0),
                    "uptime": 0,
                    "startedAt": "",
                    "lastTradeAt": ""
                }
            }
        )

    async def _process_queue(self):
        """Queue'dan eventlarni yuborish"""
        while True:
            try:
                event = await self._queue.get()
                await self._send_with_retry(event)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Webhook queue error: {e}")

    async def _send_with_retry(self, event: Dict[str, Any]) -> bool:
        """Retry bilan webhook yuborish"""
        payload = json.dumps(event)
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, payload)
        webhook_id = f"{event['data']['userBotId']}-{timestamp}-{uuid.uuid4().hex[:8]}"

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Secret": self.config.secret,
            "X-Webhook-ID": webhook_id,
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Signature": signature
        }

        event_type = event['event']
        is_important = event_type in self.PRIORITY_EVENTS

        # Muhim eventlar INFO, status_update DEBUG
        if is_important:
            logger.info(f"[WEBHOOK] Sending {event_type} to {self.config.url}")
        else:
            logger.debug(f"[WEBHOOK] Sending {event_type}")

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(
                    self.config.url,
                    data=payload,
                    headers=headers
                ) as response:
                    response_text = await response.text()

                    if response.status in (200, 201, 202):
                        if is_important:
                            logger.info(f"[WEBHOOK] {event_type} sent OK")
                        return True
                    else:
                        logger.warning(
                            f"[WEBHOOK] {event_type} FAILED (attempt {attempt + 1}): "
                            f"status={response.status}, body={response_text[:200]}"
                        )

            except asyncio.TimeoutError:
                logger.warning(f"[WEBHOOK] {event_type} TIMEOUT (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"[WEBHOOK] {event_type} CONNECTION ERROR (attempt {attempt + 1}): {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        logger.error(f"[WEBHOOK] {event_type} failed after {self.config.max_retries} attempts")
        return False
