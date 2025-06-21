import importlib
import sys
import types
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def load_server():
    # stub dependencies required by server module
    import os
    sys.path.insert(0, os.getcwd())
    keyring = types.SimpleNamespace(
        get_password=lambda *a, **k: None,
        set_password=lambda *a, **k: None,
        delete_password=lambda *a, **k: None,
    )
    sys.modules['keyring'] = keyring

    def dummy_tabulate(data, headers=None, tablefmt=None):
        lines = ["\t".join(map(str, row)) for row in data]
        if headers:
            lines.insert(0, "\t".join(headers))
        return "\n".join(lines)
    tab_module = types.SimpleNamespace(tabulate=dummy_tabulate)
    sys.modules['tabulate'] = tab_module

    # stub mcp.server.fastmcp
    mcp_mod = types.ModuleType('mcp')
    server_mod = types.ModuleType('mcp.server')
    fastmcp_mod = types.ModuleType('mcp.server.fastmcp')

    class FastMCP:
        def __init__(self, name, lifespan=None):
            self.name = name
            self.lifespan = lifespan

        def tool(self):
            def decorator(fn):
                return fn
            return decorator

    class Context:
        def __init__(self, request_context=None):
            self.request_context = request_context

    fastmcp_mod.FastMCP = FastMCP
    fastmcp_mod.Context = Context
    server_mod.fastmcp = fastmcp_mod
    mcp_mod.server = server_mod
    sys.modules['mcp'] = mcp_mod
    sys.modules['mcp.server'] = server_mod
    sys.modules['mcp.server.fastmcp'] = fastmcp_mod

    # stub exchange_calendars
    ex_mod = types.ModuleType('exchange_calendars')

    class DummyCalendar:
        def is_open_on_minute(self, dt):
            return False

        def next_open(self, dt):
            return dt + timedelta(days=1)

    ex_mod.get_calendar = lambda name: DummyCalendar()
    sys.modules['exchange_calendars'] = ex_mod

    # stub tastytrade package and submodules
    tt_mod = types.ModuleType('tastytrade')
    class Session: ...
    class Account: ...
    tt_mod.Session = Session
    tt_mod.Account = Account

    metrics_mod = types.ModuleType('tastytrade.metrics')
    metrics_mod.a_get_market_metrics = AsyncMock()
    sys.modules['tastytrade.metrics'] = metrics_mod
    tt_mod.metrics = metrics_mod

    dxfeed_mod = types.ModuleType('tastytrade.dxfeed')
    class Quote: ...
    dxfeed_mod.Quote = Quote
    sys.modules['tastytrade.dxfeed'] = dxfeed_mod
    tt_mod.dxfeed = dxfeed_mod

    instruments_mod = types.ModuleType('tastytrade.instruments')
    class Option: ...
    class Equity: ...
    class NestedOptionChain: ...
    instruments_mod.Option = Option
    instruments_mod.Equity = Equity
    instruments_mod.NestedOptionChain = NestedOptionChain
    sys.modules['tastytrade.instruments'] = instruments_mod
    tt_mod.instruments = instruments_mod

    order_mod = types.ModuleType('tastytrade.order')
    class NewOrder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    class _Status:
        def __init__(self, value):
            self.value = value

    class OrderStatus:
        LIVE = _Status('LIVE')
        RECEIVED = _Status('RECEIVED')
        CANCELLED = _Status('CANCELLED')
        REPLACED = _Status('REPLACED')
    class _Action:
        def __init__(self, value):
            self.value = value

    class OrderAction:
        BUY_TO_OPEN = _Action('BTO')
        SELL_TO_CLOSE = _Action('STC')
    class OrderTimeInForce:
        DAY = 'DAY'
    class OrderType:
        LIMIT = 'LIMIT'
        MARKET = 'MARKET'
    order_mod.NewOrder = NewOrder
    order_mod.OrderStatus = OrderStatus
    order_mod.OrderAction = OrderAction
    order_mod.OrderTimeInForce = OrderTimeInForce
    order_mod.OrderType = OrderType
    sys.modules['tastytrade.order'] = order_mod
    tt_mod.order = order_mod

    streamer_mod = types.ModuleType('tastytrade.streamer')
    class DXLinkStreamer:
        def __init__(self, session):
            self.session = session
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        async def subscribe(self, *a, **k):
            pass
        async def get_event(self, typ):
            return MagicMock(bid_price=1.0, ask_price=1.2)
    streamer_mod.DXLinkStreamer = DXLinkStreamer
    sys.modules['tastytrade.streamer'] = streamer_mod
    tt_mod.streamer = streamer_mod

    sys.modules['tastytrade'] = tt_mod

    return importlib.import_module('tasty_agent.server')


@pytest.fixture(scope='module')
def server():
    return load_server()


def make_ctx(server, account):
    sc = server.ServerContext(session=None, account=account, is_test=False)
    return types.SimpleNamespace(request_context=types.SimpleNamespace(lifespan_context=sc))


def test_place_trade_market(server):
    account = MagicMock()
    account.a_place_order = AsyncMock(return_value=MagicMock(errors=[], warnings=[], order=MagicMock(id='1')))
    ctx = make_ctx(server, account)

    instrument = MagicMock(symbol='AAPL', streamer_symbol='AAPL')
    instrument.build_leg = lambda qty, action: MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, '_create_instrument', AsyncMock(return_value=instrument))
        mp.setattr(server, '_get_quote', AsyncMock(return_value=(Decimal('1.0'), Decimal('1.1'))))
        result = asyncio.run(server.place_trade(ctx, 'Buy to Open', 100, 'AAPL', order_type='Market'))

    assert 'Order placement successful' in result
    account.a_place_order.assert_awaited()


def test_place_trade_limit(server):
    account = MagicMock()
    account.a_place_order = AsyncMock(return_value=MagicMock(errors=[], warnings=[], order=MagicMock(id='2')))
    ctx = make_ctx(server, account)

    instrument = MagicMock(symbol='TSLA', streamer_symbol='TSLA')
    instrument.build_leg = lambda qty, action: MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, '_create_instrument', AsyncMock(return_value=instrument))
        mp.setattr(server, '_get_quote', AsyncMock(return_value=(Decimal('250.0'), Decimal('255.0'))))
        result = asyncio.run(server.place_trade(
            ctx,
            'Sell to Close',
            50,
            'TSLA',
            order_price=250,
            order_type='Limit',
        ))

    assert 'Order placement successful' in result
    account.a_place_order.assert_awaited()


def test_place_trade_option_dry_run(server):
    account = MagicMock()
    account.a_place_order = AsyncMock(return_value=MagicMock(errors=[], warnings=[], order=None))
    ctx = make_ctx(server, account)

    instrument = MagicMock(symbol='NVDA', streamer_symbol='NVDA')
    instrument.build_leg = lambda qty, action: MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, '_create_instrument', AsyncMock(return_value=instrument))
        mp.setattr(server, '_get_quote', AsyncMock(return_value=(Decimal('10.0'), Decimal('12.0'))))
        result = asyncio.run(server.place_trade(
            ctx,
            'Buy to Open',
            5,
            'NVDA',
            strike_price=500,
            option_type='C',
            expiration_date='2024-12-20',
            dry_run=True,
        ))

    assert 'Dry Run' in result
    account.a_place_order.assert_awaited()


def test_get_live_orders(server):
    order = MagicMock(
        id='10',
        legs=[MagicMock(symbol='AAPL', action='BTO', quantity=1)],
        order_type='LIMIT',
        price=100,
        status='LIVE'
    )
    account = MagicMock()
    account.a_get_live_orders = AsyncMock(return_value=[order])
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_live_orders(ctx))
    assert '10' in result
    account.a_get_live_orders.assert_awaited()


def test_cancel_order(server):
    response = MagicMock(order=MagicMock(status=server.OrderStatus.CANCELLED))
    account = MagicMock()
    account.a_cancel_order = AsyncMock(return_value=response)
    ctx = make_ctx(server, account)

    result = asyncio.run(server.cancel_order(ctx, '12345'))
    assert 'Successfully cancelled' in result
    account.a_cancel_order.assert_awaited()


def test_cancel_order_dry_run(server):
    account = MagicMock()
    account.a_cancel_order = AsyncMock()
    ctx = make_ctx(server, account)

    result = asyncio.run(server.cancel_order(ctx, '999', dry_run=True))
    assert 'Dry run' in result
    account.a_cancel_order.assert_not_called()


def test_cancel_order_fallback_to_sync(server):
    response = MagicMock(order=MagicMock(status=server.OrderStatus.CANCELLED))
    account = MagicMock()
    account.cancel_order = AsyncMock(return_value=response)
    account.a_cancel_order = None
    ctx = make_ctx(server, account)

    result = asyncio.run(server.cancel_order(ctx, '55555'))
    assert 'Successfully cancelled' in result
    account.cancel_order.assert_awaited()


def test_cancel_order_sync_method(server):
    """Ensure sync cancel_order works when no async method is available."""
    response = MagicMock(order=MagicMock(status=server.OrderStatus.CANCELLED))

    def sync_cancel(session, order_id):
        return response

    account = MagicMock()
    account.cancel_order = MagicMock(side_effect=sync_cancel)
    account.a_cancel_order = None
    ctx = make_ctx(server, account)

    result = asyncio.run(server.cancel_order(ctx, '444'))
    assert 'Successfully cancelled' in result
    account.cancel_order.assert_called_once_with(ctx.request_context.lifespan_context.session, 444)


def test_modify_order_quantity(server):
    original_order = MagicMock(
        status=server.OrderStatus.LIVE,
        editable=True,
        legs=[MagicMock(symbol='AAPL', action='BTO', quantity=100)],
        time_in_force='DAY',
        order_type='LIMIT',
        price=150
    )
    account = MagicMock()
    account.a_get_order = AsyncMock(return_value=original_order)
    account.a_replace_order = AsyncMock(return_value=MagicMock(errors=[], warnings=[], order=MagicMock(id='99')))
    ctx = make_ctx(server, account)

    instrument = MagicMock(symbol='AAPL', streamer_symbol='AAPL')
    instrument.build_leg = lambda qty, action: MagicMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, '_create_instrument', AsyncMock(return_value=instrument))
        result = asyncio.run(server.modify_order(ctx, '67890', new_quantity=200))

    assert 'modified successfully' in result
    account.a_replace_order.assert_awaited()


def test_get_current_positions(server):
    pos = MagicMock(symbol='AAPL', instrument_type='Equity', quantity=10, mark_price=150, multiplier=1)
    account = MagicMock()
    account.a_get_positions = AsyncMock(return_value=[pos])
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_current_positions(ctx))
    assert 'AAPL' in result
    account.a_get_positions.assert_awaited()


def test_get_account_balances(server):
    balances = MagicMock(
        cash_balance=1000,
        equity_buying_power=5000,
        derivative_buying_power=5000,
        net_liquidating_value=10000,
        maintenance_excess=2000,
    )
    account = MagicMock()
    account.a_get_balances = AsyncMock(return_value=balances)
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_account_balances(ctx))
    assert 'Cash Balance' in result
    account.a_get_balances.assert_awaited()


def test_get_nlv_history(server):
    entry = MagicMock(
        time='2024-01-01T00:00:00Z',
        total_open=1,
        total_high=2,
        total_low=0.5,
        total_close=1.5,
    )
    account = MagicMock()
    account.a_get_net_liquidating_value_history = AsyncMock(return_value=[entry])
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_nlv_history(ctx, time_back='6m'))
    assert 'Net Liquidating Value History' in result
    account.a_get_net_liquidating_value_history.assert_awaited()


def test_get_transaction_history(server):
    txn = MagicMock(
        transaction_date=datetime(2024,1,1),
        transaction_sub_type='Trade',
        description='Buy AAPL',
        net_value=100
    )
    account = MagicMock()
    account.a_get_history = AsyncMock(return_value=[txn])
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_transaction_history(ctx, start_date='2024-01-01'))
    assert 'Transaction History' in result
    account.a_get_history.assert_awaited()


def test_get_prices(server):
    account = MagicMock()
    ctx = make_ctx(server, account)
    instrument = MagicMock(symbol='AAPL', streamer_symbol='AAPL')

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(server, '_create_instrument', AsyncMock(return_value=instrument))
        mp.setattr(server, '_get_quote', AsyncMock(return_value=(Decimal('1.5'), Decimal('1.6'))))
        result = asyncio.run(server.get_prices(
            ctx,
            'AAPL',
            expiration_date='2024-12-20',
            option_type='C',
            strike_price=200,
        ))

    assert 'Current prices for' in result


def test_get_metrics(server):
    metric = MagicMock(
        symbol='TSLA',
        implied_volatility_index_rank=0.5,
        implied_volatility_percentile=0.6,
        beta=1.2,
        liquidity_rating='Good',
        lendability='High',
        earnings=MagicMock(expected_report_date='2024-07-01', time_of_day='am')
    )
    server.metrics.a_get_market_metrics.return_value = [metric]

    account = MagicMock()
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_metrics(ctx, ['TSLA']))
    assert 'Market Metrics' in result
    server.metrics.a_get_market_metrics.assert_awaited()


def test_get_metrics_bracket_string(server):
    metric = MagicMock(
        symbol='SPY',
        implied_volatility_index_rank=0.3,
        implied_volatility_percentile=0.4,
        beta=0.9,
        liquidity_rating='High',
        lendability='Good',
        earnings=None,
    )
    server.metrics.a_get_market_metrics.return_value = [metric]

    account = MagicMock()
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_metrics(ctx, "[SPY]"))
    assert 'Market Metrics' in result
    server.metrics.a_get_market_metrics.assert_awaited_with(ctx.request_context.lifespan_context.session, ['SPY'])


def test_get_metrics_multiline_string(server):
    metric = MagicMock(
        symbol='SPY',
        implied_volatility_index_rank=0.2,
        implied_volatility_percentile=0.3,
        beta=0.8,
        liquidity_rating='High',
        lendability='Good',
        earnings=None,
    )
    server.metrics.a_get_market_metrics.return_value = [metric, metric]

    account = MagicMock()
    ctx = make_ctx(server, account)

    result = asyncio.run(server.get_metrics(ctx, "SPY,\nAAPL"))
    assert 'Market Metrics' in result
    server.metrics.a_get_market_metrics.assert_awaited_with(
        ctx.request_context.lifespan_context.session,
        ['SPY', 'AAPL']
    )


def test_check_market_status(server):
    result = asyncio.run(server.check_market_status())
    assert 'Market is currently CLOSED' in result
