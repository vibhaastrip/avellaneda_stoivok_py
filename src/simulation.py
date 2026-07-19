from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.config import SimulationConfig
from src.environment import MarketEnvironment
from src.market_maker import(
    Naive,
    InventorySkew,
    VolatilityScaled,
    AvellanedaStoikov,
    EnhanceAvellanedaStoikov
)

class Strategy(Enum):
    NAIVE = 'naive'
    INVENTORY_SKEWED = 'inventory_skewed'
    VOLATILITY_SCALED = "volatility_scaled"
    AVELLANEDA_STOIKOV = "avellaneda_stoikov"
    ENHANCED_AVELLANEDA_STOIKOV = "enhanced_avellaneda_stoikov"

@dataclass
class TradeLogEntry:
    time: float
    mid_price: float
    bid: float | None
    ask: float | None
    bid_filled: bool
    ask_filled: bool
    inventory: int
    cash: float
    wealth: float
    fees_paid: float

@dataclass
class SimulationResult:
    price_history: list[float]
    inventory_history: list[int]
    wealth_history: list[float]
    trade_log: list[TradeLogEntry]
    fees_paid: float

def build_agent(config: SimulationConfig, strategy: Strategy):
    quote_controls = {
        'inventory_limit' : config.inventory_limit,
        'min_quote_spread' : config.min_quote_spread,
        'max_quote_distance' : config.max_quote_distance
    }

    if strategy is Strategy.NAIVE:
        return Naive(
            spread = config.naive_spread,
            maker_rebate= config.maker_rebate,
            taker_fee=config.taker_fee,
            **quote_controls
        )
    
    if strategy is Strategy.INVENTORY_SKEWED:
        return InventorySkew(
            spread=config.naive_spread,
            inventory_skew=config.inventory_skew,
            maker_rebate=config.maker_rebate,
            taker_fee=config.taker_fee,
            **quote_controls
        )
    
    if strategy is Strategy.VOLATILITY_SCALED:
        return VolatilityScaled(
            spread=config.naive_spread,
            volatility_spread_multiplier= config.volatility_spread_multiplier,
            dt = config.dt,
            maker_rebate= config.maker_rebate,
            taker_fee=config.taker_fee,
            **quote_controls
        )
    
    if strategy is Strategy.AVELLANEDA_STOIKOV:
        return AvellanedaStoikov(
            T = config.T,
            sigma = config.sigma,
            gamma=config.gamma,
            k = config.k,
            maker_rebate= config.maker_rebate,
            taker_fee=config.taker_fee,
            **quote_controls
        )
    
    if strategy is Strategy.ENHANCED_AVELLANEDA_STOIKOV:
        return EnhanceAvellanedaStoikov(
            T = config.T,
            sigma = config.sigma,
            gamma=config.gamma,
            k = config.k,
            maker_rebate= config.maker_rebate,
            taker_fee=config.taker_fee,
            **quote_controls
        )
    
    raise ValueError(f"Unsupported Strategy: {strategy}")

def _realised_sigma(price_hist: list[float], dt: float, window: int):
    if len(price_hist) < 3:
        return None

    prices = np.array(price_hist[-(window + 1):])
    log_returns = np.diff(np.log(prices))

    if len(log_returns) < 2:
        return None

    return float(np.std(log_returns, ddof=1) / np.sqrt(dt))

def run_simulation(
        config: SimulationConfig,
        stratgey: Strategy,
        rng: np.random.Generator |None = None     
):
    env = MarketEnvironment(config, rng=rng)
    agent = build_agent(config=config, strategy=stratgey)
    trade_log = []

    steps = int(config.T/config.dt)

    for _ in range(steps):
        mid_price = env.step_price()
        realised_sigma = _realised_sigma(
            env.price_history,
            config.dt,
            config.volatility_window
        )
        bid, ask = agent.get_quotes(env.current_time, mid_price, realised_sigma)
        bid_filled, ask_filled = env.execute_order(bid, ask)
        agent.update_state(mid_price, bid_filled, ask_filled, bid, ask)

        trade_log.append(
            TradeLogEntry(
                time=env.current_time,
                mid_price=mid_price,
                bid=bid,
                ask=ask,
                bid_filled=bid_filled,
                ask_filled=ask_filled,
                inventory=agent.inventory,
                cash=agent.cash,
                wealth=agent.wealth_hist[-1],
                fees_paid=agent.fees_paid,
            )
        )

    return SimulationResult(
        price_history=env.price_history,
        inventory_history=agent.inventory_hist,
        wealth_history=agent.wealth_hist,
        trade_log=trade_log,
        fees_paid=agent.fees_paid
    )

