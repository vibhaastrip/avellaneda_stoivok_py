import numpy as np
from src.config import SimulationConfig

class MarketEnvironment:
    def __init__(self, config: SimulationConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.current_time = 0.0
        self.mid_price = config.start_price
        self.last_return = 0.0
        self.price_history = []
        self.tiem_history =[]

    def step_price(self):
        previous_price = self.mid_price()
        dW = self.rng.normla(0,np.sqrt(self.config.dt))
        self.mid_price += self.mid_price*self.config.sigma*dW
        self.last_return = (self.mid_price - previous_price)/previous_price
        self.current_time += self.config.dt
        self.price_history.append(self.mid_price)
        self.tiem_history.append(self.current_time)
        return self.mid_price
    
    def _fill_probabilty(self, quote):
        if quote is None:
            return 0.0
        
        delta = abs(self.mid_price - quote)
        intesity = self.config.A * np.exp(-self.config.k * delta)
        return 1.0 - np.exp(-intesity*self.config.dt)
    
    def execute_order(self, bid, ask):
        p_buy = self._fill_probabilty(bid)
        p_ask = self._fill_probabilty(ask)

        if self.config.adverse_selection_strength:
            adverse_signal = np.sign(self.last_return)
            p_buy *= np.exp(-self.config.adverse_selection_strength * adverse_signal)
            p_sell *= np.exp(self.config.adverse_selection_strength * adverse_signal)

        p_buy = np.clip(p_buy, 0.0, 1.0)
        p_sell = np.clip(p_sell, 0.0, 1.0)

        bid_filled = self.rng.random() < p_buy
        ask_filled = self.rng.random() < p_sell

        return bid_filled, ask_filled
    
    

    