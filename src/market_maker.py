import numpy as np

class MarketMaker:
    def __init__(
            self,
            maker_rebate,
            taker_fee,
            start_cash = 0,
            inventory_limit = None,
            min_quote_spread = 0.0,
            max_quote_distance = None
    ):
        self.cash = start_cash
        self.inventory = start_cash
        self.maker_rebate = maker_rebate
        self.taker_fee = taker_fee
        self.inventory_limit = inventory_limit
        self.min_quote_spread = min_quote_spread
        self.max_quote_distance = max_quote_distance

        self.fees_paid = 0.0
        self.inventory_hist = [self.inventory]
        self.wealth_hist = [self.cash]

    def update_state(self,mid_price, bid_filled, ask_filled, fill_price_bid, fill_price_ask):
        if bid_filled:
            self.inventory += 1
            if fill_price_bid >= mid_price:
                net_fee = fill_price_bid * self.taker_fee
            else:
                net_fee = - fill_price_bid * self.maker_rebate

            self.fees_paid += net_fee
            transaction_cost = fill_price_bid + net_fee
            self.cash -= transaction_cost
        
        if ask_filled:
            self.inventory -= 1
            if fill_price_ask <= mid_price:
                net_fee = fill_price_ask * self.taker_fee
            else:
                net_fee = - fill_price_ask * self.maker_rebate

            self.fees_paid += net_fee
            revenue = fill_price_ask - net_fee
            self.cash += revenue

        current_wealth = self.cash +(self.inventory*mid_price)

        self.inventory_hist.append(self.inventory)
        self.wealth_hist.append(current_wealth)

    def apply_quote_controls(self, mid_price, bid, ask):
        if self.max_quote_distance is not None:
            bid = max(bid , mid_price - self.max_quote_distance)
            ask = min(ask , mid_price+self.max_quote_distance)
        
        if self.min_quote_spread and ask-bid < self.min_quote_spread:
            center = (bid+ask)/2
            bid = center - (self.min_quote_spread/2)
            ask = center + (self.min_quote_spread/2)

        if self.inventory_limit is not None:
            if self.inventory >= self.inventory_limit:
                bid = None
            if self.inventory <= -self.inventory_limit:
                ask = None
        
        return bid,ask
    
class Naive(MarketMaker):
    def __init__(
            self,
            spread,
            maker_rebate,
            taker_fee,
            inventory_limit = None,
            min_quote_spread = 0.0,
            max_quote_distance = None
    ):
        super().__init__(
            maker_rebate=maker_rebate,
            taker_fee=taker_fee,
            inventory_limit=inventory_limit,
            min_quote_spread=min_quote_spread,
            max_quote_distance=max_quote_distance,
        )
        self.spread = spread

    def get_quotes(self, current_time, mid_price, realised_sigma = None):
        bid = mid_price - (self.spread/2)
        ask = mid_price + (self.spread/2)
        return self.apply_quote_controls(mid_price,bid,ask)
    
class InventorySkew(Naive):
    def __init__(self, spread, inventory_skew, maker_rebate, taker_fee, **kwargs):
        super().__init__(
            spread=spread,
            maker_rebate=maker_rebate,
            taker_fee=taker_fee,
            **kwargs,
        )
        self.inventory_skew = inventory_skew

    def get_quotes(self, current_time, mid_price, realized_sigma=None):
        center = mid_price - (self.inventory * self.inventory_skew)
        bid = center - (self.spread / 2)
        ask = center + (self.spread / 2)
        return self.apply_quote_controls(mid_price, bid, ask)
    
class VolatilityScaled(Naive):
    def __init__(
        self,
        spread,
        volatility_spread_multiplier,
        dt,
        maker_rebate,
        taker_fee,
        **kwargs,
    ):
        super().__init__(
            spread=spread,
            maker_rebate=maker_rebate,
            taker_fee=taker_fee,
            **kwargs,
        )
        self.volatility_spread_multiplier = volatility_spread_multiplier
        self.dt = dt

    def get_quotes(self, current_time, mid_price, realized_sigma=None):
        volatility_spread = 0.0
        if realized_sigma is not None:
            volatility_spread = (
                self.volatility_spread_multiplier
                * mid_price
                * realized_sigma
                * np.sqrt(self.dt)
            )

        spread = max(self.spread, volatility_spread)
        bid = mid_price - (spread / 2)
        ask = mid_price + (spread / 2)
        return self.apply_quote_controls(mid_price, bid, ask)

class AvellanedaStoikov(MarketMaker):
    def __init__(
        self,
        T,
        sigma,
        gamma,
        k,
        maker_rebate,
        taker_fee,
        inventory_limit=None,
        min_quote_spread=0.0,
        max_quote_distance=None,
    ):
        super().__init__(
            maker_rebate=maker_rebate,
            taker_fee=taker_fee,
            inventory_limit=inventory_limit,
            min_quote_spread=min_quote_spread,
            max_quote_distance=max_quote_distance,
        )
        self.T = T
        self.sigma = sigma
        self.gamma = gamma
        self.kappa = k
    def reservation_price(self, mid_price, t, sigma=None):
        sigma = self.sigma if sigma is None else sigma
        time_left = max(self.T-t, 0)
        return mid_price - self.inventory*self.gamma*(sigma**2)*time_left
    
    def optimal_spread(self, t, sigma = None):
        sigma = self.sigma if sigma is None else sigma
        time_left = max(self.T-t, 0)
        return self.gamma * (sigma**2) * time_left + (2.0/self.gamma)*np.log(1 + self.gamma/self.kappa)
    
    def get_quotes(self,current_time, mid_price, realised_sigma= None):
        r = self.reservation_price(mid_price , current_time)
        delta = self.optimal_spread(current_time)/2.0

        bid = r-delta
        ask = r+delta

        return self.apply_quote_controls(mid_price, bid,ask)
class EnhanceAvellanedaStoikov(AvellanedaStoikov):
    def get_quotes(self, current_time, mid_price, realized_sigma=None):
        
        sigma = self.sigma if realized_sigma is None else realized_sigma
        r = self.reservation_price(mid_price, current_time, sigma=sigma)
        delta = self.optimal_spread(current_time, sigma=sigma) / 2.0

        bid = r - delta
        ask = r + delta

        return self.apply_quote_controls(mid_price, bid, ask)

