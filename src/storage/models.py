from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    strategy : Mapped[str] = mapped_column(String, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)

    gamma: Mapped[float] = mapped_column(Float, nullable=True)
    sigma: Mapped[float] = mapped_column(Float, nullable=False)
    k: Mapped[float] = mapped_column(Float, nullable=False)
    A: Mapped[float] = mapped_column(Float, nullable=False)
    inventory_limit: Mapped[int] = mapped_column(Integer, nullable=True)
    min_quote_spread: Mapped[float] = mapped_column(Float, nullable=False)
    max_quote_distance: Mapped[float] = mapped_column(Float, nullable=True)
    volatility_spread_multiplier: Mapped[float] = mapped_column(Float, nullable=False)
    inventory_skew: Mapped[float] = mapped_column(Float, nullable=False)
    adverse_selection_strength: Mapped[float] = mapped_column(Float, nullable=False)

    net_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    sharpe: Mapped[float] = mapped_column(Float, nullable=False)
    sortino: Mapped[float] = mapped_column(Float, nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    max_abs_inventory: Mapped[float] = mapped_column(Float, nullable=False)
    fees_paid: Mapped[float] = mapped_column(Float, nullable=False)