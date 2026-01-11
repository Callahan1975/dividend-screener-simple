# -------------------------
# FAIR VALUE MODELS
# -------------------------

def fair_value_yield(price, dividend_yield, normalized_yield=0.03):
    """
    Yield reversion model
    Fair Price = Annual Dividend / Normalized Yield
    """
    if price is None or dividend_yield is None:
        return None
    if dividend_yield <= 0 or normalized_yield <= 0:
        return None

    annual_dividend = price * dividend_yield
    return round(annual_dividend / normalized_yield, 2)


def fair_value_ddm(price, dividend_yield, growth, discount_rate=0.08):
    """
    Gordon Growth DDM
    Fair Value = D1 / (r - g)
    """
    if price is None or dividend_yield is None:
        return None
    if dividend_yield <= 0:
        return None

    # cap growth to realistic long-term level
    g = min(growth if growth is not None else 0.02, 0.06)
    r = discount_rate

    if r <= g:
        return None

    annual_dividend = price * dividend_yield
    next_dividend = annual_dividend * (1 + g)

    return round(next_dividend / (r - g), 2)


def upside_pct(price, fair_value):
    if price is None or fair_value is None:
        return None
    if price <= 0:
        return None
    return round((fair_value / price - 1) * 100, 1)
