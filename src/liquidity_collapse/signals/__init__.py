"""Every signal is `build(panel) -> DataFrame[date, barrid, score]`.

`date` is a formation date from the panel, and the score may use only information
public by that day's close. Higher score = expected to outperform.
"""

from . import composite, liquidity_shock

REGISTRY = {
    "liquidity_shock": liquidity_shock.build,
    "liquidity_shock_resid": liquidity_shock.build_residual,
    "fund_composite": composite.fund,
    "fund_plus_liquidity": composite.fund_plus_liquidity,
}
