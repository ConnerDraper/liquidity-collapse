# Speaker script: When Trading Dries Up (`slides_fixed_gamma.pdf`)

The slides carry only short phrases; this is what we say. Numbers match the slides. Turnover is one-way (half of buys plus sells), the usual convention. "2019–24" is the held-back test, run once after the design was frozen.

---

## 1. Title
- We're Group 7. We built a signal from sudden collapses in how much a stock trades, and tested whether the Silver Fund could use it.
- Built on 2011–2018, then tested once on 2019–2024, years we held back the whole time.

## 2. The idea
- Most investors stay away from illiquid stocks: they're costly to trade and hard to exit. Only a small, patient group holds them.
- When a stock's trading suddenly collapses, the people holding it are no longer its natural owners, so it has to pass to that patient group.
- That hand-off is slow: with little volume, sellers can only exit a bit at a time, and each sale needs a slightly lower price. So the price drifts down for months.
- Why doesn't someone trade it away? Shorting small, illiquid stocks is expensive or impossible.
- **Likely question:** "Isn't illiquidity supposed to earn a premium?" Yes, for stocks that are *already* illiquid; we see that too. The drop happens *during* the move to those patient holders.

## 3. What is a liquidity event?
- **Illiquidity** is the Amihud measure: how much the price moves per dollar traded, averaged over the last 21 days. High means a little trading moves the price a lot.
- **Shock:** we compare it with the stock's own past year: the log of illiquidity now divided by its past-year average. Above zero means harder to trade than usual for this stock. (The average is a geometric mean of the previous twelve 21-day values, which is why the log of the ratio is exact.) A stock that's always illiquid doesn't count; only a *change* does.
- **Remove the fund's six scores:** each day we regress the shock on the fund's six scores (ivol, momentum, reversal, Barra momentum, beta, Barra reversal) and keep the residual, so we only count information the fund doesn't already have. Then we standardize across stocks.
- **Event:** a stock lands in the top 5% that day and wasn't flagged in the previous 21 trading days. The "age" of an event is the number of trading days since.

## 4. Bottom line
- Four points: the effect is real; just avoiding these stocks points the right way but is too noisy; the optimizer version works long-only after trading costs, including on the held-back years; and it's small.
- Say the small part ourselves: about +0.1–0.2% a year long-only after costs. It belongs as one input to the fund's alpha, not as a strategy on its own.

## 5. After trading dries up, the stock keeps falling
- Each line follows stocks after their trading collapsed and compares them with similar stocks: same size, equally illiquid beforehand, no collapse.
- All events fall steadily behind: about −0.5% after a month, about −2.9% after a year.
- Largest fifth barely moves: big stocks have plenty of buyers, so there's no slow hand-off.
- It's gradual, not a one-day drop, which is why the timing of trades matters later.
- **Likely question:** "Why do the blue and orange lines end at the same spot?" They really do end about the same; it isn't a labeling error.

## 6. Mostly in small, illiquid stocks
- Strongest in small and already-illiquid stocks, weak in the largest and most liquid: exactly where the hand-off story says, fewest patient buyers.
- It's also the catch: those stocks hold little weight in the benchmark, which limits how much a long-only fund can bet against them.

## 7. It is not something we already know
- Positive numbers mean flagged stocks earn *less*, which is what we bet on: about 0.18% a month less, after controlling for size, prior illiquidity, momentum and long-term reversal.
- Adding last month's return barely changes it, so it isn't short-term reversal.
- It's measured monthly, so it understates the first few weeks, where most of the effect is.
- **Likely question:** "Is this just idiosyncratic volatility?" No: the fund's volatility score is one of the six we remove first.

## 8. Just avoiding them points the right way, but is mostly noise
- The simple version: hold the benchmark, sell every stock flagged in the last 63 days to zero, and give its weight to similar stocks.
- **It is consistent with our story:** the stock-specific part of its return is positive, +0.09% a year. The flagged stocks did lag.
- **Why it's noisy:** it reweights 8–9% of the index. That creates factor tilts (size, industry, style) that carry about half its risk and earn about nothing. So its IR is only 0.1–0.2.
- **What the optimizer does differently:** it hedges those tilts with the risk model and sizes each bet by the expected drift: bigger for fresh events and more volatile stocks, smaller as the event ages. Result: more return (+0.12%) at a quarter of the risk (0.11% vs about 0.45%).
- **Likely question:** "Doesn't that contradict your idea?" No: the avoid list and the optimizer make the same bet; the optimizer just takes it without the noise. That's the transfer-coefficient idea: the closer your positions are to what the alphas call for, the more of the signal reaches the portfolio.

## 9. How we bet on it
- **Our alpha** is the Grinold–Kahn form: volatility × how far flagged stocks fell behind similar stocks, per unit of volatility, for the event's age (first month, or months two to three). Re-estimated every January from earlier years only. No recent event means zero alpha.
- **Optimizer:** expected return, minus a penalty for risk, minus trading costs; active risk capped at 1%.
  - γ (how much we dislike risk) is fixed for each year: the value that would have put the no-limits portfolio at 1% risk, from the previous year. So we bet more when many events are live.
  - Every trade pays half the bid–ask spread, times λ. λ is picked each January from earlier years (between 0.5 and 4); for 2019–24 it stayed at its 2018 value.
- **Three portfolios, same constraints except:**
  - *No limits, no costs*: the yardstick for the signal's full strength. Not meant to be traded.
  - *Limited shorting + costs*: each stock can be shorted up to 0.25% of the fund.
  - *No shorting + costs*: what the fund can actually hold.
- **Likely question:** "Why is long-only so small?" The stocks it bets against hold little benchmark weight, and it can't go below zero.

## 10. Calm stocks fell less than we expected, volatile stocks more
- An earlier version gave every flagged stock the same alpha, and the optimizer piled into calm stocks.
- Calm flagged stocks delivered about a third of what we predicted; the most volatile about one and a half times.
- So the drift scales with volatility, and we put volatility into the alpha. That was the last change before freezing the design.

## 11. Daily rebalancing
- Chart: each line scaled to the same risk so the shapes compare; shaded = held-back years. The table has the real numbers.
- **No limits:** about +1.3–1.4% a year at about 1.1–1.2% risk, IR about 1.2 in both periods.
- **Limited shorting + costs:** +0.57% a year (IR 0.59) in 2011–18, +0.35% (IR 0.31) in 2019–24. Turnover about 6× a year, so positions last about two months, as long as the drift does. It pays about 0.5% a year in spreads.
- **No shorting + costs:** +0.12% (IR 1.04) in 2011–18, +0.17% (IR 0.56) in 2019–24. Turnover about half a time a year, almost nothing in spreads.
- **TC** (about 0.8 limited, about 0.35 long-only): how much of the signal's strength survives the shorting rule and costs together.
- **Scaling:** the chart puts every line at the same risk. Long-only actually runs at only 0.1–0.3% risk, so its line is scaled up 5–10×, which it can't do in practice: to take more risk it would have to short.
- **Likely question, "why does the more constrained portfolio have the better risk-adjusted return?"** It's trading costs.
  - Before costs, the two have about the same IR: 1.12 vs 1.16 in 2011–18; in 2019–24 limited shorting is actually ahead, 0.73 vs 0.64.
  - Limited shorting trades about 6× a year, a third of it in small stocks, and spreads take about half its return (about 0.5% a year).
  - Long-only trades about 0.5× a year in larger stocks, and spreads cost it 0.01–0.02%.
  - So the higher IR comes with far less money: limited shorting still earns 2–5× more in absolute terms. Appendix slide 16 has the numbers.
- **Likely question, "is the turnover crazy?"** About 6× a year one-way is faster than value or momentum (under 1× to 4×) but slower than short-term reversal (10×+). Charging costs cut it by about 60%.

## 12. Monthly rebalancing
- Same layout, trading once a month.
- **Limited shorting + costs** holds up better monthly in 2019–24: +0.54% a year (IR 0.54), turnover about 1.5×.
- **No shorting + costs:** +0.14% (IR 1.10) in 2011–18, +0.18% (IR 0.58) in 2019–24.
- **No limits:** about +0.6% a year, IR about 0.5. Weaker than daily, because a monthly portfolio reaches most collapses late.

## 13. How much to trust it
- We changed the design several times on 2011–18, which flatters those years. That's why 2019–24 was frozen out and run once.
- Long-only after costs earned +0.17% a year in 2019–24, IR 0.56. Six years is short: the true IR could plausibly be anywhere from about −0.3 to 1.4.
- **Be ready for this:** in 2019–24 most of limited shorting's return after costs came in the first few weeks of each year. The no-cost portfolios show the same January pattern, so it's in the data, not our set-up. Long-only doesn't depend on January.

## 14. Conclusion
- When a stock's trading collapses, it lags for months, because of a slow hand-off to patient buyers.
- A long-only fund can capture it, but only with the optimizer; just avoiding the stocks is too noisy.
- It's small, so it belongs as one input to the fund's combined alpha.
- Next: combine it with the fund's six scores and test the combination with trading costs.

---

## Appendix (only if asked)

**16. Why does long-only have the higher IR?** It's trading costs: before costs the IRs are about equal (1.12 vs 1.16, then 0.73 vs 0.64); limited shorting's spreads take about half its return, long-only's almost nothing. If they push further, "but even without costs long-only's IR is higher" (1.97 vs 1.07 on 2011–18, in the no-cost runs):
  - the portfolios that can short fund their bets by going long similar small, illiquid stocks, and those lag too. In the earlier design's breakdown, the hedges gave back over half the gain from the flagged stocks;
  - long-only funds its small underweights with overweights spread over larger benchmark stocks, which gave back nothing;
  - the model actually *expected* long-only to have the lower IR (0.5–0.7 vs 1.1–1.4). Six to eight years of a portfolio this small can't tell those IRs apart with confidence, but the gap shows up in every version we ran.

**17. IC by year:** after removing the fund's scores, the signal ranks next month's stock-specific returns correctly in 12 of 13 years (mean IC 0.015, t 6.0). It's new information, not an accident of one period.

**18. What did we change, and when?** v2: a daily flag missed most of the drift, so each event got an age. v3: a noisy positive alpha for large stocks made the portfolio buy them, so positive alphas were banned and bets hedged with similar stocks. v4: alpha scales with volatility. Then frozen.

**19. How we picked the short limit:** we tried 0.05% to 2% per stock. The IRs were all within noise of each other (0.94–1.14), so we picked by risk: 0.25% sits between long-only and no limits.

**20. Every portfolio with and without costs:** costs cut turnover by about 60% for the shorting portfolios and roughly halved their return; long-only barely changes, since it trades little. A star marks what a no-cost portfolio would pay in spreads if traded.

**21. Where does the return come from?** By the age of the event behind each position: the bets against freshly flagged stocks earn; the hedges in similar stocks give some back, because small illiquid stocks lag in general.

---

## If someone pushes hard

| Question | Answer |
|---|---|
| "A tenth of a percent a year is nothing." | On its own, yes. But it's new information, and as one input its trades net against other signals' trades. |
| "Isn't this data mining?" | We changed the design on 2011–18, then froze it and tested once on six untouched years. Long-only stayed positive after costs. |
| "Why does the more constrained portfolio have the better IR?" | Trading costs. Before costs the IRs are about equal; limited shorting trades 6× a year and spreads take half its return, long-only pays almost nothing. And long-only runs at only 0.1–0.3% risk, so it makes less money. |
| "Why not just avoid the stocks?" | Same bet, but it moves 8–9% of the index and half its risk is factor tilts. The optimizer takes the same bet without the noise. |
| "What would prove you wrong?" | Returns rising after a collapse, the effect being as strong in large liquid stocks, or failure in future data. |
| "Why not just short these stocks?" | The fund is long-only, and borrowing these stocks is expensive. That's exactly why the drift exists. |

---

## Earlier deck (`slides.pdf`, 5% risk target, no costs)
Same order and text, except: slide 9 has the earlier objective (maximize expected return at a fixed 5% risk) and two portfolios; slides 11–12 show no limits and no shorting without costs (long-only daily +0.19% in 2011–18, +0.51% in 2019–24); slide 18 explains why long-only was bigger in 2019–24 (large stocks only got an alpha after 2015).
