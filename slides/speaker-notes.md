# Speaker notes: When Trading Dries Up

A few talking points per slide, plus the question most likely to come up. Numbers match the slides.

---

## 1. Title
- We're Group 7. We built a signal from sudden drops in how much a stock trades, and tested whether the Silver Fund could use it.
- Built on 2011–2018, then tested once on 2019–2024, years we held back the whole time.

## 2. The idea
- **One sentence:** most investors don't want illiquid stocks, so when a stock suddenly stops trading it has to pass to a small group of patient buyers, and that slow hand-off pushes the price down for months.
- Why slow: sellers can't sell much at once because nobody's trading, and the few buyers want a lower price for each sale.
- Why nobody trades it away: shorting small, illiquid stocks is expensive or impossible.
- We first strip out anything the fund's six existing scores already capture, so we only count new information.
- **Likely question:** "Isn't illiquidity supposed to earn a premium?" Yes, for stocks that are *already* illiquid; we see that too. The drop happens *during* the move to those patient holders.

## 3. Bottom line
- Four takeaways: it's real, the simple version fails, the optimizer version works out of sample, and it's small.
- Say the small part ourselves before anyone else does: about +0.5% a year long-only, about +0.3% after trading costs.
- Our recommendation is to use it as one input to the fund's alpha, not as a strategy on its own.

## 4. After trading dries up, the stock keeps falling
- Each line follows stocks after their trading collapsed and compares them with similar stocks: same size, equally illiquid beforehand, no collapse.
- Blue (all events): steadily falls behind, about −0.5% after a month and −2.9% after a year.
- Grey (largest fifth): barely moves, because big stocks have plenty of buyers and there's no hand-off.
- It's gradual, not a one-day drop, which is why the timing of trades matters later.
- **Likely question:** "Why do blue and orange end at the same spot?" They really do end about the same; it isn't a labeling error.

## 5. Mostly in small, illiquid stocks
- The effect is strongest in small and illiquid stocks and weak in the largest and most liquid.
- That's exactly where the hand-off story says it should be: fewest patient buyers.
- It's also the catch: those stocks hold little weight in the benchmark, which limits how much a long-only fund can bet against them.

## 6. It is not something we already know
- Positive numbers mean flagged stocks earn *less*, which is what we bet on.
- About 0.18% a month less, after controlling for size, prior illiquidity, momentum and long-term reversal.
- Adding last month's return barely changes it, so it isn't short-term reversal.
- It's measured monthly, so it understates the first few weeks, where most of the effect is.
- **Likely question:** "Is this just idiosyncratic volatility?" No: the fund's volatility score is one of the six we remove first.

## 7. The simple idea doesn't work
- The obvious version: hold the benchmark, drop every recently flagged stock, give its weight to similar stocks. No model, no optimizer.
- It earns about +0.05 to +0.10% a year, not significant.
- Why: the flagged stocks are small, so dropping them tilts the portfolio toward big stocks, and that noise is bigger than the effect.
- That motivates the next slide: the optimizer removes that noise.

## 8. How we bet on it
- **Our alpha:** the stock's volatility × how far flagged stocks historically fell behind per unit of volatility, depending on how long ago the collapse was (first month, or months two to three). This is the Grinold–Kahn idea: the same signal is worth more in a stock that moves more.
- Re-estimated every January using only earlier years. Stocks without a recent collapse get zero alpha.
- **Optimizer:** maximize expected active return at a fixed 5% active risk.
- **Two versions:** the same constraints (fully invested, zero beta, hedged within size and liquidity groups); the only difference is whether shorting is allowed.
- No-limits shows the signal's full strength. No-shorting is what the fund can actually hold.
- **Likely question:** "Why is long-only risk only 0.1–0.5%, not 5%?" To reach 5% it would have to add bets with no expected return. It takes only the risk its real bets need, and there's little benchmark weight to sell.

## 9. Calm stocks fell less than we expected, volatile stocks more
- An earlier version gave every flagged stock the same alpha. The optimizer then piled into calm stocks, since the same return looks better at lower risk.
- Calm flagged stocks delivered about a third of what we predicted; the most volatile about one and a half times.
- So the drift scales with volatility, and we put volatility into the alpha. That was the last change before freezing the design.

## 10. Daily rebalancing
- Left chart: each period's line is scaled to the same risk so the shapes are comparable; the table has the real numbers.
- No limits: IR about 1.2 in both periods.
- No shorting: +0.19% a year (IR 1.8) in 2011–18, then +0.51% a year (IR 1.05) in the held-back years.
- Transfer coefficient 0.44 and 0.36: the long-only version keeps roughly 40% of what the signal could earn without the long-only rule.
- **Likely question:** "The no-limits turnover is enormous." It isn't a strategy we'd trade; it's a yardstick for how much information the signal has. The long-only portfolio turns over only a few times a year, which is normal.
- **Likely question:** "Are those IRs inflated by thinly traded stocks?" No. Measured from monthly returns instead of daily, the long-only IR is the same, about 1.0.

## 11. Monthly rebalancing
- Same layout, trading once a month.
- No limits still works (IR about 0.5), but long-only earns nothing in 2019–2024.
- Most of the drift comes in the first weeks, and a monthly portfolio reaches most collapses too late. That's why we recommend daily.

## 12. How much to trust it
- We changed the design four times on 2011–2018, which flatters those years. That's why 2019–2024 was frozen out and run exactly once.
- The 2019–24 long-only portfolio is bigger because the model learned over time that large stocks drift too (appendix).
- After trading costs, long-only earns about +0.3% a year (IR about 0.6).
- Six years is short: the true IR could plausibly be anywhere from about 0.2 to 1.9.

## 13. Conclusion
- Trading collapses lead to months of underperformance, because of a slow hand-off to patient buyers.
- A long-only fund can use it, but only with the optimizer and daily trading.
- It's small, so it belongs as one input to the fund's combined alpha.
- Next step: combine it with the fund's six scores and test the combination with trading costs.

---

## Appendix (only if asked)

**15. IC by year:** after removing the fund's scores the signal predicts next-month returns in 12 of 13 years. That's evidence it's new information, not an accident of one period.

**16. What did we change, and when?**
1. A daily flag missed most of the drift, so each collapse got an "age".
2. A noisy positive alpha for large stocks made the portfolio buy them, so positive alphas were banned and bets were hedged with similar stocks.
3. Calm stocks under-delivered, so volatility went into the alpha.

The design was frozen before the held-back test.

**17. Why is long-only bigger in 2019–24?** Until 2015 the model gave large flagged stocks no alpha, so it could only sell small stocks with little benchmark weight. Later, large stocks got a negative alpha too, so there was more to sell. Same signal, bigger portfolio.

**18. Where does the return come from?** In the no-limits runs, the bets against freshly flagged stocks earn; the hedges in similar stocks lose some back, because small illiquid stocks lag in general. What's left is the effect of the collapse itself.

---

## If someone pushes hard

| Question | Answer |
|---|---|
| "Half a percent a year is nothing." | On its own, yes. But it's new information with a high return per unit of risk, and as one input its trades net against other signals' trades. |
| "Isn't this data mining?" | We made four rounds of changes on 2011–18, so we froze the design and tested once on six untouched years. It held up when rebalanced daily. |
| "What would prove you wrong?" | Returns rising after a collapse, the effect showing up as strongly in large liquid stocks, or failure in future out-of-sample data. |
| "Why not just short these stocks?" | The fund is long-only, and even if it could short, borrowing these stocks is expensive. That's exactly why the drift exists. |

---

# Fixed-γ deck (`slides_fixed_gamma.pdf`): notes for the slides that differ

This deck uses the new design: γ fixed each year, active risk capped at 1%, and trading costs charged. Slides not listed here are the same as above. All numbers are 2011–2018; this design hasn't been run on 2019–2024.

## 8. How we bet on it
- **Objective:** expected return minus a risk penalty minus trading costs, with active risk capped at 1%.
- **γ (how much we dislike risk)** is set once a year, from the previous year. Risk floats: we bet more when many collapses are live, less when few are.
- **Costs:** every trade pays half the bid–ask spread, times λ. λ is picked each January from earlier years: 1 at first, then 0.5–4.
- **The three portfolios:**
  - *No limits, no costs*: the yardstick for the signal's full strength.
  - *Limited shorting + costs*: each stock can be shorted up to 0.25% of the fund.
  - *No shorting + costs*: what the fund can hold.
- **Likely question:** "Why does no limits pay no costs?" It isn't meant to be traded. It measures how much the signal could earn, and TC is measured against it.

## 10. Daily rebalancing
- **No limits:** +1.30% a year at 1.1% risk (IR 1.17), turning over 35× a year.
- **Limited shorting after costs:** +0.57% a year (IR 0.59), turnover 12× a year, and it pays about 0.5% a year in spreads.
- **Long-only after costs:** +0.12% a year (IR 1.04, t ≈ 2.9), turnover under 1× a year, almost nothing in spreads.
- **TC 0.84 and 0.33:** how much of the signal survives the shorting rule and costs together.
- **Likely question, "the turnover is huge":** charging costs cut it to a third (35× → 14× with no limits, 30× → 12× with limited shorting). The rest is the price of a signal that lives for weeks.

## 11. Monthly rebalancing
- **Long-only after costs** does about as well as daily: +0.14% a year, IR 1.10, positive in 8 of 8 years.
- **Limited shorting after costs:** +0.43% a year (IR 0.60).
- **No limits:** +0.62% a year (IR 0.56). It is weaker than daily because a monthly portfolio reaches most collapses late.

## Appendix: every portfolio with and without trading costs
- **Costs cut turnover** by about 60% for no limits and limited shorting, and roughly halved the return.
- **Long-only barely changes:** it trades little to begin with.
- \* = what that portfolio would pay in spreads if traded alone, not charged.

## If someone asks why we changed the design
- **Judges asked for something between "trades too much" and "takes too little risk".**
  - Limited shorting is the in-between portfolio.
  - Charging costs is how we tamed turnover.
  - Fixing γ is the textbook set-up.
- **The 1% cap keeps every portfolio on a comparable scale.**
