# MonteCarloFinance AI Coding Instructions

## Project Overview

MonteCarloFinance is a Python library for financial market simulations using Monte Carlo methods. It models market dynamics (GBM + Jump Diffusion), bond pricing (Vasicek), and behavioral economics (Panic/Shockwaves).

## Architecture & Core Components

- **Orchestrator**: `MonteCarloSimulation` (`core/simulation.py`) manages the simulation loop, coordinating models and collecting results.
- **Configuration**: `SimulationConfig` (`core/config.py`) and sub-configs (`MarketConfig`, `BondConfig`, `LiquidityConfig`, `PGREConfig`) centralize all tunable parameters using `dataclasses`.
- **Models**:
  - **Liquidity**: `LiquidityModel` (`models/liquidity.py`) - Simulates global liquidity cycles (QE/QT), acting as the "tide" that lifts/lowers all boats.
  - **Market**: `MarketModel` (`models/market.py`) - Handles price evolution, now driven by both fundamentals and `liquidity_beta`.
  - **Bond**: `BondModel` (`models/bond.py`) - Handles yield curves and general default risk.
  - **Credit (Specific)**: `PGREModel` (`models/pgre.py`) - Models specific idiosyncratic risk for One Market Plaza (OMP) with discrete event triggers (2026 Extension, 2027 Debt Yield).
  - **Consumer**: `ConsumerModel` (`models/consumer.py`) - Models the "Real Economy" (Wages vs CoL, Debt) to track the "Hollow Middle" thesis.
  - **State**: Models maintain internal state (`_current_value`, `_history`) and expose properties for current status.
- **Triggers**: `ShockwaveTrigger` and `PanicModel` modify model parameters (volatility, drift) dynamically based on simulation state.

## Code Style & Patterns

- **Data Structures**: Prefer `dataclasses` for structured data and configuration.
- **Type Hinting**: Mandatory for all function signatures. Use `typing` module (`List`, `Dict`, `Optional`, `Callable`).
- **Documentation**: Use Google-style docstrings (Args, Returns, Attributes) for all public classes and methods.
- **Math/Numerics**: Use `numpy` for all vector operations and random number generation. Avoid raw `random` module.
- **Randomness**: Always use `utils.random.RandomGenerator` or pass a seed to ensure reproducibility.

## Critical Workflows

- **Installation**: `pip install -e ".[dev]"`
- **Testing**: Run `pytest`. Tests should ensure deterministic results by setting `random_seed` in `SimulationConfig`.
- **Simulation Loop**:
  1. Initialize `MonteCarloSimulation` with `SimulationConfig`.
  2. `run()` method executes the loop:
     - **Step 1 (The Tide)**: Update `LiquidityModel`.
     - **Step 2 (The Reaction)**: Update `ShockwaveTrigger` (checks for events) and `PanicModel` (adjusts volatility/drift).
     - **Step 3 (The Asset)**: Step `MarketModel` (influenced by Liquidity return).
     - **Step 4 (The Debt)**: Step `BondModel` and `PGREModel`.
     - **Step 5 (The People)**: Step `ConsumerModel` (Wages, CoL, Debt).
     - Records data in `SimulationResults`.

## Common Tasks

- **Adding a New Shock Type**:
  1. Add entry to `ShockType` enum in `triggers/shockwave.py`.
  2. Implement trigger logic in `ShockwaveTrigger`.
- **Extending Statistics**:
  1. Add field to `SimulationResults` dataclass.
  2. Update `SimulationResults.calculate_statistics()` in `core/simulation.py`.

## Key Files

- `monte_carlo_finance/core/config.py`: All simulation parameters.
- `monte_carlo_finance/core/simulation.py`: Main loop and result aggregation.
- `monte_carlo_finance/models/market.py`: Core math for market movements.
- `monte_carlo_finance/models/liquidity.py`: Global liquidity dynamics.
- `monte_carlo_finance/models/pgre.py`: Specific credit risk modeling (OMP).

---

## Study Criteria: Data Sources & Parametrization Factors

This section documents the external data sources and behavioral factors used to weight and parameterize the simulation. The AI assistant will help gauge appropriate weights for these inputs.

### Macroeconomic & Treasury Indicators

| Factor                                 | Source              | Purpose                                                                                        |
| -------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------------- |
| **US 10Y-2Y Treasury Spread**          | FRED (T10Y2Y)       | Yield curve inversion signal; recession probability input for `drift` and `panic_sensitivity`. |
| **US Overnight Reverse Repo (ON RRP)** | NY Fed              | Liquidity conditions; impacts `jump_intensity` and systemic stress triggers.                   |
| **US 30-Year Fixed Mortgage Rate**     | FRED (MORTGAGE30US) | Housing market stress; feeds into CMBS default probability.                                    |
| **US U6 Unemployment Rate**            | BLS                 | Broad labor market slack; influences consumer default risk and `panic_sensitivity`.            |

### Credit & Fixed Income Markets

| Factor                                               | Source                      | Purpose                                                                                           |
| ---------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------- |
| **TIC Quarterly Report Data**                        | US Treasury                 | Foreign holdings of US securities; capital flow stress indicator.                                 |
| **DTCC Sponsored Member Volume**                     | DTCC                        | DVP vs GC, Repo vs Reverse Repo volumes; measures repo market stress and collateral velocity.     |
| **US CMBS Delinquency Rates (Office & Multifamily)** | Trepp, FRED                 | Direct input to `BondConfig.credit_spread` and default probability curves.                        |
| **PGRE Quarterly Supplementary Reports**             | SEC EDGAR (Paramount Group) | Assess debt yield covenant compliance for Feb 6, 2026 extension; CMBS cascade trigger if default. |

### Equity & Alternative Indices

| Factor                                            | Source               | Purpose                                                                           |
| ------------------------------------------------- | -------------------- | --------------------------------------------------------------------------------- |
| **S&P 500 Information Technology Index**          | S&P Global           | Tech sector sentiment; correlates with AI investment P&L and market `volatility`. |
| **Gold Price (SGOL)**                             | ETF price feed       | Flight-to-safety indicator; inverse correlation to risk assets during panic.      |
| **Cambridge Associates US Private Equity Index**  | Cambridge Associates | Illiquid asset stress; delayed mark-to-market amplifies shockwaves.               |
| **Cambridge Associates US Venture Capital Index** | Cambridge Associates | VC valuation stress; impacts tech sector contagion modeling.                      |

### Regulatory & Structural Events

| Factor                              | Source                          | Effective Date | Purpose                                                                                                                                                                                     |
| ----------------------------------- | ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Stablecoin Reserve Requirements** | US Congress - S.1582 GENIUS Act | Jan 17, 2026   | Forced liquidation of non-compliant assets (e.g., CMBS holdings). Creates discrete `ShockwaveEvent` with `ShockType.REGULATORY`. Models impact on One Market Plaza and broader bond market. |

### AI & Technology Investment Metrics

| Factor                                 | Source                              | Purpose                                                                                             |
| -------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| **AI Investment Profit-to-Loss Ratio** | Earnings reports, analyst estimates | Quantifies realized vs unrealized AI capex returns; feeds tech sector `volatility` and `jump_mean`. |

### Behavioral & Sentiment Factors

These factors modify the `PanicConfig` and `ShockwaveConfig` parameters:

| Factor                                                | Quantification Approach                                                         | Maps To                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Cognitive Bias Factor (Blind Trust / Complacency)** | Inverse VIX term structure, put/call ratio divergence, retail sentiment surveys | Reduces `panic_sensitivity` during low-vol regimes; increases `jump_mean` severity when shock hits. |
| **Panic Sell Factor**                                 | VIX spikes, fund outflow data, social media sentiment velocity                  | Direct multiplier on `panic_sensitivity` and `volatility_multiplier`.                               |
| **Automated Trading/Selling Factor**                  | Market microstructure data (order flow imbalance, HFT activity estimates)       | Increases `cascade_decay` speed and `selloff_intensity`; models algorithmic amplification.          |

---

## Parameterization Workflow

1. **Data Ingestion**: Scrape/API the sources listed above into a `DataIngestion` module.
2. **Normalization**: Convert raw values to z-scores or percentile ranks for cross-factor comparability.
3. **Weight Assignment**: AI-assisted analysis to assign weights based on:
   - Historical correlation with market stress events
   - Lead/lag relationships
   - Current regime (risk-on vs risk-off)
4. **Config Generation**: Map weighted factors to `SimulationConfig` fields.
5. **Scenario Definition**: Define base case, stress case, and tail-risk scenarios using different weight profiles.

---

## Data Directory Structure & Status

All data resides in `data/`. Current holdings and format assessment:

| Directory               | Files                                                                                  | Format Status          | Notes                                                     |
| ----------------------- | -------------------------------------------------------------------------------------- | ---------------------- | --------------------------------------------------------- |
| `FRED/`                 | T10Y2Y.csv, RRPONTSYD.csv, MORTGAGE30US.csv, U6RATE.csv, DALLCCACBEP.csv, DTWEXBGS.csv | ✅ Clean CSV           | Daily/weekly series, ready for ingestion                  |
| `DTCC/`                 | SponsoredVolume-2025-12-04.csv                                                         | ⚠️ Currency formatting | Remove `$` and commas from numeric columns                |
| `TIC/`                  | HTML files                                                                             | ⚠️ Needs parsing       | Convert HTML tables to CSV                                |
| `CMBS/`                 | sec-stats-cmbs-issuances.csv                                                           | ✅ Clean CSV           | Quarterly issuance data                                   |
| `TREPP/`                | CMBS_Delinquency_2025.csv                                                              | ✅ Clean CSV           | Citation metadata included, strip `[cite_start]` prefixes |
| `Cambridge_Associates/` | private_equity.csv, venture_capital.csv, real-estate.csv                               | ✅ Clean CSV           | Quarterly returns by horizon                              |
| `Gold/`                 | AUX-USD.csv                                                                            | ⚠️ Date format         | Parse `DD-MMM-YYYY` format, 10-day intervals              |
| `AI_Losses/`            | Losses_By_AI_Company.csv                                                               | ✅ Clean CSV           | Company-level P&L estimates                               |
| `GENIUS-Act/`           | Effects.txt                                                                            | ✅ Analysis doc        | Mathematical framework for regulatory shock               |
| `iShares/`              | .xls file                                                                              | ⚠️ Convert to CSV      | S&P 500 IT Index fund data                                |
| `Microsoft/`            | 2025-financial-report.csv                                                              | ✅ Clean CSV           | Azure/AI segment data for loss attribution                |
| `MIT/`                  | State_Of_AI_in_Business.csv                                                            | ✅ Clean CSV           | AI adoption metrics, strip citation prefixes              |
| `PGRE/`                 | 10-Q/, 8-k-report-november-2025/, Schedule-14A/                                        | ✅ Clean CSV           | One Market Plaza debt: $416.5M @ 4.08%, matures 2026      |

### Critical PGRE Data Points (Feb 6, 2026 Deadline)

**Loan Structure & Servicer:**

- **Trust**: One Market Plaza Trust 2017-1MKT (single-borrower CMBS)
- **Master Servicer**: Wells Fargo Bank N.A.
- **Special Servicer**: Wells Fargo Bank N.A.
- **Loan transferred to special servicing**: Jan 8, 2024 (imminent monetary default)
- **Modification executed**: Feb 1, 2024

**One Market Plaza (OMP) - Key Risk Asset:**

- **PGRE Ownership**: 49% (consolidated)
- **PGRE Share of Debt**: $416.5M @ 4.08%, matures Feb 2026
- **Total Property Debt**: $850M (reduced from $975M via $125M paydown in Feb 2024 modification)
- **Square Feet**: 1,552,609
- **Occupancy**: 69.6% (critically low for Class A SF office)
- **Percent Leased**: 70.7%
- **Annualized Rent PSF**: $121.06

**🚨 S&P Global Ratings NCF Analysis (March 2024):**

S&P's underwriting methodology uses conservative, stress-tested assumptions:

| S&P Assumption   | Value      | Rationale                                              |
| ---------------- | ---------- | ------------------------------------------------------ |
| **Vacancy Rate** | 35.0%      | Aligned with CoStar submarket projections for 2025     |
| **Gross Rent**   | $105.90/SF | Mark-to-market adjustments (Visa: $98.47 → $70.73/SF)  |
| **OpEx Ratio**   | 39.5%      | Standard CMBS underwriting                             |
| **S&P NCF**      | **$60.5M** | 14.4% below their 2020 NCF estimate                    |
| **Cap Rate**     | 7.25%      | +50bps from 2020 for increased risk premium            |
| **S&P Value**    | $846.3M    | 17.9% below 2020 value; 52.1% below issuance appraisal |

**🚨 Debt Yield Covenant - VERIFIED (S&P Global, March 26, 2024):**

Per the Feb 2024 loan modification, the **8.5% debt yield threshold** applies to the **forbearance option after February 2027**, not the Feb 2026 maturity. The 2026-2027 extension is considered "safe" or "highly probable" because the $125M paydown in 2024 effectively "bought" this flexibility.

| Deadline        | Event                  | Conditions / Requirement                                                                              |
| --------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **Feb 6, 2026** | Extension to 2027      | **Standard Conditions**: No Event of Default, Purchase New Rate Cap, Pay Extension Fee (~0.25-0.50%). |
| **Feb 5, 2027** | Maturity / Forbearance | **Hard Trigger**: **≥8.5% debt yield required** for post-2027 forbearance.                            |

**Calculated Debt Yields:**

| Method                     | NCF    | Debt Yield | vs 8.5%      |
| -------------------------- | ------ | ---------- | ------------ |
| S&P Global (stress-tested) | $60.5M | **7.12%**  | ❌ **Below** |
| Servicer-reported (2023)   | $83.8M | 9.86%      | ✅ Above     |
| Servicer-adjusted (2023)   | $92.0M | 10.82%     | ✅ Above     |

**⚠️ Critical Risk Assessment:**

1. **S&P's 7.12% debt yield is below the 8.5% forbearance threshold** — if market conditions don't improve significantly, the post-2027 forbearance option is unavailable.

2. **Servicer-reported NCF ($83.8-92M) vs S&P NCF ($60.5M)** — The 40% gap reflects S&P's forward-looking stress assumptions for tenant rollover and submarket deterioration.

3. **Concentrated Tenant Rollover Risk** (per S&P):

   - 2025: 24.4% of NRA (includes Google - April 2025 expiry)
   - 2026: 35.4% of NRA (Morgan Lewis Feb, Autodesk June, Visa Sept)
   - **61.8% of NRA expires 2024-2026**

4. **Visa Already Vacated** — 10.1% of NRA is dark; tenant relocated to Mission Bay HQ ahead of Sept 2026 lease expiry.

**Wells Fargo NCF Calculation Methodology:**

As Master and Special Servicer, Wells Fargo Bank N.A. uses CREFC-standard CMBS underwriting for debt yield covenant tests. The key distinction:

| Metric  | Formula                               | Used For             |
| ------- | ------------------------------------- | -------------------- |
| **NOI** | Gross Revenue - Operating Expenses    | Simple cash flow     |
| **NCF** | NOI - TI/LC Reserves - CapEx Reserves | **Covenant testing** |

**Standard CMBS Debt Yield Formula:**
$$\text{Debt Yield} = \frac{\text{NCF}}{\text{Loan Balance}} \times 100\%$$

Per industry standards (Wall Street Prep, CREFC), CMBS lenders typically require **8-12% minimum debt yield**. Wells Fargo's 8.5% threshold for OMP forbearance is at the lower end of this range.

**Why S&P's NCF is Lower:**

S&P applies forward-looking stress assumptions, while servicers report trailing actuals:

| Factor         | Servicer (2023) | S&P (Stressed)                 |
| -------------- | --------------- | ------------------------------ |
| Vacancy        | 4.2%            | 35.0%                          |
| Gross Rent     | $105.90/SF      | $105.90/SF (but Visa @ $70.73) |
| TI/LC Reserves | Actual incurred | Elevated for rollover          |
| NCF Result     | $83.8-92M       | **$60.5M**                     |
| Debt Yield     | 9.86-10.82%     | **7.12%**                      |

**Conclusion:** The 8.5% covenant will be tested against **actual trailing NCF** at decision time (Feb 2027), not S&P's stress projection. However, given the tenant rollover since March 2024 (Google April 2025, massive sublease inventory), actual NCF by Feb 2027 will likely deteriorate toward S&P's projection.

**Extension Feasibility Analysis (Feb 2026):**

- **Status**: **Likely to Extend**.
- **Liquidity**: PGRE holds **$330M in Cash** (Sep 30, 2025), sufficient to cover the Extension Fee (~$2.1M) and new Rate Cap.
- **Debt Service**: Even under S&P's stressed NCF ($60.5M), the DSC is **1.74x** ($60.5M / $34.7M interest), well above the 1.0x default threshold.
- **Simulation Logic**: Treat Feb 2026 as a **Liquidity Shock** (deduct cap costs + fees) rather than a Default Event. The real default risk is the **Feb 2027 Debt Yield Test**.

**Risk Amplifiers:**

- SF submarket vacancy projected: 30.8% (2024) → 35.3% (2025) → 36.7% (2026)
- SF asking rents declining: $60.86/SF → $55.54 → $49.60 → $48.73
- 17.5% of NRA marketed for sublease (including Visa's entire space)
- S&P LTV ratio: **100.4%** (loan exceeds S&P's stressed property value)

**Contagion Trigger Conditions:**

1. If OMP fails to refinance/extend → CMBS default event
2. PGRE's SF Cash NOI (YTD 9mo): $77.8M vs total portfolio $239.9M (32% concentration)
3. Management projects 2026E Cash NOI drops to $234M from $301M 2025E (**-22% YoY**)
4. Core FFO projected: 2025E $126M → 2026E $94M (**-25% YoY**)

**PGRE Financial Stress Indicators:**

- Net loss per share: -$0.27 (9mo 2025) vs -$0.04 (9mo 2024) — **575% deterioration**
- Core FFO per share: $0.61 (9mo 2025) vs $0.48 (9mo 2024)
- Total debt: $3.38B, weighted avg rate 4.37%, weighted avg maturity 3.1 years
- Near-term maturities: 60 Wall St (2025), 111 Sutter (2025), OMP (2026), 55 Second St (2026)

### PGRE Granular Data Analysis (Source: Q3 2025 10-Q & 8-K)

**1. The "Death Spiral" Mechanics (San Francisco Portfolio):**

- **Negative Mark-to-Market**: SF leases are renewing at **-11.4%** below expiring rents (Cash basis).
- **Astronomical Concessions**:
  - **TI/LC Costs**: **$173.36 PSF** (San Francisco avg). This is exceptionally high, indicating "buying tenants" to maintain occupancy.
  - **Free Rent**: **12.5 months** average concession.
- **Cash Burn**:
  - **Unlevered Free Cash Flow (2025E)**: **$(34) Million** (Negative).
  - **Unlevered Free Cash Flow (2026E)**: **$(44) Million** (Negative).
  - _Insight_: Positive NOI is being completely consumed by the massive TI/LC capital requirements to address the 2025-2026 rollover cliff.

**2. Lease Expiration Cliff (The Trigger Zone):**

| Period         | Sq Ft Expiring | Rent PSF (Expiring) | Notes                               |
| :------------- | :------------- | :------------------ | :---------------------------------- |
| **Q4 2025**    | 539,543        | $94.90              | Major rollover event                |
| **Q1 2026**    | 513,104        | $86.04              | Continued pressure                  |
| **Q3 2026**    | 184,938        | **$110.70**         | Likely Visa expiry (High rent loss) |
| **Total 2026** | **1,180,364**  | **$88.09**          | ~1.2M sq ft repricing at -11%       |

**3. Portfolio Concentration Risk:**

- **San Francisco Dependency**: SF contributes **~32%** of PGRE's total YTD Cash NOI ($77.8M / $239.9M).
- **One Market Plaza**: 69.6% Occupied (vs 93.4% for 1633 Broadway in NY). The asset is significantly underperforming the rest of the portfolio.
- **111 Sutter Street**: **43.8% Leased**. This asset is effectively distressed/zombie status.

---

## Mathematical Framework: Behavioral Factors

### 1. Cognitive Bias Factor (Complacency Index) — $B_c$

Models "sleep at the wheel" behavior during low-volatility regimes. Based on **Prospect Theory** (Kahneman & Tversky) and **Minsky's Financial Instability Hypothesis**.

$$B_c(t) = \alpha_1 \cdot \underbrace{\left(\frac{\text{VIX}_{\text{9m}} - \text{VIX}_{\text{1m}}}{\text{VIX}_{\text{1m}}}\right)^{-1}}_{\text{Inverse VIX Term Structure}} + \alpha_2 \cdot \underbrace{\left(\frac{P/C_{\text{equity}}}{P/C_{\text{5yr avg}}}\right)}_{\text{Put/Call Divergence}} + \alpha_3 \cdot S_{\text{retail}}(t)$$

Where:

- $\text{VIX}_{\text{9m}}, \text{VIX}_{\text{1m}}$ = VIX futures at 9-month and 1-month tenors
- $P/C$ = Put/Call ratio (low = complacency)
- $S_{\text{retail}}(t)$ = Normalized retail sentiment (AAII survey, social media)
- $\alpha_1, \alpha_2, \alpha_3$ = Weights (sum to 1)

**Effect on Simulation**:

- High $B_c$ → Reduces `panic_sensitivity` by factor $(1 - 0.5 \cdot B_c)$
- When shock hits: `jump_mean` severity multiplied by $(1 + B_c)$ (complacent markets overreact)

### 2. Panic Sell Factor — $P_s$

Models herd behavior and flight-to-safety dynamics. Based on **Cont & Bouchaud herding models** and empirical fund flow analysis.

$$P_s(t) = \beta_1 \cdot \underbrace{\max\left(0, \frac{\Delta\text{VIX}_t}{\sigma_{\text{VIX}}}\right)}_{\text{Normalized VIX Spike}} + \beta_2 \cdot \underbrace{\left|\frac{F_{\text{outflow}}(t)}{F_{\text{AUM}}}\right|}_{\text{Fund Outflow Intensity}} + \beta_3 \cdot \underbrace{\frac{d S_{\text{fear}}}{dt}}_{\text{Sentiment Velocity}}$$

Where:

- $\Delta\text{VIX}_t$ = Daily VIX change
- $\sigma_{\text{VIX}}$ = Rolling 30-day VIX standard deviation
- $F_{\text{outflow}}$ = Net fund outflows (ICI data)
- $S_{\text{fear}}$ = Fear component of sentiment indices

**Effect on Simulation**:

- `panic_sensitivity` = `base_panic_sensitivity` $\times (1 + P_s)$
- `volatility_multiplier` = `base_volatility_multiplier` $\times (1 + 0.5 \cdot P_s)$

### 3. Automated Trading/Selling Factor — $A_t$

Models algorithmic amplification of market moves. Based on **Kyle's Lambda** (market impact) and order flow toxicity (VPIN).

$$A_t(t) = \gamma_1 \cdot \underbrace{\frac{|\text{OIB}_t|}{\text{Volume}_t}}_{\text{Order Imbalance}} + \gamma_2 \cdot \underbrace{\text{VPIN}_t}_{\text{Volume-Sync. PIN}} + \gamma_3 \cdot \underbrace{\frac{V_{\text{dark}}}{V_{\text{total}}}}_{\text{Dark Pool Ratio}}$$

Where:

- $\text{OIB}$ = Order Imbalance (buyer-initiated minus seller-initiated volume)
- $\text{VPIN}$ = Volume-Synchronized Probability of Informed Trading (Easley et al.)
- $V_{\text{dark}}/V_{\text{total}}$ = Dark pool volume fraction

**Effect on Simulation**:

- `cascade_decay` = `base_cascade_decay` $\times (1 - 0.3 \cdot A_t)$ (faster propagation)
- `selloff_intensity` = `base_selloff_intensity` $\times (1 + A_t)$

### Historical Calibration Events

Use these events to backtest and calibrate behavioral factor weights:

| Event                       | Date         | $B_c$ (Pre) | $P_s$ (Peak) | $A_t$ (Peak) | Market Drop    |
| --------------------------- | ------------ | ----------- | ------------ | ------------ | -------------- |
| **2008 GFC**                | Sep-Oct 2008 | High        | Extreme      | Moderate     | -40% (S&P)     |
| **2010 Flash Crash**        | May 6, 2010  | Moderate    | High         | Extreme      | -9% (intraday) |
| **2015 China Deval**        | Aug 24, 2015 | High        | High         | High         | -11% (3 days)  |
| **2018 Volmageddon**        | Feb 5, 2018  | Extreme     | High         | High         | -10% (2 days)  |
| **2020 COVID Crash**        | Mar 2020     | Moderate    | Extreme      | High         | -34% (23 days) |
| **2022 UK Gilt Crisis**     | Sep-Oct 2022 | High        | High         | Extreme      | LDI cascade    |
| **2023 SVB/Regional Banks** | Mar 2023     | High        | High         | Moderate     | -25% (KRE)     |

### Socioeconomic Thesis: The "Hollow Middle" (Source: How Money Works Uncut)

**Core Thesis**: The middle class has been dismantled by a shift from labor-based to asset-based economics (The "1971 Moment").

| Concept             | Simulation Parameter    | Mechanism                                                                                |
| :------------------ | :---------------------- | :--------------------------------------------------------------------------------------- |
| **Decoupling**      | `wage_productivity_gap` | Productivity grows, wages flatline. Surplus goes to `MarketModel` (Corporate Profits).   |
| **Bifurcation**     | `inflation_bifurcation` | Essentials (Housing/Ed) inflate faster than CPI. Squeezes `disposable_income`.           |
| **Debt Band-Aid**   | `consumer_leverage`     | Gap between wages and CoL is filled by debt. Increases `systemic_fragility`.             |
| **Two-Income Trap** | `income_fragility`      | Higher fixed costs (childcare/housing) mean small income shocks cause immediate default. |

**Modeling Implication**:

- We need a **Consumer Model** that tracks the "Real Economy" vs the "Asset Economy".
- **Feedback Loop**: High Asset Prices (Good for Market) -> High Housing Costs (Bad for Consumer) -> Increased Debt -> Eventual Consumption Collapse -> Market Crash.

### The "Everything Bubble" Thesis (Source: How Money Works Uncut)

**Core Concept**: A fundamental disconnect between the "Asset Economy" (inflated by liquidity/passive flows) and the "Real Economy" (Consumer).

| Concept                               | Simulation Component | Mechanism                                                                                                                        |
| :------------------------------------ | :------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| **Passive Investing ("Blind Trust")** | `MarketModel`        | Automatic inflows drive prices regardless of fundamentals. Creates a "floor" until it becomes a "trap door."                     |
| **Corporate Buybacks**                | `MarketModel`        | "Burning trillions" to boost EPS, masking weak consumer demand.                                                                  |
| **The Consumer Cliff**                | `ShockwaveTrigger`   | The bubble bursts when the "Hollow Middle" collapses. High `consumer_default_prob` triggers `ShockType.CONSUMER_CREDIT_FAILURE`. |
| **Rationality Return**                | `PanicModel`         | The moment CoL > Debt Capacity. Triggers a rapid repricing of assets to fundamental reality.                                     |

### Socioeconomic Thesis: The "Founder Mode" Collapse (Source: How Money Works Uncut)

**Core Thesis**: The "Genius Founder" archetype is a byproduct of ZIRP (Zero Interest Rate Policy). As rates rise, the lack of accountability and business acumen in "Founder Mode" leadership leads to value destruction.

| Concept                | Simulation Parameter           | Mechanism                                                                                                                         |
| :--------------------- | :----------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| **Founder Mode**       | `founder_mode_intensity`       | High intensity increases `volatility` (impulsive decisions) and `jump_intensity` (scandal risk).                                  |
| **ZIRP Dependency**    | `zirp_dependency`              | Measures how much a company/market relies on cheap capital. High dependency + Low Liquidity = Rapid `narrative_premium` collapse. |
| **Accountability Gap** | `moral_hazard_factor`          | Low accountability encourages risk. Increases `volatility` but delays mean reversion (problems fester longer).                    |
| **The Reality Check**  | `ShockType.MANAGEMENT_FAILURE` | Triggered when `Liquidity` drops below threshold for high `zirp_dependency` assets.                                               |

**Modeling Implication**:

- **Regime Shift**: The transition from High Liquidity (ZIRP) to Low Liquidity exposes "naked" swimmers (Buffett's adage).
- **Management Risk**: We must model the "Idiosyncratic Risk" of leadership failure, which correlates with `narrative_premium`. High narrative stocks often have high `founder_mode_intensity`.

### Socioeconomic Thesis: The "Exit Strategy" Economy (Source: How Money Works Uncut)

**Core Thesis**: The economy is designed for wealth extraction ("buck-passing") rather than value creation. Elites insulate themselves from consequences, leaving the bill for future generations.

| Concept                | Simulation Parameter         | Mechanism                                                                                                                                           |
| :--------------------- | :--------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PE Shell Game**      | `pe_dominance`               | Private Equity extraction boosts short-term `drift` (fees/stripping) but increases long-term `default_prob` and `jump_intensity` (bankruptcy risk). |
| **Insulation**         | `insulation_factor`          | High insulation reduces `panic_sensitivity` for "Smart Money" (they exit early) but increases `selloff_intensity` when the "Exit Event" triggers.   |
| **Financial Nihilism** | `financial_nihilism`         | High nihilism in consumers increases `risk_appetite` (speculation) and `debt_accumulation` (BNPL usage), delaying the consumption collapse.         |
| **The Exit**           | `ShockType.SMART_MONEY_EXIT` | A discrete event where `insulation_factor` is high. Smart money dumps assets while retail is still buying (`narrative_premium` is high).            |

**Modeling Implication**:

- **Divergence**: We must model the split between "Smart Money" (Insulated) and "Retail/Public" (Exposed).
- **Time Horizon**: PE strategies work in the short term (0-5 years) but fail in the long term. The simulation must track "Asset Health" separate from "Market Price".

### Economic Forensic Analysis: Emergent Fragility & Thermodynamic Wealth

**Core Thesis**: The economy is a complex adaptive system undergoing a "phase transition" (Ising Model). Wealth distribution follows thermodynamic laws (Boltzmann-Gibbs vs Pareto), reinforced by cognitive scarcity ("Bandwidth Tax").

| Concept                     | Simulation Parameter             | Mechanism                                                                                                                                       |
| :-------------------------- | :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Thermodynamic Wealth**    | `wealth_distribution_pareto`     | Bottom 90% follow Boltzmann-Gibbs (random exchange); Top 10% follow Pareto (compounding). Modeled via `saving_propensity` ($\lambda$).          |
| **Cognitive Bandwidth Tax** | `scarcity_tunneling`             | Low wealth reduces cognitive bandwidth (IQ drop). Forces $\lambda \to 0$ (Scarcity Trap). High wealth allows $\lambda \to 1$ (Abundance Cycle). |
| **Ising Phase Transition**  | `market_coupling_strength` ($J$) | Measures "herding" behavior. High $J$ + Low Volatility ("Temperature") = Spontaneous Symmetry Breaking (Crash).                                 |
| **Maturity Wall**           | `debt_maturity_spike`            | Discrete spikes in default risk in 2025 ($957B), 2026 ($539B), 2027 ($550B).                                                                    |
| **Risk Aversion (DRRA)**    | `risk_aversion_curve`            | Wealthy take _more_ risk in crises (buy dip); Poor take _negative-EV_ risk (lotteries) to escape scarcity.                                      |

**Modeling Implication**:

- **Wealth Distribution**: Split `ConsumerModel` into "Thermal Agents" (Low $\lambda$) and "Pareto Agents" (High $\lambda$).
- **Market Dynamics**: Implement Ising Model logic in `MarketModel`. Crash probability $P(\text{Crash}) \propto \exp(J/T)$.
- **Debt Cliff**: Hard-code maturity spikes in `BondModel` or `ShockwaveTrigger`.

---

## Additional Data Recommendations

### Missing Data (High Priority)

1. **VIX Term Structure** — Required for $B_c$ calculation. Source: CBOE VIX futures.
2. **Fund Flow Data** — Required for $P_s$. Source: ICI weekly data or EPFR.
3. **VPIN/Order Flow Data** — Required for $A_t$. Source: NYSE TAQ or commercial vendors.
4. **AAII Sentiment Survey** — Retail sentiment for $B_c$.

### Missing Data (Medium Priority)

5. **Credit Default Swap Spreads** — CMBS/IG CDS for real-time credit stress.
6. **Fed Funds Futures** — Rate expectations for drift calibration.
7. **MOVE Index** — Bond market volatility (complement to VIX).

### Data Reformatting Tasks

```python
# Priority reformatting tasks for data ingestion
REFORMAT_TASKS = {
    "DTCC": "Strip currency symbols: df.replace('[$,]', '', regex=True)",
    "TIC": "Parse HTML tables with pandas.read_html()",
    "Gold": "pd.to_datetime(date_col, format='%d-%b-%Y')",
    "iShares": "pd.read_excel() then export to CSV",
    "TREPP": "df.columns = df.columns.str.replace(r'\[cite.*?\]', '', regex=True)",
    "MIT": "Same as TREPP - strip citation markers",
}
```
