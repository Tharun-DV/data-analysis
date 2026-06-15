"""Data-Science Analysis: Types + Real-World CO2 Case Study (runnable mirror of co2_analysis.ipynb)."""

# # Data-Science Analysis: The 5 Types + a Real-World CO₂ Case Study
#
# This notebook does two things:
#
# 1. **Documents the main *types* of data-science analysis** — descriptive, diagnostic, predictive, prescriptive, and cognitive.
# 2. **Demonstrates all of them end-to-end on a real-world dataset** — global CO₂ emissions — with every step explained.
#
# > **Dataset:** *Our World in Data — CO₂ and Greenhouse Gas Emissions* (`owid-co2-data.csv`).
# > 50,411 rows × 79 columns, covering 254 countries/regions, **1750 → 2024**.
# > Source: https://github.com/owid/co2-data (CC-BY 4.0).

# ## How many types of data-science analysis are there?
#
# There is no single official count, but the **analytics maturity model** — the
# framework behind terms like *"descriptive"* and *"predictive"* — defines **5 core
# types**, in order of increasing sophistication and business value:
#
# | # | Type | Question it answers | Typical techniques |
# |---|------|---------------------|--------------------|
# | 1 | **Descriptive** | *What happened?* | Summaries, KPIs, dashboards, counts, means, trends |
# | 2 | **Diagnostic** | *Why did it happen?* | Drill-downs, correlation, segmentation, root-cause |
# | 3 | **Predictive** | *What will happen?* | Regression, time-series forecasting, ML classification |
# | 4 | **Prescriptive** | *What should we do?* | Optimisation, scenario / what-if analysis, recommen­dations |
# | 5 | **Cognitive** | *How can the system decide autonomously?* | AI / deep learning, NLP, autonomous agents |
#
# A second, **statistics-oriented** taxonomy (Leek & Peng) splits the field into six:
# **descriptive, exploratory, inferential, predictive, causal, mechanistic** — the first
# four overlap with the table above, while *causal* asks "what is the effect of X on Y?"
# and *mechanistic* derives exact equations from theory (rare outside physics/biology).
#
# **In short: 4 canonical types (descriptive → diagnostic → predictive → prescriptive),
# with *cognitive* as the emerging 5th.** This notebook walks through the canonical four
# on a single, real dataset.

# ## The real-world use case
#
# **Climate question:** *"How fast are global CO₂ emissions growing, what is driving
# them, where are they heading, and what would it take to change course?"*
#
# This is a textbook analytics problem because it maps cleanly onto the four types:
#
# - **Descriptive** — chart global emissions over 1900–2024; rank top emitters.
# - **Diagnostic** — is economic activity (GDP) the driver? how do fuel sources split?
# - **Predictive** — fit a model and forecast emissions to 2035.
# - **Prescriptive** — model what-if trajectories: business-as-usual vs. flat vs. halve-by-2035.

# ### Step 0 — Environment setup
#
# Standard data-science stack: **pandas / numpy** for data wrangling,
# **matplotlib / seaborn** for visualisation, **scipy** for statistics,
# and **scikit-learn** for the predictive model.

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted",
              rc={"figure.figsize": (10, 5), "figure.dpi": 110})
plt.rcParams["axes.titleweight"] = "bold"
pd.set_option("display.width", 120)
print("Libraries ready.")

# ### Step 1 — Load the data
#
# The CSV was downloaded from the OWID GitHub release and lives at `data/owid-co2-data.csv`.

RAW = "data/owid-co2-data.csv"
df = pd.read_csv(RAW)
print("Raw shape :", df.shape)
print("Year span :", df.year.min(), "-", df.year.max())
print("Entities  :", df.country.nunique(), "(countries + regional aggregates)")
df.head(3)

# ### Step 2 — Understand & clean
#
# OWID mixes **real countries** with **aggregates** (`World`, `Asia`, `EU-27`, income
# groups…). We keep the aggregates aside for global totals and isolate real countries
# using the `iso_code` column (only countries have one). We also restrict the
# country-level frame to the **modern era (1950+)** and a focused column set.
#
# Key null findings from the raw check: `co2` is ~42% null overall (mostly pre-industrial
# and tiny territories) and `gdp` ~70% null (GDP data is sparse before ~1990 and for many
# small nations) — both matter for the diagnostic step below.

# Global aggregate (one row per year) - keep the 'World' rows for global totals
world = df[df["country"] == "World"].copy()

# Real countries only (those carrying an ISO-3 code), modern era, focused columns
key_cols = ["country", "year", "iso_code", "population", "gdp", "co2",
            "co2_per_capita", "coal_co2", "oil_co2", "gas_co2", "cement_co2",
            "flaring_co2", "cumulative_co2", "temperature_change_from_co2", "co2_per_gdp"]
countries = (df[df["iso_code"].notna()]
             .loc[df.year >= 1950, key_cols]
             .copy())

print("Country-level rows (1950+):", countries.shape)
print(f"Null share of co2 in this frame: {countries['co2'].isna().mean()*100:.1f}%")
countries.describe().round(1)

# ---
# ## 🔵 1. Descriptive analysis — *What happened?*
#
# Goal: summarise the raw history so the shape of the problem is obvious. No modelling yet,
# just *"what does the record show?"*

# #### 1a. Global emissions trajectory, 1900–2024

w = world[world["year"] >= 1900].sort_values("year")
fig, ax = plt.subplots()
ax.plot(w["year"], w["co2"], color="#d62728", lw=2.2)
ax.fill_between(w["year"], 0, w["co2"], color="#d62728", alpha=0.12)
ax.set_title("Global CO$_2$ Emissions, 1900-2024")
ax.set_xlabel("Year"); ax.set_ylabel("CO$_2$ (million tonnes)")

peak_yr = int(w.loc[w["co2"].idxmax(), "year"]); peak_v = w["co2"].max()
ax.annotate(f"Peak {peak_v:,.0f} Mt ({peak_yr})",
            xy=(peak_yr, peak_v), xytext=(peak_yr - 45, peak_v - 3500),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
plt.tight_layout(); plt.show()

v24 = float(w[w.year == 2024]["co2"].iloc[0])
v50 = float(w[w.year == 1950]["co2"].iloc[0])
print(f"Global CO2  2024 = {v24:,.0f} Mt   |   1950 = {v50:,.0f} Mt   |   {v24/v50:.1f}x increase")

# #### 1b. Who emits the most — absolute vs. per-person (2023)
#
# Absolute emissions expose the biggest national polluters; per-capita reveals the
# most emission-intensive lifestyles. They tell very different stories.

latest = 2023  # last year with broad country-level coverage
top_em = countries[countries.year == latest].dropna(subset=["co2"]).nlargest(10, "co2")
top_pc = (countries[countries.year == latest]
          .dropna(subset=["co2_per_capita"]))
top_pc = top_pc[top_pc.population > 1e6].nlargest(10, "co2_per_capita")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=top_em, y="country", x="co2", ax=axes[0],
            hue="country", legend=False, palette="Reds_r")
axes[0].set_title(f"Top 10 Absolute Emitters ({latest})")
axes[0].set_xlabel("CO$_2$ (Mt)"); axes[0].set_ylabel("")

sns.barplot(data=top_pc, y="country", x="co2_per_capita", ax=axes[1],
            hue="country", legend=False, palette="OrRd_r")
axes[1].set_title(f"Top 10 Per-Capita Emitters ({latest}, pop > 1M)")
axes[1].set_xlabel("tonnes CO$_2$ per person"); axes[1].set_ylabel("")
plt.tight_layout(); plt.show()

print("Absolute leaders :", list(top_em["country"][:3]))
print("Per-capita leaders:", list(top_pc["country"][:3]))

# ---
# ## 🟠 2. Diagnostic analysis — *Why did it happen?*
#
# Now we ask *why* emissions are distributed this way. Two angles:
#
# - **Driver check:** is economic output (GDP) the dominant correlate of emissions?
# - **Composition check:** which fuels (coal / oil / gas / cement) make up the global total, and how is that mix shifting?

# Use 2022 - the latest year with broad GDP coverage (153 countries)
latest_g = 2022
rec = (countries[(countries.year == latest_g) & countries["gdp"].notna()
                 & countries["co2"].notna() & (countries.population > 1e6)].copy())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=rec, x="gdp", y="co2", size="population", hue="population",
                sizes=(20, 400), alpha=0.7, ax=axes[0], palette="viridis", legend=False)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_title(f"CO$_2$ vs GDP ({latest_g}, log-log)")
axes[0].set_xlabel("GDP ($)"); axes[0].set_ylabel("CO$_2$ (Mt)")
for c in ["China", "United States", "India", "Russia", "Germany", "Japan", "Brazil"]:
    r = rec[rec.country == c]
    if len(r):
        axes[0].annotate(c, (r.gdp.iloc[0], r.co2.iloc[0]), fontsize=8)

wf = (world[world.year >= 1990]
      [["year", "coal_co2", "oil_co2", "gas_co2", "cement_co2", "flaring_co2"]]
      .set_index("year").sort_index())
wf.plot.area(ax=axes[1], cmap="tab10", alpha=0.85)
axes[1].set_title("Global CO$_2$ by Fuel Source (1990-)")
axes[1].set_ylabel("CO$_2$ (Mt)")
axes[1].legend(title="", ncol=2, fontsize=8)
plt.tight_layout(); plt.show()

# Correlation & elasticity (slope in log-log ≈ %CO2 per %GDP)
r_g, r_c, r_p = np.log10(rec["gdp"]), np.log10(rec["co2"]), np.log10(rec["population"])
elasticity = np.polyfit(r_g, r_c, 1)[0]
print(f"Countries analysed ({latest_g}): {len(rec)}")
print(f"log10 corr  GDP ~ CO2 : {r_g.corr(r_c):.3f}")
print(f"log10 corr  POP ~ CO2 : {r_p.corr(r_c):.3f}")
print(f"CO2-vs-GDP elasticity : {elasticity:.2f}  (≈ %{elasticity*100:.0f} CO2 per +1% GDP)")

# **What the diagnostics show:**
# - GDP and CO₂ move almost in lockstep across countries (r ≈ 0.95) — economic scale is the
#   dominant driver. The ~1.0 elasticity means a 1% larger economy maps to ~1% more CO₂.
# - Population matters less than GDP (r ≈ 0.67): richer, not just bigger, countries emit more.
# - On the composition side, **coal is still the single largest source** of global emissions,
#   with gas and cement growing fastest since 1990.

# ---
# ## 🟢 3. Predictive analysis — *What will happen?*
#
# We fit a simple **linear regression** of global CO₂ against year using 2000–2024
# (the era where absolute growth is near-linear) and project to 2035. The R² and a
# ±2σ band communicate how confident the projection is.
#
# *(This is intentionally a transparent baseline; production climate forecasts use far
# more sophisticated integrated-assessment models, but the workflow — fit, score,
# project, quantify uncertainty — is identical.)*

hist = world[(world.year >= 2000) & (world.year <= 2024)].dropna(subset=["co2"]).sort_values("year")
X = hist[["year"]].values
y = hist["co2"].values
model = LinearRegression().fit(X, y)
pred_in = model.predict(X)
r2 = r2_score(y, pred_in)
sigma = np.std(y - pred_in)

future_yrs = np.arange(2025, 2036).reshape(-1, 1)
forecast = model.predict(future_yrs)

fig, ax = plt.subplots()
ax.plot(hist.year, y, "o-", color="#d62728", label="Actual (2000-2024)")
ax.plot(future_yrs.ravel(), forecast, "s--", color="#ff7f0e", label="Linear forecast")
ax.fill_between(future_yrs.ravel(), forecast - 2*sigma, forecast + 2*sigma,
                color="#ff7f0e", alpha=0.15, label="+/- 2 sigma band")
ax.set_title("Global CO$_2$ Forecast to 2035 (linear fit)")
ax.set_xlabel("Year"); ax.set_ylabel("CO$_2$ (Mt)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

print(f"Model R^2      : {r2:.3f}")
print(f"Learned slope  : {model.coef_[0]:,.1f} Mt per year")
print(f"Projected 2030 : {model.predict([[2030]])[0]:,.0f} Mt")
print(f"Projected 2035 : {forecast[-1]:,.0f} Mt  ({(forecast[-1]/y[-1]-1)*100:+.0f}% vs 2024)")

# ---
# ## 🟣 4. Prescriptive analysis — *What should we do?*
#
# Forecasting tells us where we are *headed*; prescription asks what action would move us
# onto a different path. Here we turn the model into a **decision lever**: given a target
# for 2035, what constant annual reduction rate is required to reach it?
#
# Three scenarios:
# 1. **Business-as-usual** — linear growth continues.
# 2. **Flat** — hold emissions at the 2024 level through 2035.
# 3. **Halve by 2035** — a Paris-style deep-cut ambition.

base_2024 = float(world[world.year == 2024]["co2"].iloc[0])
n_years = 2035 - 2024
bau_2035 = float(model.predict([[2035]])[0])

def annual_rate(start, target, n):
    return 100 * ((target / start) ** (1 / n) - 1)

flat_rate = annual_rate(base_2024, base_2024, n_years)          # 0 by construction
halve_rate = annual_rate(base_2024, base_2024 / 2, n_years)

yrs = np.arange(2024, 2036)
bau_path = model.predict(yrs.reshape(-1, 1))
flat_path = base_2024 * np.ones_like(yrs, dtype=float)
half_path = base_2024 * (0.5) ** ((yrs - 2024) / n_years)

fig, ax = plt.subplots()
ax.plot(yrs, bau_path, "o-", label="Business as usual (linear)", color="#d62728")
ax.plot(yrs, flat_path, "s--", label=f"Flat at 2024 ({flat_rate:+.1f}%/yr)", color="#1f77b4")
ax.plot(yrs, half_path, "^--", label=f"Halve by 2035 ({halve_rate:+.2f}%/yr)", color="#2ca02c")
ax.axhline(base_2024, color="grey", lw=0.7, ls=":")
ax.set_title("Emission Trajectories: What It Takes")
ax.set_xlabel("Year"); ax.set_ylabel("Global CO$_2$ (Mt)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

print(f"2024 baseline                      : {base_2024:,.0f} Mt")
print(f"Business-as-usual 2035             : {bau_2035:,.0f} Mt  ({(bau_2035/base_2024-1)*100:+.0f}% vs 2024)")
print(f"Halve by 2035 requires             : {halve_rate:+.2f}%/yr every year to 2035")

# ---
# ## Key findings & recap
#
# **On the data:**
# - Global CO₂ rose from ~5,930 Mt (1950) to **~38,600 Mt (2024)** — a **6.5× increase**.
# - **China, the United States, India** lead on absolute emissions; **Qatar, Kuwait, Bahrain**
#   lead per capita — the two rankings barely overlap.
# - **GDP is the dominant driver** (log-log r ≈ 0.95, elasticity ≈ 1.0); population is weaker (r ≈ 0.67).
# - **Coal remains the #1 fuel source**, though gas and cement have grown fastest since 1990.
#
# **On the analysis types — how each contributed here:**
#
# | Type | What it produced | Tools used |
# |------|------------------|------------|
# | **Descriptive** | The 6.5× growth curve, top-emitter rankings | `pandas` summaries, line/bar plots |
# | **Diagnostic** | GDP↔CO₂ correlation, fuel-source decomposition | scatter/log-log, correlation, stacked area |
# | **Predictive** | A linear model (R² ≈ 0.92) projecting ~45 Gt by 2035 | `scikit-learn` `LinearRegression` |
# | **Prescriptive** | A decision lever: ~**-6.1 %/yr** cuts needed to halve by 2035 | scenario / what-if modelling |
#
# **Bottom line:** the four analytics types form a ladder — *describe → explain → forecast →
# act*. Each rung builds on the previous one's evidence, turning raw history into an
# actionable recommendation.
