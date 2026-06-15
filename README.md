# Data Analysis: CO₂ Emissions (OWID)

A small data-analysis project that takes a real-world CO₂ emissions dataset and walks
through the **four main types of data analysis** — descriptive, diagnostic, predictive,
and prescriptive — in a single notebook.

The point is less "produce a novel climate finding" and more *show what each kind of
analysis actually does on a real dataset, end to end.*

## The dataset

`owid-co2-data.csv` from [Our World in Data](https://github.com/owid/co2-data) (CC-BY 4.0).

- ~50,000 rows, 79 columns
- 254 countries and regions
- years 1750 through 2024
- columns cover annual CO₂ broken down by fuel source, per-capita and per-GDP emissions,
  cumulative totals, population, GDP, and estimated warming contribution

It's committed in `data/`, so the notebook runs out of the box — no download step.

## What the notebook does

`co2_analysis.ipynb` is split into four sections, one per analysis type.

### 1. Descriptive — *what happened?*
Summarise the history. Global emissions went from ~5,900 Mt in 1950 to ~38,600 Mt in 2024,
a 6.5× increase. China, the US, and India lead on total emissions; Qatar, Kuwait, and
Bahrain lead per person — the two rankings barely overlap.

### 2. Diagnostic — *why did it happen?*
Look for drivers. CO₂ tracks GDP almost one-to-one across countries (log–log correlation
≈ 0.95, elasticity ≈ 1.0). Coal is still the single biggest source, with gas and cement
growing fastest since 1990.

### 3. Predictive — *what's next?*
Fit a plain linear regression on 2000–2024 (R² ≈ 0.92) and project forward. If current
trends hold, global emissions land around 45 Gt by 2035. This is a transparent baseline,
not a real climate model — see the caveats below.

### 4. Prescriptive — *what should we do?*
Turn the forecast into a decision number. Holding emissions flat at 2024 is one thing;
halving them by 2035 would need roughly a **6% cut every single year** through 2035.

## Files

| File | What it is |
|------|------------|
| `co2_analysis.ipynb` | the main notebook, already executed with charts |
| `co2_analysis.py` | the same analysis as a plain runnable script |
| `requirements.txt` | pinned dependencies |
| `data/owid-co2-data.csv` | the dataset |

## Running it

```
pip install -r requirements.txt
```

Open the notebook:

```
jupyter lab co2_analysis.ipynb
```

…or just run the script (prints the same numbers, saves the plots):

```
python co2_analysis.py
```

Tested on Python 3.12.

## Caveats

- **The forecast is intentionally simple** — a straight line. It exists to demonstrate the
  workflow (fit, score, project, show uncertainty), not to predict real climate outcomes.
  Actual emissions modelling uses integrated assessment models far beyond a one-liner
  regression.
- **Year choices are pragmatic.** The "top emitters" tables use 2023; the cross-country
  GDP plots use 2022, the latest year with GDP coverage for ~150 countries.
- **OWID mixes countries with aggregates.** The dataset includes regional roll-ups like
  `World`, `Asia`, `EU-27`, and income groups alongside real countries. The notebook keeps
  these separate — global totals use the `World` row; country-level work filters to rows
  that carry an ISO-3 code.

## License

Code in this repo is MIT. The dataset (`data/owid-co2-data.csv`) is © Our World in Data
and licensed CC-BY 4.0 — see their [source](https://github.com/owid/co2-data).
