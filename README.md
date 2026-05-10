# Yuan-Agent (Event-Driven ROA Forecasting System)
An AI financial analyst agent that analyzes financial reports and news to predict companies’ return on assets (ROA).

This repository contains a small, auditable research prototype for event-driven financial forecasting.
The current target is **ROA**, computed from:

```text
ROA = profit_margin * asset_turnover
```

News/event rules are only allowed to update `profit_margin` and `asset_turnover`.
`equity_multiplier` is still retained in fundamental analysis for historical ROE context, but it is not a rule target and is not used by the ROA prediction executor.

## Project Structure

```text
data/
  event_library/      Controlled event taxonomy
  fundamentals/       Offline baseline state
  news/               Local news Excel files
  reports/            Local Sungrow report PDFs
  rules/              Active ROA rule library
  supervision/        Export supervision data
scripts/
  run_fundamental_analyst_reports.py  Parse report PDFs into structured fundamentals
  run_quarterly_eda_only.py           Extract quarterly events from news
  run_quarterly_2021_2023_training.py Train rules on 2021-2023 in three reasoning passes
  run_quarterly_2024_test.py          Run the 2024 ROA backtest
src/forecasting_system/
  agents/             EDA, Fundamental Analyst, Student, Teacher, Reasoning interfaces
  schemas/            JSON schemas for records and rules
  tools/              Deterministic tools and DeepSeek helpers
tests/                Unit tests
logs/                 Generated outputs, ignored by Git
```

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For LLM-backed scripts, create a local `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then fill in `DEEPSEEK_API_KEY`. Unit tests do not require an API key.

## Run Tests

```powershell
pytest -q
```

Expected result:

```text
21 passed
```

## Train Rules And Run The 2024 Backtest

The training and backtest scripts use generated intermediate files under `logs/`.
Run the pipeline in this order:

```powershell
python scripts\run_fundamental_analyst_reports.py
python scripts\run_quarterly_eda_only.py
python scripts\run_quarterly_2021_2023_training.py
python scripts\run_quarterly_2024_test.py logs\quarterly_2021_2023_training\active_rules_final.json
```

The scripts print stage timing. DeepSeek calls also print elapsed time and token usage when the API returns usage data.

Training outputs are written to:

```text
logs/quarterly_2021_2023_training/
```

The trained rules from the third pass are written to:

```text
logs/quarterly_2021_2023_training/active_rules_final.json
```

The 2024 backtest outputs are written to:

```text
logs/quarterly_2024_test_final_rules/
```

By default, the 2024 backtest reads `data/rules/rules.json`. You can pass a trained rule file as the first argument, as shown above, to test the freshly trained rules without overwriting the base rules.

The generated `logs/` contents are ignored by Git so experiment outputs do not clutter the repository.

## Agent Roles

- **EDA Agent** extracts structured events from news.
- **Fundamental Analyst** reads disclosed reports and computes factual indicators, including `reported_roa`.
- **Student Agent** applies matched rules and computes ROA deterministically.
- **Teacher Agent** compares predicted ROA and components against supervision.
- **Reasoning Agent** may suggest bounded rule updates, but it cannot add/delete rules or change rule targets.

## Current Modeling Constraints

- Forecast target: `roa`
- Adjustable rule targets: `profit_margin`, `asset_turnover`
- Rule function: `linear_adjustment`
- Rules cannot target `equity_multiplier`
- Fundamental analysis may report ROE and `equity_multiplier`, but those are kept separate from ROA prediction.

## GitHub Notes

Do not commit `.env`, `.venv`, `__pycache__`, `.pytest_cache`, or generated `logs/` outputs.
The `.gitignore` in this repository already excludes them.
