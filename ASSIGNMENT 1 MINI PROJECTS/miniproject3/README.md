# Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model

## Contents
- `miniproject3_fish_stock_export_risk.ipynb`: full analysis notebook
- `src/fish_model.py`: reusable OOP model classes and functions
- `tests/test_fish_model.py`: pytest unit tests
- `requirements.txt`: dependencies

## Setup and run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Open `miniproject3_fish_stock_export_risk.ipynb` and choose **Kernel → Restart & Run All**.

Run unit tests from the repository root:
```bash
python -m pytest -q
```

## Model assumptions and findings
The notebook uses the assignment parameters `r=0.4`, `K=10,000` tonnes, `N(0)=4,000` tonnes and weekly proportional harvest rates `h=0.05, 0.10, 0.20, 0.30`. Price follows a reproducible bounded random walk in UGX/kg. Revenue uses harvested tonnes converted to kilograms. The notebook compares stock and revenue scenarios, estimates the 5% lower-tail annual-revenue quantile over 1,000 price paths, and tests an 8-week closed season over five years.

The classic `MSY=rK/4` formula assumes a constant-yield harvest from the logistic model. This project instead uses harvest `h*N`, so MSY is a reference rather than a directly equivalent annual target. All parameters and simulated prices are illustrative and should not be treated as a real stock assessment.

## AI-use declaration
AI assistance was used to draft and structure code, tests, explanations, and notebook cells.