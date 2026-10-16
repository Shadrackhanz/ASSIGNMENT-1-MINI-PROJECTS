# Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser

## Files
- `miniproject4_rainfall_crop_suitability.ipynb`: full analysis notebook
- `src/rainfall_model.py`: reusable `Region`, `CropRule`, and cosine similarity logic
- `tests/test_rainfall_model.py`: pytest unit tests
- `requirements.txt`: dependencies

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Open the notebook and select **Kernel → Restart & Run All**.

Run tests from the project root:
```bash
python -m pytest -q
```

## Data and interpretation
The rainfall figures for Kampala, Gulu, and Mbarara are the illustrative monthly values provided in the assignment. Crop thresholds are provisional demonstration rules and must be replaced or justified with a cited authoritative agronomy source before using the results as actual crop advice. The multi-year boxplot extension uses synthetic values solely to demonstrate the workflow; it is not observed rainfall data.

## AI-use declaration
AI assistance was used to draft and structure code, tests, explanations, and notebook cells. The student should review, run, understand, and adapt the work and be able to explain the calculations and assumptions independently.
