# Solar Micro-Grid Dispatch Planner

## Contents
- `project2_2_solar_microgrid.ipynb`: executable notebook
- `src/project.py`: reusable, typed OOP logic
- `tests/test_project.py`: unit tests
- `requirements.txt`: dependencies

## Setup and run
Use Python 3.10 or newer. From this folder run:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```
Open the notebook and choose **Kernel → Restart Kernel and Run All Cells**. Run tests with `pytest -q`.

## Data and reproducibility
Data described as illustrative/synthetic are not official measurements. Random simulations use fixed seeds. Generated plots and tables are produced by notebook execution.

## Findings and limitations
See the notebook's final interpretation section for assumptions, findings and limitations. Results are teaching examples, not operational advice.

## AI-use declaration
An AI coding assistant was used to draft and organise parts of the code, tests, and explanatory text.
