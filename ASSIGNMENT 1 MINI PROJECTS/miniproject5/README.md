# Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner

## Files
- `miniproject5_taxi_route_revenue_fleet_planner.ipynb`: complete analysis.
- `src/taxi_model.py`: reusable classes and equilibrium solver.
- `tests/test_taxi_model.py`: pytest tests.
- `requirements.txt`: dependencies.

## Run
Install dependencies with `pip install -r requirements.txt`, then open the notebook with Jupyter and choose **Restart & Run All**. Run tests using `python -m pytest -q`.

## Assumptions and limitations
Passenger counts are total daily passenger journeys for each route. Gross revenue equals journeys times fare; costs and profit are not modelled. Vehicle capacity is 14 passengers times 8 one-way trips = 112 passenger journeys per vehicle-day, and fleet size includes a 15% buffer. Ntinda counts are supplied in the assignment; Entebbe and Mukono data are illustrative. The 60-day extension uses seeded synthetic data, not measured traffic. The short 10-day series and days 4–10 validation window make forecast ranking uncertain.

## AI-use declaration
An AI coding assistant helped draft and check the notebook, module, tests and documentation. 