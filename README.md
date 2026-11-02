# GridXAI-Diffusion

## About the project

I built **GridXAI-Diffusion** as an interpretable smart-grid cybersecurity
prototype. It uses sensor readings and the relationships between grid nodes to
identify unusual readings, rank them with an anomaly score, and explain why
they were flagged.

The project currently works with synthetic data. I use it to explore how
graph-based analysis can support anomaly detection in smart-grid monitoring
without treating the model as a black box.

## What it does

- Generates a reproducible synthetic smart-grid dataset.
- Preprocesses voltage, current, frequency, and power measurements.
- Builds a graph of the grid nodes.
- Propagates anomaly evidence through the graph using diffusion.
- Flags readings above a configurable percentile threshold.
- Shows high-risk readings and explanations in a Streamlit dashboard.
- Provides separate pages for data exploration, graph analysis,
  explainability, and model evaluation.

## Tech stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- NetworkX
- Plotly, Matplotlib, and Seaborn
- Pytest

## Getting started

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the dependencies

```powershell
pip install -r requirements.txt
```

### 3. Generate the dataset

The dashboard expects the generated file at
`data/smart_grid_data.csv`.

```powershell
python data/generate_data.py
```

### 4. Start the dashboard

```powershell
streamlit run app.py
```

Streamlit will print a local URL that can be opened in a browser.

## Run the tests

```powershell
pytest
```

## Project structure

```text
GridXAl-diffusion/
├── app.py                         # Main Streamlit dashboard
├── data/
│   ├── generate_data.py           # Synthetic data generator
│   └── smart_grid_data.csv        # Generated dataset
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Graph_Analysis.py
│   ├── 3_Explainability.py
│   └── 4_Model_Evaluation.py
├── src/
│   ├── preprocessing.py
│   ├── graph_builder.py
│   ├── gridxai_diffusion.py
│   ├── explainability.py
│   ├── evaluation.py
│   └── visualization.py
└── tests/
```

## How the detection works

1. I scale the sensor features so that they can be compared.
2. I calculate a local sensor score from the scaled measurements.
3. I create a graph representing the grid-node connectivity.
4. I diffuse node-level evidence through that graph.
5. I combine the local and diffused scores into one anomaly score.
6. I flag the highest-scoring readings using a percentile threshold.

## Important limitation

This is a research and learning prototype that uses synthetic data. It has
not been validated for operational grid security, safety-critical decisions,
or deployment on a real power system.