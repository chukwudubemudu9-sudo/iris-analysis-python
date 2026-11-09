# Iris analysis in Python — DTS 201

Exploratory analysis of the Iris dataset and a k-nearest-neighbours (k-NN) classifier. The notebook includes saved outputs; the script reproduces the analysis and writes charts, tables, and model metrics to `results/`.

## Run the script

From this folder:

```bash
python -m pip install -r requirements.txt
python iris_analysis.py
```

To open the notebook, install Jupyter and run `iris_analysis.ipynb`. The bundled `iris.csv` supports offline execution.

## Recorded result

The stratified test split contains 45 flowers. Training-only cross-validation selects `k = 7`; the saved test accuracy is 42/45 (93.3%).

