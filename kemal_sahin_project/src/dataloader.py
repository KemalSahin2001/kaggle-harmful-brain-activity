# src/data_loader.py
import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    return data

# Additional functions for data loading can be added here.
