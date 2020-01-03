import pandas as pd
import numpy as np

variables_df = pd.read_csv("data/internal/variables/simulation_variables.csv", names=["variable","value"])

variables_df = variables_df.set_index("variable", inplace=False)

print(variables_df)

print(variables_df.loc["proportion_transfer"].value)

print(variables_df.loc["angio_reperfusion_time_median"].value)
