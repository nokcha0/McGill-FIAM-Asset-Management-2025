"""
This code snippet takes a 4-month lead of the accounting ratios and adds back to the original sample
The lead ratios can be used as left-hand-side variables
"""

import pandas as pd

# read the original sample
data = pd.read_csv(
    "/teamspace/uploads/ret_sample.csv",
    parse_dates=["date", "ret_eom", "char_date", "char_eom"],
    low_memory=False,
)

# read the list of ratios
ratio_list = pd.read_csv("/teamspace/uploads/acc_ratios.csv")
ratio_list = ratio_list["Variable"].tolist()

# keep the necessary identifiers
ratios = data[["id", "char_eom"] + ratio_list]

# move the ratios 4 month ahead
ratios = ratios.copy()
ratios["char_eom"] = ratios["char_eom"] + pd.DateOffset(months=4)
ratios = ratios.rename(columns={col: col + "_lead4" for col in ratio_list})
ratios = ratios.rename(
    columns={"char_eom": "ret_eom"}
)  # to merge with the correct date for left-hand-side variables

# merge with the original data and save
final_data = pd.merge(data, ratios, on=["id", "ret_eom"], how="left")
final_data.to_csv("output/data/final_data.csv", index=False)
