# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Recipe inputs
input = dataiku.Dataset("input_dataset")
input_df = input.get_dataframe()

# Convert to json
input_json = input_df.to_json()

# Convert json to a one row, one column data frame
input_json_df = pd.DataFrame(data=[input_json], columns=['json'])

# Write new data frame back to Dataiku dataset
output = dataiku.Dataset("output_dataset")
output.write_with_schema(input_json_df)

# Write json to external file
f = open('output_file', 'w')
f.write(input_json)
f.close()