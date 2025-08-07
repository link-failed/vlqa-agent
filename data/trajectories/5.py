"""
Which issuing country has the highest number of transactions?
Answer must be just the country code. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'.
"""

import pandas as pd

# Load the payments data
payments_df = pd.read_csv('./test_case/dabstep_data/payments.csv')

# Count transactions by issuing country
country_counts = payments_df['issuing_country'].value_counts()

# Get the country with the highest number of transactions
highest_country = country_counts.index[0]

# The answer must be directly stored in answer variable.
answer = highest_country
    