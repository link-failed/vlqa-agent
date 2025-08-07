"""
What is the fee ID or IDs that apply to account_type = R and aci = B?
Answer must be a list of values in comma separated list, eg: A, B, C. If the answer is an empty list, reply with an empty string. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'.
"""

import json

# Load the fees data
with open('./test_case/dabstep_data/fees.json', 'r') as file:
    fees_data = json.load(file)

# Define the filtering criteria
account_type = 'R'
aci = 'B'

fee_ids = []

for entry in fees_data:
    # Check if the entry has the required fields
    if 'account_type' in entry and 'aci' in entry and 'ID' in entry:
        # Check if account_type contains 'R' or is None/empty, and aci contains 'B' or is None/empty
        # If a field is set to null it means that it applies to all possible values of that field. E.g. null value in aci means that the rules applies for all possible values of aci.
        if ((entry['account_type'] is None or entry['account_type'] == [] or 'R' in entry['account_type']) and \
            (entry['aci'] is None or entry['aci'] == [] or 'B' in entry['aci'])):
            fee_ids.append(str(entry['ID']))

# Format the answer
if fee_ids:
    answer = ', '.join(fee_ids)
else:
    answer = 'Not Applicable'

print(answer)
