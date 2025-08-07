"""
For account type H and the MCC description: Eating Places and Restaurants, what would be the average fee that the card scheme GlobalCard would charge for a transaction value of 10 EUR? Provide the answer in EUR and 6 decimals.
Answer must be just a number expressed in EUR rounded to 6 decimals. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'.
"""

import json
import csv

"""
If a field is set to null it means that it applies to all possible values of that field. E.g. null value in aci means that the rules applies for all possible values of aci.
"""

# Load fees data
with open('./test_case/dabstep_data/fees.json', 'r') as f:
    fees_data = json.load(f)

# Load merchant category codes
with open('./test_case/dabstep_data/merchant_category_codes.csv', 'r') as f:
    mcc_reader = csv.DictReader(f)
    mcc_map = {row['mcc']: row['description'] for row in mcc_reader}
    
# Filter MCC for Eating Places and Restaurants
eating_places_mcc = [mcc for mcc, desc in mcc_map.items() if desc == "Eating Places and Restaurants"]

# Filter fees for GlobalCard, account type H, and relevant MCC, considering null values and lists for all fields
# If a field is set to null it means that it applies to all possible values of that field. E.g. null value in aci means that the rules applies for all possible values of aci.
relevant_fees = [
    fee for fee in fees_data
    if (fee['card_scheme'] == "" or fee['card_scheme'] == 'GlobalCard') and (
        fee['account_type'] == [] or 'H' in fee['account_type']
    ) and (
        fee['merchant_category_code'] == [] or any(mcc in fee['merchant_category_code'] for mcc in map(int, eating_places_mcc))
    )
]

if not relevant_fees:
    print("Not Applicable")
    exit()

# Calculate fees for a transaction value of 10 EUR using fees.json
transaction_value = 10
fees = []
for fee in relevant_fees:
    fixed_fee = fee['fixed_amount']
    # fee = fixed_amount + rate * transaction_value / 10000
    rate_fee = (fee['rate'] / 10000) * transaction_value
    total_fee = fixed_fee + rate_fee
    fees.append(total_fee)
    
if not fees:
    answer = "Not Applicable"
else:
    average_fee = sum(fees) / len(fees)
    # the answer must be directly stored in answer variable
    answer = f"{average_fee:.6f}"