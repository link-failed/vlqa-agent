"""
For credit transactions, what would be the average fee that the card scheme GlobalCard would charge for a transaction value of 10 EUR?
Answer must be just a number expressed in EUR rounded to 6 decimals. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'.
"""

import json
import os

def calculate_globalcard_credit_fee(schema=None):
    # Load fees data
    fees_file = os.path.join(os.path.dirname(__file__), '.', 'test_case', 'dabstep_data', 'fees.json')

    with open(fees_file, 'r') as f:
        fees = json.load(f)

    # Match all types if schema is empty
    schema = schema or [fee.get('card_scheme') for fee in fees]

    # Filter for GlobalCard credit transactions or all types if is_credit is None
    globalcard_credit_fees = [
        fee for fee in fees
        if (fee.get('card_scheme') in [None, "GlobalCard"]) \
           and (fee.get('is_credit') is None or fee.get('is_credit'))
    ]

    if not globalcard_credit_fees:
        return 'Not Applicable'

    transaction_amount = 10.0
    total_fee = sum(
        fee.get('fixed_amount', 0) + (fee.get('rate', 0) / 10000) * transaction_amount
        for fee in globalcard_credit_fees
    )

    return round(total_fee / len(globalcard_credit_fees), 6)

# The answer must be directly stored in answer variable
answer = calculate_globalcard_credit_fee()
print(answer)
