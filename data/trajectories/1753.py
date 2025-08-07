"""
What are the applicable fee IDs for Belles_cookbook_store in March 2023?
Answer must be a list of values in comma separated list, eg: A, B, C. If the answer is an empty list, reply with an empty string. If a question does not have a relevant or applicable answer for the task, please respond with 'Not Applicable'.
"""

"""
If a field is set to null it means that it applies to all possible values of that field. E.g. null value in aci means that the rules applies for all possible values of aci.
"""

import json
import csv
import re
from datetime import datetime

# Load fees data from the JSON file
with open('./test_case/dabstep_data/fees.json', 'r') as f:
    fees_data = json.load(f)

def parse_capture_delay(fee_capture_delay):
    """Parse the capture_delay value from fees.json to handle fuzzy match cases."""
    if fee_capture_delay is None:
        return range(0, 100)  # Apply to all possible values, assuming 0-99 as practical range

    if isinstance(fee_capture_delay, str):
        if fee_capture_delay.startswith("<"):
            upper_limit = int(fee_capture_delay[1:])
            return range(0, upper_limit)
        elif fee_capture_delay.startswith(">"):
            lower_limit = int(fee_capture_delay[1:]) + 1
            return range(lower_limit, 100)  # Assuming 100 as a practical upper bound
        elif "-" in fee_capture_delay:
            lower_limit, upper_limit = map(int, fee_capture_delay.split("-"))
            return range(lower_limit, upper_limit + 1)
    return [str(fee_capture_delay)]

def parse_percentage_range(percentage_range):
    """Parse a percentage range string like '<8.3%' or '7.2%-7.7%' into a tuple."""
    if percentage_range is None:
        return None
    if percentage_range.startswith('<'):
        return (0, float(percentage_range[1:-1]))
    elif percentage_range.startswith('>'):
        return (float(percentage_range[1:-1]), float('inf'))
    elif '-' in percentage_range:
        lower, upper = map(float, re.findall(r'\d+\.\d+', percentage_range))
        return (lower, upper)
    return None

def parse_volume_range(volume_range):
    """Parse a volume range string like '<100k' or '1m-5m' into a tuple."""
    if volume_range is None:
        return None
    multiplier = {'k': 1_000, 'm': 1_000_000}
    if volume_range.startswith('<'):
        value = float(re.findall(r'\d+', volume_range)[0]) * multiplier[volume_range[-1]]
        return (0, value)
    elif volume_range.startswith('>'):
        value = float(re.findall(r'\d+', volume_range)[0]) * multiplier[volume_range[-1]]
        return (value, float('inf'))
    elif '-' in volume_range:
        lower, upper = re.findall(r'\d+[km]', volume_range)
        lower_value = float(re.findall(r'\d+', lower)[0]) * multiplier[lower[-1]]
        upper_value = float(re.findall(r'\d+', upper)[0]) * multiplier[upper[-1]]
        return (lower_value, upper_value)
    return None

def is_within_range(value, range_tuple):
    """Check if a value is within a given range tuple."""
    if range_tuple is None:
        return True
    return range_tuple[0] <= value <= range_tuple[1]

def get_applicable_fee_ids():
    # Define characteristics for Belles_cookbook_store in March 2023
    target_merchant = "Belles_cookbook_store"
    target_year = 2023

    # Calculate the range of days for March 2023
    march_start = datetime(target_year, 3, 1).timetuple().tm_yday
    march_end = datetime(target_year, 3, 31).timetuple().tm_yday

    # Load merchant data
    with open('./test_case/dabstep_data/merchant_data.json', 'r') as f:
        merchant_data = json.load(f)

    # Extract Belles_cookbook_store's information from merchant_data
    merchant_info = next((m for m in merchant_data if m['merchant'] == target_merchant), None)
    if not merchant_info:
        print("Merchant information not found.")
        return "Not Applicable"

    capture_delay = str(merchant_info["capture_delay"])  # Normalize capture_delay to string
    merchant_category_code = merchant_info['merchant_category_code']
    account_type = merchant_info['account_type']

    # Load payments data
    with open('./test_case/dabstep_data/payments.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        relevant_transactions = [
            row for row in reader
            if row['merchant'] == target_merchant and march_start <= int(row['day_of_year']) <= march_end and int(row['year']) == target_year
        ]

    # Calculate monthly fraud level and volume
    total_volume = sum(float(tx['eur_amount']) for tx in relevant_transactions)
    fraud_transactions = [tx for tx in relevant_transactions if tx['has_fraudulent_dispute'] == 'TRUE']
    fraud_volume = sum(float(tx['eur_amount']) for tx in fraud_transactions)
    monthly_fraud_level = (fraud_volume / total_volume) * 100 if total_volume > 0 else 0

    applicable_fees = set()

    for transaction in relevant_transactions:
        transaction_aci = transaction['aci'].strip()
        transaction_card_scheme = transaction['card_scheme'].strip()
        transaction_is_credit = (
            None if transaction['is_credit'] == '' else
            True if transaction['is_credit'] == 'TRUE' else
            False
        )
        # intracountry is 1.0 if issuer country == acquiring country, and 0.0 when they are different
        transaction_intracountry = 1.0 if transaction['issuing_country'] == transaction['acquirer_country'] else 0.0

        for fee in fees_data:
            fee_aci_list = fee.get('aci', [])
            fee_card_scheme = fee.get('card_scheme')
            fee_is_credit = fee.get('is_credit')
            fee_intracountry = fee.get('intracountry')
            fee_capture_delay = fee.get('capture_delay')
            fee_capture_delay_range = parse_capture_delay(fee_capture_delay)
            fee_account_type_list = fee.get('account_type', [])
            fee_merchant_category_code_list = fee.get('merchant_category_code', [])
            # Monthly volumes and rates are computed always in natural months (e.g. January, February), starting always in day 1 and ending in the last natural day of the month (i.e. 28 for February, 30 or 31).
            fee_monthly_fraud_level_range = parse_percentage_range(fee.get('monthly_fraud_level'))
            fee_monthly_volume_range = parse_volume_range(fee.get('monthly_volume'))

            # If a field is set to null it means that it applies to all possible values of that field. E.g. null value in aci means that the rules applies for all possible values of aci.
            if (
                (fee_aci_list == [] or transaction_aci in fee_aci_list) and
                transaction_card_scheme == fee_card_scheme and
                (fee_is_credit is None or fee_is_credit == transaction_is_credit) and
                (fee_intracountry is None or fee_intracountry == transaction_intracountry) and
                # "capture_delay": string type. Possible values are '3-5' (between 3 and 5 days), '>5', '3', 'immediate', or 'manual'
                (fee_capture_delay_range is None or str(capture_delay) in map(str, fee_capture_delay_range)) and
                (fee_account_type_list == [] or account_type in fee_account_type_list) and
                (fee_merchant_category_code_list == [] or str(merchant_category_code) in fee_merchant_category_code_list) and
                is_within_range(monthly_fraud_level, fee_monthly_fraud_level_range) and
                is_within_range(total_volume, fee_monthly_volume_range)
            ):
                fee_id = str(fee['ID'])
                applicable_fees.add(fee_id)

    if not applicable_fees:
        return ""
    return ", ".join(sorted(applicable_fees))

if __name__ == "__main__":
    # the answer must be directly stored in answer variable, without any other chars
    answer = get_applicable_fee_ids()
    answer = answer if answer else "Not Applicable"
    print(answer)