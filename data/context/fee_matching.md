# Fee Matching Configuration

This document describes how to match fee conditions against payment and merchant data. Each fee condition has specific matching rules that must be evaluated to determine if a fee applies to a given transaction.

## Fee Matching Rules

### merchant_category_code
- **Type**: List of strings
- **Null behavior**: When null, fee applies to all merchant category codes
- **Data sources**:
  - `merchant_data.mcc` - Direct MCC value from merchant data
  - `payments.merchant` - Requires lookup in merchant_data.json to get MCC
- **Matching logic**: The merchant's MCC must exactly match one of the values in the list

### monthly_fraud_level
- **Type**: Percentage range (e.g., "0-5%", "5-10%")
- **Null behavior**: When null, fee applies to all fraud levels
- **Calculation**: `(fraudulent_transactions_count / total_transactions_count) * 100` for the month
- **Data sources**:
  - `payments.has_fraudulent_dispute` - Boolean indicating if transaction is fraudulent
  - `day_of_year` - Used to determine the month for aggregation
- **Matching logic**: Calculate monthly fraud percentage and check if it falls within the specified range

### monthly_volume
- **Type**: Amount range (e.g., "0-10000", "10000-50000")
- **Null behavior**: When null, fee applies to all volume levels
- **Calculation**: Sum of all `payments.eur_amount` for the merchant in that month
- **Data sources**:
  - `payments.eur_amount` - Transaction amount in EUR
  - `day_of_year` - Used to determine the month for aggregation
- **Matching logic**: Calculate monthly total volume and check if it falls within the specified range

### is_credit
- **Type**: Boolean (true/false)
- **Null behavior**: When null, fee applies to both credit and debit transactions
- **Data sources**:
  - `payments.is_credit` - Boolean indicating if transaction is credit
- **Matching logic**: Direct boolean value matching

### intracountry
- **Type**: Boolean (1.0 for true, 0.0 for false)
- **Null behavior**: When null, fee applies to both domestic and international transactions
- **Calculation**: 
  - `1.0` when `payments.issuing_country == payments.acquirer_country`
  - `0.0` when `payments.issuing_country != payments.acquirer_country`
- **Data sources**:
  - `payments.issuing_country` - Country where card was issued
  - `payments.acquirer_country` - Country of the acquiring bank
- **Matching logic**: Compare the calculated intracountry value with the fee condition

### aci
- **Type**: List of strings
- **Empty behavior**: When empty list, fee applies to all ACI values
- **Data sources**:
  - `payments.aci` - Authorization Characteristics Indicator
- **Matching logic**: The payment's ACI must exactly match one of the values in the list

### account_type
- **Type**: List of strings
- **Empty behavior**: When empty list, fee applies to all account types
- **Data sources**:
  - `merchant_data.account_type` - Type of merchant account
- **Matching logic**: The merchant's account type must exactly match one of the values in the list

### capture_delay
- **Type**: Range (e.g., "<3", "3-5") or string (e.g., "manual")
- **Null behavior**: When null, fee applies to all capture delays
- **Data sources**:
  - `merchant_data.capture_delay` - Number of days or "manual"
- **Matching logic**:
  - For ranges: Check if the merchant's capture_delay falls within the specified range
  - For strings: Exact string matching (e.g., "manual")

## General Matching Principles

1. **Null/Empty Values**: When a fee condition is null or empty, it acts as a wildcard and matches all possible values for that field.

2. **Data Lookups**: Some conditions require joining data from multiple sources:
   - Merchant information requires looking up merchant_data.json using the merchant identifier
   - Monthly calculations require aggregating data by month using day_of_year

3. **Calculation Requirements**: Some conditions require real-time calculations:
   - Monthly fraud level percentage
   - Monthly transaction volume
   - Intracountry status determination
