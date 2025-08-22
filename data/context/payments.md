# Payment Transaction Data Dictionary

This document describes the columns in the payment transaction dataset, including their descriptions, domain knowledge, data types, and sample values.

## Columns

### psp_reference
- **Description**: Unique identifier for each payment transaction, used for tracking and reconciliation.
- **Data Type**: string
- **Sample Values**: `20034594130`, `36926127356`
- **Domain Knowledge**: 
  - Each payment is assigned a unique reference for traceability and dispute resolution.

### merchant
- **Description**: Name of the merchant processing the transaction; may be a brand or platform.
- **Data Type**: string
- **Sample Values**: `Crossfit_Hanna`, `Belles_cookbook_store`, `Golfclub_Baron_Friso`
- **Relationships**: 
  - Exact match against merchant names in `merchant_data.merchant`
- **Domain Knowledge**: 
  - Merchant names are used for reporting, analytics, and fee calculation.
  - Merchant names may be mapped to merchant category codes (MCC) for risk and pricing.

### card_scheme
- **Description**: Card network or scheme used for the transaction (e.g., MasterCard, Visa, Amex, Other).
- **Data Type**: string
- **Sample Values**: `NexPay`, `GlobalCard`, `SwiftCharge`, `TransactPlus`
- **Domain Knowledge**: 
  - Card schemes determine network rules, interchange rates, and acceptance policies.

### year
- **Description**: Year when the payment was initiated.
- **Data Type**: integer
- **Sample Values**: `2023`
- **Domain Knowledge**: 
  - Year is used for time-based reporting and trend analysis.

### hour_of_day
- **Description**: Hour of the day (0-23) when the payment was initiated.
- **Data Type**: integer
- **Sample Values**: `16`, `23`, `4`, `3`
- **Domain Knowledge**: 
  - Hour of payment can reveal peak transaction times and consumer behavior.

### minute_of_hour
- **Description**: Minute of the hour (0-59) when the payment was initiated.
- **Data Type**: integer
- **Sample Values**: `21`, `58`, `30`, `5`
- **Domain Knowledge**: 
  - Minute granularity can be useful for fraud detection and system monitoring.

### day_of_year
- **Description**: Day of the year (1-366) when the payment was initiated.
- **Data Type**: integer
- **Sample Values**: `12`, `75`, `96`, `77`
- **Domain Knowledge**: 
  - Day of year is used for aggregating transactions by month or season.

### is_credit
- **Description**: Boolean indicator if the transaction used a credit card (True) or not (False). Values are 'True' or 'False' as strings in the CSV.
- **Data Type**: string
- **Sample Values**: `True`, `False`
- **Domain Knowledge**: 
  - Credit transactions may incur higher fees and different risk profiles than debit.
  - Fee rules may depend on whether the transaction is credit or debit.

### eur_amount
- **Description**: Transaction amount in euros.
- **Data Type**: float
- **Sample Values**: `151.74`, `45.7`, `14.11`, `238.42`
- **Possible Formulas**: 
  - `monthly_volume = sum(eur_amount) over month by merchant`
- **Domain Knowledge**: 
  - Transaction value is used for fee calculation, reporting, and risk assessment.

### ip_country
- **Description**: Country where the shopper was located at the time of transaction, determined by IP address.
- **Data Type**: string
- **Sample Values**: `SE`, `NL`, `LU`, `IT`, `BE`, `FR`, `GR`, `ES`
- **Domain Knowledge**: 
  - IP country can be used for fraud detection, compliance, and geo-based analytics.

### issuing_country
- **Description**: Country of the bank that issued the card used in the transaction.
- **Data Type**: string
- **Sample Values**: `SE`, `NL`, `LU`, `IT`, `BE`, `FR`, `GR`, `ES`
- **Possible Formulas**: 
  - `intracountry = 1.0 if issuing_country == acquirer_country else 0.0`
- **Domain Knowledge**: 
  - Issuing country is used to determine if a transaction is domestic or cross-border, impacting fees and risk.

### device_type
- **Description**: Type of device used by the shopper (e.g., Windows, Linux, MacOS, iOS, Android, Other).
- **Data Type**: string
- **Sample Values**: `Windows`, `Linux`, `MacOS`, `iOS`, `Android`, `Other`
- **Domain Knowledge**: 
  - Device type can indicate channel (mobile, desktop) and is useful for fraud analytics.

### ip_address
- **Description**: Hashed value of the shopper's IP address for privacy and tracking.
- **Data Type**: string
- **Sample Values**: `pKPYzJqqwB8TdpY0jiAeQw`, `uzUknOkIqExYsWv4X14GUg`, `3VO1v_RndDg6jzEiPjfvoQ`
- **Domain Knowledge**: 
  - IP address is anonymized for privacy but can be used for session tracking and fraud detection.

### email_address
- **Description**: Hashed value of the shopper's email address for privacy and deduplication.
- **Data Type**: string
- **Sample Values**: `0AKXyaTjW7H4m1hOWmOKBQ`, `_Gm8at1k2ojYAM_wSEptNw`, `hzw3CbkxazpVg38re7jchQ`
- **Domain Knowledge**: 
  - Email address is anonymized for privacy and can be used for identifying repeat customers or fraud.

### card_number
- **Description**: Hashed value of the card number for privacy and security.
- **Data Type**: string
- **Sample Values**: `uRofX46FuLUrSOTz8AW5UQ`, `6vqQ89zfCeFk6s4VOoWZFQ`, `EmxSN8-GXQw3RG_2v7xKxQ`
- **Domain Knowledge**: 
  - Card number is hashed to protect sensitive data and prevent misuse.

### shopper_interaction
- **Description**: Type of shopper interaction: 'Ecommerce' (online) or 'POS' (in-person/in-store).
- **Data Type**: string
- **Sample Values**: `Ecommerce`, `POS`
- **Domain Knowledge**: 
  - Shopper interaction type affects risk, authentication requirements, and fee structure.

### card_bin
- **Description**: Bank Identification Number (BIN) of the card, used to identify issuing bank and card type.
- **Data Type**: string
- **Sample Values**: `4802`, `4920`, `4571`, `4017`
- **Domain Knowledge**: 
  - BIN is used for card scheme routing, fraud detection, and analytics.

### has_fraudulent_dispute
- **Description**: Boolean indicator if the transaction was disputed as fraudulent by the issuing bank. Values are 'True' or 'False' as strings in the CSV.
- **Data Type**: string
- **Sample Values**: `True`, `False`
- **Possible Formulas**: 
  - `monthly_fraud_rate = sum(has_fraudulent_dispute == 'True') / count(*) over month by merchant`
- **Domain Knowledge**: 
  - Fraudulent disputes impact merchant fees, risk scoring, and may trigger reviews.

### is_refused_by_adyen
- **Description**: Boolean indicator if the transaction was refused by Adyen (the payment processor). Values are 'True' or 'False' as strings in the CSV.
- **Data Type**: string
- **Sample Values**: `True`, `False`
- **Domain Knowledge**: 
  - Refusals may be due to risk, compliance, or technical issues and are important for monitoring approval rates.

### aci
- **Description**: Authorization Characteristics Indicator (ACI) code describing transaction authentication context.
- **Data Type**: string
- **Sample Values**: `F`, `D`, `G`, `E`, `B`, `A`
- **Domain Knowledge**: 
  - ACI codes indicate card presence, authentication method, and risk level.
  - Correct ACI selection can optimize fees and reduce transaction friction.

### acquirer_country
- **Description**: Country where the acquiring bank (the merchant's bank) is located.
- **Data Type**: string
- **Sample Values**: `NL`, `SE`, `IT`, `LU`, `BE`, `FR`, `GR`, `ES`
- **Possible Formulas**: 
  - `intracountry = 1.0 if issuing_country == acquirer_country else 0.0`
- **Domain Knowledge**: 
  - Acquirer country is used to determine if a transaction is local or cross-border, impacting fees and compliance.
