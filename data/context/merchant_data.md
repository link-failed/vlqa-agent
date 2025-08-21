{
  "columns": {
    "merchant": {
      "data_type": "string",
      "data_sample": ["Crossfit_Hanna", "Belles_cookbook_store"],
      "description": "Merchant name."
    },
    "capture_delay": {
      "data_type": "string to indicate type or number or range",
      "data_sample": ["manual", "immediate", "1", "2-3", "<7"],
      "description": "Capture delay for settlement, can be a label or a number.",
      "domain_knowledge": "Time from authorization to settlement. Faster capture (e.g. immediate) is more expensive."
    },
    "acquirer": {
      "data_type": "list of strings",
      "data_sample": [["gringotts", "medici"], ["bank_of_springfield"]],
      "description": "List of acquirers for the merchant.",
      "domain_knowledge": "Acquirer processes card payments. Local acquiring (same country as issuer) reduces fees."
    },
    "merchant_category_code": {
      "data_type": "integer",
      "data_sample": [7997, 5812, 5942],
      "description": "Merchant Category Code (MCC) for the merchant.",
      "domain_knowledge": "MCC categorizes business, affects risk, fraud, and fees."
    },
    "account_type": {
      "data_type": "string",
      "data_sample": ["F", "H", "R", "D", "S"],
      "description": "Account type for the merchant.",
      "domain_knowledge": "Account type (R, D, H, F, S, O) is based on business model/industry and affects fee rules."
    }
  }
}
