# Fraud assumptions
This document contains assumptions regarding fraud are used in the data generation process.

- Fraud is highly dynamic in nature, i.e., new types of fraud are encountered frequently.
    - S1: once fraudsters learn that a certain type of fraud is known/blocked they will switch to a new approach, akin to whack-a-mole
- Fraud is imbalanced, ranging from highly imbalanced to slightly imbalanced.
    - S1: roughly 1 / 1e5 payments will be fraud, for insurance fraud the ratio is expected to be closer to 25%

#### Sources:
S1: A participant in the DEP program works on payment fraud for a financial institution.

# Suggestions:
- labels are not binary but also indicate if it was a reasonable suspect for fraud
