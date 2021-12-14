```sql
LEFT JOIN RangedExchangeRates RER 
ON (T.Date > RER.DateFrom) AND (T.Date <= RER.DateTo)
```