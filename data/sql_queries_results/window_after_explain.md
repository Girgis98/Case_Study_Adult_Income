|   id |   parent |   notused | detail                               |
|-----:|---------:|----------:|:-------------------------------------|
|    3 |        0 |         0 | CO-ROUTINE (subquery-3)              |
|    6 |        3 |         0 | CO-ROUTINE (subquery-4)              |
|    8 |        6 |         0 | CO-ROUTINE agg                       |
|   15 |        8 |       223 | SCAN adult USING INDEX idx_workclass |
|   18 |        8 |         0 | USE TEMP B-TREE FOR GROUP BY         |
|   62 |        6 |        82 | SCAN agg                             |
|   76 |        6 |         0 | USE TEMP B-TREE FOR ORDER BY         |
|   95 |        3 |        81 | SCAN (subquery-4)                    |
|  179 |        0 |        81 | SCAN (subquery-3)                    |
|  255 |        0 |         0 | USE TEMP B-TREE FOR ORDER BY         |