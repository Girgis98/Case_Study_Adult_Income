
SELECT workclass,
    COUNT(*) AS cnt,
    AVG(income) AS high_income_prob
FROM adult
GROUP BY workclass
ORDER BY high_income_prob DESC
LIMIT 5;