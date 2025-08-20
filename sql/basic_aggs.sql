SELECT workclass,
    COUNT(*) AS cnt,
    AVG("hours-per-week") AS avg_hours,
    AVG(age) AS avg_age,
    AVG(income) AS high_income_prob
FROM adult
GROUP BY workclass
ORDER BY cnt DESC;