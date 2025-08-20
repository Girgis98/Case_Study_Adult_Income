WITH agg AS (
  SELECT workclass, occupation,
         AVG("hours-per-week") AS avg_hours,
         COUNT(*)            AS cnt
  FROM adult
  GROUP BY workclass, occupation
)
SELECT workclass, occupation, avg_hours, cnt,
       RANK() OVER (PARTITION BY workclass ORDER BY avg_hours DESC) AS rnk
FROM agg
WHERE cnt >= 100
ORDER BY workclass, rnk;
