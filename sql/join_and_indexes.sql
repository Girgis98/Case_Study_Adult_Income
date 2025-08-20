SELECT
  cr.region,
  COUNT(*)             AS cnt,
  AVG(income)          AS pct_high_income,
  AVG("hours-per-week") AS avg_hours
FROM adult a
LEFT JOIN country_region cr
  ON a."native-country" = cr.native_country
GROUP BY cr.region
ORDER BY cnt DESC;
