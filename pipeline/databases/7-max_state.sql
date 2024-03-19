-- select mac temperture per state
SELECT State, MAX(value) AS max_temp
FROM temperatures
GROUP BY State
ORDER BY State ASC;
