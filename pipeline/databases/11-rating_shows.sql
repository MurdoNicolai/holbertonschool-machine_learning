-- list all show by rating
SELECT ts.title, SUM(tsr.rate) AS rating_sum
FROM tv_shows AS ts
INNER JOIN tv_show_ratings AS tsr ON ts.id = tsr.show_id
GROUP BY ts.title
ORDER BY rating_sum DESC;
