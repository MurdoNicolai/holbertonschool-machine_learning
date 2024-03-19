-- list all show by rating
SELECT ts.title AS title, SUM(tsr.rate) AS rating
FROM tv_shows AS ts
INNER JOIN tv_show_ratings AS tsr ON ts.id = tsr.show_id
GROUP BY title
ORDER BY rating DESC;
